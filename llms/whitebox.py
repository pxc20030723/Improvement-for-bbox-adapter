import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate.state import PartialState
from accelerate.utils import release_memory, InitProcessGroupKwargs
from peft import LoraConfig, TaskType, get_peft_model
import datasets
from datasets import Dataset
datasets.disable_progress_bar()
from datetime import timedelta
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_LOG_MODEL"] = "false"

from tqdm.auto import tqdm

from utils.util import get_answer_start_idx
from utils.loggers import loggers

from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_constant_schedule_with_warmup,
)

from accelerate import Accelerator

torch.cuda.empty_cache()
torch.set_printoptions(threshold=10_000)


class MoEScoringHead(nn.Module):
    """
    Shared encoder + multiple scoring heads.
    The input is the pooled representation from the encoder and the output keeps
    the same scalar-logit interface as the original classifier head.
    """
    def __init__(self, hidden_size, config):
        super().__init__()
        self.num_experts = config.get("moe_num_experts", 4)
        gating_hidden = config.get("moe_gating_hidden_size") or hidden_size
        expert_hidden = config.get("moe_expert_hidden_size") or hidden_size
        dropout = config.get("moe_dropout", 0.1)
        self.load_balance_coef = config.get("moe_load_balance_coef", 0.0)
        self.entropy_coef = config.get("moe_entropy_coef", 0.0)
        self.score_aware_gate = config.get("moe_score_aware_gate", False)
        self.expert_dropout = config.get("moe_expert_dropout", 0.0)
        self.top_k = config.get("moe_top_k")


        gate_input_size = hidden_size + self.num_experts if self.score_aware_gate else hidden_size

        self.gating_net = nn.Sequential(
            nn.Linear(gate_input_size, gating_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, self.num_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, 1),
            )
            for _ in range(self.num_experts)
        ])

        self.latest_aux_loss = None
        self.latest_gate_probs = None
        self.latest_expert_scores = None
        self.latest_expert_usage = None

    def forward(self, pooled_representation):
        expert_scores = torch.cat(
            [expert(pooled_representation) for expert in self.experts],
            dim=-1,
        )
        if self.score_aware_gate:
          gate_input = torch.cat([pooled_representation, expert_scores.detach()], dim=-1)
        else:
          gate_input = pooled_representation

        gate_logits = self.gating_net(gate_input)
        if self.training and self.expert_dropout > 0:
            drop_mask = torch.rand(self.num_experts, device=gate_logits.device) < self.expert_dropout
            if drop_mask.all():
                keep_idx = torch.randint(self.num_experts, (1,), device=gate_logits.device)
                drop_mask[keep_idx] = False
            gate_logits = gate_logits.masked_fill(drop_mask.unsqueeze(0), -1e4)

        gate_probs = torch.softmax(gate_logits, dim=-1)
        top_k = self.config.get("moe_top_k")
        if top_k is not None and top_k < self.num_experts:
            top_values, top_indices = torch.topk(gate_probs, k=top_k, dim=-1)

            sparse_gate = torch.zeros_like(gate_probs)
            sparse_gate.scatter_(dim=-1, index=top_indices, src=top_values)

            gate_probs = sparse_gate / sparse_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        
        final_score = (gate_probs * expert_scores).sum(dim=-1, keepdim=True)
        expert_usage = gate_probs.mean(dim=0)

        self.latest_gate_probs = gate_probs.detach()
        self.latest_expert_scores = expert_scores.detach()
        self.latest_expert_usage = expert_usage.detach()

        aux_loss = pooled_representation.new_tensor(0.0)
        if self.load_balance_coef > 0:
            uniform = torch.full_like(expert_usage, 1.0 / self.num_experts)
            aux_loss = aux_loss + self.load_balance_coef * torch.sum((expert_usage - uniform) ** 2)
        if self.entropy_coef > 0:
            entropy = -(gate_probs * torch.log(gate_probs.clamp_min(1e-8))).sum(dim=-1).mean()
            aux_loss = aux_loss + self.entropy_coef * entropy #- -》 +

        self.latest_aux_loss = aux_loss
        return final_score


class Whitebox_LLM:
    """
    This class implements the Whitebox_LLM model, such as Llama / DeBERTa.
    """
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["critic_model"],
            truncation_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=96000))
        self.accelerator = Accelerator(
            split_batches=False,
            mixed_precision="fp16",
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            log_with="wandb" if self.config.get("log_with_wandb", False) else None,
            project_dir="logs" if self.config.get("log_with_wandb", False) else None,
            device_placement=True,
            kwargs_handlers=[kwargs],
        )

        self.mode = config["critic_mode"]
        if self.mode == "generation":
            self.model = AutoModelForCausalLM.from_pretrained(
                config["critic_model"],
                trust_remote_code=True,
            )

        elif self.mode == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config["critic_model"],
                trust_remote_code=True,
                num_labels=1,
            )
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

            if self.config.get("use_moe_head", False):
                self._replace_classifier_with_moe()
                for head_name in ["classifier", "score", "qa_outputs"]:
                    if hasattr(self.model, head_name):
                        self.accelerator.print(
                            f"{head_name} type: {type(getattr(self.model, head_name))}"
                        )
        else:
            raise NotImplementedError

        if self.config.get("use_lora", False):
            self.model = self._apply_lora(self.model)

        self.model.config.use_cache = False if "phi" in self.config["critic_model"].lower() else True
        self.model.config.pretraining_tp = 1

        if self.tokenizer.pad_token is None:
            self.accelerator.print("Adding pad token to the tokenizer...")
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.answer_token = self.tokenizer.encode(
            "\nA: ", return_tensors="pt", add_special_tokens=False
        )[0, 1:]

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["learning_rate"] * self.accelerator.gradient_accumulation_steps,
            weight_decay=0.01,
        )
        self.lr_scheduler = get_constant_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"],
        )

        self.accelerator.print(
            f"Distributed: {self.accelerator.distributed_type}, Mixed precision: {self.accelerator.mixed_precision}"
        )
        if self.config.get("use_lora", False):
            self.model.print_trainable_parameters()

        self.print_parameter_stats()


    def print_parameter_stats(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.print(f"Total params: {total_params:,}")
        self.accelerator.print(f"Trainable params: {trainable_params:,}")


    def _replace_classifier_with_moe(self):
        hidden_size = self.model.config.hidden_size
        moe_head = MoEScoringHead(hidden_size=hidden_size, config=self.config)
        for head_name in ["classifier", "score", "qa_outputs"]:
            if hasattr(self.model, head_name):
                setattr(self.model, head_name, moe_head)
                self.accelerator.print(
                    f"Replaced classifier head `{head_name}` with MoE scoring head ({self.config.get('moe_num_experts', 4)} experts)."
                )
                return
        raise ValueError("Could not find a supported classification head to replace with MoE.")

    def _apply_lora(self, model):
        task_type = TaskType.SEQ_CLS if self.mode == "classification" else TaskType.CAUSAL_LM
        target_modules = self.config.get("lora_target_modules", "all-linear")
        modules_to_save = self.config.get("lora_modules_to_save")

        if modules_to_save is None and self.mode == "classification":
            candidate_modules = ["classifier", "score", "qa_outputs"]
            present_modules = {name for name, _ in model.named_modules()}
            modules_to_save = [name for name in candidate_modules if name in present_modules]

        lora_config = LoraConfig(
            task_type=task_type,
            r=self.config.get("lora_r", 8),
            lora_alpha=self.config.get("lora_alpha", 16),
            lora_dropout=self.config.get("lora_dropout", 0.0),
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            bias=self.config.get("lora_bias", "none"),
        )
        return get_peft_model(model, lora_config)

    @PartialState().on_main_process
    def build_dataset(self, positive_texts, negative_texts, save_to):
        pos_len, neg_len = len(positive_texts), len(negative_texts)
        labels = -torch.ones(pos_len + neg_len)
        labels[:pos_len] = 1.0

        input_texts = positive_texts + negative_texts
        temp_dataset = Dataset.from_dict({
            "texts": input_texts,
            "labels": labels,
        }).with_format("torch")

        batch_dataset = temp_dataset.map(
            lambda x: self.tokenizer(
                x["texts"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=self.config["add_special_tokens"],
            ),
            remove_columns=["texts"],
            batched=True,
        )

        batch_dataset.save_to_disk(save_to)
        print(f"\nDataset saved to {save_to}\n")

        tr_text = (
            f"\npos_len: {pos_len}\nneg_len: {neg_len}\n\n"
            + f"\n\n{'-' * 20}\n\n".join([t + f" (Label: {l})" for t, l in zip(input_texts, labels.tolist())])
        )
        loggers["train"].info(f"\n{'=' * 20}\n{tr_text}\n\n")

    def build_dataloader(self, batch_dataset):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader_params = {
            "batch_size": self.config["batch_size"],
            "collate_fn": data_collator,
            "num_workers": 0,
            "pin_memory": True,
            "shuffle": True,
        }
        batch_dataloader = self.accelerator.prepare(DataLoader(batch_dataset, **dataloader_params))
        return batch_dataloader

    def _get_classifier_head(self, model):
        def unwrap_head(module):
            if module is None:
                return None

            if isinstance(module, MoEScoringHead):
                return module

            if hasattr(module, "modules_to_save") and len(module.modules_to_save) > 0:
                for _, submodule in module.modules_to_save.items():
                    if isinstance(submodule, MoEScoringHead):
                        return submodule

            if hasattr(module, "original_module") and isinstance(module.original_module, MoEScoringHead):
                return module.original_module

            return module

        for head_name in ["classifier", "score", "qa_outputs"]:
            if hasattr(model, head_name):
                return unwrap_head(getattr(model, head_name))

        base_model = getattr(model, "base_model", None)
        if base_model is not None:
            for inner_name in ["model", "base_model"]:
                inner = getattr(base_model, inner_name, None)
                if inner is not None:
                    for head_name in ["classifier", "score", "qa_outputs"]:
                        if hasattr(inner, head_name):
                            return unwrap_head(getattr(inner, head_name))

        return None
    def get_moe_diagnostics(self):
        if not self.config.get("use_moe_head", False):
            return None

        classifier_head = self._get_classifier_head(self.model)
        if classifier_head is None:
            return None

        gate_probs = getattr(classifier_head, "latest_gate_probs", None)
        expert_scores = getattr(classifier_head, "latest_expert_scores", None)

        if gate_probs is None or expert_scores is None:
            return None

        return {
            "gate_probs": gate_probs.detach().cpu(),
            "expert_scores": expert_scores.detach().cpu(),
            "top_expert": gate_probs.argmax(dim=-1).detach().cpu(),
        }


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").type(torch.LongTensor).to(self.accelerator.device)

        inputs = inputs.to(self.accelerator.device)
        outputs = model(**inputs)
        output_logits = outputs.get("logits")

        input_ids = inputs["input_ids"].detach()
        attention_mask = inputs["attention_mask"].detach()

        alpha = self.config["l2_reg_coef"]
        energy_temp = self.config["energy_temp"]
        l2_loss = 0.0
        aux_loss = 0.0
        classifier_head = None

        if self.mode == "generation":
            logits = output_logits.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

            for i in range(input_ids.shape[0]):
                answer_start_from = get_answer_start_idx(input_ids[i], self.answer_token)
                attention_mask[i, :answer_start_from] = 0

            energies = -(logits * attention_mask).mean(axis=-1)

        if self.mode == "classification":
            energies = -output_logits.squeeze(-1)

            if self.config.get("use_moe_head", False):
                classifier_head = self._get_classifier_head(model)
                print("classifier_head type:", type(classifier_head))
                print("latest_expert_usage:", getattr(classifier_head, "latest_expert_usage", None))

                if classifier_head is not None and getattr(classifier_head, "latest_aux_loss", None) is not None:
                    aux_loss = classifier_head.latest_aux_loss

        pos_energy = energies[labels > 0] / energy_temp
        neg_energy = energies[labels < 0] / energy_temp

        if pos_energy.shape[0] == 0:
            pos_energy = torch.zeros(1).to(self.accelerator.device)
        if neg_energy.shape[0] == 0:
            neg_energy = torch.zeros(1).to(self.accelerator.device)

        ml_loss = pos_energy.mean() - neg_energy.mean()

        if alpha != 0:
            l2_loss = alpha * energies.square().mean()

        loss = ml_loss + l2_loss + aux_loss

        self.accelerator.log({"total_loss": loss.item()})
        self.accelerator.log({"l2_loss": l2_loss.item() if alpha > 0 else 0.0})
        self.accelerator.log({"ml_loss": ml_loss.item()})
        self.accelerator.log({"aux_loss": aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss})
        self.accelerator.log({"pos_energy": pos_energy.mean().item()})
        self.accelerator.log({"neg_energy": neg_energy.mean().item()})

        if self.mode == "classification" and self.config.get("use_moe_head", False) and classifier_head is not None:
            expert_usage = getattr(classifier_head, "latest_expert_usage", None)
            print("expert_usage before wandb log:", expert_usage)
            if expert_usage is not None:
                for expert_idx, usage in enumerate(expert_usage.tolist()):
                    print(f"logging expert_{expert_idx}_usage =", usage)
                    self.accelerator.log({f"expert_{expert_idx}_usage": usage})

        loggers["tensor"].info(
            f"\n{'=' * 20}\n\ninput_ids:\n{input_ids}\ninput_id_shape: {inputs['input_ids'].shape}\nattention_mask:\n{attention_mask}"
            + f"\nlogits:\n{output_logits}, shape:\n{output_logits.shape}"
            + f"\npositive_energy:\n{pos_energy.shape}\nnegative_energy:\n{neg_energy.shape}"
            + f"\nargmax_logit:\n{torch.argmax(output_logits, dim=-1)}"
            + f"\nmax_logit:\n{output_logits.gather(dim=-1, index=torch.argmax(output_logits, dim=-1).unsqueeze(-1)).squeeze(-1)}"
        )

        return (loss, outputs) if return_outputs else loss

    def train_step(self, train_loader):
        progress_bar = tqdm(range(len(train_loader)), desc="Training", disable=not self.accelerator.is_local_main_process)
        avg_loss = 0.0

        self.model.train()
        for step_idx, batch in enumerate(train_loader):
            with self.accelerator.accumulate(self.model):
                loss = self.compute_loss(
                    model=self.model,
                    inputs=batch,
                )

                avg_loss += loss.item()
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.accelerator.log({"gradient_norm": grad_norm.mean()})
                    self.accelerator.log({"avg_loss": avg_loss / self.accelerator.gradient_accumulation_steps})
                    avg_loss = 0.0

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_description(f"Loss: {loss.item():.3f}")

            self.accelerator.log({"learning_rate": self.lr_scheduler.get_last_lr()[0]})
            self.accelerator.log({"update_step": step_idx})

        release_memory()

    def input_text_process(self, input_texts):
        return input_texts

    def get_scores_from_texts(self, input_texts, mode="sum_logits"):
        input_texts = self.input_text_process(input_texts)
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            add_special_tokens=self.config["add_special_tokens"],
            padding=True,
            truncation=True,
        ).to(self.accelerator.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            output_logits = outputs.get("logits")
            output_log_probs = torch.log_softmax(output_logits.float(), dim=-1)

            if self.mode == "generation":
                answer_start_from = get_answer_start_idx(input_ids, self.answer_token)
                answer_ids = input_ids[:, answer_start_from:]
                answer_mask = attention_mask[:, answer_start_from:]
                total_len = input_ids.shape[1]

                logits = output_logits.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1) * answer_mask
                log_probs = output_log_probs.gather(dim=-1, index=answer_ids.unsqueeze(-1)).squeeze(-1) * answer_mask

                if mode == "sum_logits":
                    return logits.sum(dim=-1).detach()
                if mode == "mean_logits":
                    return logits.mean(dim=-1).detach()
                if mode == "log_prob":
                    return log_probs.sum(dim=-1).detach()
                if mode == "neg_ppl":
                    return -torch.exp(-log_probs.sum(dim=-1) / total_len).detach()

            if self.mode == "classification":
                return output_logits.detach().squeeze(-1)

        
        
        
    
