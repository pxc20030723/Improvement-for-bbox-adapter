# Colab A100 Runbook

This repo can be run in Colab with an A100. The lowest-cost path is:

1. Use `gpt-4o-mini` as the generator model.
2. Start with `StrategyQA`.
3. Run a small subset first.
4. Enable LoRA on the white-box critic only.

## 1. Open a Colab notebook

Set runtime to:

- GPU
- A100 if available

## 2. Setup cells

### Clone and install

```bash
!git clone https://github.com/haotiansun14/BBox-Adapter.git
%cd BBox-Adapter
!pip install -r requirements.txt
```

### Optional: login to Weights & Biases

```python
# import wandb
# wandb.login()
```

### Set the API key

```python
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: ")
```

### Sanity check GPU

```bash
!nvidia-smi
```

## 3. First run

Use the Colab-friendly config:

```bash
!python main.py --config configs/strategyqa_colab_lora.yaml
```

## 4. What changed from the original repo

- Added optional LoRA support for the white-box critic in `llms/whitebox.py`
- Added `max_train_samples` and `max_eval_samples` so you can run cheap subset experiments
- Added `configs/strategyqa_colab_lora.yaml` as a cheap first-pass config

## 5. Recommended progression

1. Run `strategyqa_colab_lora.yaml`
2. Confirm the full train/eval loop works
3. Increase `max_train_samples` and `max_eval_samples`
4. Increase `num_candidates` and `beam_size`
5. Move to `scienceqa` or `gsm8k`

## 6. Important notes

- LoRA is applied to the local critic model, not the black-box OpenAI model.
- TruthfulQA is not the best first target because the original setup depends on an older judge-model workflow.
- Your main cost driver is still OpenAI API usage, so keep candidate counts small at first.
