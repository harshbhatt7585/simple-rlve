# Minimal RLVR (1B Instruct Model) Example

This is a smallest-practical RLVR pipeline for learning research:

- **Model**: `meta-llama/Llama-3.2-1B-Instruct` (1B-class)
- **Environment**: generated arithmetic reasoning tasks
- **Trainer**: Hugging Face TRL `GRPOTrainer`
- **Reward**: verifiable correctness + output format bonus
- **Logs**: per-episode reward logs + trainer metrics logs

## 1) Setup environment

```bash
cd /teamspace/studios/this_studio/rlvr
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional for faster rollout generation
pip install "vllm==0.12.0"
```

Or one command:

```bash
cd /teamspace/studios/this_studio/rlvr
./setup_env.sh
source .venv/bin/activate
```

## 2) Train

```bash
python train.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --output_dir rlvr_outputs/llama32_1b_instruct_rlvr \
  --num_episodes 256 \
  --max_steps 60
```

If outputs drift from strict format, try a lower sampling temperature, for example: `--temperature 0.7`.

For `meta-llama/Llama-3.2-1B-Instruct`, make sure your Hugging Face account has accepted the model license and your environment is authenticated (`huggingface-cli login`).

### CPU-only run (slow, but useful for debugging)

```bash
python train.py --device cpu --max_steps 5 --num_episodes 32
```

### Faster training profile (Unsloth-style settings)

```bash
python train.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --output_dir rlvr_outputs/llama32_1b_instruct_rlvr_fast \
  --num_episodes 256 \
  --max_steps 60
```

Notes:
- vLLM is enabled by default when `--device cuda` is used.
- vLLM settings are currently hardcoded in `train.py`:
  - `vllm_mode="colocate"`
  - `vllm_gpu_memory_utilization=0.4`
  - `vllm_enable_sleep_mode=False`
  - `vllm_max_model_length=512`

### Cleaner terminal logs

```bash
python train.py \
  --terminal_log_every 1 \
  --sample_log_every 1 \
  --prediction_log_count 1 \
  --sample_chars 160
```

This prints per-step:
- reward summary
- prediction lines with `reward`, `expected`, `predicted`, and completion text snippet

Default is quiet (`--terminal_log_every 0 --sample_log_every 0`) while still writing JSONL logs.

Default behavior hides noisy raw trainer dict logs. If you want them back:

```bash
python train.py --keep_trainer_logs
```

Default behavior also suppresses external noise (HTTP request logs, model-load chatter, warning spam).
If you want full external logs while debugging, use:

```bash
python train.py --show_external_logs
```

### W&B logging

```bash
python train.py --wandb
```

W&B is fixed to:
- project: `RLVR`
- run name: environment name (`arithemetic_reasoning`)

Offline W&B mode:

```bash
python train.py --wandb --wandb_mode offline
```

Note: W&B needs local socket support. In restricted sandboxes it will auto-fallback to normal training without W&B.

## 3) Outputs

All outputs go under `--output_dir`:

- `episode_rewards.jsonl`: one line per generated episode with:
  - prompt/question
  - completion
  - expected answer vs predicted answer
  - correctness reward, format reward, total reward
- `training_metrics.jsonl`: trainer metrics each log step
- `final_model/`: final trained model + tokenizer

## Notes

- This is intentionally minimal for research learning.
- LoRA is enabled with hardcoded defaults in `train.py` (`r=16`, `alpha=32`, `dropout=0.05`, `bias="none"`).
