# Minimal RLVR (1B Model) Example

This is a smallest-practical RLVR pipeline for learning research:

- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1B-class)
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
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output_dir rlvr_outputs/tinyllama_rlvr \
  --num_episodes 256 \
  --max_steps 60
```

### CPU-only run (slow, but useful for debugging)

```bash
python train.py --use_cpu --max_steps 5 --num_episodes 32
```

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
- For low-memory GPUs, keep LoRA enabled (default).
- If you want full fine-tuning instead of LoRA: add `--disable_lora` (requires more VRAM).
