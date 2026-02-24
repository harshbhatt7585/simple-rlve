from __future__ import annotations

import argparse
import logging
import operator
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from peft import LoraConfig

warnings.filterwarnings("ignore", message=r"CUDA initialization.*")
warnings.filterwarnings("ignore", message=r"Can't initialize NVML")

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from trl import GRPOConfig, GRPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_logging import EpisodeRewardLogger, MetricsJSONLCallback, configure_external_logs

LOGGER = logging.getLogger("rlvr")
WANDB_PROJECT = "RLVR"
USE_VLLM = True
VLLM_MODE = "colocate"
VLLM_GPU_MEMORY_UTILIZATION = 0.4
VLLM_ENABLE_SLEEP_MODE = False
VLLM_MAX_MODEL_LENGTH = 512
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
NUM_GENERATIONS = 2
MAX_COMPLETION_LENGTH = 256
LEARNING_RATE = 1e-5
TEMPERATURE = 1.0
BETA = 0.0
SAVE_STEPS = 20
NUM_ITERATIONS = 1
STEPS_PER_GENERATION = 2
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--output_dir", default="rlvr_outputs/run")
    p.add_argument("--num_episodes", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Execution device (defaults to cuda).",
    )
    p.add_argument("--load_in_4bit", action="store_true", help="Load the base model in 4-bit (bitsandbytes).")
    p.add_argument("--bnb_4bit_quant_type", default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--bnb_4bit_use_double_quant", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--bnb_4bit_compute_dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Compute dtype used by 4-bit kernels (auto picks bf16/fp16/fp32 from hardware).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--terminal_log_every", type=int, default=1)
    p.add_argument("--sample_log_every", type=int, default=1)
    p.add_argument("--wandb", action="store_true")

    return p.parse_args()


@dataclass
class Episode:
    prompt: str
    question: str
    answer: str


class ArithmeticEnv:
    OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
    NAME = "arithemetic_reasoning"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def sample(self) -> Episode:
        a, b, c = [self.rng.randint(0, 20) for _ in range(3)]
        op1, op2 = self.rng.choices(list(self.OPS), k=2)
        result = self.OPS[op2](self.OPS[op1](a, b), c)
        question = f"({a} {op1} {b}) {op2} {c}"
        prompt = (
            "Solve the arithmetic problem.\n"
            "Return exactly this XML format:\n"
            "<reasoning>short step-by-step reasoning</reasoning>\n"
            "<answer>final integer</answer>\n"
            f"Problem: {question}"
        )
        return Episode(prompt=prompt, question=question, answer=str(result))

    def build_dataset(self, n: int) -> Dataset:
        rows = [vars(self.sample()) for _ in range(n)]
        return Dataset.from_list(rows)


def make_lora_config():
    if not LORA_TARGET_MODULES:
        raise ValueError("LORA_TARGET_MODULES must contain at least one module name")
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )


def make_model_init_kwargs(args: argparse.Namespace, dtype: torch.dtype, device: str) -> dict:
    kwargs = {"torch_dtype": dtype}
    if not args.load_in_4bit:
        return kwargs
    if device != "cuda":
        raise ValueError("--load_in_4bit requires CUDA; set --device cuda and run on a CUDA GPU.")

    from transformers import BitsAndBytesConfig
    
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map.get(args.bnb_4bit_compute_dtype, dtype)
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    kwargs["device_map"] = "auto"
    return kwargs


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    LOGGER.setLevel(logging.INFO)
    configure_external_logs(show_external_logs=False)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if args.device == "cuda" and not torch.cuda.is_available():
        fallback = "mps" if mps_available else "cpu"
        LOGGER.warning("CUDA requested via --device cuda but unavailable; falling back to %s.", fallback)
        args.device = fallback
    if args.device == "mps" and not mps_available:
        LOGGER.warning("MPS requested via --device mps but unavailable; falling back to CPU.")
        args.device = "cpu"

    has_cuda = args.device == "cuda"
    use_mps = args.device == "mps"
    use_cpu = args.device == "cpu"
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    use_vllm = USE_VLLM

    # vLLM currently requires CUDA; disable it on non-CUDA devices.
    if args.device != "cuda" and use_vllm:
        LOGGER.warning("Disabling vLLM because --device is set to %s.", args.device)
        use_vllm = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = ArithmeticEnv(seed=args.seed)
    train_dataset = env.build_dataset(args.num_episodes)
    if args.wandb:
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_fn = EpisodeRewardLogger(
        output_dir / "episode_rewards.jsonl",
        terminal_log_every=args.terminal_log_every,
        sample_log_every=args.sample_log_every,
    )
    callback = MetricsJSONLCallback(
        output_dir / "training_metrics.jsonl",
        max_steps=args.max_steps,
        terminal_log_every=args.terminal_log_every,
    )

    model_init_kwargs = make_model_init_kwargs(args=args, dtype=dtype, device=args.device)

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        run_name=env.NAME,
        report_to="wandb" if args.wandb else "none",
        remove_unused_columns=False,
        use_cpu=use_cpu,
        use_mps_device=use_mps,
        bf16=use_bf16,
        fp16=use_fp16,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=TEMPERATURE,
        beta=BETA,
        use_vllm=use_vllm,
        model_init_kwargs=model_init_kwargs,
        log_completions=False,
        vllm_mode=VLLM_MODE,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        vllm_enable_sleep_mode=VLLM_ENABLE_SLEEP_MODE,
        vllm_max_model_length=VLLM_MAX_MODEL_LENGTH,
        steps_per_generation=STEPS_PER_GENERATION,
        num_iterations=NUM_ITERATIONS,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
        peft_config=make_lora_config(),
    )
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)

    LOGGER.info(
        "Starting training | device=%s bf16=%s 4bit=%s",
        args.device,
        use_bf16,
        args.load_in_4bit,
    )
    trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    LOGGER.info("Done. Model saved to %s", final_dir)


if __name__ == "__main__":
    main()
