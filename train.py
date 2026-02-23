#!/usr/bin/env python3
"""Minimal RLVR training pipeline using TRL's GRPOTrainer."""

from __future__ import annotations

import argparse
import logging
import operator
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore", message=r"CUDA initialization.*")
warnings.filterwarnings("ignore", message=r"Can't initialize NVML")

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from trl import GRPOConfig, GRPOTrainer

from training_logging import EpisodeRewardLogger, MetricsJSONLCallback, configure_external_logs

LOGGER = logging.getLogger("rlvr")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--output_dir", default="rlvr_outputs/run")
    p.add_argument("--num_episodes", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--num_generations", type=int, default=2)
    p.add_argument("--max_completion_length", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--use_cpu", action="store_true")
    p.add_argument("--disable_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", default="none", choices=["none", "all", "lora_only"])
    p.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA adapters.",
    )
    p.add_argument("--save_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--terminal_log_every", type=int, default=0)
    p.add_argument("--sample_log_every", type=int, default=0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--vllm_mode", default="server", choices=["server", "colocate"])
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.55)
    p.add_argument("--vllm_enable_sleep_mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_iterations", type=int, default=1)
    p.add_argument("--steps_per_generation", type=int, default=1)
    
    return p.parse_args()


@dataclass
class Episode:
    prompt: str
    question: str
    answer: str


class ArithmeticEnv:
    OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
    NAME = "arithmetic_reasoning"

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


def make_lora_config(args: argparse.Namespace):
    if args.disable_lora:
        return None
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError("LoRA requested but peft is not installed. Install with: pip install peft") from exc
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("--lora_target_modules must contain at least one module name")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    LOGGER.setLevel(logging.INFO)
    configure_external_logs(show_external_logs=False)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cpu = args.use_cpu or not torch.cuda.is_available()
    has_cuda = not use_cpu
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = ArithmeticEnv(seed=args.seed)
    train_dataset = env.build_dataset(args.num_episodes)

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

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="wandb" if args.wandb else "none",
        remove_unused_columns=False,
        use_cpu=use_cpu,
        bf16=use_bf16,
        fp16=use_fp16,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        use_vllm=args.use_vllm,
        model_init_kwargs={"torch_dtype": dtype},
        log_completions=False,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        steps_per_generation=args.steps_per_generation,
        num_iterations=args.num_iterations,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
        peft_config=make_lora_config(args),
    )
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)

    LOGGER.info("Starting training | cpu=%s bf16=%s lora=%s", use_cpu, use_bf16, not args.disable_lora)
    trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    LOGGER.info("Done. Model saved to %s", final_dir)


if __name__ == "__main__":
    main()
