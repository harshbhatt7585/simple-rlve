#!/usr/bin/env python3
"""Minimal RLVR training pipeline for a 1B-class language model.

This script trains a model on a simple arithmetic reasoning environment using
verifiable rewards via TRL's GRPOTrainer.
"""

from __future__ import annotations

import argparse
import logging
import operator
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from trl import GRPOConfig, GRPOTrainer

from training_logging import EpisodeRewardLogger, MetricsJSONLCallback

LOGGER = logging.getLogger("rlvr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RLVR training with a 1B model.")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output_dir", type=str, default="rlvr_outputs/tinyllama_rlvr")
    parser.add_argument("--num_episodes", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=128,
        help="Reserved for compatibility across TRL versions (not used in trl==0.28.0).",
    )
    parser.add_argument("--max_completion_length", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--episodes_log_name", type=str, default="episode_rewards.jsonl")
    parser.add_argument("--metrics_log_name", type=str, default="training_metrics.jsonl")
    parser.add_argument(
        "--terminal_log_every",
        type=int,
        default=1,
        help="Print compact terminal summaries every N training steps.",
    )
    parser.add_argument(
        "--sample_log_every",
        type=int,
        default=5,
        help="Print one sampled completion every N training steps (0 disables sample logs).",
    )
    parser.add_argument(
        "--sample_chars",
        type=int,
        default=160,
        help="Max characters to show for sampled completions in terminal logs.",
    )
    parser.add_argument(
        "--keep_trainer_logs",
        action="store_true",
        help="Keep default Hugging Face trainer progress/dict logs in terminal.",
    )
    return parser.parse_args()


@dataclass
class Episode:
    prompt: str
    question: str
    answer: str


class ArithmeticReasoningEnv:
    """Very small environment that emits arithmetic reasoning tasks."""

    OPS = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
    }

    def __init__(self, seed: int = 42, min_value: int = 0, max_value: int = 20) -> None:
        self.rng = random.Random(seed)
        self.min_value = min_value
        self.max_value = max_value

    def sample(self) -> Episode:
        a = self.rng.randint(self.min_value, self.max_value)
        b = self.rng.randint(self.min_value, self.max_value)
        c = self.rng.randint(self.min_value, self.max_value)
        op1 = self.rng.choice(list(self.OPS))
        op2 = self.rng.choice(list(self.OPS))

        first = self.OPS[op1](a, b)
        result = self.OPS[op2](first, c)
        question = f"({a} {op1} {b}) {op2} {c}"
        prompt = (
            "Solve the arithmetic problem.\n"
            "Return exactly this XML format:\n"
            "<reasoning>short step-by-step reasoning</reasoning>\n"
            "<answer>final integer</answer>\n"
            f"Problem: {question}"
        )
        return Episode(prompt=prompt, question=question, answer=str(result))

    def build_dataset(self, num_episodes: int) -> Dataset:
        rows = []
        for _ in range(num_episodes):
            item = self.sample()
            rows.append({"prompt": item.prompt, "question": item.question, "answer": item.answer})
        return Dataset.from_list(rows)


def maybe_make_lora_config(disable_lora: bool):
    if disable_lora:
        return None
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA mode. Install with: pip install peft"
        ) from exc

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_log_path = output_dir / args.episodes_log_name
    metrics_log_path = output_dir / args.metrics_log_name

    env = ArithmeticReasoningEnv(seed=args.seed)
    train_dataset = env.build_dataset(args.num_episodes)
    LOGGER.info("Dataset built with %s episodes", len(train_dataset))

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load tokenizer/model config. Use a valid local model path or enable internet access."
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cpu = args.use_cpu or not torch.cuda.is_available()
    has_cuda = not use_cpu and torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(has_cuda and not use_bf16)
    dtype = torch.float32
    if use_bf16:
        dtype = torch.bfloat16
    elif use_fp16:
        dtype = torch.float16

    reward_fn = EpisodeRewardLogger(
        episodes_log_path,
        terminal_log_every=args.terminal_log_every,
        sample_log_every=args.sample_log_every,
        sample_chars=args.sample_chars,
    )
    callback = MetricsJSONLCallback(
        metrics_log_path,
        max_steps=args.max_steps,
        terminal_log_every=args.terminal_log_every,
    )
    peft_config = maybe_make_lora_config(disable_lora=args.disable_lora)

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        run_name="minimal_rlvr_1b",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        use_cpu=use_cpu,
        bf16=use_bf16,
        fp16=use_fp16,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        model_init_kwargs={"torch_dtype": dtype},
        log_completions=False,
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
        peft_config=peft_config,
    )
    if not args.keep_trainer_logs:
        trainer.remove_callback(ProgressCallback)
        trainer.remove_callback(PrinterCallback)

    LOGGER.info(
        "Starting RLVR training | use_cpu=%s bf16=%s fp16=%s lora=%s",
        use_cpu,
        use_bf16,
        use_fp16,
        not args.disable_lora,
    )
    LOGGER.info(
        (
            "logging config | terminal_log_every=%s | sample_log_every=%s | sample_chars=%s "
            "| episodes_log=%s | metrics_log=%s"
        ),
        args.terminal_log_every,
        args.sample_log_every,
        args.sample_chars,
        episodes_log_path,
        metrics_log_path,
    )
    trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    LOGGER.info("Training complete")
    LOGGER.info("Episode logs: %s", episodes_log_path)
    LOGGER.info("Metrics logs: %s", metrics_log_path)
    LOGGER.info("Model saved to: %s", final_dir)


if __name__ == "__main__":
    main()
