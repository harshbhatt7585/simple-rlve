#!/usr/bin/env python3
"""Minimal RLVR training pipeline for a 1B-class language model.

This script trains a model on a simple arithmetic reasoning environment using
verifiable rewards via TRL's GRPOTrainer.
"""

from __future__ import annotations

import argparse
import json
import logging
import operator
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

LOGGER = logging.getLogger("rlvr")
ANSWER_PATTERN = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.IGNORECASE | re.DOTALL)


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


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return str(value[0].get("content", ""))
        return " ".join(str(x) for x in value)
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def _extract_answer(text: str) -> str | None:
    match = ANSWER_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _clip_text(text: str, max_chars: int) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max(0, max_chars - 3)] + "..."


class EpisodeRewardLogger:
    """Custom reward function that logs every generated episode."""

    def __init__(
        self,
        log_path: Path,
        terminal_log_every: int = 1,
        sample_log_every: int = 5,
        sample_chars: int = 160,
    ) -> None:
        self.log_path = log_path
        self.episode_id = 0
        self.__name__ = "episode_reward"
        self.terminal_log_every = max(1, terminal_log_every)
        self.sample_log_every = max(0, sample_log_every)
        self.sample_chars = max(40, sample_chars)
        self.running_reward_sum = 0.0
        self.running_episode_count = 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_header()

    def _write_header(self) -> None:
        self.log_path.write_text("")

    def __call__(
        self,
        prompts: list[Any],
        completions: list[Any],
        answer: list[str],
        question: list[str],
        trainer_state=None,
        **_: Any,
    ) -> list[float]:
        rewards: list[float] = []
        step = int(trainer_state.global_step) if trainer_state is not None else -1
        correct_count = 0
        format_count = 0
        sample_record: dict[str, Any] | None = None

        for prompt, completion, expected, q in zip(prompts, completions, answer, question, strict=True):
            completion_text = _as_text(completion)
            predicted = _extract_answer(completion_text)

            format_ok = "<reasoning>" in completion_text and "</reasoning>" in completion_text
            format_ok = format_ok and "<answer>" in completion_text and "</answer>" in completion_text
            format_reward = 0.25 if format_ok else 0.0

            correct = predicted == expected
            correctness_reward = 1.0 if correct else -0.25
            total_reward = correctness_reward + format_reward
            rewards.append(total_reward)
            correct_count += int(correct)
            format_count += int(format_ok)
            if sample_record is None:
                sample_record = {
                    "question": q,
                    "expected_answer": expected,
                    "predicted_answer": predicted,
                    "completion": completion_text,
                }

            log_record = {
                "episode_id": self.episode_id,
                "global_step": step,
                "question": q,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "is_correct": correct,
                "format_reward": format_reward,
                "correctness_reward": correctness_reward,
                "total_reward": total_reward,
                "prompt": _as_text(prompt),
                "completion": completion_text,
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_record) + "\n")
            self.episode_id += 1

        if rewards:
            batch_size = len(rewards)
            reward_mean = sum(rewards) / batch_size
            reward_min = min(rewards)
            reward_max = max(rewards)
            accuracy = correct_count / batch_size
            format_rate = format_count / batch_size
            self.running_reward_sum += sum(rewards)
            self.running_episode_count += batch_size
            running_reward = self.running_reward_sum / self.running_episode_count

            logical_step = max(step, 0)
            if (logical_step + 1) % self.terminal_log_every == 0:
                LOGGER.info(
                    (
                        "episode_stats step=%s | batch=%s | reward(mean=%.3f min=%.3f max=%.3f) "
                        "| acc=%.1f%% | format=%.1f%% | running_reward=%.3f"
                    ),
                    step,
                    batch_size,
                    reward_mean,
                    reward_min,
                    reward_max,
                    accuracy * 100.0,
                    format_rate * 100.0,
                    running_reward,
                )
                if self.sample_log_every > 0 and (logical_step + 1) % self.sample_log_every == 0 and sample_record:
                    LOGGER.info(
                        "sample step=%s | q=%s | expected=%s predicted=%s | completion=%s",
                        step,
                        sample_record["question"],
                        sample_record["expected_answer"],
                        sample_record["predicted_answer"],
                        _clip_text(str(sample_record["completion"]), self.sample_chars),
                    )

        return rewards


class MetricsJSONLCallback(TrainerCallback):
    """Writes trainer logs to JSONL for easy plotting."""

    def __init__(self, path: Path, max_steps: int, terminal_log_every: int = 1) -> None:
        self.path = path
        self.max_steps = max_steps
        self.terminal_log_every = max(1, terminal_log_every)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not state.is_local_process_zero or not logs:
            return
        numeric_logs = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in logs.items()
            if isinstance(v, (int, float, str))
        }
        payload = {"global_step": int(state.global_step), "epoch": float(state.epoch or 0.0), **numeric_logs}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        step = int(payload["global_step"])
        logical_step = max(step, 1)
        should_log = (logical_step % self.terminal_log_every == 0) or ("train_runtime" in payload)
        if not should_log:
            return

        if "reward" in payload:
            progress = f"{step}/{self.max_steps}" if self.max_steps > 0 else str(step)
            progress_pct = (100.0 * step / self.max_steps) if self.max_steps > 0 else 0.0
            LOGGER.info(
                (
                    "train step=%s (%.1f%%) | reward=%.3f | reward_std=%.3f | "
                    "lr=%.2e | entropy=%.3f | comp_len=%.1f | step_time=%.2fs"
                ),
                progress,
                progress_pct,
                float(payload.get("reward", 0.0)),
                float(payload.get("reward_std", 0.0)),
                float(payload.get("learning_rate", 0.0)),
                float(payload.get("entropy", 0.0)),
                float(payload.get("completions/mean_length", 0.0)),
                float(payload.get("step_time", 0.0)),
            )
        else:
            LOGGER.info(
                (
                    "train_done runtime=%.2fs | steps_per_sec=%.3f | "
                    "samples_per_sec=%.3f | train_loss=%.4f"
                ),
                float(payload.get("train_runtime", 0.0)),
                float(payload.get("train_steps_per_second", 0.0)),
                float(payload.get("train_samples_per_second", 0.0)),
                float(payload.get("train_loss", 0.0)),
            )


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
