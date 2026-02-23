#!/usr/bin/env python3
"""Minimal RLVR training pipeline for a 1B-class language model.

This script trains a model on a simple arithmetic reasoning environment using
verifiable rewards via TRL's GRPOTrainer.
"""

from __future__ import annotations

import argparse
import logging
import operator
import os
import random
import socket
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# Suppress noisy torch CUDA probe warnings before importing torch/trl.
warnings.filterwarnings(
    "ignore",
    message=r"CUDA initialization: Unexpected error from cudaGetDeviceCount.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Can't initialize NVML",
)

# Unsloth standby should be set before training stack imports when using vLLM.
if "--use_vllm" in sys.argv and "--no-unsloth_vllm_standby" not in sys.argv:
    os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")


def _raise_dependency_error(exc: Exception) -> None:
    message = str(exc)
    if "numpy.dtype size changed" in message or "multiarray failed to import" in message:
        raise RuntimeError(
            "Dependency import failed due to binary incompatibility (likely numpy/pandas/scipy/sklearn). "
            "Recreate the environment and install from requirements.txt."
        ) from exc
    raise RuntimeError("Failed to import training dependencies.") from exc


try:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.trainer_callback import PrinterCallback, ProgressCallback
    from trl import GRPOConfig, GRPOTrainer
except Exception as exc:
    _raise_dependency_error(exc)


from training_logging import EpisodeRewardLogger, MetricsJSONLCallback, configure_external_logs

LOGGER = logging.getLogger("rlvr")
WANDB_PROJECT = "RLVR"


def setup_rlvr_logger(spaced_logs: bool = True) -> None:
    """Configure rlvr logger with optional blank-line spacing between records."""
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    LOGGER.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    if spaced_logs:
        handler.terminator = "\n\n"
    LOGGER.addHandler(handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RLVR training with a 1B model.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", type=str, default="rlvr_outputs/llama32_1b_instruct_rlvr")
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for generation to speed up rollout throughput (GPU only).",
    )
    parser.add_argument(
        "--vllm_mode",
        type=str,
        default="server",
        choices=["server", "colocate"],
        help="vLLM backend mode used by TRL GRPO.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory reserved for vLLM cache/model (0, 1].",
    )
    parser.add_argument(
        "--vllm_enable_sleep_mode",
        action="store_true",
        help="Enable vLLM sleep mode to reduce memory pressure between generation bursts.",
    )
    parser.add_argument(
        "--steps_per_generation",
        type=int,
        default=None,
        help="Number of optimization steps to run before triggering a fresh generation pass.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="GRPO inner-loop iterations per generated batch. Keep at 1 for max speed.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL regularization coefficient. 0.0 avoids reference-model overhead.",
    )
    parser.add_argument(
        "--mask_truncated_completions",
        action="store_true",
        help="Exclude truncated completions from loss computation.",
    )
    parser.add_argument(
        "--unsloth_vllm_standby",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set UNSLOTH_VLLM_STANDBY=1 when using vLLM.",
    )
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
        default=1,
        help="Print one sampled completion every N training steps (0 disables sample logs).",
    )
    parser.add_argument(
        "--sample_chars",
        type=int,
        default=160,
        help="Max characters to show for sampled completions in terminal logs.",
    )
    parser.add_argument(
        "--prediction_log_count",
        type=int,
        default=1,
        help="How many predictions to print per sampled step.",
    )
    parser.add_argument(
        "--keep_trainer_logs",
        action="store_true",
        help="Keep default Hugging Face trainer progress/dict logs in terminal.",
    )
    parser.add_argument(
        "--show_external_logs",
        action="store_true",
        help="Show external library logs (HTTP requests, download/loading messages, warnings).",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases tracking.")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode. Use offline if you want local-only logging.",
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
    NAME = "arithmetic_reasoning"

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


def can_use_local_sockets() -> bool:
    """W&B service requires local socket support."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.close()
        return True
    except OSError:
        return False


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
    )
    setup_rlvr_logger(spaced_logs=True)
    configure_external_logs(show_external_logs=args.show_external_logs)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.vllm_gpu_memory_utilization <= 0 or args.vllm_gpu_memory_utilization > 1:
        raise ValueError("--vllm_gpu_memory_utilization must be in (0, 1].")
    if args.steps_per_generation is not None and args.steps_per_generation <= 0:
        raise ValueError("--steps_per_generation must be > 0 when provided.")
    if args.num_iterations <= 0:
        raise ValueError("--num_iterations must be > 0.")
    if args.beta < 0:
        raise ValueError("--beta must be >= 0.")

    use_cpu = args.use_cpu or not torch.cuda.is_available()
    if args.use_vllm and use_cpu:
        raise ValueError("--use_vllm requires CUDA. Remove --use_cpu and run on a GPU.")
    if args.use_vllm:
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise ImportError("--use_vllm requested but vllm is not installed. Install with: pip install vllm") from exc
    if args.use_vllm and args.unsloth_vllm_standby:
        os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    has_cuda = not use_cpu and torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(has_cuda and not use_bf16)
    dtype = torch.float32
    if use_bf16:
        dtype = torch.bfloat16
    elif use_fp16:
        dtype = torch.float16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_log_path = output_dir / args.episodes_log_name
    metrics_log_path = output_dir / args.metrics_log_name
    env = ArithmeticReasoningEnv(seed=args.seed)
    env_name = getattr(env, "NAME", env.__class__.__name__)

    wandb_run = None
    report_to: str | list[str] = "none"
    run_name = env_name
    if args.wandb:
        if not can_use_local_sockets():
            LOGGER.warning("wandb unavailable in this runtime (local sockets disabled); continuing without wandb")
        else:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError("wandb is not installed. Install it with: pip install wandb") from exc

            if not args.show_external_logs:
                os.environ.setdefault("WANDB_SILENT", "true")
                os.environ.setdefault("WANDB_CONSOLE", "off")
                os.environ.setdefault("WANDB_QUIET", "true")
            os.environ.setdefault("WANDB_DIR", str(output_dir / "wandb"))
            os.environ.setdefault("WANDB_CACHE_DIR", str(output_dir / "wandb_cache"))

            wandb_kwargs: dict[str, object] = {
                "project": WANDB_PROJECT,
                "name": env_name,
                "mode": args.wandb_mode,
                "dir": str(output_dir),
                "config": vars(args),
            }
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            try:
                wandb_run = wandb.init(**wandb_kwargs)
                report_to = "wandb"
                LOGGER.info("wandb enabled | project=%s | run=%s | mode=%s", WANDB_PROJECT, env_name, args.wandb_mode)
            except Exception as exc:
                wandb_run = None
                report_to = "none"
                LOGGER.warning("wandb init failed, continuing without wandb: %s", str(exc).splitlines()[0])

    train_dataset = env.build_dataset(args.num_episodes)
    LOGGER.info("Dataset built with %s episodes", len(train_dataset))

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    except OSError as exc:
        raise RuntimeError(
            (
                "Failed to load tokenizer/model config. Use a valid local model path, enable internet access, "
                "and for gated models (for example meta-llama) ensure Hugging Face access is granted and "
                "authentication is configured."
            )
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_fn = EpisodeRewardLogger(
        episodes_log_path,
        terminal_log_every=args.terminal_log_every,
        sample_log_every=args.sample_log_every,
        sample_chars=args.sample_chars,
        prediction_log_count=args.prediction_log_count,
        wandb_run=wandb_run,
    )
    callback = MetricsJSONLCallback(
        metrics_log_path,
        max_steps=args.max_steps,
        terminal_log_every=args.terminal_log_every,
    )
    peft_config = maybe_make_lora_config(disable_lora=args.disable_lora)

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=report_to,
        remove_unused_columns=False,
        use_cpu=use_cpu,
        bf16=use_bf16,
        fp16=use_fp16,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        steps_per_generation=args.steps_per_generation,
        num_iterations=args.num_iterations,
        beta=args.beta,
        mask_truncated_completions=args.mask_truncated_completions,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
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
    if not args.show_external_logs:
        LOGGER.info("external logs: suppressed (use --show_external_logs to enable)")
    LOGGER.info(
        (
            "logging config | terminal_log_every=%s | sample_log_every=%s | sample_chars=%s | prediction_log_count=%s "
            "| temperature=%s | episodes_log=%s | metrics_log=%s"
        ),
        args.terminal_log_every,
        args.sample_log_every,
        args.sample_chars,
        args.prediction_log_count,
        args.temperature,
        episodes_log_path,
        metrics_log_path,
    )
    LOGGER.info(
        (
            "speed config | use_vllm=%s | vllm_mode=%s | vllm_gpu_mem_util=%.2f | vllm_sleep=%s | "
            "unsloth_standby=%s | steps_per_generation=%s | num_iterations=%s | beta=%s | mask_truncated=%s"
        ),
        args.use_vllm,
        args.vllm_mode,
        args.vllm_gpu_memory_utilization,
        args.vllm_enable_sleep_mode,
        bool(args.use_vllm and args.unsloth_vllm_standby),
        args.steps_per_generation,
        args.num_iterations,
        args.beta,
        args.mask_truncated_completions,
    )
    trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    LOGGER.info("Training complete")
    LOGGER.info("Episode logs: %s", episodes_log_path)
    LOGGER.info("Metrics logs: %s", metrics_log_path)
    LOGGER.info("Model saved to: %s", final_dir)
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
