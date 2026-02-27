from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from peft import LoraConfig

warnings.filterwarnings("ignore", message=r"CUDA initialization.*")
warnings.filterwarnings("ignore", message=r"Can't initialize NVML")

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from trl import GRPOConfig, GRPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_logging import MetricsJSONLCallback, configure_external_logs, format_terminal_log

LOGGER = logging.getLogger("rlvr")
WANDB_PROJECT = "RLVR"
USE_VLLM = True
VLLM_MODE = "colocate"
VLLM_GPU_MEMORY_UTILIZATION = 0.4
VLLM_ENABLE_SLEEP_MODE = False
VLLM_MAX_MODEL_LENGTH = 512
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_GENERATIONS = 2
MAX_COMPLETION_LENGTH = 96
LEARNING_RATE = 1e-5
TEMPERATURE = 0.2
BETA = 0.04
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

DATASET_ID = "namesarnav/time_expressions_dataset"
DATASET_CACHE_DIR = PROJECT_ROOT / ".hf_datasets_cache"
PROMPT_TEMPLATE = """You are given a sentence that may contain a relative time expression.
Normalize the expression to the final calendar date.

Output requirements:
- Return JSON only.
- No explanation, no markdown, no extra keys.
- Use this exact schema: {"date":"YYYY-MM-DD"}
"""
DATE_VALUE_PATTERN = re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", default="rlvr_outputs/date_normalization")
    p.add_argument("--num_episodes", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
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
    p.add_argument("--log_after_every", type=int, default=1)
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


@dataclass
class Episode:
    prompt: str
    question: str
    answer: str


def _parse_date_candidate(candidate: str) -> datetime | None:
    text = candidate.strip()
    if not text:
        return None

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass

    normalized = text.replace("/", "-").replace(".", "-")
    formats = (
        "%Y-%m-%d",
        "%d-%m-%y",
        "%d-%m-%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%B %d, %Y",
    )
    for fmt in formats:
        for variant in (normalized, text):
            try:
                return datetime.strptime(variant, fmt)
            except ValueError:
                continue
    return None


def _normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    candidates = [text]
    candidates.extend(DATE_VALUE_PATTERN.findall(text))
    for candidate in candidates:
        parsed = _parse_date_candidate(candidate)
        if parsed is not None:
            return parsed.strftime("%Y-%m-%d")
    return None


def _extract_json_response(completion_text: str) -> tuple[bool, str | None]:
    raw = completion_text.strip()
    if not raw:
        return False, None

    try:
        data = json.loads(raw)
        if isinstance(data, dict) and set(data.keys()) == {"date"} and isinstance(data.get("date"), str):
            return True, data["date"].strip()
    except Exception:
        pass
    return False, None


def _extract_expected_date(target_output: Any) -> str | None:
    data = json.loads(target_output)
    first = data[0]
    resolved_value = str(first.get("resolved_value", "")).strip()
    return resolved_value


class DateNormalizationEnv:
    NAME = "date_normalization"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.split = self._load_split()
        if len(self.split) == 0:
            raise ValueError("Loaded dataset split is empty.")

    @staticmethod
    def _build_prompt(question: str) -> str:
        return f"{PROMPT_TEMPLATE}\nSentence: {question}"

    def _load_split(self) -> Dataset:
        DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(DATASET_ID, cache_dir=str(DATASET_CACHE_DIR))
        if isinstance(dataset, Dataset):
            return dataset
        for split_name in ("train", "validation", "test"):
            if split_name in dataset:
                return dataset[split_name]
        return next(iter(dataset.values()))

    def sample(self) -> Episode:
        for _ in range(1024):
            row = self.split[self.rng.randrange(len(self.split))]
            question = str(row.get("input_text", "")).strip()
            answer = _extract_expected_date(row.get("target_output"))
            if question and answer is not None:
                return Episode(prompt=self._build_prompt(question), question=question, answer=answer)
        raise ValueError("Could not sample a valid date-normalization example from dataset.")

    def build_dataset(self, n: int) -> Dataset:
        rows = [vars(self.sample()) for _ in range(n)]
        return Dataset.from_list(rows)


class DateExtractionRewardFunction:
    """Reward function with strict JSON checks and built-in logging."""

    def __init__(self, log_path: Path, log_after_every: int = 1) -> None:
        self.log_path = log_path
        self.log_after_every = max(0, log_after_every)
        self.episode_id = 0
        self.__name__ = "date_extraction_reward"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
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
        json_valid_count = 0
        correct_count = 0
        sample_expected: str | None = None
        sample_predicted: str | None = None

        for prompt, completion, expected, q in zip(prompts, completions, answer, question, strict=True):
            completion_text = str(completion)
            expected_norm = _normalize_date(expected)
            json_valid, json_date_raw = _extract_json_response(completion_text)

            if json_valid:
                strict_predicted = _normalize_date(json_date_raw)
                is_correct = expected_norm is not None and strict_predicted == expected_norm
                reward = 1.0 if is_correct else 0.0
            else:
                strict_predicted = None
                is_correct = False
                reward = -0.25

            logged_predicted = strict_predicted if strict_predicted is not None else _normalize_date(completion_text)

            rewards.append(reward)
            json_valid_count += int(json_valid)
            correct_count += int(is_correct)
            if sample_expected is None:
                sample_expected = expected_norm
                sample_predicted = logged_predicted

            record = {
                "episode_id": self.episode_id,
                "steps": step,
                "question": q,
                "expected_date": expected_norm,
                "predicted_date": logged_predicted,
                "json_valid": json_valid,
                "is_correct": is_correct,
                "total_reward": reward,
                "prompt": str(prompt),
                "completion": completion_text,
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self.episode_id += 1

        if rewards:
            reward_mean = sum(rewards) / len(rewards)
            json_rate = json_valid_count / len(rewards)
            acc = correct_count / len(rewards)
            logical_step = max(step, 0)
            if self.log_after_every > 0 and (logical_step + 1) % self.log_after_every == 0:
                LOGGER.info(
                    format_terminal_log(
                        "episode",
                        [
                            ("steps", step),
                            ("reward", f"{reward_mean:.3f}"),
                            ("json", f"{json_rate * 100.0:.1f}%"),
                            ("acc", f"{acc * 100.0:.1f}%"),
                            ("expected", sample_expected),
                            ("predicted", sample_predicted if sample_predicted is not None else "null"),
                        ],
                        color_code="34",
                    )
                )

        return rewards


def make_lora_config() -> LoraConfig:
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    LOGGER.setLevel(logging.INFO)
    configure_external_logs(show_external_logs=False)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested via --device cuda but no CUDA device is available; falling back to CPU.")
        args.device = "cpu"

    has_cuda = args.device == "cuda"
    use_cpu = not has_cuda
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    use_vllm = USE_VLLM
    if use_cpu and use_vllm:
        LOGGER.warning("Disabling vLLM because --device is set to cpu.")
        use_vllm = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = DateNormalizationEnv(seed=args.seed)
    train_dataset = env.build_dataset(args.num_episodes)

    if args.wandb:
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_fn = DateExtractionRewardFunction(
        output_dir / "episode_rewards.jsonl",
        log_after_every=args.log_after_every,
    )
    callback = MetricsJSONLCallback(
        output_dir / "training_metrics.jsonl",
        max_steps=args.max_steps,
        terminal_log_every=args.log_after_every,
    )

    model_init_kwargs = make_model_init_kwargs(args=args, dtype=dtype, device=args.device)

    grpo_kwargs = dict(
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
    grpo_signature = inspect.signature(GRPOConfig).parameters
    unsupported_keys = sorted(k for k in grpo_kwargs if k not in grpo_signature)
    if unsupported_keys:
        LOGGER.warning("Skipping unsupported GRPOConfig args: %s", ", ".join(unsupported_keys))
    grpo_kwargs = {k: v for k, v in grpo_kwargs.items() if k in grpo_signature}
    grpo_args = GRPOConfig(**grpo_kwargs)

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
        "Starting training | device=%s bf16=%s 4bit=%s log_after_every=%s",
        args.device,
        use_bf16,
        args.load_in_4bit,
        args.log_after_every,
    )
    trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    LOGGER.info("Done. Model saved to %s", final_dir)


if __name__ == "__main__":
    main()
