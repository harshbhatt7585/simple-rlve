from __future__ import annotations

import argparse
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

from training_logging import MetricsJSONLCallback, configure_external_logs

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

DATASET_ID = "namesarnav/time_expressions_dataset"
DATASET_CACHE_DIR = PROJECT_ROOT / ".hf_datasets_cache"
PROMPT_TEMPLATE = """You are given a sentence that may contain a date.
Identify the date mentioned in the sentence and extract it in the format YYYY-MM-DD.

Return the output JSON object only, strictly in the following JSON format:
{
  "date": "YYYY-MM-DD"
}
"""
DATE_VALUE_PATTERN = re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b")


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
    p.add_argument("--terminal_log_every", type=int, default=1)
    p.add_argument("--sample_log_every", type=int, default=1)
    p.add_argument("--wandb", action="store_true")

    return p.parse_args()


@dataclass
class Episode:
    prompt: str
    question: str
    answer: str


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


def _clip_text(text: str, max_chars: int) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max(0, max_chars - 3)] + "..."


def _normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    candidates: list[str] = [text]
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "date" in data:
                candidates.insert(0, str(data["date"]).strip())
        except Exception:
            pass
    candidates.extend(DATE_VALUE_PATTERN.findall(text))

    for candidate in candidates:
        parsed = _parse_date_candidate(candidate)
        if parsed is not None:
            return parsed.strftime("%Y-%m-%d")
    return None


def _parse_date_candidate(candidate: str) -> datetime | None:
    c = candidate.strip()
    if not c:
        return None

    try:
        return datetime.fromisoformat(c.replace("Z", "+00:00"))
    except ValueError:
        pass

    c_dash = c.replace("/", "-").replace(".", "-")
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
        for candidate_variant in (c_dash, c):
            try:
                return datetime.strptime(candidate_variant, fmt)
            except ValueError:
                continue
    return None


def _extract_json_date(completion_text: str) -> tuple[bool, str | None]:
    raw = completion_text.strip()
    json_candidates = [raw]
    if "{" in raw and "}" in raw:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < end:
            json_candidates.append(raw[start : end + 1])
    json_candidates.extend(re.findall(r"\{[^{}]*\}", raw, flags=re.DOTALL))

    for chunk in json_candidates:
        try:
            data = json.loads(chunk)
        except Exception:
            continue
        if isinstance(data, dict):
            if "date" in data:
                return True, str(data["date"])
            return True, None
        return True, None
    return False, None


class DateExtractionRewardLogger:
    """Reward function with dense scoring for JSON validity + answer correctness."""

    def __init__(
        self,
        log_path: Path,
        terminal_log_every: int = 1,
        sample_log_every: int = 1,
        sample_chars: int = 160,
        prediction_log_count: int = 1,
    ) -> None:
        self.log_path = log_path
        self.episode_id = 0
        self.__name__ = "date_extraction_reward"
        self.terminal_log_every = max(0, terminal_log_every)
        self.sample_log_every = max(0, sample_log_every)
        self.sample_chars = max(40, sample_chars)
        self.prediction_log_count = max(1, prediction_log_count)
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
        sample_records: list[dict[str, Any]] = []

        for prompt, completion, expected, q in zip(prompts, completions, answer, question, strict=True):
            completion_text = _as_text(completion)
            expected_norm = _normalize_date(expected)
            json_valid, json_date_raw = _extract_json_date(completion_text)

            predicted_norm = _normalize_date(json_date_raw) if json_valid else _normalize_date(completion_text)
            is_correct = expected_norm is not None and predicted_norm == expected_norm

            # Reward structure requested by user:
            # 1) invalid JSON + wrong answer = -0.25
            # 2) valid JSON + wrong answer = 0.0
            # 3) invalid JSON + correct answer = 0.5
            # 4) valid JSON + correct answer = 1.0
            if json_valid and is_correct:
                total_reward = 1.0
            elif json_valid and not is_correct:
                total_reward = 0.0
            elif (not json_valid) and is_correct:
                total_reward = 0.5
            else:
                total_reward = -0.25

            rewards.append(total_reward)
            json_valid_count += int(json_valid)
            correct_count += int(is_correct)

            if len(sample_records) < self.prediction_log_count:
                sample_records.append(
                    {
                        "question": q,
                        "expected_date": expected_norm,
                        "predicted_date": predicted_norm,
                        "json_valid": json_valid,
                        "reward": total_reward,
                        "completion": completion_text,
                    }
                )

            log_record = {
                "episode_id": self.episode_id,
                "global_step": step,
                "question": q,
                "expected_date": expected_norm,
                "predicted_date": predicted_norm,
                "json_valid": json_valid,
                "is_correct": is_correct,
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
            json_valid_rate = json_valid_count / batch_size
            accuracy = correct_count / batch_size
            logical_step = max(step, 0)

            if self.terminal_log_every > 0 and (logical_step + 1) % self.terminal_log_every == 0:
                LOGGER.info(
                    (
                        "episode_stats step=%s | reward(mean=%.3f min=%.3f max=%.3f) "
                        "| json_valid=%.1f%% | date_acc=%.1f%%"
                    ),
                    step,
                    reward_mean,
                    reward_min,
                    reward_max,
                    json_valid_rate * 100.0,
                    accuracy * 100.0,
                )
                if self.sample_log_every > 0 and (logical_step + 1) % self.sample_log_every == 0:
                    for record in sample_records:
                        LOGGER.info(
                            "prediction | reward=%.3f | json_valid=%s | expected=%s predicted=%s | text=%s",
                            float(record["reward"]),
                            record["json_valid"],
                            record["expected_date"],
                            record["predicted_date"],
                            _clip_text(str(record["completion"]), self.sample_chars),
                        )

        return rewards


class DateNormalizationEnv:
    NAME = "date_normalization"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _load_split(self) -> Dataset:
        DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(DATASET_ID, cache_dir=str(DATASET_CACHE_DIR))
        if isinstance(dataset, Dataset):
            return dataset

        for split_name in ("train", "validation", "test"):
            if split_name in dataset:
                return dataset[split_name]
        return next(iter(dataset.values()))

    def _resolve_columns(self, split: Dataset) -> tuple[str, str]:
        columns = list(split.column_names)
        text_candidates = ("expression", "input_text", "sentence", "text", "input", "prompt", "question")
        answer_candidates = ("resolved_value", "target_output", "output_text", "date", "normalized_date", "target", "answer", "label")

        text_col = next((name for name in text_candidates if name in columns), None)
        answer_col = next((name for name in answer_candidates if name in columns and name != text_col), None)

        if text_col is None and columns:
            text_col = columns[0]
        if answer_col is None:
            answer_col = next((name for name in columns if name != text_col), None)

        if text_col is None or answer_col is None:
            raise ValueError(f"Could not infer text/date columns from dataset columns: {columns}")
        return text_col, answer_col


    def build_dataset(self, n: int) -> Dataset:
        split = self._load_split()
        text_col, answer_col = self._resolve_columns(split)
        total = len(split)
        if total == 0:
            raise ValueError("Loaded dataset split is empty.")

        if n <= total:
            indices = self.rng.sample(range(total), k=n)
        else:
            indices = [self.rng.randrange(total) for _ in range(n)]

        rows: list[dict[str, str]] = []
        for idx in indices:
            row = split[int(idx)]
            if "type" in row and str(row["type"]).strip().lower() != "date":
                continue
            question = str(row[text_col]).strip()
            answer = _normalize_date(row[answer_col])
            if not question or answer is None:
                continue

            prompt = f"{PROMPT_TEMPLATE}\nSentence: {question}"
            rows.append(vars(Episode(prompt=prompt, question=question, answer=answer)))

        if not rows:
            raise ValueError(
                "No valid rows were produced from the dataset. "
                "Check source columns and date format normalization."
            )
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
    if device == "cpu":
        raise ValueError("--load_in_4bit requires CUDA; set --device cuda and run on a GPU.")

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

    reward_fn = DateExtractionRewardLogger(
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
