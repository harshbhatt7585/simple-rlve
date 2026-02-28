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

from training_logging import MetricsJSONLCallback, WeaveTraceLogger, configure_external_logs, format_terminal_log

LOGGER = logging.getLogger("rlvr")
WANDB_PROJECT = "RLVR"
USE_VLLM = True
VLLM_MODE = "colocate"
VLLM_GPU_MEMORY_UTILIZATION = 0.5
VLLM_ENABLE_SLEEP_MODE = False
VLLM_MAX_MODEL_LENGTH = 512
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 4096
LEARNING_RATE = 1e-5
TEMPERATURE = 0.7
BETA = 0.04
SAVE_STEPS = 20
NUM_ITERATIONS = 1
STEPS_PER_GENERATION = 4
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
DEEPSEEK_R1_DISTILL_QWEN_1_5B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEEPSEEK_R1_QWEN_CHAT_TEMPLATE = """{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = '' -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{{- bos_token if bos_token is not none else '' -}}
{{- system_message + '\n' if system_message else '' -}}
{%- for message in loop_messages -%}
    {%- if message['role'] == 'user' -%}
        {{- '<\uff5cUser\uff5c>' + message['content'] -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- '<\uff5cAssistant\uff5c>' + message['content'] + '<\uff5cend\u2581of\u2581sentence\uff5c>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<\uff5cAssistant\uff5c>' -}}
{%- endif -%}
"""
PROMPT_TEMPLATE = """You are given a sentence that may contain a relative time expression.
Normalize the expression to the final calendar date.

Output requirements:
- Return JSON only.
- No explanation, no markdown, no extra keys.
- Use this exact schema: {"date":"YYYY-MM-DD"}
"""
DATE_VALUE_PATTERN = re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
THINK_BLOCK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_PATTERN = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=DEEPSEEK_R1_DISTILL_QWEN_1_5B)
    p.add_argument("--output_dir", default="rlvr_outputs/date_normalization")
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device (defaults to cuda).",
    )
    p.add_argument("--load_in_8bit", action="store_true", help="Load the base model in 8-bit (bitsandbytes).")
    p.add_argument("--load_in_4bit", action="store_true", help="Load the base model in 4-bit (bitsandbytes).")
    p.add_argument(
        "--llm_int8_threshold",
        type=float,
        default=6.0,
        help="Outlier threshold used by 8-bit quantization kernels.",
    )
    p.add_argument(
        "--llm_int8_has_fp16_weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether 8-bit modules keep FP16 main weights.",
    )
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
    p.add_argument(
        "--weave",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log per-completion traces to Weave/W&B. Use --no-weave to disable.",
    )
    p.add_argument(
        "--weave_project",
        default=None,
        help="Weave project name. Accepts either `project` or `entity/project`.",
    )
    return p.parse_args()


@dataclass
class Episode:
    prompt: list[dict[str, str]]
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


def _strip_reasoning_blocks(text: str) -> str:
    cleaned = THINK_BLOCK_PATTERN.sub(" ", text)
    cleaned = THINK_TAG_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    candidates.extend(match.group(0) for match in JSON_OBJECT_PATTERN.finditer(text))
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _extract_json_response(completion_text: str, strict_json_only: bool = True) -> tuple[bool, str | None]:
    raw = completion_text.strip()
    if not raw:
        return False, None
    raw = _strip_reasoning_blocks(raw)
    if not raw:
        return False, None

    def normalize_candidate(value: Any) -> str | None:
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

    for candidate in _json_candidates(raw):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and set(data.keys()) == {"date"} and isinstance(data.get("date"), str):
                return True, normalize_candidate(data.get("date"))
        except Exception:
            continue

    if not strict_json_only:
        return False, normalize_candidate(raw)

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
    def _build_prompt(question: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": PROMPT_TEMPLATE.strip()},
            {"role": "user", "content": f"Sentence: {question}"},
        ]

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

    def __init__(
        self,
        log_path: Path,
        log_after_every: int = 1,
        weave_logger: WeaveTraceLogger | None = None,
    ) -> None:
        self.log_path = log_path
        self.log_after_every = max(0, log_after_every)
        self.weave_logger = weave_logger
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
            if isinstance(completion, list) and completion and isinstance(completion[0], dict):
                completion_text = str(completion[0].get("content", ""))
            else:
                completion_text = str(completion)
            expected_norm = str(expected).strip() if expected is not None else None
            json_valid, strict_predicted = _extract_json_response(completion_text, strict_json_only=True)

            if json_valid:
                year_correct = False
                month_correct = False
                day_correct = False
                if (
                    expected_norm is not None
                    and strict_predicted is not None
                    and ISO_DATE_PATTERN.fullmatch(expected_norm)
                    and ISO_DATE_PATTERN.fullmatch(strict_predicted)
                ):
                    exp_year, exp_month, exp_day = expected_norm.split("-")
                    pred_year, pred_month, pred_day = strict_predicted.split("-")
                    year_correct = exp_year == pred_year
                    month_correct = exp_month == pred_month
                    day_correct = exp_day == pred_day
                reward = (int(year_correct) + int(month_correct) + int(day_correct)) / 3.0
                is_correct = year_correct and month_correct and day_correct
            else:
                strict_predicted = None
                is_correct = False
                reward = -0.25
                year_correct = False
                month_correct = False
                day_correct = False

            if strict_predicted is not None:
                logged_predicted = strict_predicted
            else:
                _, logged_predicted = _extract_json_response(completion_text, strict_json_only=False)

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
                "year_correct": year_correct,
                "month_correct": month_correct,
                "day_correct": day_correct,
                "is_correct": is_correct,
                "total_reward": reward,
                "prompt": prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False),
                "completion": completion_text,
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if self.weave_logger is not None:
                self.weave_logger.log_llm_completion(
                    prompt=record["prompt"],
                    expected_date=expected_norm,
                    completion=completion_text,
                    reward=reward,
                )
            self.episode_id += 1

        if rewards:
            reward_mean = sum(rewards) / len(rewards)
            json_rate = json_valid_count / len(rewards)
            logical_step = max(step, 0)
            if self.log_after_every > 0 and (logical_step + 1) % self.log_after_every == 0:
                LOGGER.info(
                    format_terminal_log(
                        "episode",
                        [
                            ("steps", step),
                            ("reward", f"{reward_mean:.3f}"),
                            ("json", f"{json_rate * 100.0:.1f}%"),
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
    use_quantized = args.load_in_4bit or args.load_in_8bit
    # QLoRA-style path: keep adapter math in BF16 and use BF16 compute for quantized kernels.
    kwargs = {"torch_dtype": torch.bfloat16 if use_quantized else dtype}
    if not args.load_in_4bit and not args.load_in_8bit:
        return kwargs
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one quantization mode at a time: --load_in_4bit or --load_in_8bit.")
    if device != "cuda":
        quant_mode = "--load_in_4bit" if args.load_in_4bit else "--load_in_8bit"
        raise ValueError(f"{quant_mode} requires CUDA; set --device cuda and run on a CUDA GPU.")

    from transformers import BitsAndBytesConfig

    if args.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=args.llm_int8_threshold,
            llm_int8_has_fp16_weight=args.llm_int8_has_fp16_weight,
        )
        kwargs["device_map"] = "auto"
        return kwargs

    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    kwargs["device_map"] = "auto"
    return kwargs


def _use_deepseek_r1_qwen_chat_template(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized.startswith("deepseek-ai/deepseek-r1-distill-qwen")


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
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one quantization mode at a time: --load_in_4bit or --load_in_8bit.")

    has_cuda = args.device == "cuda"
    use_quantized = args.load_in_4bit or args.load_in_8bit
    if use_quantized and not has_cuda:
        raise ValueError("Quantized training requires CUDA. Set --device cuda and run on a CUDA GPU.")
    if use_quantized and not torch.cuda.is_bf16_supported():
        raise ValueError("QLoRA-style BF16 training requires a BF16-capable CUDA GPU.")
    if args.load_in_4bit and args.bnb_4bit_compute_dtype != "bfloat16":
        LOGGER.warning("Overriding --bnb_4bit_compute_dtype=%s to bfloat16 for QLoRA BF16 mode.", args.bnb_4bit_compute_dtype)

    use_cpu = not has_cuda
    use_bf16 = has_cuda and (torch.cuda.is_bf16_supported() or use_quantized)
    use_fp16 = has_cuda and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    use_vllm = USE_VLLM
    if use_cpu and use_vllm:
        LOGGER.warning("Disabling vLLM because --device is set to cpu.")
        use_vllm = False
    if args.load_in_8bit and use_vllm:
        LOGGER.warning("Disabling vLLM because --load_in_8bit is not supported with colocated vLLM.")
        use_vllm = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = DateNormalizationEnv(seed=args.seed)
    train_dataset = env.build_dataset(args.num_episodes)

    if args.wandb:
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    weave_logger = WeaveTraceLogger(
        enabled=args.weave,
        project_name=args.weave_project or os.environ.get("WANDB_PROJECT") or WANDB_PROJECT,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if _use_deepseek_r1_qwen_chat_template(args.model_name):
        # Keep the reasoning tokens in the generated text so reward parsing can strip them explicitly.
        tokenizer.chat_template = DEEPSEEK_R1_QWEN_CHAT_TEMPLATE

    reward_fn = DateExtractionRewardFunction(
        output_dir / "episode_rewards.jsonl",
        log_after_every=args.log_after_every,
        weave_logger=weave_logger,
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
        vllm_importance_sampling_correction=True,
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
        "Starting training | device=%s bf16=%s 8bit=%s 4bit=%s log_after_every=%s",
        args.device,
        use_bf16,
        args.load_in_8bit,
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
