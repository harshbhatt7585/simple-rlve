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
from typing import Any, List

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
MAX_ROLLOUT_TURNS = 3
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
    p.add_argument("--terminal_log_every", type=int, default=1)
    p.add_argument("--sample_log_every", type=int, default=1)
    p.add_argument("--wandb", action="store_true")

    return p.parse_args()


@dataclass
class Message:
    prompt: str
    question: str
    answer: str


@dataclass
class Episode:
    messages: List[Message]


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
        for candidate_variant in (normalized, text):
            try:
                return datetime.strptime(candidate_variant, fmt)
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
        if not isinstance(data, dict):
            return True, None
        if "date" not in data:
            return True, None
        return True, str(data["date"])
    return False, None


def _extract_expected_date(target_output: Any) -> str | None:
    raw = str(target_output).strip() if target_output is not None else ""
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except Exception:
        return None

    resolved_value: str | None = None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        resolved_value = str(data[0].get("resolved_value", "")).strip() or None
    elif isinstance(data, dict):
        resolved_value = str(data.get("resolved_value", "")).strip() or None

    if resolved_value is None:
        return None
    if not ISO_DATE_PATTERN.fullmatch(resolved_value):
        return None
    return resolved_value


def _extract_messages(input_text: Any, target_output: Any) -> list[Message]:
    question = str(input_text).strip()
    answer = _extract_expected_date(target_output)
    if not question or answer is None:
        return []

    prompt = f"{PROMPT_TEMPLATE}\nSentence: {question}"
    return [Message(prompt=prompt, question=question, answer=answer)]


class DateExtractionRewardLogger:
    """Reward function with strict JSON parsing and date correctness checks."""

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
        trajectory_reward: list[float] | None = None,
        trajectory_done: list[bool] | None = None,
        trajectory_turns: list[int] | None = None,
        global_step: list[int] | None = None,
        trainer_state=None,
        **_: Any,
    ) -> list[float]:
        rewards: list[float] = []
        rollout_global_step = (
            int(global_step[0]) if global_step is not None and len(global_step) > 0 else
            (int(trainer_state.global_step) if trainer_state is not None else -1)
        )
        json_valid_count = 0
        correct_count = 0
        trajectory_done_count = 0
        trajectory_turn_sum = 0
        has_trajectory_done = trajectory_done is not None
        has_trajectory_turns = trajectory_turns is not None
        sample_records: list[dict[str, Any]] = []

        for idx, (prompt, completion, expected, q) in enumerate(zip(prompts, completions, answer, question, strict=True)):
            completion_text = _as_text(completion)
            expected_norm = _normalize_date(expected)
            json_valid, json_date_raw = _extract_json_date(completion_text)
            predicted_norm = _normalize_date(json_date_raw) if json_valid else None
            is_correct = expected_norm is not None and predicted_norm == expected_norm
            rollout_done = bool(trajectory_done[idx]) if has_trajectory_done and idx < len(trajectory_done) else None
            rollout_turns = int(trajectory_turns[idx]) if has_trajectory_turns and idx < len(trajectory_turns) else None

            if trajectory_reward is not None and idx < len(trajectory_reward):
                total_reward = float(trajectory_reward[idx])
            else:
                if not json_valid:
                    total_reward = -0.25
                elif is_correct:
                    total_reward = 1.0
                else:
                    total_reward = 0.0

            rewards.append(total_reward)
            json_valid_count += int(json_valid)
            correct_count += int(is_correct)
            if rollout_done is not None:
                trajectory_done_count += int(rollout_done)
            if rollout_turns is not None:
                trajectory_turn_sum += rollout_turns

            if len(sample_records) < self.prediction_log_count:
                sample_records.append(
                    {
                        "question": q,
                        "expected_date": expected_norm,
                        "predicted_date": predicted_norm,
                        "json_valid": json_valid,
                        "reward": total_reward,
                        "trajectory_done": rollout_done,
                        "trajectory_turns": rollout_turns,
                        "completion": completion_text,
                    }
                )

            log_record = {
                "episode_id": self.episode_id,
                "global_step": rollout_global_step,
                "question": q,
                "expected_date": expected_norm,
                "predicted_date": predicted_norm,
                "json_valid": json_valid,
                "is_correct": is_correct,
                "trajectory_done": rollout_done,
                "trajectory_turns": rollout_turns,
                "total_reward": total_reward,
                "prompt": _as_text(prompt),
                "completion": completion_text,
            }
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(log_record) + "\n")
            self.episode_id += 1

        if rewards:
            batch_size = len(rewards)
            reward_mean = sum(rewards) / batch_size
            json_valid_rate = json_valid_count / batch_size
            accuracy = correct_count / batch_size
            logical_global_step = max(rollout_global_step, 0)
            done_rate = (trajectory_done_count / batch_size) if has_trajectory_done else None
            avg_turns = (trajectory_turn_sum / batch_size) if has_trajectory_turns else None

            if self.terminal_log_every > 0 and (logical_global_step + 1) % self.terminal_log_every == 0:
                if done_rate is not None and avg_turns is not None:
                    LOGGER.info(
                        (
                            "episode_stats global_step=%s | reward=%.3f | json=%.1f%% | acc=%.1f%% "
                            "| done=%.1f%% | turns=%.2f"
                        ),
                        rollout_global_step,
                        reward_mean,
                        json_valid_rate * 100.0,
                        accuracy * 100.0,
                        done_rate * 100.0,
                        avg_turns,
                    )
                else:
                    LOGGER.info(
                        (
                            "episode_stats global_step=%s | reward=%.3f | json=%.1f%% | acc=%.1f%%"
                        ),
                        rollout_global_step,
                        reward_mean,
                        json_valid_rate * 100.0,
                        accuracy * 100.0,
                    )

                if self.sample_log_every > 0 and (logical_global_step + 1) % self.sample_log_every == 0:
                    for record in sample_records:
                        LOGGER.info(
                            (
                                "sample | reward=%.3f | expected=%s | predicted=%s | json=%s "
                                "| done=%s | turns=%s | text=%s"
                            ),
                            float(record["reward"]),
                            record["expected_date"],
                            record["predicted_date"],
                            record["json_valid"],
                            record["trajectory_done"],
                            record["trajectory_turns"],
                            _clip_text(str(record["completion"]), self.sample_chars),
                        )

        return rewards


class DateNormalizationEnv:
    NAME = "date_normalization"

    def __init__(self, num_episodes: int, seed: int = 42):
        self.rng = random.Random(seed)
        self.episodes, self.dataset = self.build_dataset(num_episodes)
        self.episode_by_prompt = {
            episode.messages[0].prompt: episode for episode in self.episodes if episode.messages
        }
        self.current_episode: Episode | None = None
        self.current_step = 0
        self.done = False
        if self.episodes:
            self.reset(episode=self.episodes[0])

    def _load_split(self) -> Dataset:
        DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(DATASET_ID, cache_dir=str(DATASET_CACHE_DIR))
        if isinstance(dataset, Dataset):
            return dataset
        for split_name in ("train", "validation", "test"):
            if split_name in dataset:
                return dataset[split_name]
        return next(iter(dataset.values()))

    def reset(self, episode: Episode | None = None, prompt: str | None = None) -> Message:
        if not self.episodes:
            raise ValueError("No episodes available; dataset loading returned zero valid rows.")

        if episode is None and prompt is not None:
            episode = self.episode_by_prompt.get(prompt)
        if episode is None:
            episode = self.rng.choice(self.episodes)

        self.current_episode = episode
        self.current_step = 0
        self.done = False
        if not self.current_episode.messages:
            raise ValueError("Episode has no messages.")
        return self.current_episode.messages[self.current_step]

    def _normalize_action(self, action: Any) -> str | None:
        json_valid, json_date = _extract_json_date(_as_text(action))
        if not json_valid:
            return None
        return _normalize_date(json_date)

    def step(self, action: Any) -> dict[str, Any]:
        if self.current_episode is None:
            raise RuntimeError("Call reset() before step().")

        current = self.current_episode.messages[self.current_step]
        ground_truth = current.answer
        output = self._normalize_action(action)

        if output is None:
            reward = -0.25
        elif output == ground_truth:
            reward = 1.0
            self.done = True
        else:
            reward = 0.0

        return {
            "output": output,
            "ground_truth": ground_truth,
            "reward": reward,
            "done": self.done,
            "step": self.current_step,
        }

    def build_dataset(self, n: int) -> tuple[list[Episode], Dataset]:
        split = self._load_split()
        text_col, answer_col = ("input_text", "target_output")
        total = len(split)
        if total == 0:
            raise ValueError("Loaded dataset split is empty.")

        if n <= total:
            indices = self.rng.sample(range(total), k=n)
        else:
            indices = [self.rng.randrange(total) for _ in range(n)]

        episodes: list[Episode] = []
        rows: list[dict[str, str]] = []
        for idx in indices:
            row = split[int(idx)]
            messages = _extract_messages(row[text_col], row[answer_col])
            if not messages:
                continue
            episode = Episode(messages=messages)
            episodes.append(episode)
            for message in episode.messages:
                rows.append(vars(message))

        if not rows:
            raise ValueError(
                "No valid rows were produced from the dataset. "
                "Check source columns and date format normalization."
            )
        return episodes, Dataset.from_list(rows)


class MultiTurnGRPOTrainer(GRPOTrainer):
    """GRPO trainer with explicit environment rollouts over multiple turns."""

    def __init__(self, *args, rollout_env: DateNormalizationEnv, max_rollout_turns: int = MAX_ROLLOUT_TURNS, **kwargs):
        self.rollout_env = rollout_env
        self.max_rollout_turns = max(1, int(max_rollout_turns))
        self.global_rollout_step = 0
        super().__init__(*args, **kwargs)

    @staticmethod
    def _retry_feedback(transition: dict[str, Any]) -> str:
        if transition.get("output") is None:
            return 'Invalid JSON. Return only {"date":"YYYY-MM-DD"}.'
        return 'Wrong date. Return only {"date":"YYYY-MM-DD"}.'

    def _next_turn_prompt(
        self,
        base_prompt: str,
        turn_idx: int,
        action_text: str,
        transition: dict[str, Any],
    ) -> str:
        feedback = self._retry_feedback(transition)
        return (
            f"{base_prompt}\n\n"
            f"Attempt {turn_idx + 1}: {action_text}\n"
            f"Feedback: {feedback}\n"
            "Try again."
        )

    @staticmethod
    def _align_logprobs(logprobs: list[float] | None, completion_ids: list[int]) -> list[float]:
        if logprobs is None:
            return [0.0] * len(completion_ids)
        if len(logprobs) == len(completion_ids):
            return logprobs
        if len(logprobs) > len(completion_ids):
            return logprobs[: len(completion_ids)]
        return logprobs + [0.0] * (len(completion_ids) - len(logprobs))

    def _generate_single_turn(self, prompts: list):
        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        logprobs: list[list[float]] | None = [] if self.use_vllm else None
        trajectory_reward: list[float] = []
        trajectory_done: list[bool] = []
        trajectory_turns: list[int] = []

        for raw_prompt in prompts:
            prompt_text = _as_text(raw_prompt)
            current_message = self.rollout_env.reset(prompt=prompt_text)
            base_prompt = current_message.prompt
            current_prompt = base_prompt

            total_reward = 0.0
            done = False
            turns_used = 0
            last_prompt_ids: list[int] = []
            last_completion_ids: list[int] = [self.eos_token_id]
            last_logprobs: list[float] | None = None

            for turn_idx in range(self.max_rollout_turns):
                turns_used = turn_idx + 1
                turn_prompt_ids, turn_completion_ids, turn_logprobs, _ = super()._generate_single_turn([current_prompt])
                last_prompt_ids = turn_prompt_ids[0]
                generated_ids = turn_completion_ids[0]
                last_completion_ids = generated_ids if generated_ids else [self.eos_token_id]
                if turn_logprobs is not None:
                    last_logprobs = turn_logprobs[0]

                action_text = self.processing_class.decode(last_completion_ids, skip_special_tokens=True)
                transition = self.rollout_env.step(action_text)
                self.global_rollout_step += 1
                total_reward += float(transition["reward"])
                done = bool(transition["done"])
                if done:
                    break
                current_prompt = self._next_turn_prompt(
                    base_prompt=base_prompt,
                    turn_idx=turn_idx,
                    action_text=action_text,
                    transition=transition,
                )

            prompt_ids.append(last_prompt_ids)
            completion_ids.append(last_completion_ids)
            trajectory_reward.append(total_reward)
            trajectory_done.append(done)
            trajectory_turns.append(turns_used)
            if logprobs is not None:
                logprobs.append(self._align_logprobs(last_logprobs, last_completion_ids))

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["global_step"].append(float(self.global_rollout_step))
        extra_fields = {
            "trajectory_reward": trajectory_reward,
            "trajectory_done": trajectory_done,
            "trajectory_turns": trajectory_turns,
            "global_step": self.global_rollout_step,
        }
        return prompt_ids, completion_ids, logprobs, extra_fields


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

    env = DateNormalizationEnv(args.num_episodes, seed=args.seed)
    train_dataset = env.dataset

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
    grpo_args = GRPOConfig(**grpo_kwargs)

    trainer = MultiTurnGRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
        peft_config=make_lora_config(),
        rollout_env=env,
        max_rollout_turns=MAX_ROLLOUT_TURNS,
    )
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)

    LOGGER.info(
        "Starting training | device=%s bf16=%s 4bit=%s max_rollout_turns=%s",
        args.device,
        use_bf16,
        args.load_in_4bit,
        MAX_ROLLOUT_TURNS,
    )
    trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    LOGGER.info("Done. Model saved to %s", final_dir)


if __name__ == "__main__":
    main()
