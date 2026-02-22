#!/usr/bin/env python3
"""Logging utilities for RLVR training."""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

LOGGER = logging.getLogger("rlvr")
ANSWER_PATTERN = re.compile(r"<answer>\s*(-?\d+)\s*</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_FALLBACK_PATTERNS = (
    re.compile(r"\bfinal answer\s*[:=]\s*(-?\d[\d,]*)\b", re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]\s*(-?\d[\d,]*)\b", re.IGNORECASE),
)


def configure_external_logs(show_external_logs: bool = False) -> None:
    """Reduce noisy third-party logs so terminal output stays focused."""
    if show_external_logs:
        return

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    warnings.filterwarnings(
        "ignore",
        message=r"The tokenizer has new PAD/BOS/EOS tokens.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Passing `generation_config` together with generation-related arguments.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Unable to fetch remote file due to the following error .*silently ignoring the lookup for the file config\.json.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Could not find a config file in .* - will assume that the vocabulary was not modified\.",
    )

    # Silence hub/http informational request logs.
    for logger_name in (
        "httpx",
        "urllib3",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "huggingface_hub.utils._http",
        "transformers",
        "accelerate",
        "wandb",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    try:
        from huggingface_hub.utils import disable_progress_bars
        from huggingface_hub.utils import logging as hf_hub_logging

        disable_progress_bars()
        hf_hub_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        pass


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
    if match:
        return match.group(1).strip()

    for pattern in ANSWER_FALLBACK_PATTERNS:
        matches = pattern.findall(text)
        if not matches:
            continue
        candidate = matches[-1].replace(",", "").strip()
        if re.fullmatch(r"-?\d+", candidate):
            return candidate
    return None


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
        prediction_log_count: int = 1,
        wandb_run: Any | None = None,
    ) -> None:
        self.log_path = log_path
        self.episode_id = 0
        self.__name__ = "episode_reward"
        self.terminal_log_every = max(1, terminal_log_every)
        self.sample_log_every = max(0, sample_log_every)
        self.sample_chars = max(40, sample_chars)
        self.prediction_log_count = max(1, prediction_log_count)
        self.wandb_run = wandb_run
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
        sample_records: list[dict[str, Any]] = []

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
            if len(sample_records) < self.prediction_log_count:
                sample_records.append(
                    {
                        "question": q,
                        "expected_answer": expected,
                        "predicted_answer": predicted,
                        "completion": completion_text,
                        "reward": total_reward,
                    }
                )

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
                        "episode_stats step=%s | reward(mean=%.3f min=%.3f max=%.3f) "
                        "| acc=%.1f%% | format=%.1f%% | running_reward=%.3f"
                    ),
                    step,
                    reward_mean,
                    reward_min,
                    reward_max,
                    accuracy * 100.0,
                    format_rate * 100.0,
                    running_reward,
                )
                if self.sample_log_every > 0 and (logical_step + 1) % self.sample_log_every == 0:
                    for record in sample_records:
                        LOGGER.info(
                            (
                                "prediction | reward=%.3f | expected=%s predicted=%s "
                                "| text=%s"
                            ),
                            float(record["reward"]),
                            record["expected_answer"],
                            record["predicted_answer"],
                            _clip_text(str(record["completion"]), self.sample_chars),
                        )

            if self.wandb_run is not None:
                wandb_step = logical_step + 1
                payload: dict[str, Any] = {
                    "episode/reward_mean": reward_mean,
                    "episode/reward_min": reward_min,
                    "episode/reward_max": reward_max,
                    "episode/accuracy": accuracy,
                    "episode/format_rate": format_rate,
                    "episode/running_reward": running_reward,
                }
                if sample_records:
                    payload["episode/prediction_text"] = _clip_text(str(sample_records[0]["completion"]), self.sample_chars)
                    payload["episode/prediction_reward"] = float(sample_records[0]["reward"])
                    payload["episode/predicted_answer"] = str(sample_records[0]["predicted_answer"])
                    payload["episode/expected_answer"] = str(sample_records[0]["expected_answer"])
                try:
                    self.wandb_run.log(payload, step=wandb_step)
                except Exception:
                    pass

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
        excluded_keys = {"reward_std", "learning_rate", "lr"}
        numeric_logs = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in logs.items()
            if isinstance(v, (int, float, str)) and k not in excluded_keys
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
                    "train step=%s (%.1f%%) | reward=%.3f | "
                    "entropy=%.3f | comp_len=%.1f | step_time=%.2fs"
                ),
                progress,
                progress_pct,
                float(payload.get("reward", 0.0)),
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
