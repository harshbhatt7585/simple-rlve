#!/usr/bin/env python3
"""Logging utilities for RLVR training."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
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
    warnings.filterwarnings(
        "ignore",
        message=r"Merge lora module to 4-bit linear may get different generations due to rounding errors\.",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Unmerge lora module to 4-bit linear may get different generations due to rounding errors\.",
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


def _supports_ansi_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return sys.stdout.isatty()


def _color(text: str, ansi_code: str | None) -> str:
    if ansi_code is None or not _supports_ansi_color():
        return text
    return f"\033[{ansi_code}m{text}\033[0m"


def format_terminal_log(
    label: str,
    fields: list[tuple[str, Any]],
    color_code: str | None = None,
) -> str:
    tag = _color(f"[{label.upper():<8}]", color_code)
    rendered_fields = []
    for key, value in fields:
        if value is None:
            continue
        rendered_fields.append(f"{key}={value}")
    return f"{tag} " + "  ".join(rendered_fields)


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
        self.terminal_log_every = max(0, terminal_log_every)
        self.sample_log_every = max(0, sample_log_every)
        self.sample_chars = max(40, sample_chars)
        self.prediction_log_count = max(1, prediction_log_count)
        self.wandb_run = wandb_run
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
                "steps": step,
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
            batch_reward_sum = sum(rewards)
            reward_mean = batch_reward_sum / batch_size
            reward_min = min(rewards)
            reward_max = max(rewards)
            accuracy = correct_count / batch_size
            format_rate = format_count / batch_size
            self.running_episode_count += batch_size

            logical_step = max(step, 0)
            if self.terminal_log_every > 0 and (logical_step + 1) % self.terminal_log_every == 0:
                LOGGER.info(
                    format_terminal_log(
                        "episode",
                        [
                            ("steps", step),
                            ("reward", f"{reward_mean:.3f}"),
                            ("acc", f"{accuracy * 100.0:.1f}%"),
                            ("format", f"{format_rate * 100.0:.1f}%"),
                        ],
                        color_code="36",
                    )
                )
                if self.sample_log_every > 0 and (logical_step + 1) % self.sample_log_every == 0:
                    for record in sample_records:
                        LOGGER.info(
                            format_terminal_log(
                                "sample",
                                [
                                    ("reward", f"{float(record['reward']):.3f}"),
                                    ("expected", record["expected_answer"]),
                                    ("predicted", record["predicted_answer"]),
                                    ("text", _clip_text(str(record["completion"]), self.sample_chars)),
                                ],
                                color_code="35",
                            )
                        )

            if self.wandb_run is not None:
                wandb_step = logical_step + 1
                payload: dict[str, Any] = {
                    "episode/reward_mean": reward_mean,
                    "episode/reward_sum": batch_reward_sum,
                    "episode/reward_min": reward_min,
                    "episode/reward_max": reward_max,
                    "episode/accuracy": accuracy,
                    "episode/format_rate": format_rate,
                    "episode/episodes_seen": self.running_episode_count,
                }
                if sample_records:
                    payload["episode/prediction_text"] = _clip_text(str(sample_records[0]["completion"]), self.sample_chars)
                    payload["episode/prediction_reward"] = float(sample_records[0]["reward"])
                    payload["episode/predicted_answer"] = str(sample_records[0]["predicted_answer"])
                    payload["episode/expected_answer"] = str(sample_records[0]["expected_answer"])
                payload["episode/rollout_steps"] = wandb_step
                try:
                    # Do not pass explicit `step` because trainer integrations may already advance
                    # internal wandb steps, which can trigger out-of-order warnings.
                    self.wandb_run.log(payload)
                except Exception:
                    pass

        return rewards


class MetricsJSONLCallback(TrainerCallback):
    """Writes trainer logs to JSONL for easy plotting."""

    def __init__(self, path: Path, max_steps: int, terminal_log_every: int = 1) -> None:
        self.path = path
        self.max_steps = max_steps
        self.terminal_log_every = max(0, terminal_log_every)
        self._wandb_metrics_initialized = False
        self._last_rollout_steps = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def _log_to_wandb(self, payload: dict[str, Any], steps: int) -> None:
        try:
            import wandb
        except Exception:
            return

        run = getattr(wandb, "run", None)
        if run is None:
            return

        if not self._wandb_metrics_initialized:
            try:
                wandb.define_metric("steps")
                wandb.define_metric("reward", step_metric="steps")
                wandb.define_metric("/train/reward", step_metric="steps")
                wandb.define_metric("entropy", step_metric="steps")
                wandb.define_metric("completions/mean_length", step_metric="steps")
                wandb.define_metric("step_time", step_metric="steps")
            except Exception:
                pass
            self._wandb_metrics_initialized = True

        wandb_payload = {
            key: value for key, value in payload.items() if isinstance(value, (int, float))
        }
        if "steps" in wandb_payload:
            wandb_payload["rollout_steps"] = wandb_payload["steps"]
        wandb_payload["steps"] = steps
        if "reward" in payload:
            wandb_payload["/train/reward"] = float(payload["reward"])
        try:
            # Keep custom x-axis in payload while letting wandb own internal `_step`.
            run.log(wandb_payload)
        except Exception:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not state.is_local_process_zero or not logs:
            return
        excluded_keys = {"reward_std", "learning_rate", "lr"}
        numeric_logs = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in logs.items()
            if isinstance(v, (int, float, str)) and k not in excluded_keys
        }
        trainer_steps = int(state.global_step)
        payload = {"steps": trainer_steps, "epoch": float(state.epoch or 0.0), **numeric_logs}

        rollout_candidate: int | None = None
        if "steps" in numeric_logs:
            try:
                rollout_candidate = int(float(numeric_logs["steps"]))
            except Exception:
                rollout_candidate = None
        elif "global_step" in numeric_logs:
            try:
                rollout_candidate = int(float(numeric_logs["global_step"]))
            except Exception:
                rollout_candidate = None

        if rollout_candidate is not None:
            # Prefer rollout/global-step semantics when they move faster than trainer global_step,
            # but never let a stale low value (e.g., always 1) pin the terminal step counter.
            payload["steps"] = max(trainer_steps, rollout_candidate, self._last_rollout_steps)
        elif self._last_rollout_steps > 0:
            payload["steps"] = max(trainer_steps, self._last_rollout_steps)
        else:
            payload["steps"] = trainer_steps

        try:
            payload["steps"] = int(float(payload["steps"]))
        except Exception:
            payload["steps"] = int(max(self._last_rollout_steps, trainer_steps))

        if payload["steps"] < self._last_rollout_steps:
            payload["steps"] = self._last_rollout_steps
        self._last_rollout_steps = payload["steps"]

        payload.pop("global_step", None)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        steps = int(payload["steps"])
        logical_step = max(steps, 1)
        should_log = ("train_runtime" in payload) or (
            self.terminal_log_every > 0 and logical_step % self.terminal_log_every == 0
        )
        if not should_log:
            return

        # Keep wandb step aligned with Trainer global_step to avoid out-of-order step warnings.
        self._log_to_wandb(payload, steps=int(state.global_step))

        if "reward" in payload:
            LOGGER.info(
                format_terminal_log(
                    "train",
                    [
                        ("steps", steps),
                        ("reward", f"{float(payload.get('reward', 0.0)):.3f}"),
                    ],
                    color_code="32",
                )
            )
        else:
            LOGGER.info(
                format_terminal_log(
                    "done",
                    [
                        ("runtime", f"{float(payload.get('train_runtime', 0.0)):.2f}s"),
                        ("train_loss", f"{float(payload.get('train_loss', 0.0)):.4f}"),
                    ],
                    color_code="33",
                )
            )
