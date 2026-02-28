#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "simple-rlvr-date-normalization"
LOCAL_PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = Path("/root/project")
REMOTE_OUTPUTS_DIR = REMOTE_PROJECT_DIR / "rlvr_outputs"
REMOTE_DATASETS_CACHE_DIR = REMOTE_PROJECT_DIR / ".hf_datasets_cache"
REMOTE_HF_HOME = Path("/root/.cache/huggingface")

DEFAULT_GPU = os.environ.get("MODAL_GPU", "A10")
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_OUTPUT_DIR = "rlvr_outputs/date_normalization"

output_volume = modal.Volume.from_name(
    "simple-rlvr-date-normalization-outputs",
    create_if_missing=True,
)
dataset_cache_volume = modal.Volume.from_name(
    "simple-rlvr-date-normalization-datasets-cache",
    create_if_missing=True,
)
hf_home_volume = modal.Volume.from_name(
    "simple-rlvr-date-normalization-hf-home",
    create_if_missing=True,
)

secret_names = [
    value
    for value in (
        os.environ.get("MODAL_HF_SECRET_NAME"),
        os.environ.get("MODAL_WANDB_SECRET_NAME"),
    )
    if value
]
secrets = [modal.Secret.from_name(name) for name in secret_names]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install(
        "vllm==0.12.0",
        "bitsandbytes>=0.43.0",
    )
    .add_local_dir(
        str(LOCAL_PROJECT_DIR),
        remote_path=str(REMOTE_PROJECT_DIR),
        copy=True,
    )
)

app = modal.App(name=APP_NAME)


def _build_train_command(
    *,
    model_name: str,
    output_dir: str,
    num_episodes: int,
    max_steps: int,
    device: str,
    seed: int,
    log_after_every: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    wandb: bool,
    weave: bool,
    extra_args: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "envs/date_normalization.py",
        "--model_name",
        model_name,
        "--output_dir",
        output_dir,
        "--num_episodes",
        str(num_episodes),
        "--max_steps",
        str(max_steps),
        "--device",
        device,
        "--seed",
        str(seed),
        "--log_after_every",
        str(log_after_every),
    ]

    if load_in_4bit:
        cmd.append("--load_in_4bit")
    if load_in_8bit:
        cmd.append("--load_in_8bit")
    if wandb:
        cmd.append("--wandb")

    cmd.append("--weave" if weave else "--no-weave")

    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    return cmd


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=8.0,
    memory=32768,
    timeout=24 * 60 * 60,
    secrets=secrets,
    volumes={
        str(REMOTE_OUTPUTS_DIR): output_volume,
        str(REMOTE_DATASETS_CACHE_DIR): dataset_cache_volume,
        str(REMOTE_HF_HOME): hf_home_volume,
    },
    env={
        "HF_HOME": str(REMOTE_HF_HOME),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    },
)
def run_date_normalization(
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_episodes: int = 20,
    max_steps: int = 60,
    device: str = "cuda",
    seed: int = 42,
    log_after_every: int = 1,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    wandb: bool = False,
    weave: bool = False,
    extra_args: str = "",
) -> dict[str, str]:
    command = _build_train_command(
        model_name=model_name,
        output_dir=output_dir,
        num_episodes=num_episodes,
        max_steps=max_steps,
        device=device,
        seed=seed,
        log_after_every=log_after_every,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        wandb=wandb,
        weave=weave,
        extra_args=extra_args,
    )

    print(f"Running on Modal GPU={DEFAULT_GPU}")
    print(f"Working directory: {REMOTE_PROJECT_DIR}")
    print(f"Command: {shlex.join(command)}")

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(REMOTE_HF_HOME))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        subprocess.run(
            command,
            cwd=str(REMOTE_PROJECT_DIR),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
    finally:
        output_volume.commit()
        dataset_cache_volume.commit()
        hf_home_volume.commit()

    resolved_output_dir = (
        str((REMOTE_PROJECT_DIR / output_dir).resolve())
        if not Path(output_dir).is_absolute()
        else output_dir
    )
    return {
        "gpu": DEFAULT_GPU,
        "output_dir": resolved_output_dir,
    }


@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_episodes: int = 20,
    max_steps: int = 60,
    device: str = "cuda",
    seed: int = 42,
    log_after_every: int = 1,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    wandb: bool = False,
    weave: bool = False,
    extra_args: str = "",
):
    result = run_date_normalization.remote(
        model_name=model_name,
        output_dir=output_dir,
        num_episodes=num_episodes,
        max_steps=max_steps,
        device=device,
        seed=seed,
        log_after_every=log_after_every,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        wandb=wandb,
        weave=weave,
        extra_args=extra_args,
    )
    print(json.dumps(result, indent=2))
