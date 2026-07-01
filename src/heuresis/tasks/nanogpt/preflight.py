"""NanoGPT task-specific preflight checks.

Validates GPU memory, data caches, and agent config BEFORE the first harness.run()
so we fail fast with a clear message instead of crashing deep inside bwrap.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "autoresearch"


def check_nanogpt(gpus: list[int]) -> list[str]:
    """Return a list of error strings. Empty list = OK."""
    errors: list[str] = []
    errors.extend(_check_data_cache())
    errors.extend(_check_tokenizer_cache())
    if gpus:
        errors.extend(_check_gpu_memory_free(gpus))
        errors.extend(_check_flash_attn_cache())
    errors.extend(_check_opencode_config())
    return errors


def _check_data_cache() -> list[str]:
    data_dir = CACHE_DIR / "data"
    if not data_dir.is_dir():
        return [
            f"nanogpt data dir missing: {data_dir}. "
            f"Run `bash scripts/setup.sh nanogpt` to download."
        ]
    shard_files = list(data_dir.glob("*.parquet"))
    if not shard_files:
        return [
            f"no *.parquet training shards in {data_dir}. "
            f"Run `bash scripts/setup.sh nanogpt` to download."
        ]
    return []


def _check_tokenizer_cache() -> list[str]:
    tok_dir = CACHE_DIR / "tokenizer"
    if not tok_dir.is_dir():
        return [
            f"nanogpt tokenizer dir missing: {tok_dir}. "
            f"Run `bash scripts/setup.sh nanogpt` to download."
        ]
    return []


def _check_gpu_memory_free(gpus: list[int], min_free_mib: int = 20_000) -> list[str]:
    nvs = shutil.which("nvidia-smi")
    if nvs is None:
        return ["nvidia-smi not found on PATH — cannot verify GPU memory"]
    try:
        out = subprocess.run(
            [nvs, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        return [f"nvidia-smi failed: {e.stderr.strip()}"]
    free_by_idx: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx_s, mib_s = [p.strip() for p in line.split(",", 1)]
        free_by_idx[int(idx_s)] = int(mib_s)
    errors = []
    for g in gpus:
        free = free_by_idx.get(g)
        if free is None:
            errors.append(f"GPU {g} not visible to nvidia-smi")
        elif free < min_free_mib:
            errors.append(
                f"GPU {g} has only {free} MiB free (need >= {min_free_mib}). "
                f"Another process may be using it."
            )
    return errors


def _check_flash_attn_cache() -> list[str]:
    """Verify flash-attn3 kernel is pre-cached so HF_HUB_OFFLINE=1 works in-sandbox."""
    hf_hub = (
        Path(
            os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        )
        / "hub"
    )
    if hf_hub.is_dir() and any("flash-attn" in d.name for d in hf_hub.iterdir()):
        return []
    return [
        f"Flash-attn3 not cached in {hf_hub}\n"
        f"  Fix: venvs/nanogpt/bin/python scripts/preflight_nanogpt.py"
    ]


def _check_opencode_config() -> list[str]:
    # We don't fail hard here — opencode might be configured differently per user.
    # Just warn if the common location is missing.
    cfg = Path.home() / ".config" / "opencode"
    if not cfg.exists():
        return [
            f"opencode config dir missing at {cfg}. "
            f"Run opencode once on this machine to initialize."
        ]
    return []
