"""DiscoGen task helpers.

Utilities for baseline loading, workspace file setup, and meta-test
workspace preparation.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from heuresis.workspace import Workspace

logger = logging.getLogger(__name__)

_DEFAULT_EXCLUDE = {"requirements.txt", "install.sh"}

# Bump when the set/content of patches applied by
# ``patch_modelunlearning_workspace`` or ``prefetch_modelunlearning_data``
# changes in a way that requires rebuilding any cached baseline templates.
# Old caches whose key embeds a different version are ignored (and lazily
# pruned by ``ensure_modelunlearning_baseline_template``).
BASELINE_TEMPLATE_VERSION = "v3"  # v3: rebuild after fast-eval hardlink pollution fix (2026-05-10)

_RUN_MAIN_WALK_BUG = 'dirs[:] = [d for d in dirs if d != "data"]'
_RUN_MAIN_WALK_FIX = (
    'dirs[:] = [d for d in dirs if d != "data" and not d.startswith(".")]'
)

# Upstream make_dataset.py checks if `data/wmdp/wmdp-corpora/` exists and
# skips re-download if so — but a partial extraction (interrupted by an
# agent's `timeout 60 python run_main.py` compile-check) leaves the dir
# present with only some JSONL files, and the skip then masks the
# corruption. Replace the existence check with one that requires every
# expected JSONL file. Idempotent.
_MAKE_DATASET_BUG = (
    "    # Check if extracted dataset folder already exists\n"
    "    extracted_folder = os.path.join(dest_dir, \"wmdp-corpora\")\n"
    "    if os.path.exists(extracted_folder):\n"
    '        print(f"Dataset already exists in {extracted_folder}, skipping download.")\n'
    "        return"
)
_MAKE_DATASET_FIX = (
    "    # Check that the extracted dataset folder contains every expected\n"
    "    # JSONL file -- partial extraction (interrupted compile-check) is\n"
    "    # detected here and triggers a fresh re-download instead of silently\n"
    "    # proceeding with missing data.\n"
    "    #\n"
    "    # NB: the upstream wmdp-corpora.zip ships THREE jsonl files:\n"
    "    # bio-retain, cyber-forget, cyber-retain. bio-forget-corpus.jsonl is\n"
    "    # NOT in this zip -- it lives in a separate wmdp-bio dataset. So we\n"
    "    # check only the 3 files that are actually expected here.\n"
    "    extracted_folder = os.path.join(dest_dir, \"wmdp-corpora\")\n"
    "    _expected = (\n"
    "        \"bio-retain-corpus.jsonl\",\n"
    "        \"cyber-forget-corpus.jsonl\",\n"
    "        \"cyber-retain-corpus.jsonl\",\n"
    "    )\n"
    "    if os.path.exists(extracted_folder) and all(\n"
    "        os.path.isfile(os.path.join(extracted_folder, _f)) for _f in _expected\n"
    "    ):\n"
    '        print(f"Dataset already exists in {extracted_folder}, skipping download.")\n'
    "        return\n"
    "    # If we got here with an existing-but-incomplete extracted_folder,\n"
    "    # remove it so the unzip below starts from a clean slate.\n"
    "    if os.path.exists(extracted_folder):\n"
    "        import shutil as _shutil\n"
    "        _shutil.rmtree(extracted_folder, ignore_errors=True)"
)

# v1 of the make_dataset fix (incorrect: listed bio-forget which is not in
# the wmdp-cyber zip — caused agents to misdiagnose the prefetch as
# incomplete and monkeypatch around it). Kept as a string so the patch can
# forward-migrate any baseline_dir that still has v1.
_MAKE_DATASET_FIX_V1 = (
    "    # Check that the extracted dataset folder contains every expected\n"
    "    # JSONL file -- partial extraction (interrupted compile-check) is\n"
    "    # detected here and triggers a fresh re-download instead of silently\n"
    "    # proceeding with missing data.\n"
    "    extracted_folder = os.path.join(dest_dir, \"wmdp-corpora\")\n"
    "    _expected = (\n"
    "        \"bio-forget-corpus.jsonl\", \"bio-retain-corpus.jsonl\",\n"
    "        \"cyber-forget-corpus.jsonl\", \"cyber-retain-corpus.jsonl\",\n"
    "    )\n"
    "    if os.path.exists(extracted_folder) and all(\n"
    "        os.path.isfile(os.path.join(extracted_folder, _f)) for _f in _expected\n"
    "    ):\n"
    '        print(f"Dataset already exists in {extracted_folder}, skipping download.")\n'
    "        return\n"
    "    # If we got here with an existing-but-incomplete extracted_folder,\n"
    "    # remove it so the unzip below starts from a clean slate.\n"
    "    if os.path.exists(extracted_folder):\n"
    "        import shutil as _shutil\n"
    "        _shutil.rmtree(extracted_folder, ignore_errors=True)"
)


_MAIN_PY_PRINT_LOOP = """    for _, evaluator in evaluators.items():
        summary_file = evaluator.get_logs_file_path(eval_dir, suffix="SUMMARY")
        with open(summary_file, 'r') as f:
            print(json.dumps(json.load(f)))"""

_MAIN_PY_PRINT_MERGED = """    _merged_summary = {}
    for _, evaluator in evaluators.items():
        summary_file = evaluator.get_logs_file_path(eval_dir, suffix="SUMMARY")
        with open(summary_file, 'r') as f:
            _merged_summary.update(json.load(f))
    print(json.dumps(_merged_summary))"""

_MAIN_PY_HF_TOKEN_REQUIRED = '''    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_login(hf_token)
    else:
        print("HF_TOKEN environment variable not set. Aborting.")
        exit(1)'''

_MAIN_PY_HF_TOKEN_OPTIONAL_V1 = '''    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_login(hf_token)
    else:
        print("HF_TOKEN not set; proceeding unauthenticated. Public datasets/models only.")'''

_MAIN_PY_HF_TOKEN_OPTIONAL = '''    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            hf_login(hf_token)
        except Exception as _hf_login_err:
            # An invalid/expired token in the host env should not abort the
            # run -- WMDP-cyber data is a public S3 zip and Qwen2.5-1.5B is
            # open. Warn and proceed; gated downloads will surface a clearer
            # 401 at fetch time if encountered.
            print(f"HF login failed ({_hf_login_err}); proceeding unauthenticated.")
    else:
        print("HF_TOKEN not set; proceeding unauthenticated. Public datasets/models only.")'''

_MODEL_CFG_FLASH_ATTN_LINE = "  attn_implementation: 'flash_attention_2'"
_MODEL_CFG_SDPA_LINE = "  attn_implementation: 'sdpa'"

_MAIN_PY_RUNTIME_COPIES = '''# Copy dataset_config and trainer_config to configs/ before Hydra loads the main config
_base_dir = Path(__file__).resolve().parent
shutil.copy2(_base_dir / "main_config.yaml", _base_dir / "configs" / "main_config.yaml")
shutil.copy2(_base_dir / "trainer_config.yaml", _base_dir / "configs/trainer" / "custom.yaml")
(_base_dir / "configs/model").mkdir(parents=True, exist_ok=True)
shutil.copy2(_base_dir / "model_config.yaml", _base_dir / "configs/model" / "model_conf.yaml")'''

_MAIN_PY_RUNTIME_COPIES_REMOVED = (
    "# (config copies pre-staged by patch_modelunlearning_workspace; "
    "main.py runs on a read-only dataset bind at execute time)"
)


def patch_modelunlearning_workspace(
    src_dir: Path,
    *,
    fast_eval: bool = False,
    fast_eval_limit: int = 200,
    fast_max_steps: int = 10,
) -> None:
    """Apply ModelUnlearning-specific patches that make the workspace runnable
    under our framework.

    Idempotent patches, each restricted to non-editable infrastructure:

    1. ``main.py``: merge per-evaluator SUMMARY JSON dicts into a single dict
       before the final print. Upstream prints ``len(evaluators)`` separate
       JSON lines; ``run_main_performance.py`` only captures the LAST line,
       silently dropping all but one metric. We collect every evaluator
       summary into ``_merged_summary`` and print exactly one JSON.

    2. ``main.py``: relax the ``HF_TOKEN`` ``exit(1)`` to a warning. WMDP-cyber
       data ships from a public S3 zip, Qwen2.5-1.5B-Instruct is open, and
       lm_eval's MMLU subset is public — so an unauthenticated run completes
       fine. If a real auth wall is hit later (gated model swap), the run
       fails naturally at download with a clear HF error.

    3. ``model_config.yaml``: switch ``attn_implementation`` from
       ``flash_attention_2`` to ``sdpa``. flash-attn 2.6.3 (the version
       upstream pins in ``install.sh``) does not build against torch 2.11 +
       CUDA 13 — the build aborts with "too old version" errors deep in
       ``torch.utils.cpp_extension``. SDPA is bundled with PyTorch, ships
       BF16 fused kernels on A100/H100, and is the same fallback the
       upstream ``toggle_attn.py --platform mac`` script applies. The
       throughput cost vs flash-attn for our small model + short seq is
       small (≲15%) and irrelevant to algorithmic ablation.

    4. ``main.py`` + on-disk pre-staging: upstream ``main.py`` copies
       ``main_config.yaml``/``trainer_config.yaml``/``model_config.yaml``
       into ``configs/`` at *runtime* before Hydra loads. Under
       ``Workspace.lock_down_edits=True`` the per-dataset directory is
       bind-mounted **read-only** in the executor sandbox, so those
       runtime copies fail with ``OSError: [Errno 30]``. We pre-stage
       the same files at patch time (writable filesystem) and strip the
       runtime copies from main.py — Hydra reads pre-staged configs,
       no runtime writes into the dataset dir.

    5. (opt-in) ``fast_eval=True``: shrink eval + training for smoke runs.
       Sets ``eval_on_start: false`` (saves the ~5 min pre-train eval that
       isn't used in scoring), adds ``limit: <fast_eval_limit>`` to lm_eval's
       ``simple_evaluate_args`` (caps WMDP+MMLU eval samples; full WMDP is
       1,987 Qs, MMLU-STEM ~3k Qs), and overrides ``max_steps`` to
       ``<fast_max_steps>``. Cuts a typical run from ~25 min to ~5-7 min at
       the cost of noisier scores and an under-trained model — appropriate
       for *infrastructure* smoke, NOT algorithm evaluation. Off by default.

    Raises ``RuntimeError`` if any expected upstream substring is absent
    (upstream format changed).

    Args:
        src_dir: ModelUnlearning workspace root (contains per-dataset
            subdirectories like ``wmdp_cyber_<model>/``).
        fast_eval: If True, apply patch 5 (eval+training shrink). Default False.
        fast_eval_limit: lm_eval ``limit`` parameter when ``fast_eval=True``.
        fast_max_steps: trainer ``max_steps`` override when ``fast_eval=True``.
    """
    targets = sorted(src_dir.glob("*/main.py"))
    if not targets:
        raise RuntimeError(
            f"patch_modelunlearning_workspace: no per-dataset main.py found "
            f"under {src_dir}; expected at least one *_<model_id>/main.py."
        )
    for target in targets:
        text = target.read_text()
        original = text
        if _MAIN_PY_PRINT_LOOP in text:
            text = text.replace(_MAIN_PY_PRINT_LOOP, _MAIN_PY_PRINT_MERGED)
        elif _MAIN_PY_PRINT_MERGED not in text:
            raise RuntimeError(
                f"patch_modelunlearning_workspace: expected evaluator print "
                f"loop not found in {target}; upstream main.py format may "
                f"have changed."
            )
        if _MAIN_PY_HF_TOKEN_REQUIRED in text:
            text = text.replace(
                _MAIN_PY_HF_TOKEN_REQUIRED, _MAIN_PY_HF_TOKEN_OPTIONAL
            )
        elif _MAIN_PY_HF_TOKEN_OPTIONAL_V1 in text:
            # Forward-migrate v1 (warned on missing token but still crashed on
            # invalid token) to the v2 form (try/except wraps hf_login).
            text = text.replace(
                _MAIN_PY_HF_TOKEN_OPTIONAL_V1, _MAIN_PY_HF_TOKEN_OPTIONAL
            )
        elif _MAIN_PY_HF_TOKEN_OPTIONAL not in text:
            raise RuntimeError(
                f"patch_modelunlearning_workspace: expected HF_TOKEN block "
                f"not found in {target}; upstream main.py format may have "
                f"changed."
            )
        if text != original:
            target.write_text(text)

    model_cfgs = sorted(src_dir.glob("*/model_config.yaml"))
    if not model_cfgs:
        raise RuntimeError(
            f"patch_modelunlearning_workspace: no model_config.yaml found "
            f"under {src_dir}/*/."
        )
    for cfg in model_cfgs:
        text = cfg.read_text()
        if _MODEL_CFG_FLASH_ATTN_LINE in text:
            cfg.write_text(
                text.replace(_MODEL_CFG_FLASH_ATTN_LINE, _MODEL_CFG_SDPA_LINE)
            )
        elif _MODEL_CFG_SDPA_LINE not in text:
            raise RuntimeError(
                f"patch_modelunlearning_workspace: expected "
                f"attn_implementation line not found in {cfg}; upstream "
                f"model_config.yaml format may have changed."
            )

    # Patch 4: pre-stage configs that main.py would otherwise copy at
    # runtime, then strip those runtime copies from main.py.
    import shutil as _shutil  # local import — keep top-level imports clean

    for ds_dir in sorted(d for d in src_dir.iterdir() if d.is_dir()):
        main_py = ds_dir / "main.py"
        if not main_py.is_file():
            continue
        # Pre-stage configs (idempotent).
        configs_dir = ds_dir / "configs"
        if (ds_dir / "main_config.yaml").is_file():
            _shutil.copy2(ds_dir / "main_config.yaml", configs_dir / "main_config.yaml")
        if (ds_dir / "trainer_config.yaml").is_file():
            (configs_dir / "trainer").mkdir(parents=True, exist_ok=True)
            _shutil.copy2(
                ds_dir / "trainer_config.yaml",
                configs_dir / "trainer" / "custom.yaml",
            )
        if (ds_dir / "model_config.yaml").is_file():
            (configs_dir / "model").mkdir(parents=True, exist_ok=True)
            _shutil.copy2(
                ds_dir / "model_config.yaml",
                configs_dir / "model" / "model_conf.yaml",
            )
        text = main_py.read_text()
        if _MAIN_PY_RUNTIME_COPIES in text:
            main_py.write_text(
                text.replace(_MAIN_PY_RUNTIME_COPIES, _MAIN_PY_RUNTIME_COPIES_REMOVED)
            )
        elif _MAIN_PY_RUNTIME_COPIES_REMOVED not in text:
            raise RuntimeError(
                f"patch_modelunlearning_workspace: expected runtime config-"
                f"copy block not found in {main_py}; upstream main.py "
                f"format may have changed."
            )

    # Patch 5 (always-on): harden upstream make_dataset.py to detect
    # partial extraction. Required for ModelUnlearning datasets that ship
    # data via a S3 zip (e.g. wmdp_cyber). A no-op for any per-dataset
    # workspace whose make_dataset.py does NOT contain the upstream skip
    # block (e.g. tofu, muse — which use HF datasets).
    for ds_dir in sorted(d for d in src_dir.iterdir() if d.is_dir()):
        md = ds_dir / "make_dataset.py"
        if not md.is_file():
            continue
        text = md.read_text()
        if _MAKE_DATASET_BUG in text:
            md.write_text(text.replace(_MAKE_DATASET_BUG, _MAKE_DATASET_FIX))
        elif _MAKE_DATASET_FIX_V1 in text:
            # Forward-migrate v1 (wrong _expected list including bio-forget,
            # which isn't in the wmdp-cyber zip) to the corrected v2 form.
            md.write_text(text.replace(_MAKE_DATASET_FIX_V1, _MAKE_DATASET_FIX))

    if fast_eval:
        apply_fast_eval_patches(
            src_dir,
            limit=fast_eval_limit,
            max_steps=fast_max_steps,
        )


def prefetch_modelunlearning_data(
    src_dir: Path,
    *,
    venv_python: Path,
    timeout: int = 600,
) -> None:
    """Pre-fetch per-dataset data into baseline_dir before any executor spawns.

    Each per-dataset workspace under ``src_dir`` contains a ``make_dataset.py``
    that downloads + extracts the canonical corpus into ``./data/`` relative to
    the script's cwd. We invoke each ``make_dataset.py`` once with cwd =
    ``src_dir`` (so data lands at ``src_dir/data/...``), populating the
    baseline workspace. ``Workspace.files`` then hardlinks ``data/`` read-only
    into every executor sandbox — eliminating both the per-executor S3
    download (~5 min × N executors) and the make_dataset race that would
    otherwise corrupt extraction when an agent's ``timeout 60 python
    run_main.py`` compile-check kills mid-unzip.

    No-op when no ``make_dataset.py`` is present (e.g. a fresh clone before
    create_task ran). The make_dataset path is task-domain specific; this
    helper only runs for ModelUnlearning workspaces.

    Args:
        src_dir: Baseline workspace root (must already have been populated by
            ``discogen.create_task()`` and patched by
            ``patch_modelunlearning_workspace()`` so the hardened
            ``make_dataset.py`` is in place).
        venv_python: Path to the task venv's python — the upstream
            ``make_dataset.py`` is stdlib-only but kept consistent with how
            other framework helpers shell out.
        timeout: Wall-clock seconds for each make_dataset.py invocation.

    Raises:
        subprocess.CalledProcessError: A make_dataset.py exited non-zero.
        subprocess.TimeoutExpired: Took longer than ``timeout`` seconds.
    """
    import subprocess as _sp

    for ds_dir in sorted(d for d in src_dir.iterdir() if d.is_dir()):
        md = ds_dir / "make_dataset.py"
        if not md.is_file():
            continue
        # ``make_dataset.py`` is a module that *defines* ``make_dataset()`` —
        # the actual call lives in main.py. Running ``python make_dataset.py``
        # would be a no-op (no ``__main__`` block). We import the module and
        # invoke ``make_dataset()`` explicitly. cwd=src_dir so the relative
        # ``data/`` path inside the function lands at src_dir/data/.
        runner = (
            "import importlib.util as _u, sys as _s\n"
            f"_spec = _u.spec_from_file_location('mds', r'{md}')\n"
            "_m = _u.module_from_spec(_spec); _s.modules['mds'] = _m\n"
            "_spec.loader.exec_module(_m)\n"
            "_m.make_dataset()\n"
        )
        _sp.run(
            [str(venv_python), "-c", runner],
            cwd=src_dir,
            check=True,
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# Baseline-template cache (C2 refactor)
# ---------------------------------------------------------------------------
#
# Pre-build a fully-resolved baseline workspace once per (domain, train_pair,
# test_pair, template_backend, BASELINE_TEMPLATE_VERSION). Subsequent
# experiments hardlink that template via ``cp -al`` instead of regenerating
# from upstream templates + patching + re-prefetching every launch. Eliminates
# the make_dataset race, the agent-vs-read-only-mount confusion, and the
# 6-patches-in-sequence runtime overhead.


def _baseline_template_cache_key(
    *,
    domain: str,
    train_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]] | None,
    template_backend: str,
    use_base: bool,
) -> str:
    """Compute a stable cache key for a baseline template.

    The key captures every input that would change the template's contents:
    domain, train/test pairs, backend, use_base flag, and a framework patch
    version so a bump invalidates all caches at once. The short SHA suffix
    keeps the directory name readable while still uniquely identifying the
    config tuple.
    """
    parts = [
        BASELINE_TEMPLATE_VERSION,
        domain,
        template_backend,
        f"use_base={use_base}",
        "train=" + ",".join(f"{t}@{m}" for t, m in train_pairs),
        "test=" + ",".join(f"{t}@{m}" for t, m in (test_pairs or [])),
    ]
    raw = "|".join(parts).encode()
    digest = hashlib.sha256(raw).hexdigest()[:12]
    safe_train = "+".join(t for t, _ in train_pairs).replace("/", "-")
    return f"{domain}_{safe_train}_{template_backend}_{digest}"


def ensure_modelunlearning_baseline_template(
    *,
    domain: str,
    train_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]] | None = None,
    template_backend: str = "default",
    use_base: bool = True,
    template_root: Path,
    venv_python: Path,
    timeout: int = 1800,
) -> Path:
    """Return the path to a fully-resolved baseline template, building it
    if absent.

    On first call for a given ``(domain, train_pairs, test_pairs, backend,
    use_base, BASELINE_TEMPLATE_VERSION)`` combination, this method:

    1. Calls ``discogen.create_task()`` to materialize the upstream task
       template into a staging directory.
    2. Applies all framework patches via ``patch_run_main_walk`` and
       ``patch_modelunlearning_workspace`` (no ``fast_eval`` — that overlay
       is applied per-experiment after cloning).
    3. Pre-fetches the canonical dataset via ``prefetch_modelunlearning_data``.
    4. Atomically promotes the staging directory to the final cache path,
       writing a ``.baseline_ready`` sentinel.

    Subsequent calls (typically < 100 ms) just verify the sentinel and
    return. Concurrent builders are serialized via an ``fcntl.flock`` on a
    per-cache-key lock file so two parallel ``run.py`` launches share the
    same template without racing.

    Args:
        domain: Discogen task domain (e.g. ``"ModelUnlearning"``).
        train_pairs: List of ``(task_id, model_id)`` for the training task(s).
        test_pairs: Optional held-out test pairs; folded into the cache key
            so a config change rebuilds.
        template_backend: Template backend (typically ``"default"``).
        use_base: Whether discogen should materialize baseline implementations
            rather than empty interface stubs.
        template_root: Cache root. Convention:
            ``venvs/discogen/<domain>/baseline_template/``.
        venv_python: Path to the task venv's python — used by ``make_dataset.py``
            for the corpus download/unzip step.
        timeout: Wall-clock seconds for the prefetch step.

    Returns:
        Path to the cache directory. Caller should hardlink (``cp -al``) this
        into the per-experiment ``baseline_dir`` via ``clone_baseline_template``.

    Raises:
        Whatever the underlying builders raise (subprocess errors,
        ``RuntimeError`` from patches when upstream format drifts, etc.).
        On failure, partial staging directories are cleaned up so a retry
        starts from a clean slate.
    """
    key = _baseline_template_cache_key(
        domain=domain,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        template_backend=template_backend,
        use_base=use_base,
    )
    template_dir = template_root / key
    sentinel = template_dir / ".baseline_ready"

    # Fast path — no lock needed for the common case.
    if sentinel.is_file():
        return template_dir

    template_root.mkdir(parents=True, exist_ok=True)
    lock_path = template_root / f"{key}.lock"
    logger.info(
        "Baseline template not yet cached at %s — acquiring build lock",
        template_dir,
    )
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        # Re-check under the lock: another process may have built it while
        # we were waiting.
        if sentinel.is_file():
            return template_dir

        # Wipe any partial state from a prior crashed builder.
        if template_dir.exists():
            shutil.rmtree(template_dir)

        # Build into a staging dir to make the promotion atomic.
        staging = template_root / f"{key}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        try:
            from discogen import create_task

            config_dict: dict[str, Any] = {
                "train_task_id": [t for t, _ in train_pairs],
                "test_task_id": [t for t, _ in (test_pairs or [])],
                "train_model_id": [m for _, m in train_pairs],
                "test_model_id": [m for _, m in (test_pairs or [])],
                "source_path": str(staging),
                "template_backend": template_backend,
                # Editable-modules flags need to be set explicitly; the
                # ModelUnlearning domain only has ``change_loss``. Other
                # discogen domains call this helper with their own flags.
                "change_loss": True,
            }
            logger.info("Building baseline template at %s ...", staging)
            create_task(
                task_domain=domain,
                test=False,
                config_dict=config_dict,
                no_data=False,
                use_base=use_base,
            )
            patch_run_main_walk(staging)
            # fast_eval is NOT applied here -- it's a per-experiment overlay
            # that the caller (run.py) applies after cloning.
            patch_modelunlearning_workspace(staging)
            prefetch_modelunlearning_data(
                staging, venv_python=venv_python, timeout=timeout
            )
            (staging / ".baseline_ready").write_text(
                f"version={BASELINE_TEMPLATE_VERSION}\n"
                f"key={key}\n"
            )
            # Atomic promote.
            staging.rename(template_dir)
            logger.info("Baseline template ready: %s", template_dir)
        except BaseException:
            # Best-effort cleanup so the next call starts from a clean state.
            shutil.rmtree(staging, ignore_errors=True)
            raise

    return template_dir


def clone_baseline_template(template_dir: Path, dest: Path) -> None:
    """Materialize a baseline template into ``dest`` via ``cp -al``.

    Hardlinks every file under ``template_dir`` into ``dest`` (including the
    pre-extracted corpus). Total disk impact per experiment is the size of
    new inodes / metadata only — the multi-GB data files share inodes with
    the template.

    Args:
        template_dir: Path returned by ``ensure_modelunlearning_baseline_template``.
        dest: New baseline directory (typically ``exp.dir / "src"``). MUST
            NOT already exist.

    Raises:
        FileExistsError: ``dest`` already exists.
        RuntimeError: ``cp -al`` failed. The framework requires hardlink-
            capable filesystems (CephFS supports this); a slow-copy fallback
            is intentionally not provided because it would mask a configuration
            problem.
    """
    if dest.exists():
        raise FileExistsError(
            f"clone_baseline_template: {dest} already exists; refusing to clobber"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["cp", "-al", str(template_dir), str(dest)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cp -al {template_dir} -> {dest} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()!r}. The framework requires baseline_dir on "
            f"a hardlink-capable filesystem that shares devices with the template "
            f"cache (CephFS is supported)."
        )


def apply_fast_eval_patches(src_dir: Path, *, limit: int, max_steps: int) -> None:
    """Shrink eval + training for smoke runs (opt-in via fast_eval=True).

    Three idempotent edits applied to the per-dataset configs (already
    pre-staged under ``configs/`` by patch 4):

    * ``configs/trainer/custom.yaml``: add ``eval_on_start: False`` (skips the
      pre-train evaluation; baseline metrics aren't used in scoring anyway)
      and override ``max_steps`` to ``max_steps``.
    * ``configs/eval/<*>.yaml`` (the active eval config): add ``limit: <limit>``
      under ``simple_evaluate_args`` so lm_eval evaluates only the first N
      examples per task.

    Hardlink safety: ``src_dir`` is typically a ``cp -al`` clone of the
    cached baseline template, so files share inodes with the template.
    Every edit goes through :func:`_write_breaking_hardlink` to ensure
    we never mutate the template via a shared inode.
    """
    for ds_dir in sorted(d for d in src_dir.iterdir() if d.is_dir()):
        # Trainer overrides
        custom = ds_dir / "configs" / "trainer" / "custom.yaml"
        if custom.is_file():
            text = custom.read_text()
            new_text = _yaml_set_arg(text, "eval_on_start", "False")
            new_text = _yaml_set_arg(new_text, "max_steps", str(max_steps))
            if new_text != text:
                _write_breaking_hardlink(custom, new_text)
        # Eval limit -- patch every eval/*.yaml that has simple_evaluate_args.
        eval_dir = ds_dir / "configs" / "eval"
        if eval_dir.is_dir():
            for eval_cfg in sorted(eval_dir.glob("*.yaml")):
                text = eval_cfg.read_text()
                if "simple_evaluate_args:" not in text:
                    continue
                new_text = _yaml_set_subkey(
                    text, "simple_evaluate_args", "limit", str(limit)
                )
                if new_text != text:
                    _write_breaking_hardlink(eval_cfg, new_text)


def _write_breaking_hardlink(path: Path, content: str) -> None:
    """Write *content* to *path* without mutating any hardlinked twin.

    ``cp -al`` materialises per-run baselines as hardlink trees over the
    cached template, so naive ``Path.write_text`` truncates the shared
    inode and the edit propagates back into the template (silent cache
    pollution). We unlink first so the new write goes to a fresh inode;
    the template keeps the original inode.
    """
    if path.exists():
        path.unlink()
    path.write_text(content)


def _yaml_set_arg(text: str, key: str, value: str) -> str:
    """Idempotently set ``args.<key>: <value>`` in a trainer custom.yaml.

    Replaces an existing line matching ``  <key>: ...`` (two-space indent
    used by upstream's args block) or appends one inside the ``args:`` block.
    """
    line = f"  {key}: {value}"
    pattern = re.compile(rf"^  {re.escape(key)}:.*$", re.M)
    if pattern.search(text):
        return pattern.sub(line, text)
    # Append inside args block. The block ends at EOF or the next top-level key.
    if re.search(r"^args:\s*$", text, re.M):
        return text.rstrip() + "\n" + line + "\n"
    # No args block (shouldn't happen for trainer custom.yaml) — append top-level.
    return text.rstrip() + "\n" + line + "\n"


def _yaml_set_subkey(text: str, parent: str, subkey: str, value: str) -> str:
    """Idempotently set ``<parent>.<subkey>: <value>`` in an eval yaml.

    Operates inside the parent block (two-space-indented children).
    """
    sub_line = f"  {subkey}: {value}"
    sub_pattern = re.compile(rf"^  {re.escape(subkey)}:.*$", re.M)
    parent_re = re.compile(rf"^{re.escape(parent)}:\s*$", re.M)
    parent_match = parent_re.search(text)
    if not parent_match:
        return text
    # Look for existing subkey within reasonable scope (next 50 lines)
    head = text[: parent_match.end()]
    rest = text[parent_match.end():]
    if sub_pattern.search(rest):
        rest = sub_pattern.sub(sub_line, rest, count=1)
        return head + rest
    # Append subkey just after the parent line.
    return head + "\n" + sub_line + rest


def load_unlearning_baselines(
    domain: str,
    backend: str,
    train_pairs: list[tuple[str, str]] | None = None,
    test_pairs: list[tuple[str, str]] | None = None,
) -> tuple[dict[str, dict[str, tuple[float, str]]], dict[str, dict[str, tuple[float, str]]]]:
    """Load ModelUnlearning baselines into a multi-metric, per-objective shape.

    Upstream's ``baseline_scores.yaml`` for ModelUnlearning is keyed
    metric-first (``wmdp_cyber/acc``, ``mmlu_stemp/acc``, …) with each metric
    carrying its own ``objective``. This is fundamentally different from
    OnPolicyRL's single-metric layout and incompatible with
    :func:`load_baselines`. This helper inverts that structure into a form
    the :class:`~heuresis.tasks.discogen.grader_unlearning.ModelUnlearningGrader`
    consumes directly.

    Output structure::

        {
            "./wmdp_cyber_Qwen2.5-1.5B-Instruct": {
                "wmdp_cyber/acc": (0.24761, "min"),
                "mmlu_stemp/acc": (0.51792, "max"),
            },
            ...
        }

    The dataset key matches what discogen's per-dataset workspace dir is named
    (``<task_id>_<model_id>``) and what ``run_main.py``'s ``os.walk`` reports
    via the ``"./<dirname>"`` prefix.

    Args:
        domain: ``"ModelUnlearning"`` (other domains will raise).
        backend: Template backend, typically ``"default"``.
        train_pairs: Train (task_id, model_id) tuples to include. ``None``
            yields an empty dict.
        test_pairs: Test (task_id, model_id) tuples to include. ``None``
            yields an empty dict.

    Returns:
        ``(train_baselines, test_baselines)`` ready for ``ModelUnlearningGrader``.

    Raises:
        ValueError: A requested ``(task_id, model_id)`` has no metrics in
            the upstream YAML for the given backend.
    """
    import discogen

    baselines_path = (
        Path(discogen.__file__).parent
        / "domains"
        / domain
        / "utils"
        / "baseline_scores.yaml"
    )
    with open(baselines_path) as f:
        raw = yaml.safe_load(f)

    # Upstream `baseline_scores.yaml` carries known metric-name typos that
    # disagree with what the per-dataset evaluator actually emits. Apply
    # well-known fixups at load time so the grader matches what main.py
    # prints. Add new entries here as upstream typos surface.
    _METRIC_TYPO_FIXUPS = {
        "mmlu_stemp/acc": "mmlu_stem/acc",  # upstream WMDP-cyber baseline
    }

    # Invert: walk metric → backend → task_id, build {task_id: {metric: (val, obj)}}
    by_task: dict[str, dict[str, tuple[float, str]]] = {}
    for metric_name, metric_data in raw.items():
        if not isinstance(metric_data, dict):
            continue
        objective = metric_data.get("objective")
        if not isinstance(objective, str):
            continue
        backend_data = metric_data.get(backend)
        if not isinstance(backend_data, dict):
            continue
        canonical_name = _METRIC_TYPO_FIXUPS.get(metric_name, metric_name)
        for task_id, score in backend_data.items():
            by_task.setdefault(task_id, {})[canonical_name] = (float(score), objective)

    def _restrict(pairs: list[tuple[str, str]] | None) -> dict[str, dict[str, tuple[float, str]]]:
        if pairs is None:
            return {}
        out: dict[str, dict[str, tuple[float, str]]] = {}
        missing: list[tuple[str, str]] = []
        for task_id, model_id in pairs:
            if task_id not in by_task:
                missing.append((task_id, model_id))
                continue
            # Workspace dir name and run_main.py walk key.
            dataset_key = f"./{task_id}_{model_id}"
            out[dataset_key] = by_task[task_id]
        if missing:
            raise ValueError(
                f"ModelUnlearning pairs without baselines: {missing}. "
                f"Available task_ids in YAML: {sorted(by_task)}."
            )
        return out

    return _restrict(train_pairs), _restrict(test_pairs)


def patch_run_main_walk(src_dir: Path) -> None:
    """Patch the os.walk filter in upstream discogen's ``run_main.py``.

    Upstream ``run_main_{performance,time,energy}.py`` excludes only
    directories literally named ``data``, so ``os.walk`` descends into
    ``.venv/`` and tries to execute ``.venv/lib/.../scipy/_lib/cobyqa/main.py``
    as a task entrypoint, which crashes on relative imports. This patch
    extends the filter to also skip hidden directories.

    Idempotent: a re-run after patching is a no-op. Raises if the
    expected upstream substring is absent (e.g. upstream format changed).

    Call this immediately after ``discogen.create_task()`` returns so that
    both the baseline copy under ``<exp.dir>/src/`` and every executor
    workspace materialized from it inherit the fix.
    """
    target = src_dir / "run_main.py"
    text = target.read_text()
    if _RUN_MAIN_WALK_FIX in text:
        return
    if _RUN_MAIN_WALK_BUG not in text:
        raise RuntimeError(
            f"patch_run_main_walk: expected substring not found in {target}; "
            "upstream discogen run_main.py format may have changed."
        )
    target.write_text(text.replace(_RUN_MAIN_WALK_BUG, _RUN_MAIN_WALK_FIX))


def _filter_baselines_by_task_ids(
    baselines: dict[str, float],
    train_task_ids: list[str],
) -> dict[str, float]:
    """Restrict ``baselines`` to entries matching ``train_task_ids``.

    Each task id is normalized to the ``./<task_id>`` form used as keys in
    ``baselines``. Raises ``ValueError`` if any requested id is absent from
    the upstream baselines so that config typos surface at startup rather
    than producing silently smaller-than-intended baseline sets.
    """
    wanted = {f"./{tid}" for tid in train_task_ids}
    missing = wanted - baselines.keys()
    if missing:
        raise ValueError(
            f"train_task_ids contains entries not present in upstream baselines: "
            f"{sorted(missing)}. Available: {sorted(baselines)}"
        )
    return {k: v for k, v in baselines.items() if k in wanted}


def load_baselines(
    domain: str,
    backend: str,
    train_task_ids: list[str] | None = None,
    test_task_ids: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, float], str]:
    """Load baseline scores from discogen's installed package.

    Args:
        domain: DiscoGen task domain name (e.g. ``"OnPolicyRL"``).
        backend: Template backend name (e.g. ``"default"``, ``"recurrent"``,
            ``"transformer"``).
        train_task_ids: Optional list of task ids (e.g.
            ``["MinAtar/Breakout"]``). When provided, ``train_baselines``
            is restricted to these ids so the grader does not require
            datasets the workspace never materializes. When ``None``
            (default), ``train_baselines`` is empty. Raises
            ``ValueError`` if any id is not in the upstream YAML.
        test_task_ids: Optional list of held-out task ids consumed by
            the meta-test phase. When provided, ``test_baselines`` is
            restricted to these ids; when ``None`` (default),
            ``test_baselines`` is empty (meta-test will skip).

    Returns:
        Tuple ``(train_baselines, test_baselines, objective)`` mapping
        dataset path prefixes (e.g. ``"./MinAtar/Breakout"``) to
        baseline scores, plus ``objective`` (``"max"`` or ``"min"``).
        Either dict is empty when its corresponding ``*_task_ids`` arg
        is ``None``.
    """
    import discogen

    baselines_path = (
        Path(discogen.__file__).parent
        / "domains"
        / domain
        / "utils"
        / "baseline_scores.yaml"
    )
    with open(baselines_path) as f:
        raw = yaml.safe_load(f)

    objective = "max"
    baselines: dict[str, float] = {}
    train_baselines: dict[str, float] = {}
    test_baselines: dict[str, float] = {}
    for metric_data in raw.values():
        if not isinstance(metric_data, dict):
            continue
        if "objective" in metric_data and isinstance(metric_data["objective"], str):
            objective = metric_data["objective"]
        if backend in metric_data:
            for task_id, score in metric_data[backend].items():
                baselines[f"./{task_id}"] = score

    if train_task_ids is not None:
        train_baselines = _filter_baselines_by_task_ids(baselines, train_task_ids)
    if test_task_ids is not None:
        test_baselines = _filter_baselines_by_task_ids(baselines, test_task_ids)
    return train_baselines, test_baselines, objective


def build_workspace_files(
    src_dir: Path,
    exclude: set[str] | None = None,
) -> dict[str, Path]:
    """Walk top-level entries of a discogen task directory.

    Args:
        src_dir: Path to the generated discogen task root (contains
            ``discovered/``, dataset dirs, ``run_main.py``, etc.).
        exclude: Names to skip. Defaults to
            ``{"requirements.txt", "install.sh"}`` (setup-only files).

    Returns:
        Dict mapping top-level entry name to its absolute path, suitable
        for ``Workspace.files``.
    """
    exclude = exclude if exclude is not None else _DEFAULT_EXCLUDE
    return {p.name: p for p in src_dir.iterdir() if p.name not in exclude}


def setup_meta_test_workspace(
    elite_dir: Path,
    test_dir: Path,
    config: dict[str, Any],
    venv_path: Path,
    requirements_path: Path,
    install_args: list[str] | None = None,
) -> None:
    """Prepare a meta-test workspace from an elite's executor dir.

    The flow:
        1. Copy ``discovered/`` from ``elite_dir`` into ``test_dir/discovered/``.
        2. Call ``create_task(test=True, config_dict=..., source_path=test_dir)``
           to generate the test dataset directories and symlinks.
        3. Call ``Workspace.setup()`` on ``test_dir`` to stamp the two-tier
           venv marker (``.venv/`` placeholder dir + ``.venv_source`` pointer).

    Ordering matters: ``create_task(test=True)`` deletes everything except
    ``discovered/``, so venv setup must happen after ``create_task``.

    IMPORTANT: the ``test_dir/.venv/`` directory this leaves behind is a
    *stub*, not a real venv. It exists so that bwrap-mounted runs can
    bind-mount the real venv over it. Meta-test runs on the host (no
    bwrap), so callers must invoke the interpreter directly at
    ``venv_path / "bin" / "python"`` rather than ``test_dir / ".venv"
    / "bin" / "python"``.

    Args:
        elite_dir: Path to the elite's executor workspace.
        test_dir: Destination directory for the meta-test workspace.
        config: DiscoGen YAML config (must include ``_domain`` or ``source_path``).
        venv_path: Host-side venv path. This is also the path meta-test
            callers should use directly when invoking ``python run_main.py``.
        requirements_path: Requirements file for venv auto-creation.
        install_args: Extra args for ``uv pip install`` (e.g.
            ``["--prerelease", "allow"]``).
    """
    from discogen import create_task

    test_dir.mkdir(parents=True, exist_ok=True)
    src_discovered = elite_dir / "discovered"
    dst_discovered = test_dir / "discovered"
    if dst_discovered.exists():
        shutil.rmtree(dst_discovered)
    shutil.copytree(src_discovered, dst_discovered, symlinks=True)

    test_config = dict(config)
    test_config["source_path"] = str(test_dir)
    domain = config.get("_domain", "OnPolicyRL")
    create_task(
        task_domain=domain,
        test=True,
        config_dict=test_config,
        no_data=False,
    )
    patch_run_main_walk(test_dir)

    ws = Workspace(
        venv=venv_path,
        requirements=requirements_path,
        install_args=install_args or [],
    )
    ws.setup(test_dir)
