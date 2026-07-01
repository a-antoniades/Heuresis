"""Unit tests for ModelUnlearning-specific helpers.

Covers ``load_unlearning_baselines`` (multi-metric/per-objective baseline
inversion) and ``patch_modelunlearning_workspace`` (idempotent main.py +
model_config.yaml patches).
"""

from pathlib import Path

import pytest

from heuresis.tasks.discogen.helpers import (
    _baseline_template_cache_key,
    apply_fast_eval_patches,
    clone_baseline_template,
    ensure_modelunlearning_baseline_template,
    load_unlearning_baselines,
    patch_modelunlearning_workspace,
    prefetch_modelunlearning_data,
)


def test_load_unlearning_baselines_wmdp_cyber():
    train, test = load_unlearning_baselines(
        "ModelUnlearning",
        "default",
        train_pairs=[("wmdp_cyber", "Qwen2.5-1.5B-Instruct")],
        test_pairs=[("wmdp_cyber", "Qwen2.5-1.5B-Instruct")],
    )
    key = "./wmdp_cyber_Qwen2.5-1.5B-Instruct"
    assert key in train
    metrics = train[key]
    # WMDP-cyber upstream YAML carries two metrics for this dataset.
    assert "wmdp_cyber/acc" in metrics
    # Upstream baseline yaml has the metric as `mmlu_stemp/acc` (typo) but the
    # lm_eval subset is actually `mmlu_stem/acc`; load_unlearning_baselines
    # applies a known-typo fixup so the grader matches what main.py emits.
    assert "mmlu_stem/acc" in metrics
    assert "mmlu_stemp/acc" not in metrics
    forget_baseline, forget_obj = metrics["wmdp_cyber/acc"]
    retain_baseline, retain_obj = metrics["mmlu_stem/acc"]
    assert forget_obj == "min"
    assert retain_obj == "max"
    assert 0 < forget_baseline < 1
    assert 0 < retain_baseline < 1
    # test = train when same pair given (the ModelUnlearning meta-test
    # phase isn't used in this experiment; trivially equal here).
    assert train == test


def test_load_unlearning_baselines_none_pairs_returns_empty():
    train, test = load_unlearning_baselines(
        "ModelUnlearning", "default", train_pairs=None, test_pairs=None,
    )
    assert train == {}
    assert test == {}


def test_load_unlearning_baselines_missing_task_raises():
    with pytest.raises(ValueError, match="without baselines"):
        load_unlearning_baselines(
            "ModelUnlearning",
            "default",
            train_pairs=[("nonexistent_task", "Qwen2.5-1.5B-Instruct")],
        )


def test_patch_workspace_idempotent_full_pipeline(tmp_path: Path):
    """Run create_task → patch → patch-again, verify all three patches stick."""
    from discogen import create_task
    from heuresis.tasks.discogen.helpers import patch_run_main_walk

    src = tmp_path / "src"
    config = {
        "train_task_id": ["wmdp_cyber"],
        "test_task_id": ["wmdp_cyber"],
        "train_model_id": ["Qwen2.5-1.5B-Instruct"],
        "test_model_id": ["Qwen2.5-1.5B-Instruct"],
        "source_path": str(src),
        "template_backend": "default",
        "change_loss": True,
    }
    create_task(
        task_domain="ModelUnlearning",
        test=False,
        config_dict=config,
        no_data=True,
        use_base=True,
    )
    patch_run_main_walk(src)
    patch_modelunlearning_workspace(src)
    # Second call should be a no-op (idempotent).
    patch_modelunlearning_workspace(src)

    main_py = src / "wmdp_cyber_Qwen2.5-1.5B-Instruct" / "main.py"
    text = main_py.read_text()
    # Patch 1: merged-summary print loop
    assert "_merged_summary = {}" in text
    assert "_merged_summary.update(json.load(f))" in text
    # Patch 2: relaxed HF_TOKEN handling
    assert "Aborting." not in text
    assert "proceeding unauthenticated" in text

    # Patch 3: SDPA in model_config.yaml
    model_cfg = src / "wmdp_cyber_Qwen2.5-1.5B-Instruct" / "model_config.yaml"
    cfg_text = model_cfg.read_text()
    assert "attn_implementation: 'sdpa'" in cfg_text
    assert "flash_attention_2" not in cfg_text

    # Patch 4: configs/ pre-staged + runtime copies removed from main.py
    ds_dir = src / "wmdp_cyber_Qwen2.5-1.5B-Instruct"
    assert (ds_dir / "configs" / "main_config.yaml").is_file()
    assert (ds_dir / "configs" / "trainer" / "custom.yaml").is_file()
    assert (ds_dir / "configs" / "model" / "model_conf.yaml").is_file()
    main_py_text = (ds_dir / "main.py").read_text()
    # Runtime copies must be gone (otherwise main.py crashes under read-only
    # bind in the executor sandbox) and replaced by an explanatory marker.
    assert "shutil.copy2" not in main_py_text
    assert "pre-staged by patch_modelunlearning_workspace" in main_py_text


def test_patch_workspace_fast_eval_shrinks_training_and_eval(tmp_path: Path):
    """fast_eval=True must shrink eval + training; default is unchanged."""
    from discogen import create_task

    src = tmp_path / "src"
    config = {
        "train_task_id": ["wmdp_cyber"],
        "test_task_id": ["wmdp_cyber"],
        "train_model_id": ["Qwen2.5-1.5B-Instruct"],
        "test_model_id": ["Qwen2.5-1.5B-Instruct"],
        "source_path": str(src),
        "template_backend": "default",
        "change_loss": True,
    }
    create_task(
        task_domain="ModelUnlearning",
        test=False,
        config_dict=config,
        no_data=True,
        use_base=True,
    )
    from heuresis.tasks.discogen.helpers import patch_run_main_walk
    patch_run_main_walk(src)
    patch_modelunlearning_workspace(
        src, fast_eval=True, fast_eval_limit=50, fast_max_steps=5
    )
    # Idempotent: applying twice is a no-op
    patch_modelunlearning_workspace(
        src, fast_eval=True, fast_eval_limit=50, fast_max_steps=5
    )

    ds_dir = src / "wmdp_cyber_Qwen2.5-1.5B-Instruct"
    trainer_text = (ds_dir / "configs" / "trainer" / "custom.yaml").read_text()
    assert "eval_on_start: False" in trainer_text
    assert "max_steps: 5" in trainer_text
    # The original max_steps: 80 must be replaced, not appended.
    assert trainer_text.count("max_steps:") == 1

    # Eval limit must land in simple_evaluate_args block.
    eval_text = (ds_dir / "configs" / "eval" / "wmdp_cyber+mmlu.yaml").read_text()
    assert "limit: 50" in eval_text
    # Ensure it's under simple_evaluate_args, not at top level.
    assert (
        eval_text.index("simple_evaluate_args:")
        < eval_text.index("limit: 50")
    )


def test_patch_workspace_fast_eval_default_off(tmp_path: Path):
    """fast_eval defaults to False — workspace stays at upstream defaults."""
    from discogen import create_task

    src = tmp_path / "src"
    config = {
        "train_task_id": ["wmdp_cyber"],
        "test_task_id": ["wmdp_cyber"],
        "train_model_id": ["Qwen2.5-1.5B-Instruct"],
        "test_model_id": ["Qwen2.5-1.5B-Instruct"],
        "source_path": str(src),
        "template_backend": "default",
        "change_loss": True,
    }
    create_task(
        task_domain="ModelUnlearning",
        test=False,
        config_dict=config,
        no_data=True,
        use_base=True,
    )
    from heuresis.tasks.discogen.helpers import patch_run_main_walk
    patch_run_main_walk(src)
    patch_modelunlearning_workspace(src)  # default: fast_eval=False

    ds_dir = src / "wmdp_cyber_Qwen2.5-1.5B-Instruct"
    trainer_text = (ds_dir / "configs" / "trainer" / "custom.yaml").read_text()
    assert "eval_on_start: False" not in trainer_text
    assert "max_steps: 80" in trainer_text  # upstream default unchanged
    eval_text = (ds_dir / "configs" / "eval" / "wmdp_cyber+mmlu.yaml").read_text()
    assert "limit:" not in eval_text  # no limit added by default


def test_patch_workspace_hardens_make_dataset_existence_check(tmp_path: Path):
    """Patch 5: replaces folder-existence skip with all-files-present check."""
    from discogen import create_task
    from heuresis.tasks.discogen.helpers import patch_run_main_walk

    src = tmp_path / "src"
    config = {
        "train_task_id": ["wmdp_cyber"],
        "test_task_id": ["wmdp_cyber"],
        "train_model_id": ["Qwen2.5-1.5B-Instruct"],
        "test_model_id": ["Qwen2.5-1.5B-Instruct"],
        "source_path": str(src),
        "template_backend": "default",
        "change_loss": True,
    }
    create_task(
        task_domain="ModelUnlearning",
        test=False,
        config_dict=config,
        no_data=True,
        use_base=True,
    )
    patch_run_main_walk(src)
    patch_modelunlearning_workspace(src)
    # Idempotent
    patch_modelunlearning_workspace(src)

    md = (src / "wmdp_cyber_Qwen2.5-1.5B-Instruct" / "make_dataset.py").read_text()
    # Old skip path replaced
    assert "if os.path.exists(extracted_folder):\n        print" not in md
    # Extract the _expected tuple block and check exactly what's in it.
    # bio-forget legitimately appears in the EXPLANATORY COMMENT (so we
    # can't just check it's absent from the whole file); what matters is
    # that the runtime tuple doesn't list it.
    import re as _re
    m = _re.search(r"_expected\s*=\s*\(([^)]*)\)", md, _re.S)
    assert m is not None, "_expected tuple not found in patched make_dataset.py"
    tuple_body = m.group(1)
    assert "cyber-forget-corpus.jsonl" in tuple_body
    assert "cyber-retain-corpus.jsonl" in tuple_body
    assert "bio-retain-corpus.jsonl" in tuple_body
    # bio-forget is NOT in the wmdp-cyber zip; listing it here would cause
    # every prefetch to look incomplete and trigger agent misdiagnosis.
    assert "bio-forget-corpus.jsonl" not in tuple_body
    # Partial extraction is removed
    assert "_shutil.rmtree" in md


def test_prefetch_calls_make_dataset_function_with_correct_cwd(tmp_path: Path):
    """``prefetch_modelunlearning_data`` must INVOKE make_dataset(), not
    just import the module.

    Upstream ``make_dataset.py`` is a module that *defines* ``make_dataset()``
    but doesn't call it (the actual call lives in ``main.py``'s ``__main__``
    block). A naive ``python make_dataset.py`` would be a silent no-op.

    Replaces the real make_dataset.py with a stub whose ``make_dataset()``
    function records cwd to a marker file, so we can verify (a) the function
    was invoked, (b) cwd was the parent ``src`` dir (so relative ``data/``
    paths land at src/data/, not inside the dataset dir).
    """
    import sys
    src = tmp_path / "src"
    ds = src / "wmdp_cyber_Qwen2.5-1.5B-Instruct"
    ds.mkdir(parents=True)
    marker = src / "ran.log"
    (ds / "make_dataset.py").write_text(
        "import os\n"
        "def make_dataset():\n"
        f"    open(r'{marker}', 'a').write(os.getcwd() + '\\n')\n"
        # Sentinel: this line should NOT execute -- if the prefetch runs
        # the file as a script, this would create the file.
        f"open(r'{tmp_path / 'BAD_module_level_executed'}', 'a').write('x')\n"
    )

    prefetch_modelunlearning_data(src, venv_python=Path(sys.executable))

    # make_dataset() was actually called
    assert marker.is_file(), "make_dataset() was not invoked"
    cwd = marker.read_text().strip()
    assert Path(cwd).resolve() == src.resolve(), (
        f"prefetch ran with wrong cwd: {cwd}"
    )
    # Module-level code DID execute (importlib does that), but if we ever
    # switch back to "run as script" the sentinel would still fire — so
    # this assertion isn't strict. The marker assertion above is the load-
    # bearing one.


# ---------------------------------------------------------------------------
# C2: baseline-template cache
# ---------------------------------------------------------------------------


def test_baseline_template_cache_key_stable_and_diff_sensitive():
    """Cache key must be deterministic for same inputs and change for any
    semantically-meaningful diff (pairs, backend, use_base, version)."""
    base = dict(
        domain="ModelUnlearning",
        train_pairs=[("wmdp_cyber", "Qwen2.5-1.5B-Instruct")],
        test_pairs=[("wmdp_cyber", "Qwen2.5-1.5B-Instruct")],
        template_backend="default",
        use_base=True,
    )
    k0 = _baseline_template_cache_key(**base)
    # Determinism
    assert k0 == _baseline_template_cache_key(**base)
    # Backend changes the key
    assert _baseline_template_cache_key(**{**base, "template_backend": "transformer"}) != k0
    # use_base changes the key
    assert _baseline_template_cache_key(**{**base, "use_base": False}) != k0
    # train pair changes the key
    assert _baseline_template_cache_key(
        **{**base, "train_pairs": [("tofu", "Qwen2.5-1.5B-Instruct")]}
    ) != k0
    # test pair changes the key
    assert _baseline_template_cache_key(
        **{**base, "test_pairs": [("tofu", "Qwen2.5-1.5B-Instruct")]}
    ) != k0
    # Key embeds the framework version so a bump invalidates all caches.
    # (Version is hashed in; verify via monkeypatch.)
    import heuresis.tasks.discogen.helpers as _h
    saved = _h.BASELINE_TEMPLATE_VERSION
    try:
        _h.BASELINE_TEMPLATE_VERSION = "vNEXT"
        k_next = _baseline_template_cache_key(**base)
    finally:
        _h.BASELINE_TEMPLATE_VERSION = saved
    assert k_next != k0


def test_ensure_baseline_template_builds_and_caches(tmp_path: Path, monkeypatch):
    """First call builds; second call returns cached path quickly.

    Stubs ``discogen.create_task`` and ``prefetch_modelunlearning_data`` so the
    test doesn't hit the network or the real upstream code. Verifies the
    sentinel-based caching, atomic staging-then-rename, and that a second
    call skips the build (asserted via a build-counter).
    """
    import discogen
    template_root = tmp_path / "templates"
    train_pairs = [("wmdp_cyber", "Qwen2.5-1.5B-Instruct")]
    test_pairs = [("wmdp_cyber", "Qwen2.5-1.5B-Instruct")]
    build_calls = {"create_task": 0, "prefetch": 0}

    def fake_create_task(*, task_domain, test, config_dict, no_data, use_base):
        build_calls["create_task"] += 1
        staging = Path(config_dict["source_path"])
        _materialize_stub_workspace(staging)

    monkeypatch.setattr(discogen, "create_task", fake_create_task)

    # Skip the real network-touching prefetch by patching the helper itself.
    import heuresis.tasks.discogen.helpers as _helpers

    def fake_prefetch(src_dir, *, venv_python, timeout=600):
        build_calls["prefetch"] += 1
        # Materialize a minimal data dir so cloning has something to copy.
        d = src_dir / "data" / "wmdp" / "wmdp-corpora"
        d.mkdir(parents=True, exist_ok=True)
        for f in ("cyber-forget-corpus.jsonl", "cyber-retain-corpus.jsonl", "bio-retain-corpus.jsonl"):
            (d / f).write_text("{}\n")

    monkeypatch.setattr(_helpers, "prefetch_modelunlearning_data", fake_prefetch)

    import sys
    template = ensure_modelunlearning_baseline_template(
        domain="ModelUnlearning",
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        template_root=template_root,
        venv_python=Path(sys.executable),
    )

    assert template.is_dir()
    assert (template / ".baseline_ready").is_file()
    assert (template / "discovered" / "loss.py").is_file()
    assert (template / "data" / "wmdp" / "wmdp-corpora" / "cyber-forget-corpus.jsonl").is_file()
    assert build_calls == {"create_task": 1, "prefetch": 1}

    # Second call hits the cache — no rebuild.
    template2 = ensure_modelunlearning_baseline_template(
        domain="ModelUnlearning",
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        template_root=template_root,
        venv_python=Path(sys.executable),
    )
    assert template2 == template
    assert build_calls == {"create_task": 1, "prefetch": 1}, (
        "Second call should not rebuild"
    )

    # A different config rebuilds (separate cache entry).
    ensure_modelunlearning_baseline_template(
        domain="ModelUnlearning",
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        template_root=template_root,
        venv_python=Path(sys.executable),
        use_base=False,  # different from above
    )
    assert build_calls == {"create_task": 2, "prefetch": 2}


def test_ensure_baseline_template_atomic_on_failure(tmp_path: Path, monkeypatch):
    """If the build raises mid-way, no partial template_dir is left so the
    next call rebuilds from scratch (rather than serving a half-built tree)."""
    import discogen
    import heuresis.tasks.discogen.helpers as _helpers

    template_root = tmp_path / "templates"
    train_pairs = [("wmdp_cyber", "Qwen2.5-1.5B-Instruct")]

    def fake_create_task(*, task_domain, test, config_dict, no_data, use_base):
        staging = Path(config_dict["source_path"])
        _materialize_stub_workspace(staging)

    monkeypatch.setattr(discogen, "create_task", fake_create_task)

    def fake_prefetch(src_dir, **kw):
        raise RuntimeError("simulated S3 outage")

    monkeypatch.setattr(_helpers, "prefetch_modelunlearning_data", fake_prefetch)

    import sys
    with pytest.raises(RuntimeError, match="simulated S3 outage"):
        ensure_modelunlearning_baseline_template(
            domain="ModelUnlearning",
            train_pairs=train_pairs,
            template_root=template_root,
            venv_python=Path(sys.executable),
        )

    # No template_dir, no sentinel, no leftover staging dir.
    key = _baseline_template_cache_key(
        domain="ModelUnlearning",
        train_pairs=train_pairs,
        test_pairs=None,
        template_backend="default",
        use_base=True,
    )
    assert not (template_root / key).exists()
    assert not (template_root / f"{key}.staging").exists()


def test_clone_baseline_template_hardlinks(tmp_path: Path):
    """``clone_baseline_template`` should produce hardlinks (shared inodes)
    not copies, so per-experiment disk impact is minimal."""
    template = tmp_path / "template"
    template.mkdir()
    big = template / "data" / "corpus.jsonl"
    big.parent.mkdir()
    big.write_text("x" * 1_000_000)
    (template / "discovered").mkdir()
    (template / "discovered" / "loss.py").write_text("# stub\n")
    (template / ".baseline_ready").write_text("v\n")

    dest = tmp_path / "clone"
    clone_baseline_template(template, dest)

    assert (dest / "data" / "corpus.jsonl").is_file()
    assert (dest / "discovered" / "loss.py").is_file()
    # Hardlinks share inodes (only valid on same FS — should be on tmpfs/tmp).
    assert (dest / "data" / "corpus.jsonl").stat().st_ino == big.stat().st_ino


def test_clone_baseline_template_refuses_existing_dest(tmp_path: Path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "x").write_text("\n")
    dest = tmp_path / "existing"
    dest.mkdir()
    with pytest.raises(FileExistsError):
        clone_baseline_template(template, dest)


def test_apply_fast_eval_patches_does_not_mutate_template_via_hardlink(tmp_path: Path):
    """Regression for the 2026-05-10 cache-pollution incident.

    ``cp -al`` produces hardlinked clones over the cached baseline
    template. If ``apply_fast_eval_patches`` writes in place
    (``Path.write_text`` truncates and writes to the existing inode),
    the edit propagates back into the template and every subsequent
    run inherits the smoke patches — even with MU_FAST_EVAL=0. The
    fix unlinks each file before writing so a fresh inode is allocated
    for the clone, leaving the template untouched.
    """
    template = tmp_path / "template"
    ds = template / "wmdp_cyber_Qwen2.5-1.5B-Instruct"
    (ds / "configs" / "trainer").mkdir(parents=True)
    (ds / "configs" / "eval").mkdir(parents=True)
    (ds / "configs" / "trainer" / "custom.yaml").write_text(
        "args:\n  max_steps: 80\n  eval_on_start: True\n"
    )
    (ds / "configs" / "eval" / "wmdp_cyber+mmlu.yaml").write_text(
        "simple_evaluate_args:\n  tasks: [wmdp_cyber]\n"
    )

    # Hardlink-clone — same pattern as ``clone_baseline_template`` /
    # the run.py production path (``cp -al``).
    clone = tmp_path / "clone"
    clone_baseline_template(template, clone)

    custom_tpl = ds / "configs" / "trainer" / "custom.yaml"
    custom_clone = clone / "wmdp_cyber_Qwen2.5-1.5B-Instruct" / "configs" / "trainer" / "custom.yaml"
    eval_tpl = ds / "configs" / "eval" / "wmdp_cyber+mmlu.yaml"
    eval_clone = clone / "wmdp_cyber_Qwen2.5-1.5B-Instruct" / "configs" / "eval" / "wmdp_cyber+mmlu.yaml"

    # Pre-condition: clone and template share inodes (cp -al worked).
    assert custom_tpl.stat().st_ino == custom_clone.stat().st_ino
    assert eval_tpl.stat().st_ino == eval_clone.stat().st_ino

    template_custom_before = custom_tpl.read_text()
    template_eval_before = eval_tpl.read_text()

    apply_fast_eval_patches(clone, limit=200, max_steps=10)

    # Clone is patched.
    assert "max_steps: 10" in custom_clone.read_text()
    assert "eval_on_start: False" in custom_clone.read_text()
    assert "limit: 200" in eval_clone.read_text()

    # Template is unchanged — the load-bearing invariant.
    assert custom_tpl.read_text() == template_custom_before, (
        "fast-eval patches mutated the cached template via shared inode"
    )
    assert eval_tpl.read_text() == template_eval_before, (
        "fast-eval patches mutated the cached eval config via shared inode"
    )

    # Clone and template should no longer share inodes for patched files
    # (the unlink + write broke the hardlink).
    assert custom_tpl.stat().st_ino != custom_clone.stat().st_ino
    assert eval_tpl.stat().st_ino != eval_clone.stat().st_ino


def test_apply_fast_eval_patches_idempotent_no_template_drift(tmp_path: Path):
    """Re-applying patches to an already-patched clone must not touch the template."""
    template = tmp_path / "template"
    ds = template / "wmdp_cyber_Qwen2.5-1.5B-Instruct"
    (ds / "configs" / "trainer").mkdir(parents=True)
    (ds / "configs" / "trainer" / "custom.yaml").write_text(
        "args:\n  max_steps: 80\n"
    )

    clone = tmp_path / "clone"
    clone_baseline_template(template, clone)
    apply_fast_eval_patches(clone, limit=200, max_steps=10)
    template_text_after_first = (ds / "configs" / "trainer" / "custom.yaml").read_text()
    apply_fast_eval_patches(clone, limit=200, max_steps=10)
    template_text_after_second = (ds / "configs" / "trainer" / "custom.yaml").read_text()
    assert template_text_after_first == template_text_after_second


def _materialize_stub_workspace(staging: Path) -> None:
    """Create a minimal ModelUnlearning baseline workspace that
    ``patch_modelunlearning_workspace`` can successfully patch — matches
    every upstream substring those patches look for, but contains no real
    discogen / hydra dependencies."""
    staging.mkdir(parents=True, exist_ok=True)
    ds = staging / "wmdp_cyber_Qwen2.5-1.5B-Instruct"
    ds.mkdir()
    (ds / "main.py").write_text(_MINIMAL_MAIN_PY)
    (ds / "main_config.yaml").write_text("# stub\n")
    (ds / "trainer_config.yaml").write_text("# stub\n")
    (ds / "model_config.yaml").write_text(
        "model_args:\n  attn_implementation: 'flash_attention_2'\n"
    )
    (ds / "make_dataset.py").write_text(_MINIMAL_MAKE_DATASET_PY)
    # Pre-staging step copies into these subdirs of configs/.
    (ds / "configs" / "trainer").mkdir(parents=True)
    (ds / "configs" / "model").mkdir(parents=True)
    (staging / "discovered").mkdir()
    (staging / "discovered" / "loss.py").write_text("# stub\n")
    (staging / "description.md").write_text("# desc\n")
    (staging / "run_main.py").write_text(_MINIMAL_RUN_MAIN_PY)
    (staging / "requirements.txt").write_text("# stub\n")
    (staging / "install.sh").write_text("#!/bin/bash\n")


_MINIMAL_MAIN_PY = '''import os
import json
import shutil
from pathlib import Path
from huggingface_hub import login as hf_login

# Copy dataset_config and trainer_config to configs/ before Hydra loads the main config
_base_dir = Path(__file__).resolve().parent
shutil.copy2(_base_dir / "main_config.yaml", _base_dir / "configs" / "main_config.yaml")
shutil.copy2(_base_dir / "trainer_config.yaml", _base_dir / "configs/trainer" / "custom.yaml")
(_base_dir / "configs/model").mkdir(parents=True, exist_ok=True)
shutil.copy2(_base_dir / "model_config.yaml", _base_dir / "configs/model" / "model_conf.yaml")


def unlearn():
    evaluators = {}
    eval_dir = "/tmp"
    print('Dumping final evaluation results...')
    for _, evaluator in evaluators.items():
        summary_file = evaluator.get_logs_file_path(eval_dir, suffix="SUMMARY")
        with open(summary_file, 'r') as f:
            print(json.dumps(json.load(f)))


if __name__ == "__main__":
    print('Logging in to Hugging Face...')
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_login(hf_token)
    else:
        print("HF_TOKEN environment variable not set. Aborting.")
        exit(1)
    print('Creating dataset...')
    print('Begin unlearning...')
    unlearn()
'''

_MINIMAL_MAKE_DATASET_PY = '''import zipfile
import os
import urllib.request
import subprocess

def make_dataset():
    url = "https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip"
    dest_dir = "data/wmdp"
    zip_path = os.path.join(dest_dir, "wmdp-corpora.zip")

    # Check if extracted dataset folder already exists
    extracted_folder = os.path.join(dest_dir, "wmdp-corpora")
    if os.path.exists(extracted_folder):
        print(f"Dataset already exists in {extracted_folder}, skipping download.")
        return
'''

_MINIMAL_RUN_MAIN_PY = '''import os

def run_all_main_py(start_dir="."):
    for root, dirs, files in os.walk(start_dir):
        dirs[:] = [d for d in dirs if d != "data"]


if __name__ == "__main__":
    run_all_main_py()
'''
