from heuresis.experiment import Settings
from heuresis.experiment_cli import DiscoGenTaskConfig
from heuresis.tasks.discogen.adapter import OnPolicyRLAdapter


def _cfg(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("train_task_id: [0]\ntest_task_id: [1]\ntemplate_backend: default\n")
    return DiscoGenTaskConfig(config=p, domain="OnPolicyRL")


def test_identity_and_name_rewrite(tmp_path):
    a = OnPolicyRLAdapter(_cfg(tmp_path))
    assert a.name == "discogen_onpolicyrl"
    assert a.metric_label == "score"
    s = Settings(experiment_name="discogen-onpolicyrl-linear")
    a.normalize_settings(s)
    assert s.experiment_name == "discogen-OnPolicyRL-linear"


def test_store_task_is_domain(tmp_path):
    a = OnPolicyRLAdapter(_cfg(tmp_path))
    assert a.store_task == "OnPolicyRL"


def test_runs_root_includes_domain(tmp_path):
    a = OnPolicyRLAdapter(_cfg(tmp_path))
    assert a.runs_root.parts[-2:] == ("discogen", "OnPolicyRL")


def test_snapshot_lists_discovered(tmp_path):
    a = OnPolicyRLAdapter(_cfg(tmp_path))
    exec_dir = tmp_path / "exec_000"
    (exec_dir / "discovered").mkdir(parents=True)
    (exec_dir / "discovered" / "algo.py").write_text("x")
    snap = a.snapshot_files(exec_dir)
    assert "discovered/algo.py" in snap
    assert "run.log" in snap and "notes.md" in snap and "novelty.json" in snap


def test_ideator_task_vars_emit_description_and_direction(tmp_path):
    # regression: discogen ideator template uses {{ description }} + is_lower_better;
    # the thin base only adds `description` when it differs from problem_text.
    a = OnPolicyRLAdapter(_cfg(tmp_path))
    a._description = "the discogen task description"
    a.lower_is_better = True
    iv = a.ideator_task_vars(timeout_minutes=20, memory_on=False)
    assert iv["description"] == "the discogen task description"
    assert iv["is_lower_better"] is True
    ev = a.executor_task_vars(idea="x", timeout_minutes=20, gpu_count=2, memory_on=False)
    assert ev["description"] == "the discogen task description"


def test_missing_config_preflight_errors(tmp_path):
    cfg = DiscoGenTaskConfig(config=tmp_path / "nope.yaml", domain="OnPolicyRL")
    a = OnPolicyRLAdapter(cfg)
    errs = a.preflight(Settings(gpus=[0]))
    assert errs  # non-empty: config not found
