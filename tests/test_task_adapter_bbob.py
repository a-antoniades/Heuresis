from heuresis.experiment import Settings
from heuresis.tasks.bbob.adapter import BBOBAdapter


def _adapter():
    return BBOBAdapter()


def test_identity_and_paths():
    a = _adapter()
    assert a.name == "bbob"
    assert a.metric_label == "mean_log_gap"
    assert a.lower_is_better is True
    assert a.uses_gpu is False
    assert a.task_dir.name == "bbob"
    assert a.runs_root.name == "bbob"
    assert a.store_path.name == "store.db"
    assert a.store_task == "bbob"


def test_normalize_sets_default_ideators():
    a = _adapter()
    s = Settings()                       # num_ideators defaults to 1
    a.normalize_settings(s)
    assert s.num_ideators == 4


def test_no_judge_no_mounts_no_snapshot(tmp_path):
    a = _adapter()
    assert a.make_judge(Settings()) is None
    assert a.mounts() == []
    assert a.snapshot_files(tmp_path) is None


def test_task_vars_use_problem_text():
    a = _adapter()
    a.on_experiment(exp=None, state=None, settings=Settings())  # loads PROBLEM
    iv = a.ideator_task_vars(timeout_minutes=5, memory_on=False)
    assert "problem" in iv and iv["memory"] is False
    ev = a.executor_task_vars(idea="x", timeout_minutes=5, gpu_count=0,
                              memory_on=False)
    assert ev["idea"] == "x" and "problem" in ev


def test_grader_is_bbob(tmp_path):
    from heuresis.tasks.bbob import BBOBGrader
    a = _adapter()
    g = a.make_grader(tmp_path / ".grade.sock")
    assert isinstance(g, BBOBGrader)
