from heuresis.experiment import Settings
from heuresis.tasks.nanogpt.adapter import NanoGPTAdapter


def test_identity():
    a = NanoGPTAdapter()
    assert a.name == "nanogpt"
    assert a.metric_label == "val_bpb"
    assert a.lower_is_better is True
    assert a.uses_gpu is True
    assert a.runs_root.name == "nanogpt"
    assert a.store_task == "nanogpt"


def test_normalize_defaults_ideators_to_gpu_count():
    a = NanoGPTAdapter()
    s = Settings(gpus=[0, 1, 2, 3])      # num_ideators stays 1 by default
    a.normalize_settings(s)
    assert s.num_ideators == 4


def test_mounts_are_data_and_tokenizer():
    a = NanoGPTAdapter()
    ms = a.mounts()
    targets = {m.target for m in ms}
    assert targets == {
        "/workspace/.cache/autoresearch/data",
        "/workspace/.cache/autoresearch/tokenizer",
    }


def test_judge_enabled_returns_hackerjudge():
    from heuresis import HackerJudge
    a = NanoGPTAdapter()
    j = a.make_judge(Settings(enable_judge=True))
    assert isinstance(j, HackerJudge)
    assert a.make_judge(Settings(enable_judge=False)) is None


def test_executor_task_vars_include_gpu_info():
    a = NanoGPTAdapter()
    a.on_experiment(exp=None, state=None, settings=Settings())
    ev = a.executor_task_vars(idea="x", timeout_minutes=35, gpu_count=8,
                              memory_on=False)
    assert ev["gpu_info"] == "8x A100 40GB"
    assert ev["idea"] == "x"
