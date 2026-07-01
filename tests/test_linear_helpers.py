from heuresis.loops.linear import DeltaQueue, annotate_runs


class _Rec:
    def __init__(self, run_id, score, idea, iteration, workspace):
        self.run_id, self.score, self.idea = run_id, score, idea
        self.iteration, self.workspace = iteration, workspace


def test_annotate_runs_reads_workspace_id(tmp_path):
    ws = tmp_path / "exec_000"
    ws.mkdir()
    (ws / ".workspace_id").write_text("abc123\n")
    rec = _Rec("exec_000", 0.9, "do X", 0, str(ws))
    out = annotate_runs([rec])
    assert out == [{"run_id": "exec_000", "executor_id": "abc123",
                    "score": 0.9, "idea": "do X"}]


def test_annotate_runs_missing_id_is_empty(tmp_path):
    rec = _Rec("exec_001", None, "y", 1, str(tmp_path / "nope"))
    assert annotate_runs([rec])[0]["executor_id"] == ""


def test_delta_queue_reports_new_since_last(tmp_path):
    a = _Rec("a", 1.0, "", 0, str(tmp_path))
    b = _Rec("b", 2.0, "", 1, str(tmp_path))
    dq = DeltaQueue(initial_ids={"a"})
    new = dq.new_since_last([a, b])
    assert [r.run_id for r in new] == ["b"]
    # second call: nothing new
    assert dq.new_since_last([a, b]) == []
