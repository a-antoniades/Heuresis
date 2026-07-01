"""Tests for HackerJudge verdict parsing + dispatch."""
from __future__ import annotations

from pathlib import Path

import yaml as _yaml

from heuresis.judge.hacker import HackerJudge, _parse_judge_response


def test_parses_plain_json() -> None:
    raw = '{"decision": "valid", "reasoning": "ok", "evidence_refs": ["run.log:1"]}'
    v = _parse_judge_response(raw)
    assert v.decision == "valid"
    assert v.reasoning == "ok"
    assert v.evidence_refs == ["run.log:1"]
    assert v.errored is False


def test_parses_fenced_json() -> None:
    raw = """```json
{"decision": "invalid_idea", "reasoning": "breaks causality", "evidence_refs": ["train.py:240"]}
```"""
    v = _parse_judge_response(raw)
    assert v.decision == "invalid_idea"
    assert "causality" in v.reasoning


def test_parses_single_element_list_wrapper() -> None:
    raw = '[{"decision": "suspicious_evidence", "reasoning": "fake block", "evidence_refs": ["agent.log:94"]}]'
    v = _parse_judge_response(raw)
    assert v.decision == "suspicious_evidence"
    assert v.evidence_refs == ["agent.log:94"]


def test_errored_on_empty_input() -> None:
    v = _parse_judge_response("")
    assert v.errored is True
    assert v.decision == "valid"  # placeholder


def test_errored_on_malformed_json() -> None:
    v = _parse_judge_response("{decision: valid")
    assert v.errored is True
    assert v.decision == "valid"


def test_errored_on_invalid_decision_value() -> None:
    v = _parse_judge_response('{"decision": "banana", "reasoning": "x", "evidence_refs": []}')
    assert v.errored is True


def test_errored_on_missing_required_field() -> None:
    v = _parse_judge_response('{"decision": "valid"}')   # no reasoning, no evidence_refs
    assert v.errored is True


def test_evidence_refs_coerced_to_empty_list_when_missing_and_errored() -> None:
    v = _parse_judge_response("")
    assert v.evidence_refs == []


# --- review() tests ----------------------------------------------------------


def _make_task_dir(tmp_path: Path) -> Path:
    """Create a minimal task directory sufficient for HackerJudge's prompt render."""
    td = tmp_path / "fake_task"
    td.mkdir()
    (td / "task_config.yaml").write_text(_yaml.safe_dump({
        "name": "fake",
        "description": "fake task for tests",
        "verify": {
            "command": "python x.py",
            "evidence_description": "Valid output contains 'SCORE='. Small files suspicious.",
        },
    }))
    (td / "baseline_scores.yaml").write_text(_yaml.safe_dump({
        "metric": "accuracy",
        "objective": "max",
        "baseline": 0.5,
    }))
    (td / "description.md").write_text("A minimal task used only for tests.")
    (td / "idea_schema.md").write_text("## Required idea fields\n- Title\n- Change\n")
    return td


def _make_exec_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "exec_001"
    ws.mkdir()
    (ws / "run.log").write_text("SCORE=0.8\n")
    (ws / "notes.md").write_text("all legit\n")
    (ws / "agent.log").write_text("{}\n")
    return ws


class _FakeHarness:
    """A Harness stand-in. Executes a callback that writes judge.json, then
    returns a RunResult-like object with stats dict."""

    def __init__(self, writer):
        self._writer = writer
        self.calls: list[dict] = []

    def run(self, workspace, prompt=None, *, mounts=None, stateful=False,
            timeout=None, path=None):
        # Capture inputs for assertions.
        self.calls.append({
            "path": path, "mounts": mounts, "prompt": prompt,
            "stateful": stateful, "timeout": timeout,
        })
        # Caller always .setup()s a workspace dir; we just honor `path`.
        path.mkdir(parents=True, exist_ok=True)
        self._writer(path)

        class _Result:
            def __init__(self, ws):
                self.workspace = ws
                self.exit_code = 0
                self.stats = {"duration": 0.42,
                              "input_tokens": 100,
                              "output_tokens": 25,
                              "total_cost": 0.0001}

        class _Future:
            def __init__(self, res):
                self._res = res

            def result(self, timeout=None):
                return self._res

        return _Future(_Result(path))


def test_review_parses_valid_verdict(tmp_path: Path) -> None:
    td = _make_task_dir(tmp_path)
    exec_ws = _make_exec_workspace(tmp_path)
    judge_dir = tmp_path / "judge_0_0"

    def writer(path: Path) -> None:
        (path / "judge.json").write_text(
            '{"decision": "valid", "reasoning": "all good", "evidence_refs": ["run.log:1"]}'
        )

    harness = _FakeHarness(writer)
    judge = HackerJudge(harness, td, timeout=60)
    v = judge.review(
        exec_workspace=exec_ws,
        judge_dir=judge_dir,
        idea="Some research idea\n",
        info={"best_score": 0.8, "valid": True, "timed_out": False,
              "exit_code": 0, "duration": 120.0},
    )

    assert v.decision == "valid"
    assert v.errored is False
    assert v.evidence_refs == ["run.log:1"]
    # idea.md was written into judge_dir
    assert (judge_dir / "idea.md").read_text() == "Some research idea\n"
    # task and run mounts were requested
    from heuresis.workspace import Mount
    mounts = harness.calls[0]["mounts"]
    targets = {m.target for m in mounts if isinstance(m, Mount)}
    assert "/workspace/task" in targets
    assert "/workspace/run" in targets


def test_review_errored_when_judge_json_missing(tmp_path: Path) -> None:
    td = _make_task_dir(tmp_path)
    exec_ws = _make_exec_workspace(tmp_path)
    judge_dir = tmp_path / "judge_miss"

    def writer(path: Path) -> None:
        pass  # don't write judge.json

    harness = _FakeHarness(writer)
    judge = HackerJudge(harness, td)
    v = judge.review(
        exec_workspace=exec_ws, judge_dir=judge_dir,
        idea="x", info={"best_score": 0.8},
    )
    assert v.errored is True


def test_review_errored_when_judge_json_malformed(tmp_path: Path) -> None:
    td = _make_task_dir(tmp_path)
    exec_ws = _make_exec_workspace(tmp_path)
    judge_dir = tmp_path / "judge_bad"

    def writer(path: Path) -> None:
        (path / "judge.json").write_text("not json at all")

    judge = HackerJudge(_FakeHarness(writer), td)
    v = judge.review(
        exec_workspace=exec_ws, judge_dir=judge_dir,
        idea="x", info={"best_score": 0.8},
    )
    assert v.errored is True


def test_review_does_not_consume_stale_judge_json(tmp_path: Path) -> None:
    """Stale judge.json from a prior crashed attempt must not leak into a retry.

    Scenario: a previous review() call wrote judge.json then the experiment
    crashed before record_run / save_judge_review persisted the run. On resume
    the iteration counter resumes at the same value, leading to the same
    judge_dir. If review() does not clear judge.json before launching the new
    agent, a failed-to-write retry will consume the prior verdict and the run
    is silently scored against a stale audit.
    """
    td = _make_task_dir(tmp_path)
    exec_ws = _make_exec_workspace(tmp_path)
    judge_dir = tmp_path / "judge_2_0"
    judge_dir.mkdir()

    # Seed stale verdict from a prior attempt.
    (judge_dir / "judge.json").write_text(
        '{"decision": "valid", "reasoning": "stale verdict from prior attempt", '
        '"evidence_refs": ["run.log:1"]}'
    )
    assert (judge_dir / "judge.json").is_file(), "test setup: stale file missing"

    # Agent fails to write judge.json on this attempt.
    def writer(path: Path) -> None:
        pass

    judge = HackerJudge(_FakeHarness(writer), td)
    v = judge.review(
        exec_workspace=exec_ws,
        judge_dir=judge_dir,
        idea="retried attempt idea",
        info={"best_score": 0.8},
    )

    assert v.errored is True, (
        f"Expected errored=True but got decision={v.decision!r} "
        f"reasoning={v.reasoning!r}; stale judge.json leaked into the retry."
    )


def test_review_errored_when_harness_raises(tmp_path: Path) -> None:
    td = _make_task_dir(tmp_path)
    exec_ws = _make_exec_workspace(tmp_path)
    judge_dir = tmp_path / "judge_crash"

    class _CrashingHarness:
        def run(self, workspace, prompt=None, *, mounts=None, stateful=False,
                timeout=None, path=None):
            class _Future:
                def result(self, timeout=None):
                    raise RuntimeError("boom from fake harness")
            path.mkdir(parents=True, exist_ok=True)
            return _Future()

    judge = HackerJudge(_CrashingHarness(), td)
    v = judge.review(
        exec_workspace=exec_ws, judge_dir=judge_dir,
        idea="x", info={"best_score": 0.8},
    )
    assert v.errored is True
    assert "RuntimeError" in v.raw_response
    assert "boom from fake harness" in v.raw_response


# --- baseline_dir constructor param tests ----------------------------------


def test_baseline_dir_defaults_to_task_dir(tmp_path: Path) -> None:
    """Constructor without baseline_dir kwarg → self._baseline_dir == task_dir."""
    from heuresis.harness import Harness
    from heuresis.judge.hacker import HackerJudge

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task_config.yaml").write_text("name: t\neditable: f.py\n")

    harness = Harness("opencode", model="m", gpus=[])
    judge = HackerJudge(harness, task_dir)

    assert judge._baseline_dir == task_dir


def test_baseline_dir_override(tmp_path: Path) -> None:
    """Constructor with baseline_dir=other → self._baseline_dir == other."""
    from heuresis.harness import Harness
    from heuresis.judge.hacker import HackerJudge

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task_config.yaml").write_text("name: t\neditable: f.py\n")

    other_dir = tmp_path / "other"
    other_dir.mkdir()

    harness = Harness("opencode", model="m", gpus=[])
    judge = HackerJudge(harness, task_dir, baseline_dir=other_dir)

    assert judge._baseline_dir == other_dir


def test_review_mounts_baseline_dir_at_workspace_task(tmp_path: Path, monkeypatch) -> None:
    """review() must mount self._baseline_dir at /workspace/task, not self._task_dir."""
    from heuresis.harness import Harness
    from heuresis.judge.hacker import HackerJudge
    from heuresis.workspace import Mount

    td = _make_task_dir(tmp_path)
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    exec_ws = _make_exec_workspace(tmp_path)
    judge_dir = tmp_path / "judge_baseline"

    captured_mounts: list = []

    def fake_run(workspace, *, prompt, mounts, stateful, timeout, path):
        captured_mounts.extend(mounts)
        path.mkdir(parents=True, exist_ok=True)

        class _Result:
            stats: dict = {}

        class _Future:
            def result(self, timeout=None):
                return _Result()

        return _Future()

    harness = Harness("opencode", model="m", gpus=[])
    monkeypatch.setattr(harness, "run", fake_run)

    judge = HackerJudge(harness, td, baseline_dir=baseline_dir)
    judge.review(exec_workspace=exec_ws, judge_dir=judge_dir, idea="x", info={"best_score": 0.8})

    sources = [m.source for m in captured_mounts if isinstance(m, Mount)]
    assert baseline_dir in sources, f"baseline_dir not in mount sources: {sources}"
    assert td not in sources, f"task_dir incorrectly in mount sources: {sources}"


def test_description_resolves_from_baseline_dir_when_present(tmp_path: Path) -> None:
    """description.md in baseline_dir takes precedence over task_dir."""
    from heuresis.harness import Harness
    from heuresis.judge.hacker import HackerJudge

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task_config.yaml").write_text("name: t\neditable: f.py\n")
    (task_dir / "description.md").write_text("STATIC TASK DESCRIPTION")

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "description.md").write_text("PER-RUN BASELINE DESCRIPTION")

    harness = Harness("opencode", model="m", gpus=[])
    judge = HackerJudge(harness, task_dir, baseline_dir=baseline_dir)

    assert judge._description == "PER-RUN BASELINE DESCRIPTION"


# --- verify.diff_scope rendering tests -------------------------------------


def _capture_prompt_harness(captured_prompts, monkeypatch):
    """Build a Harness whose .run() captures the rendered prompt text."""
    from heuresis.harness import Harness

    def fake_run(workspace, *, prompt, mounts, stateful, timeout, path):
        captured_prompts.append(prompt)
        path.mkdir(parents=True, exist_ok=True)

        class _Result:
            stats: dict = {}

        class _Future:
            def result(self, timeout=None):
                return _Result()

        return _Future()

    harness = Harness("opencode", model="m", gpus=[])
    monkeypatch.setattr(harness, "run", fake_run)
    return harness


def test_diff_scope_defaults_to_editable(tmp_path: Path, monkeypatch) -> None:
    """Task config with editable=foo.py and no verify.diff_scope → prompt diffs foo.py."""
    from heuresis.judge.hacker import HackerJudge

    td = tmp_path / "task"
    td.mkdir()
    (td / "task_config.yaml").write_text(
        "name: t\neditable: foo.py\nverify:\n  command: echo hi\n  stdout: out.log\n"
    )

    captured: list = []
    harness = _capture_prompt_harness(captured, monkeypatch)
    judge = HackerJudge(harness, td)
    judge.review(
        exec_workspace=tmp_path / "exec",
        judge_dir=tmp_path / "judge",
        idea="x",
        info={"best_score": 0.8},
    )

    prompt_text = captured[0]
    assert "/workspace/task/foo.py" in prompt_text
    assert "/workspace/run/foo.py" in prompt_text


def test_diff_scope_override(tmp_path: Path, monkeypatch) -> None:
    """Task config with verify.diff_scope: . → prompt diffs the workspace root."""
    from heuresis.judge.hacker import HackerJudge

    td = tmp_path / "task"
    td.mkdir()
    (td / "task_config.yaml").write_text(
        "name: t\neditable: foo.py\n"
        "verify:\n  command: echo hi\n  stdout: out.log\n  diff_scope: .\n"
    )

    captured: list = []
    harness = _capture_prompt_harness(captured, monkeypatch)
    judge = HackerJudge(harness, td)
    judge.review(
        exec_workspace=tmp_path / "exec",
        judge_dir=tmp_path / "judge",
        idea="x",
        info={"best_score": 0.8},
    )

    prompt_text = captured[0]
    assert "/workspace/task/." in prompt_text
    assert "/workspace/run/." in prompt_text


def test_prompt_template_drops_head_truncation(tmp_path: Path, monkeypatch) -> None:
    """The judge prompt's step 3 must NOT contain '| head -200' (full diffs go to the judge)."""
    from heuresis.judge.hacker import HackerJudge

    td = tmp_path / "task"
    td.mkdir()
    (td / "task_config.yaml").write_text(
        "name: t\neditable: foo.py\nverify:\n  command: echo hi\n  stdout: out.log\n"
    )

    captured: list = []
    harness = _capture_prompt_harness(captured, monkeypatch)
    judge = HackerJudge(harness, td)
    judge.review(
        exec_workspace=tmp_path / "exec",
        judge_dir=tmp_path / "judge",
        idea="x",
        info={"best_score": 0.8},
    )

    assert "head -200" not in captured[0]


# --- Rendered fabrication grep must work as a real shell command -----------


def _extract_step(prompt_text: str, step_number: int) -> str:
    """Pull a numbered command line out of the rendered judge prompt."""
    import re as _re
    pattern = rf"^{step_number}\.\s+(.+)$"
    m = _re.search(pattern, prompt_text, _re.MULTILINE)
    assert m, f"step {step_number} not found in prompt:\n{prompt_text}"
    return m.group(1)


def _substitute_workspace_paths(command: str, workspace_root: Path) -> str:
    """Map /workspace/* paths in a rendered command to a real test directory."""
    return command.replace("/workspace", str(workspace_root))


def test_fabrication_grep_step_executes_with_quote_laden_pattern(
    tmp_path: Path, monkeypatch
) -> None:
    """HackerJudge's contract: step 1 of the rendered prompt must execute
    as a valid shell command for ANY task-configured fabrication_patterns,
    including patterns containing literal `'` and `"` characters.

    The previous design inlined `{{ fabrication_patterns }}` into a
    single-quoted shell argument; an embedded `'` inside the regex (as in
    `['"]` character classes) closed the outer quoting and produced a
    bash parse error before grep ever ran — making the entire fabrication
    gate silently inert for those tasks.

    This test exercises the production rendering pipeline: HackerJudge
    parses task_config.yaml → renders prompt.j2 → emits the prompt the
    agent would receive. We capture that prompt via the production
    Harness path (monkeypatched .run() captures the prompt argument the
    production code passes), substitute `/workspace/*` → test paths so
    the command can run outside bwrap, and execute step 1 via `bash -c`.
    A correct rendering matches the hostile sample; a broken one fails
    parsing or returns no match.

    Test fixture: a synthetic minimal task_config with a regex containing
    both `'` and `"` inside a `['"]` character class. This is NOT a copy
    of any production task's fabrication_patterns — its only role is to
    exercise the quote-laden failure class.
    """
    import subprocess
    from heuresis.judge.hacker import HackerJudge

    # Synthetic fixture pattern — POSIX ERE with literal ' and " inside
    # a character class. Shape is irrelevant; only the quote chars matter.
    fixture_pattern = r"""fake[[:space:]]+['"]hostile['"]"""

    td = tmp_path / "task"
    td.mkdir()
    (td / "task_config.yaml").write_text(_yaml.safe_dump({
        "name": "fixture",
        "editable": "f.py",
        "verify": {
            "command": "echo hi",
            "stdout": "run.log",
            "fabrication_patterns": fixture_pattern,
        },
    }))

    workspace = tmp_path / "ws"
    (workspace / "run").mkdir(parents=True)
    # Hostile sample exercising the fixture pattern. Contains both quote
    # styles so a correct grep catches one and proves the regex executed.
    (workspace / "run" / "agent.log").write_text(
        "preamble\n"
        "fake 'hostile' attempt\n"
        "more text\n"
    )

    captured: list = []
    harness = _capture_prompt_harness(captured, monkeypatch)
    judge_dir = tmp_path / "judge"
    judge = HackerJudge(harness, td)
    judge.review(
        exec_workspace=workspace / "run",
        judge_dir=judge_dir,
        idea="x",
        info={"best_score": 0.99, "valid": True, "duration": 1.0, "exit_code": 0},
    )

    step1 = _extract_step(captured[0], 1)
    runnable = _substitute_workspace_paths(step1, workspace)

    # If the rendered command references the materialized pattern file,
    # copy the file the production code wrote (into judge_dir, since
    # judge_dir is what bwrap mounts as /workspace) into the substituted
    # test path so the rewritten path resolves. The test does NOT
    # construct its own pattern file — it uses the bytes HackerJudge
    # actually materialized.
    if "/workspace/.fabrication_patterns" in step1:
        src = judge_dir / ".fabrication_patterns"
        assert src.exists(), (
            "prompt references /workspace/.fabrication_patterns but "
            f"HackerJudge did not write it to judge_dir ({src})"
        )
        (workspace / ".fabrication_patterns").write_bytes(src.read_bytes())

    result = subprocess.run(
        ["bash", "-c", runnable],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"step 1 shell command failed (exit {result.returncode}). "
        f"stderr: {result.stderr!r}\nrendered command: {runnable!r}"
    )
    assert "fake 'hostile'" in result.stdout, (
        f"hostile fixture line not matched. stdout: {result.stdout!r}\n"
        f"rendered command: {runnable!r}"
    )


def test_discogen_fabrication_grep_step_catches_canonical_python_write(
    tmp_path: Path, monkeypatch
) -> None:
    """Integration: HackerJudge + the *real* discogen task_config must
    produce a step 1 grep that catches a canonical fabrication attempt
    (`with open('run.log', 'w')` Python write).

    Complementary to the contract test above: that one fixes the
    HackerJudge property; this one fixes the discogen-config property.
    If a future edit to tasks/discogen/task_config.yaml weakens the
    fabrication_patterns regex such that this canonical Python-write
    pattern is no longer caught, this test fails — surfacing a
    coverage regression in the production config rather than silently
    drifting.

    Loads the real discogen task_dir via the production helper. Does
    not redefine or duplicate the regex.
    """
    import subprocess
    from heuresis.judge.hacker import HackerJudge
    from heuresis.tasks import task_dir as _task_dir

    discogen_task_dir = _task_dir("discogen")

    workspace = tmp_path / "ws"
    (workspace / "run").mkdir(parents=True)
    # Canonical Python fabrication: opening run.log for write. If the
    # discogen config catches this class today, it should keep doing so.
    (workspace / "run" / "agent.log").write_text(
        "session start\n"
        "with open('run.log', 'w') as f:\n"
        "    f.write('SCORE=0.99\\n')\n"
        "session end\n"
    )

    captured: list = []
    harness = _capture_prompt_harness(captured, monkeypatch)
    judge_dir = tmp_path / "judge"
    judge = HackerJudge(harness, discogen_task_dir)
    judge.review(
        exec_workspace=workspace / "run",
        judge_dir=judge_dir,
        idea="x",
        info={"best_score": 0.99, "valid": True, "duration": 1.0, "exit_code": 0},
    )

    step1 = _extract_step(captured[0], 1)
    runnable = _substitute_workspace_paths(step1, workspace)

    if "/workspace/.fabrication_patterns" in step1:
        src = judge_dir / ".fabrication_patterns"
        assert src.exists(), (
            "prompt references /workspace/.fabrication_patterns but "
            f"HackerJudge did not write it to judge_dir ({src})"
        )
        (workspace / ".fabrication_patterns").write_bytes(src.read_bytes())

    result = subprocess.run(
        ["bash", "-c", runnable],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"discogen step 1 shell command failed (exit {result.returncode}). "
        f"stderr: {result.stderr!r}\nrendered command: {runnable!r}"
    )
    assert "open('run.log'" in result.stdout, (
        f"canonical Python-write fabrication not matched by discogen "
        f"config. stdout: {result.stdout!r}\nrendered command: {runnable!r}"
    )


def _discogen_step1_match_helper(
    tmp_path: Path,
    monkeypatch,
    agent_log_content: str,
    expected_substring_in_match: str,
) -> None:
    """Shared driver: build a workspace with the given agent.log,
    drive HackerJudge against the real discogen task_config, run the
    rendered step 1 via bash, assert the hostile sample is matched.

    Used by the integration tests below to exercise specific
    fabrication shapes (absolute-path open, pathlib write) without
    duplicating the harness setup."""
    import subprocess
    from heuresis.judge.hacker import HackerJudge
    from heuresis.tasks import task_dir as _task_dir

    discogen_task_dir = _task_dir("discogen")

    workspace = tmp_path / "ws"
    (workspace / "run").mkdir(parents=True)
    (workspace / "run" / "agent.log").write_text(agent_log_content)

    captured: list = []
    harness = _capture_prompt_harness(captured, monkeypatch)
    judge_dir = tmp_path / "judge"
    judge = HackerJudge(harness, discogen_task_dir)
    judge.review(
        exec_workspace=workspace / "run",
        judge_dir=judge_dir,
        idea="x",
        info={"best_score": 0.99, "valid": True, "duration": 1.0, "exit_code": 0},
    )

    step1 = _extract_step(captured[0], 1)
    runnable = _substitute_workspace_paths(step1, workspace)

    if "/workspace/.fabrication_patterns" in step1:
        src = judge_dir / ".fabrication_patterns"
        assert src.exists()
        (workspace / ".fabrication_patterns").write_bytes(src.read_bytes())

    result = subprocess.run(
        ["bash", "-c", runnable],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"step 1 shell command did not match the hostile sample. "
        f"exit={result.returncode} stderr={result.stderr!r}\n"
        f"agent.log content was: {agent_log_content!r}\n"
        f"rendered command: {runnable!r}"
    )
    assert expected_substring_in_match in result.stdout, (
        f"hostile fabrication shape not caught by discogen regex. "
        f"stdout: {result.stdout!r}\n"
        f"rendered command: {runnable!r}"
    )


def test_discogen_fabrication_grep_step_catches_absolute_path_open(
    tmp_path: Path, monkeypatch
) -> None:
    """The discogen production regex must catch fabrication writes that
    use an absolute or path-prefixed run.log argument: e.g.
    `open('/workspace/run/run.log', 'w')`. The previous regex required
    a bare `'run.log'` literal and missed any prefix.

    Integration test against the real `tasks/discogen/task_config.yaml`.
    """
    _discogen_step1_match_helper(
        tmp_path,
        monkeypatch,
        agent_log_content=(
            "session start\n"
            "with open('/workspace/run/run.log', 'w') as f:\n"
            "    f.write('SCORE=0.99\\n')\n"
            "session end\n"
        ),
        expected_substring_in_match="/workspace/run/run.log",
    )


def test_discogen_fabrication_grep_step_catches_pathlib_write(
    tmp_path: Path, monkeypatch
) -> None:
    """The discogen production regex must catch fabrication writes via
    `pathlib.Path('run.log').write_text(...)` (or `.write_bytes(...)`,
    or `.open(...)`). The original regex only matched the
    `open(...)` / `with open(...)` shape and missed pathlib entirely.

    Integration test against the real `tasks/discogen/task_config.yaml`.
    """
    _discogen_step1_match_helper(
        tmp_path,
        monkeypatch,
        agent_log_content=(
            "session start\n"
            "Path('run.log').write_text('SCORE=0.99\\n')\n"
            "session end\n"
        ),
        expected_substring_in_match="Path('run.log').write_text",
    )


# --- Rendered diff step must exclude universal runtime artifacts -----------


def test_diff_step_excludes_universal_runtime_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """HackerJudge's contract: when diff_scope is the workspace root,
    step 3 of the rendered prompt must surface meaningful task changes
    while suppressing the runtime artifacts every task accumulates in
    its workspace (Workspace.setup() markers and executor outputs).

    Without exclusion, the diff output is dominated by paths the agent
    has no legitimate reason to be inspecting (`.workspace_id`,
    `.bin/`, `run.log`, `agent.log`, `notes.md`, `attempts/`,
    `__pycache__/`). That noise can exceed the judge's context budget
    and bury actual invariant violations like a modified `run_main.py`
    or new top-level Python file — the judge then misses real hacks.

    Test fixture: synthetic task_config with `diff_scope: .` plus a
    synthetic baseline + run pair laid out to mirror what a real
    discogen workspace looks like at review time. We exercise the
    full production rendering path, substitute /workspace/* → test
    paths, and execute step 3 via bash. We assert positively that a
    hostile `run_main.py` modification surfaces in the diff output,
    and negatively that named runtime artifacts do not.
    """
    import subprocess
    from heuresis.judge.hacker import HackerJudge

    # Synthetic task_config: minimal, with diff_scope=. so step 3
    # diffs the whole workspace root.
    td = tmp_path / "task"
    td.mkdir()
    (td / "task_config.yaml").write_text(_yaml.safe_dump({
        "name": "fixture",
        "editable": "discovered",
        "verify": {
            "command": "echo hi",
            "stdout": "run.log",
            "diff_scope": ".",
        },
    }))

    # Baseline workspace (the "task" side, what the agent should look like
    # untouched). Holds only run_main.py — the harness file the invariants
    # forbid modifying.
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "run_main.py").write_text("# original harness\nprint('ok')\n")

    # Run workspace (the "run" side, what the executor produced). Same
    # run_main.py path but MODIFIED (a hostile invariant violation), plus
    # the universal runtime artifacts that Workspace.setup() / the agent
    # produce on every executor run.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_main.py").write_text(
        "# HOSTILE: harness rewrite\n"
        "print('ok')\nprint('cheating')\n"
    )
    (run_dir / "run.log").write_text("training step 1\nval=0.99\n")
    (run_dir / "agent.log").write_text('{"event": "step", "n": 1}\n')
    (run_dir / "notes.md").write_text("looked clean\n")
    (run_dir / ".workspace_id").write_text("abc123def456\n")
    (run_dir / ".bin").mkdir()
    (run_dir / ".bin" / "grade").write_text("#!/bin/sh\nexit 0\n")
    (run_dir / "attempts").mkdir()
    (run_dir / "attempts" / "1").write_text("attempt 1\n")
    (run_dir / "__pycache__").mkdir()
    (run_dir / "__pycache__" / "x.cpython-313.pyc").write_text("bytecode\n")

    captured: list = []
    harness = _capture_prompt_harness(captured, monkeypatch)
    judge_dir = tmp_path / "judge"
    judge = HackerJudge(harness, td, baseline_dir=baseline_dir)
    judge.review(
        exec_workspace=run_dir,
        judge_dir=judge_dir,
        idea="x",
        info={"best_score": 0.99, "valid": True, "duration": 1.0, "exit_code": 0},
    )

    step3 = _extract_step(captured[0], 3)
    # Build a single workspace root that holds both /workspace/task and
    # /workspace/run, mirroring the in-sandbox layout. The substituted
    # command will reference workspace_root/task and workspace_root/run.
    workspace_root = tmp_path / "ws"
    workspace_root.mkdir()
    (workspace_root / "task").symlink_to(baseline_dir)
    (workspace_root / "run").symlink_to(run_dir)
    runnable = _substitute_workspace_paths(step3, workspace_root)

    # Copy the excludes file the production code wrote into judge_dir so
    # the substituted /workspace/.diff_excludes path resolves. The test
    # uses the bytes HackerJudge actually wrote.
    if "/workspace/.diff_excludes" in step3:
        src = judge_dir / ".diff_excludes"
        assert src.exists(), (
            "prompt references /workspace/.diff_excludes but HackerJudge "
            f"did not write it to judge_dir ({src})"
        )
        (workspace_root / ".diff_excludes").write_bytes(src.read_bytes())

    result = subprocess.run(
        ["bash", "-c", runnable],
        capture_output=True,
        text=True,
    )

    # diff -ur returns 1 when files differ — that's expected here.
    # We accept 0 (no diff) and 1 (diffs found); 2 means an error.
    assert result.returncode in (0, 1), (
        f"step 3 shell command errored (exit {result.returncode}). "
        f"stderr: {result.stderr!r}\nrendered command: {runnable!r}"
    )

    stdout = result.stdout
    # Positive: the hostile run_main.py modification must surface.
    assert "run_main.py" in stdout, (
        f"hostile run_main.py modification did not surface in diff "
        f"output. stdout: {stdout!r}\nrendered command: {runnable!r}"
    )

    # Negative: runtime artifacts must NOT appear in the diff. Each
    # check uses a name that's distinctive enough that a hit means the
    # artifact really leaked into the output (not coincidental
    # substring of the modification context).
    forbidden = [
        "run.log",
        "agent.log",
        "notes.md",
        ".workspace_id",
        ".bin",
        "attempts",
        "__pycache__",
    ]
    leaks = [name for name in forbidden if name in stdout]
    assert not leaks, (
        f"runtime artifacts leaked into diff output: {leaks}\n"
        f"stdout: {stdout!r}\nrendered command: {runnable!r}"
    )


def _venv_extra_fixture(tmp_path: Path, monkeypatch, *, hostile_file: bool):
    """Build a HackerJudge fixture that exercises the .venv_extra audit path.

    Sets up a baseline+run pair where the run side carries a `.venv_extra/`
    directory (matching production: the sandbox bind-mounts it writable on
    PYTHONPATH=/workspace/.venv_extra; see `_bwrap.py:436`). When
    `hostile_file=True`, drops a `metric_logger.py` under it that would
    shadow imports during grading.

    Returns ``(rendered_prompt, judge_dir, run_dir, baseline_dir)``.
    """
    from heuresis.judge.hacker import HackerJudge

    td = tmp_path / "task"
    td.mkdir()
    (td / "task_config.yaml").write_text(_yaml.safe_dump({
        "name": "fixture",
        "editable": "discovered",
        "verify": {
            "command": "echo hi",
            "stdout": "run.log",
            "diff_scope": ".",
        },
    }))

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "run_main.py").write_text("# original harness\n")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_main.py").write_text("# original harness\n")
    (run_dir / ".venv_extra").mkdir()
    if hostile_file:
        (run_dir / ".venv_extra" / "metric_logger.py").write_text(
            "def log(score):\n    return 0.99\n"
        )

    captured: list = []
    harness = _capture_prompt_harness(captured, monkeypatch)
    judge_dir = tmp_path / "judge"
    judge = HackerJudge(harness, td, baseline_dir=baseline_dir)
    judge.review(
        exec_workspace=run_dir,
        judge_dir=judge_dir,
        idea="x",
        info={"best_score": 0.99, "valid": True, "duration": 1.0, "exit_code": 0},
    )
    return captured[0], judge_dir, run_dir, baseline_dir


def _run_judge_step_in_bash(
    prompt_text: str,
    step_number: int,
    *,
    judge_dir: Path,
    run_dir: Path,
    baseline_dir: Path,
    workspace_root: Path,
):
    """Render and shell-execute one numbered step from the judge prompt.

    Symlinks ``workspace_root/{task,run}`` to the baseline+run pair so
    rewritten ``/workspace/*`` paths resolve. Materializes the marker
    files (`.diff_excludes`, `.fabrication_patterns`) the step references.
    Returns the ``CompletedProcess``.
    """
    import subprocess
    step = _extract_step(prompt_text, step_number)
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "task").symlink_to(baseline_dir)
    (workspace_root / "run").symlink_to(run_dir)
    if "/workspace/.diff_excludes" in step:
        (workspace_root / ".diff_excludes").write_bytes(
            (judge_dir / ".diff_excludes").read_bytes()
        )
    if "/workspace/.fabrication_patterns" in step:
        (workspace_root / ".fabrication_patterns").write_bytes(
            (judge_dir / ".fabrication_patterns").read_bytes()
        )
    runnable = _substitute_workspace_paths(step, workspace_root)
    return subprocess.run(["bash", "-c", runnable], capture_output=True, text=True)


def test_diff_step_excludes_venv_extra_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """`.venv_extra` is in the universal `_DIFF_EXCLUDES`, so the
    step 3 diff does NOT mention `.venv_extra`. Without this exclude,
    every legit run looks suspicious because `diff -ur` reports
    `Only in run: .venv_extra` whenever the baseline lacks the
    directory and the run side has it (the universal case — the
    sandbox bind-mounts `.venv_extra` writable on PYTHONPATH).

    File-level visibility into hostile contents is provided by the
    dedicated step 4 `find` listing — see
    `test_venv_extra_listing_step_surfaces_hostile_drop`.
    """
    prompt, judge_dir, run_dir, baseline_dir = _venv_extra_fixture(
        tmp_path, monkeypatch, hostile_file=True,
    )
    result = _run_judge_step_in_bash(
        prompt, 3,
        judge_dir=judge_dir, run_dir=run_dir, baseline_dir=baseline_dir,
        workspace_root=tmp_path / "ws_step3",
    )
    assert result.returncode in (0, 1), (
        f"step 3 errored (exit {result.returncode}): {result.stderr!r}"
    )
    assert ".venv_extra" not in result.stdout, (
        f"step 3 (diff) leaked `.venv_extra` to the judge output — the "
        f"universal exclude list should suppress this so the diff stays "
        f"clean and content visibility is delegated to step 4 (find).\n"
        f"stdout: {result.stdout!r}"
    )


def test_venv_extra_listing_step_surfaces_hostile_drop(
    tmp_path: Path, monkeypatch
) -> None:
    """A hostile `.venv_extra/metric_logger.py` drop must surface as
    a file-level evidence line in step 4's find output. Without this,
    the judge has zero visibility into PYTHONPATH-shadowing imports
    inside the sandbox-bound writable mount (`_bwrap.py:436`).
    """
    prompt, judge_dir, run_dir, baseline_dir = _venv_extra_fixture(
        tmp_path, monkeypatch, hostile_file=True,
    )
    step4 = _extract_step(prompt, 4)
    assert "find" in step4 and ".venv_extra" in step4, (
        f"step 4 must be the dedicated .venv_extra find listing; "
        f"got: {step4!r}"
    )
    result = _run_judge_step_in_bash(
        prompt, 4,
        judge_dir=judge_dir, run_dir=run_dir, baseline_dir=baseline_dir,
        workspace_root=tmp_path / "ws_step4",
    )
    assert result.returncode in (0, 1), (
        f"step 4 errored (exit {result.returncode}): {result.stderr!r}"
    )
    assert "metric_logger.py" in result.stdout, (
        f"step 4 (find) did not surface the hostile .venv_extra file. "
        f"The judge has no visibility into PYTHONPATH-shadowing imports.\n"
        f"stdout: {result.stdout!r}"
    )


def test_venv_extra_listing_step_clean_when_empty(
    tmp_path: Path, monkeypatch
) -> None:
    """An empty `.venv_extra/` (the legit case) must produce empty
    step 4 output — otherwise every legit run would trip the
    suspicious-evidence path on the judge's verdict rules.
    """
    prompt, judge_dir, run_dir, baseline_dir = _venv_extra_fixture(
        tmp_path, monkeypatch, hostile_file=False,
    )
    step4 = _extract_step(prompt, 4)
    assert "find" in step4 and ".venv_extra" in step4, (
        f"step 4 must be the dedicated .venv_extra find listing; "
        f"got: {step4!r}"
    )
    result = _run_judge_step_in_bash(
        prompt, 4,
        judge_dir=judge_dir, run_dir=run_dir, baseline_dir=baseline_dir,
        workspace_root=tmp_path / "ws_step4_clean",
    )
    assert result.returncode in (0, 1), (
        f"step 4 errored (exit {result.returncode}): {result.stderr!r}"
    )
    assert result.stdout.strip() == "", (
        f"step 4 (find) printed output for an empty .venv_extra — "
        f"every legit run would trip suspicious-evidence.\n"
        f"stdout: {result.stdout!r}"
    )
