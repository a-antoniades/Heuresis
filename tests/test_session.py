"""Tests for agent profiles, harness command building, session logic, and bwrap integration.

All tests are fast unit tests -- no actual agent processes are spawned.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from heuresis.agent import (
    CLAUDE_CODE,
    CURSOR_AGENT,
    OPENCODE,
    AgentProfile,
    DataMount,
    detect_profile,
    generate_session_id,
    _parse_session_id_from_log,
)
from heuresis import _bwrap
from heuresis.models import RunResult, TokenUsage


# ── AgentProfile ────────────────────────────────────────────────────

class TestAgentProfile:
    def test_opencode_profile(self):
        assert OPENCODE.name == "opencode"
        assert OPENCODE.run_cmd == ["run"]
        assert OPENCODE.format_args == ["--format", "json"]
        assert OPENCODE.model_flag == "-m"
        assert OPENCODE.session_flag == "-s"
        assert OPENCODE.session_id_flag is None
        assert OPENCODE.create_session_cmd is None
        assert len(OPENCODE.data_mounts) == 2

    def test_claude_code_profile(self):
        assert CLAUDE_CODE.name == "claude"
        assert CLAUDE_CODE.run_cmd == ["-p"]
        assert CLAUDE_CODE.format_args == ["--output-format", "json"]
        assert CLAUDE_CODE.model_flag == "--model"
        assert CLAUDE_CODE.session_flag == "-r"
        assert CLAUDE_CODE.session_id_flag == "--session-id"
        assert CLAUDE_CODE.create_session_cmd is None
        assert "--dangerously-skip-permissions" in CLAUDE_CODE.permission_args

    def test_cursor_agent_profile(self):
        assert CURSOR_AGENT.name == "agent"
        assert CURSOR_AGENT.run_cmd == ["-p"]
        assert CURSOR_AGENT.session_flag == "--resume"
        assert CURSOR_AGENT.create_session_cmd == ["create-chat"]
        assert "--force" in CURSOR_AGENT.permission_args
        assert "--trust" in CURSOR_AGENT.permission_args

    def test_detect_profile(self):
        assert detect_profile("opencode") is OPENCODE
        assert detect_profile("claude") is CLAUDE_CODE
        assert detect_profile("agent") is CURSOR_AGENT
        assert detect_profile("unknown") is None

    def test_profiles_are_frozen(self):
        with pytest.raises(AttributeError):
            OPENCODE.name = "modified"


class TestGenerateSessionId:
    def test_opencode_returns_none(self):
        assert generate_session_id(OPENCODE, "/usr/bin/opencode") is None

    def test_claude_code_returns_uuid(self):
        sid = generate_session_id(CLAUDE_CODE, "/usr/bin/claude")
        assert sid is not None
        assert len(sid) == 36  # dashed UUID

    @patch("heuresis.agent._create_cursor_session", return_value="chat_abc123")
    def test_cursor_agent_calls_create_chat(self, mock_create):
        sid = generate_session_id(CURSOR_AGENT, "/usr/bin/agent")
        assert sid == "chat_abc123"
        mock_create.assert_called_once_with("/usr/bin/agent")


class TestParseSessionId:
    def test_parses_session_id_field(self, tmp_path):
        log = tmp_path / "agent.log"
        log.write_text(json.dumps({"session_id": "sess_12345678"}) + "\n")
        assert _parse_session_id_from_log(log) == "sess_12345678"

    def test_parses_sessionId_field(self, tmp_path):
        log = tmp_path / "agent.log"
        log.write_text(json.dumps({"sessionId": "abcdef1234567890"}) + "\n")
        assert _parse_session_id_from_log(log) == "abcdef1234567890"

    def test_skips_short_ids(self, tmp_path):
        log = tmp_path / "agent.log"
        log.write_text(json.dumps({"id": "short"}) + "\n")
        assert _parse_session_id_from_log(log) is None

    def test_skips_non_json_lines(self, tmp_path):
        log = tmp_path / "agent.log"
        log.write_text("not json\n" + json.dumps({"session_id": "valid_id_here"}) + "\n")
        assert _parse_session_id_from_log(log) == "valid_id_here"

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _parse_session_id_from_log(tmp_path / "nope.log") is None

    def test_returns_none_for_empty_log(self, tmp_path):
        log = tmp_path / "agent.log"
        log.write_text("")
        assert _parse_session_id_from_log(log) is None


# ── Harness command building ────────────────────────────────────────

class TestBuildInnerCmd:
    """Test _build_inner_cmd without constructing a full Harness (which runs preflights)."""

    def _make_harness_stub(self, profile, binary="/usr/bin/test", model=None):
        """Minimal Harness-like object for testing _build_inner_cmd."""
        from heuresis.harness import Harness

        class Stub:
            pass

        stub = Stub()
        stub.profile = profile
        stub._binary = binary
        stub.model = model
        stub._build_inner_cmd = Harness._build_inner_cmd.__get__(stub)
        return stub

    def test_opencode_basic(self):
        h = self._make_harness_stub(OPENCODE, "/usr/bin/opencode", model="openai/gpt-5")
        cmd = h._build_inner_cmd("hello world")
        assert cmd[0] == "/usr/bin/opencode"
        assert cmd[1] == "run"
        assert "-m" in cmd
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "openai/gpt-5"
        assert "hello world" in cmd
        assert cmd[-2:] == ["--format", "json"]

    def test_claude_code_basic(self):
        h = self._make_harness_stub(CLAUDE_CODE, "/usr/bin/claude", model="sonnet")
        cmd = h._build_inner_cmd("do something")
        assert cmd[0] == "/usr/bin/claude"
        assert cmd[1] == "-p"
        assert "--model" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert cmd[-2:] == ["--output-format", "json"]

    def test_cursor_agent_basic(self):
        h = self._make_harness_stub(CURSOR_AGENT, "/usr/bin/agent", model="gpt-5")
        cmd = h._build_inner_cmd("fix bug")
        assert cmd[0] == "/usr/bin/agent"
        assert cmd[1] == "-p"
        assert "--force" in cmd
        assert "--trust" in cmd

    def test_session_id_flag(self):
        h = self._make_harness_stub(OPENCODE, "/usr/bin/opencode")
        cmd = h._build_inner_cmd("msg", session_id="sess_123")
        assert "-s" in cmd
        idx = cmd.index("-s")
        assert cmd[idx + 1] == "sess_123"

    def test_claude_session_flag(self):
        h = self._make_harness_stub(CLAUDE_CODE, "/usr/bin/claude")
        cmd = h._build_inner_cmd("msg", session_id="uuid_abc")
        assert "--session-id" in cmd
        idx = cmd.index("--session-id")
        assert cmd[idx + 1] == "uuid_abc"

    def test_cursor_session_flag(self):
        h = self._make_harness_stub(CURSOR_AGENT, "/usr/bin/agent")
        cmd = h._build_inner_cmd("msg", session_id="chat_xyz")
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "chat_xyz"

    def test_extra_cmd_args(self):
        h = self._make_harness_stub(CLAUDE_CODE, "/usr/bin/claude")
        cmd = h._build_inner_cmd("msg", extra_cmd_args=["--session-id", "my_uuid"])
        assert "--session-id" in cmd
        idx = cmd.index("--session-id")
        assert cmd[idx + 1] == "my_uuid"

    def test_no_model_when_none(self):
        h = self._make_harness_stub(OPENCODE, "/usr/bin/opencode", model=None)
        cmd = h._build_inner_cmd("msg")
        assert "-m" not in cmd


# ── Bwrap integration ──────────────────────────────────────────────

class TestBwrapAgentMounts:
    def test_add_agent_mounts_always_writable(self, tmp_path):
        data_dir = tmp_path / ".local" / "share" / "opencode"
        data_dir.mkdir(parents=True)
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)

        profile = AgentProfile(
            name="test",
            run_cmd=["run"],
            format_args=["--format", "json"],
            model_flag="-m",
            session_flag="-s",
            continue_flag="-c",
            data_mounts=[
                DataMount(str(data_dir), "/workspace/.local/share/opencode"),
                DataMount(str(config_dir), "/workspace/.config/opencode"),
            ],
        )

        cmd: list[str] = []
        _bwrap._add_agent_mounts(cmd, profile, session_mode=False)

        assert "--bind" in cmd
        assert "--ro-bind" not in cmd
        assert str(data_dir) in cmd
        assert str(config_dir) in cmd

    def test_add_agent_mounts_session_mode_also_writable(self, tmp_path):
        data_dir = tmp_path / ".claude"
        data_dir.mkdir()

        profile = AgentProfile(
            name="test",
            run_cmd=["-p"],
            format_args=["--output-format", "json"],
            model_flag="--model",
            session_flag="-r",
            continue_flag="--continue",
            data_mounts=[DataMount(str(data_dir), "/workspace/.claude")],
        )

        cmd: list[str] = []
        _bwrap._add_agent_mounts(cmd, profile, session_mode=True)

        assert "--bind" in cmd
        assert "--ro-bind" not in cmd

    def test_skips_nonexistent_dirs(self, tmp_path):
        profile = AgentProfile(
            name="test",
            run_cmd=["-p"],
            format_args=[],
            model_flag="--model",
            session_flag="-r",
            continue_flag="--continue",
            data_mounts=[DataMount(str(tmp_path / "nope"), "/workspace/nope")],
        )

        cmd: list[str] = []
        _bwrap._add_agent_mounts(cmd, profile, session_mode=False)
        assert len(cmd) == 0

    def test_build_command_uses_profile(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / ".agentdata"
        data_dir.mkdir()

        profile = AgentProfile(
            name="test",
            run_cmd=["run"],
            format_args=["--format", "json"],
            model_flag="-m",
            session_flag="-s",
            continue_flag="-c",
            data_mounts=[DataMount(str(data_dir), "/workspace/.agentdata")],
        )

        cmd = _bwrap.build_command(
            workspace=workspace,
            inner_cmd=["echo", "test"],
            profile=profile,
            session_mode=False,
        )

        joined = " ".join(cmd)
        assert "--ro-bind" in joined
        assert str(data_dir) in joined

    def test_build_command_session_mode_writable(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        data_dir = tmp_path / ".agentdata"
        data_dir.mkdir()

        profile = AgentProfile(
            name="test",
            run_cmd=["run"],
            format_args=["--format", "json"],
            model_flag="-m",
            session_flag="-s",
            continue_flag="-c",
            data_mounts=[DataMount(str(data_dir), "/workspace/.agentdata")],
        )

        cmd = _bwrap.build_command(
            workspace=workspace,
            inner_cmd=["echo", "test"],
            profile=profile,
            session_mode=True,
        )

        data_mount_idx = cmd.index(str(data_dir))
        bind_flag = cmd[data_mount_idx - 1]
        assert bind_flag == "--bind", f"Expected --bind for session_mode, got {bind_flag}"


# ── Session logic ──────────────────────────────────────────────────

class TestSession:
    def _make_mock_harness(self, profile=OPENCODE):
        harness = MagicMock()
        harness.profile = profile
        harness._binary = "/usr/bin/test"
        harness.run.return_value = RunResult(
            workspace=Path("/tmp/ws"),
            exit_code=0,
            duration=1.0,
            log_path=Path("/tmp/ws/agent.log"),
            tag=None,
            tokens=TokenUsage(),
        )
        return harness

    def test_first_turn_creates_session(self, tmp_path):
        from heuresis.session import Session

        harness = self._make_mock_harness(OPENCODE)
        session = Session(harness, tmp_path, MagicMock())

        assert session.session_id is None
        assert session.turn == 0

        log = tmp_path / "agent.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        harness.run.return_value = RunResult(
            workspace=tmp_path, exit_code=0, duration=1.0,
            log_path=log, tag=None, tokens=TokenUsage(),
        )
        log.write_text(json.dumps({"session_id": "oc_session_1"}) + "\n")

        session.run("hello")

        assert session.turn == 1
        assert session.session_id == "oc_session_1"
        call_kwargs = harness.run.call_args[1]
        assert call_kwargs["session_id"] is None
        assert call_kwargs["extra_cmd_args"] == []

    def test_second_turn_passes_session_id(self, tmp_path):
        from heuresis.session import Session

        harness = self._make_mock_harness(OPENCODE)
        log = tmp_path / "agent.log"
        log.write_text(json.dumps({"session_id": "oc_sess_42"}) + "\n")
        harness.run.return_value = RunResult(
            workspace=tmp_path, exit_code=0, duration=1.0,
            log_path=log, tag=None, tokens=TokenUsage(),
        )

        session = Session(harness, tmp_path, MagicMock())
        session.run("turn 1")
        session.run("turn 2")

        assert session.turn == 2
        second_call = harness.run.call_args_list[1][1]
        assert second_call["session_id"] == "oc_sess_42"

    def test_reset_clears_state(self, tmp_path):
        from heuresis.session import Session

        harness = self._make_mock_harness(OPENCODE)
        log = tmp_path / "agent.log"
        log.write_text(json.dumps({"session_id": "oc_sess_99"}) + "\n")
        harness.run.return_value = RunResult(
            workspace=tmp_path, exit_code=0, duration=1.0,
            log_path=log, tag=None, tokens=TokenUsage(),
        )

        session = Session(harness, tmp_path, MagicMock())
        session.run("turn 1")
        assert session.session_id == "oc_sess_99"

        session.reset()
        assert session.session_id is None
        assert session.turn == 0

    def test_claude_first_turn_passes_session_id_flag(self, tmp_path):
        from heuresis.session import Session

        harness = self._make_mock_harness(CLAUDE_CODE)
        log = tmp_path / "agent.log"
        log.write_text("")
        harness.run.return_value = RunResult(
            workspace=tmp_path, exit_code=0, duration=1.0,
            log_path=log, tag=None, tokens=TokenUsage(),
        )

        session = Session(harness, tmp_path, MagicMock())
        session.run("ideate")

        call_kwargs = harness.run.call_args[1]
        assert call_kwargs["session_id"] is None
        extra = call_kwargs["extra_cmd_args"]
        assert extra[0] == "--session-id"
        assert len(extra[1]) == 36  # dashed UUID

    def test_claude_second_turn_uses_session_flag(self, tmp_path):
        from heuresis.session import Session

        harness = self._make_mock_harness(CLAUDE_CODE)
        log = tmp_path / "agent.log"
        log.write_text("")
        harness.run.return_value = RunResult(
            workspace=tmp_path, exit_code=0, duration=1.0,
            log_path=log, tag=None, tokens=TokenUsage(),
        )

        session = Session(harness, tmp_path, MagicMock())
        session.run("turn 1")

        pre_generated_id = session.session_id
        assert pre_generated_id is not None

        session.run("turn 2")
        second_call = harness.run.call_args_list[1][1]
        assert second_call["session_id"] == pre_generated_id
        assert second_call["extra_cmd_args"] == []

    @patch("heuresis.agent._create_cursor_session", return_value="chat_pre_123")
    def test_cursor_first_turn_resumes_pre_created_chat(self, mock_create, tmp_path):
        from heuresis.session import Session

        harness = self._make_mock_harness(CURSOR_AGENT)
        log = tmp_path / "agent.log"
        log.write_text("")
        harness.run.return_value = RunResult(
            workspace=tmp_path, exit_code=0, duration=1.0,
            log_path=log, tag=None, tokens=TokenUsage(),
        )

        session = Session(harness, tmp_path, MagicMock())
        session.run("first message")

        call_kwargs = harness.run.call_args[1]
        assert call_kwargs["session_id"] == "chat_pre_123"
        assert call_kwargs["extra_cmd_args"] == []

    def test_close_is_equivalent_to_reset(self, tmp_path):
        from heuresis.session import Session

        harness = self._make_mock_harness(OPENCODE)
        log = tmp_path / "agent.log"
        log.write_text(json.dumps({"session_id": "oc_close"}) + "\n")
        harness.run.return_value = RunResult(
            workspace=tmp_path, exit_code=0, duration=1.0,
            log_path=log, tag=None, tokens=TokenUsage(),
        )

        session = Session(harness, tmp_path, MagicMock())
        session.run("turn 1")
        session.close()
        assert session.session_id is None
        assert session.turn == 0

    def test_reset_then_new_session(self, tmp_path):
        from heuresis.session import Session

        harness = self._make_mock_harness(OPENCODE)
        call_count = [0]
        session_ids = ["sess_first", "sess_second"]

        def mock_run(**kwargs):
            log = tmp_path / "agent.log"
            log.write_text(json.dumps({"session_id": session_ids[call_count[0] // 1]}) + "\n")
            call_count[0] += 1
            return RunResult(
                workspace=tmp_path, exit_code=0, duration=1.0,
                log_path=log, tag=None, tokens=TokenUsage(),
            )

        harness.run.side_effect = lambda **kw: mock_run(**kw)

        session = Session(harness, tmp_path, MagicMock())
        session.run("first session")
        assert session.session_id == "sess_first"

        session.reset()
        session.run("new session")
        assert session.session_id == "sess_second"


# ── Harness factory ────────────────────────────────────────────────

class TestHarnessFactory:
    @patch("heuresis.preflight.check_agent", return_value=[])
    @patch("shutil.which", return_value="/usr/bin/opencode")
    def test_session_factory(self, mock_which, mock_check, tmp_path):
        from heuresis.harness import Harness

        h = Harness("opencode", model="test-model")
        s = h.session(workspace=tmp_path, sandbox=MagicMock())

        from heuresis.session import Session
        assert isinstance(s, Session)

    @patch("heuresis.preflight.check_agent", return_value=[])
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_auto_detect_claude_profile(self, mock_which, mock_check):
        from heuresis.harness import Harness

        h = Harness("claude", model="sonnet")
        assert h.profile is CLAUDE_CODE

    @patch("heuresis.preflight.check_agent", return_value=[])
    @patch("shutil.which", return_value="/usr/bin/agent")
    def test_auto_detect_cursor_profile(self, mock_which, mock_check):
        from heuresis.harness import Harness

        h = Harness("agent", model="gpt-5")
        assert h.profile is CURSOR_AGENT

    def test_unknown_agent_raises(self):
        from heuresis.harness import Harness

        with pytest.raises(ValueError, match="No built-in profile"):
            Harness("unknown_agent")

    @patch("heuresis.preflight.check_agent", return_value=[])
    @patch("shutil.which", return_value="/usr/bin/custom")
    def test_explicit_profile(self, mock_which, mock_check):
        from heuresis.harness import Harness

        custom = AgentProfile(
            name="custom",
            run_cmd=["--run"],
            format_args=["--json"],
            model_flag="-m",
            session_flag="--sess",
            continue_flag="--cont",
        )
        h = Harness("custom", profile=custom)
        assert h.profile is custom
