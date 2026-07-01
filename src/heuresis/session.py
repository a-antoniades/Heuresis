"""Small stateful wrapper around :class:`heuresis.harness.Harness`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from heuresis.agent import _parse_session_id_from_log, generate_session_id


class Session:
    """Compatibility helper for multi-turn agent sessions.

    Newer code can use ``Harness.run(..., stateful=True)`` directly. This class
    keeps the explicit session API available for tests and older callers.
    """

    def __init__(self, harness: Any, workspace: Path, sandbox: Any | None = None) -> None:
        self.harness = harness
        self.workspace = Path(workspace)
        self.sandbox = sandbox
        self.session_id: str | None = None
        self.turn = 0

    def run(self, prompt: str) -> Any:
        extra_cmd_args: list[str] = []
        session_id = self.session_id

        if self.turn == 0 and session_id is None:
            generated = generate_session_id(self.harness.profile, self.harness._binary)
            if generated is not None:
                self.session_id = generated
                session_id = generated
                if self.harness.profile.session_id_flag:
                    extra_cmd_args = [self.harness.profile.session_id_flag, generated]
                    session_id = None

        result = self.harness.run(
            prompt=prompt,
            path=self.workspace,
            session_id=session_id,
            extra_cmd_args=extra_cmd_args,
        )

        # Harness.run may return a RunFuture in production or a RunResult in tests.
        if hasattr(result, "result") and callable(result.result):
            result = result.result()

        if self.session_id is None:
            log_path = getattr(result, "log_path", None) or self.workspace / "agent.log"
            parsed = _parse_session_id_from_log(Path(log_path))
            if parsed:
                self.session_id = parsed

        self.turn += 1
        return result

    def reset(self) -> None:
        self.session_id = None
        self.turn = 0

    def close(self) -> None:
        self.reset()
