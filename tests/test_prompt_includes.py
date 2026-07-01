"""Tests for Jinja includes in workspace prompt templates."""

from __future__ import annotations

from pathlib import Path

from heuresis.workspace import Workspace


def test_prompt_path_can_include_repo_relative_template(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.j2"
    prompt.write_text(
        '{% include "heuresis/prompts/common/_test_fragment.j2" %}\n'
        "Hello {{ name }}!"
    )
    fragment = Path("src/heuresis/prompts/common/_test_fragment.j2")
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text("Included {{ name }}.")
    try:
        assert Workspace(prompt=prompt).render_prompt({"name": "Ada"}) == (
            "Included Ada.\nHello Ada!"
        )
    finally:
        fragment.unlink()
