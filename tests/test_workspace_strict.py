"""Tests for Workspace.render_prompt with StrictUndefined."""
from __future__ import annotations

import pytest

from heuresis.workspace import Workspace


def test_missing_var_raises():
    ws = Workspace(prompt="Hello {{ name }}!")
    with pytest.raises(Exception) as exc_info:
        ws.render_prompt({})  # missing 'name'
    # jinja2.exceptions.UndefinedError
    assert "undefined" in str(exc_info.value).lower() or "name" in str(exc_info.value).lower()


def test_all_vars_renders():
    ws = Workspace(prompt="Hello {{ name }}!")
    assert ws.render_prompt({"name": "world"}) == "Hello world!"
