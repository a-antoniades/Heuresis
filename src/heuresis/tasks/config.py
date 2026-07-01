"""YAML-based task configuration helpers (discogen-aligned).

Tasks live at ``src/heuresis/tasks/<name>/`` and are described by small
YAML files (``task_config.yaml``, ``baseline_scores.yaml``) plus a free-form
``description.md``. There is no ``Task`` class — the layout is the contract.
Consumers read what they need by path; these helpers cover the most common
lookups so the ``"min"``/``"max"`` convention stays in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_TASKS_DIR = Path(__file__).parent


def task_dir(name: str) -> Path:
    """Absolute path to a task domain directory.

    Raises:
        ValueError: If no directory matches ``name`` under the tasks package.
    """
    p = _TASKS_DIR / name
    if not p.is_dir():
        raise ValueError(f"Unknown task: {name!r} (looked in {_TASKS_DIR})")
    return p


def load_yaml(td: Path, filename: str) -> dict[str, Any]:
    """Read and parse a YAML file from a task directory."""
    return yaml.safe_load((td / filename).read_text())


def baseline_scores(td: Path) -> dict[str, Any]:
    """Read the task's ``baseline_scores.yaml`` as a dict."""
    return load_yaml(td, "baseline_scores.yaml")


def lower_is_better(td: Path) -> bool:
    """True iff the task's metric should be minimized.

    Hides the raw ``"min"``/``"max"`` string convention from consumers.
    """
    return baseline_scores(td)["objective"] == "min"


def novelty_anchor(td: Path) -> dict[str, Any]:
    """Return the ``novelty_anchor`` block from ``task_config.yaml``, or ``{}``.

    Used by the analyzing-search-runs skill to task-anchor the 1-4 novelty
    rubric. Absence is intentional: tasks without an anchor fall back to the
    implicit "all of ML literature" reference frame (legacy behavior).
    """
    cfg = load_yaml(td, "task_config.yaml")
    return cfg.get("novelty_anchor", {}) or {}


def novelty_anchor_markdown(td: Path) -> str:
    """Format the task's novelty_anchor as a markdown block for prompts.

    Returns an empty string if no anchor is configured. Designed to be dropped
    into the classifier prompt's ``{task_preamble}`` slot.
    """
    anchor = novelty_anchor(td)
    if not anchor:
        return ""

    lines: list[str] = ["## Novelty reference frame (task-anchored)", ""]
    ctx = anchor.get("search_context") or {}
    if ctx:
        lines.append(f"**Task domain:** {anchor.get('task_domain', 'unknown')}")
        for k, v in ctx.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    catalog = anchor.get("catalog") or {}
    if catalog:
        lines.append("**Known-technique catalog** (an idea built only from these in their "
                     "usual configurations is at most level 2):")
        for group, items in catalog.items():
            items_str = ", ".join(items) if isinstance(items, list) else str(items)
            lines.append(f"- *{group}*: {items_str}")
        lines.append("")

    l3 = anchor.get("level3_examples") or []
    l4 = anchor.get("level4_examples") or []
    if l3:
        lines.append("**Level-3 examples** (material extensions of a catalog primitive):")
        for ex in l3:
            lines.append(f"- {ex}")
        lines.append("")
    if l4:
        lines.append("**Level-4 examples** (new core mechanism, no clear ancestor):")
        for ex in l4:
            lines.append(f"- {ex}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
