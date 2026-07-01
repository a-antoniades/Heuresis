"""Tiny rubric loader for the analyzing-search-runs skill.

Each rubric file at `rubrics/<name>.md` has YAML frontmatter (delimited by
`---` lines) followed by the prompt body. We parse the frontmatter with the
stdlib only — values are simple scalars / lists / single-level dicts, so a
hand-rolled parser is fine and keeps the skill PyYAML-free.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Default rubric — used when no --rubric flag is passed and no rubric is
# stamped on a cache. The user-facing default for the skill.
DEFAULT_RUBRIC = "gupta_pruthi_2025"


def rubrics_dir() -> Path:
    """Resolve the rubrics directory next to this script's parent."""
    return Path(__file__).resolve().parent.parent / "rubrics"


def list_rubrics() -> list[str]:
    """List rubric names available on disk (without `.md` extension)."""
    d = rubrics_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.md"))


def rubric_path(name: str) -> Path:
    """Resolve a rubric name to its file path."""
    return rubrics_dir() / f"{name}.md"


def load_rubric(name: str) -> dict[str, Any]:
    """Load a rubric file by name. Returns dict with frontmatter fields plus
    a `prompt_body` key holding the markdown body and `path` key holding the
    source path. Raises FileNotFoundError if the rubric does not exist.
    """
    p = rubric_path(name)
    if not p.exists():
        available = ", ".join(list_rubrics()) or "<none>"
        raise FileNotFoundError(
            f"Rubric '{name}' not found at {p}. Available: {available}"
        )

    text = p.read_text()
    frontmatter, body = _split_frontmatter(text)
    parsed = _parse_yaml_subset(frontmatter)
    parsed["prompt_body"] = body
    parsed["path"] = str(p)
    parsed.setdefault("name", name)
    _validate(parsed)
    return parsed


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a markdown file with `---` YAML frontmatter into (frontmatter, body)."""
    if not text.startswith("---"):
        raise ValueError("Rubric file must start with '---' frontmatter delimiter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Rubric frontmatter must be closed with a second '---' line")
    return parts[1].strip("\n"), parts[2].lstrip("\n")


# ---------------------------------------------------------------------------
# Minimal YAML subset parser (scalars, single-level lists, simple inline dicts)
# ---------------------------------------------------------------------------

_INLINE_DICT_RE = re.compile(r"^\{(.*)\}$")


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used in rubric frontmatter.

    Supported:
      key: scalar
      key: [a, b, c]   # inline list
      key: "quoted"
      key:
        - item1
        - item2
      key:
        sub: scalar
        sub2: {name: "X", color: "#abcdef"}   # inline dict value
    Comments (#...) and blank lines are skipped.
    """
    result: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = _strip_comment(raw)
        if not line.strip():
            i += 1
            continue
        if not _is_top_level(line):
            # Should not happen at this entry point.
            i += 1
            continue
        key, sep, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if not sep:
            i += 1
            continue
        if rest:
            # Inline value
            result[key] = _parse_scalar_or_list(rest)
            i += 1
        else:
            # Block — peek next non-blank line
            block_lines: list[str] = []
            j = i + 1
            while j < len(lines):
                nxt = _strip_comment(lines[j])
                if not nxt.strip():
                    j += 1
                    continue
                if _is_top_level(nxt):
                    break
                block_lines.append(nxt)
                j += 1
            result[key] = _parse_block(block_lines)
            i = j
    return result


def _strip_comment(line: str) -> str:
    """Strip `# comment` from a line, preserving `#` inside quotes."""
    in_quote = False
    quote = ""
    out = []
    for ch in line:
        if ch in ("'", '"'):
            if in_quote and ch == quote:
                in_quote = False
            elif not in_quote:
                in_quote = True
                quote = ch
        if ch == "#" and not in_quote:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _is_top_level(line: str) -> bool:
    return bool(line) and not line.startswith((" ", "\t"))


def _parse_scalar_or_list(s: str) -> Any:
    s = s.strip()
    if not s:
        return ""
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in _split_top_level(inner, ",")]
    if s.startswith("{") and s.endswith("}"):
        return _parse_inline_dict(s)
    return _parse_scalar(s)


def _parse_block(block_lines: list[str]) -> Any:
    """Block can be a list (- items) or a mapping (key: value)."""
    stripped = [ln for ln in block_lines if ln.strip()]
    if not stripped:
        return None
    first = stripped[0].lstrip()
    if first.startswith("- "):
        items: list[Any] = []
        for ln in stripped:
            t = ln.lstrip()
            if not t.startswith("- "):
                continue
            items.append(_parse_scalar_or_list(t[2:].strip()))
        return items
    # Mapping — single level
    out: dict[str, Any] = {}
    for ln in stripped:
        t = ln.lstrip()
        key, sep, rest = t.partition(":")
        if not sep:
            continue
        out[_parse_scalar(key.strip())] = _parse_scalar_or_list(rest.strip())
    return out


def _parse_inline_dict(s: str) -> dict[str, Any]:
    inner = s[1:-1].strip()
    if not inner:
        return {}
    out: dict[str, Any] = {}
    for piece in _split_top_level(inner, ","):
        key, sep, rest = piece.partition(":")
        if not sep:
            continue
        out[_parse_scalar(key.strip())] = _parse_scalar(rest.strip())
    return out


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split on `sep` ignoring separators inside quotes / brackets / braces."""
    parts: list[str] = []
    depth = 0
    in_quote = False
    quote_ch = ""
    cur: list[str] = []
    for ch in s:
        if in_quote:
            cur.append(ch)
            if ch == quote_ch:
                in_quote = False
            continue
        if ch in ("'", '"'):
            in_quote = True
            quote_ch = ch
            cur.append(ch)
            continue
        if ch in "[{(":
            depth += 1
            cur.append(ch)
            continue
        if ch in "]})":
            depth -= 1
            cur.append(ch)
            continue
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if not s:
        return ""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s.lower() in ("null", "~", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ("name", "score_range", "direction", "novel_threshold", "levels")


def _validate(rubric: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in rubric]
    if missing:
        raise ValueError(
            f"Rubric '{rubric.get('name', '?')}' missing required keys: {missing}"
        )
    direction = rubric["direction"]
    if direction not in ("higher_is_more_novel", "higher_is_more_plagiarized"):
        raise ValueError(
            f"Rubric '{rubric['name']}' has invalid direction='{direction}'"
        )
    sr = rubric["score_range"]
    if not (isinstance(sr, list) and len(sr) == 2 and all(isinstance(x, int) for x in sr)):
        raise ValueError(
            f"Rubric '{rubric['name']}' score_range must be [int, int]"
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def is_novel_score(rubric: dict[str, Any], score: int) -> bool:
    """Return True if score is on the 'novel' side of the rubric."""
    threshold = rubric["novel_threshold"]
    if rubric["direction"] == "higher_is_more_novel":
        return score >= threshold
    # higher_is_more_plagiarized → novel side is the LOW end
    return score <= threshold


def level_range(rubric: dict[str, Any]) -> list[str]:
    """Return distribution bucket keys (as strings) covering the score range."""
    lo, hi = rubric["score_range"]
    return [str(i) for i in range(lo, hi + 1)]


def level_name(rubric: dict[str, Any], score: int | str) -> str:
    """Look up a level's display name; falls back to str(score) if missing."""
    levels = rubric.get("levels", {}) or {}
    entry = levels.get(str(score))
    if isinstance(entry, dict):
        return entry.get("name", str(score))
    return str(score)


def level_color(rubric: dict[str, Any], score: int | str) -> str | None:
    levels = rubric.get("levels", {}) or {}
    entry = levels.get(str(score))
    if isinstance(entry, dict):
        return entry.get("color")
    return None
