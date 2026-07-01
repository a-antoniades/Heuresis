"""Single-idea spotlight: lineage chain + idea text, side-by-side.

Pulls one run from the store, traces its primary parent chain back to a
root, and renders a one-page card with the chain on the left and the
idea Markdown on the right.

Usage:
    python -m libs.lineage_idea_card <experiment_id> <run_id> <iteration>
                                     [--out path.png]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

RESEARCH_AGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = RESEARCH_AGENT_ROOT / "runs" / "nanogpt" / "store.db"


def _split(raw: str | None) -> list[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def fetch_chain(
    experiment_id: str,
    run_id: str,
    iteration: int,
    *,
    db_path: Path = DEFAULT_DB,
) -> list[dict]:
    """Walk primary-parent links back to a root. Returns ancestors-first."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    def row_for(rid: str, before_iter: int) -> dict | None:
        r = conn.execute(
            "SELECT run_id, iteration, score, valid, parent_ids, generation, "
            "       idea, metadata "
            "FROM runs WHERE experiment_id=? AND run_id=? AND iteration<? "
            "AND run_type='executor' "
            "ORDER BY iteration DESC LIMIT 1",
            (experiment_id, rid, before_iter),
        ).fetchone()
        return dict(r) if r else None

    head = conn.execute(
        "SELECT run_id, iteration, score, valid, parent_ids, generation, "
        "       idea, metadata "
        "FROM runs WHERE experiment_id=? AND run_id=? AND iteration=? "
        "AND run_type='executor'",
        (experiment_id, run_id, iteration),
    ).fetchone()
    if head is None:
        conn.close()
        raise SystemExit(f"no run {run_id}@{iteration} in {experiment_id}")
    chain = [dict(head)]

    cur = chain[0]
    while True:
        parents = _split(cur["parent_ids"])
        if not parents:
            break
        prev = row_for(parents[0], cur["iteration"])
        if prev is None or any(c["run_id"] == prev["run_id"] for c in chain):
            break
        chain.append(prev)
        cur = prev

    conn.close()
    chain.reverse()
    for c in chain:
        c["meta"] = json.loads(c["metadata"]) if c["metadata"] else {}
    return chain


def _wrap_idea(text: str, width: int = 80) -> str:
    """Light cleanup so long lines wrap nicely in matplotlib text."""
    out_lines = []
    for line in text.splitlines():
        if not line.strip():
            out_lines.append("")
            continue
        if line.lstrip().startswith(("#", "-", "*", "1.", "2.", "3.", "```")):
            indent = len(line) - len(line.lstrip())
            out_lines.append(" " * indent + line.strip())
            continue
        wrapped = textwrap.fill(
            line, width=width,
            break_long_words=False, break_on_hyphens=False,
            replace_whitespace=False,
        )
        out_lines.append(wrapped)
    return "\n".join(out_lines)


def render_card(
    chain: list[dict],
    out_path: str | Path,
    *,
    title: str | None = None,
    lower_is_better: bool = True,
    score_range: tuple[float, float] | None = None,
) -> Path:
    """Two-column figure: lineage chain on the left, idea Markdown on the right."""
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.04)
    ax_chain = fig.add_subplot(gs[0, 0])
    ax_idea = fig.add_subplot(gs[0, 1])

    # --- left: lineage chain ---
    n = len(chain)
    ys = list(range(n - 1, -1, -1))  # root at bottom, target at top

    s_vals = [c["score"] for c in chain if c["score"] is not None]
    if score_range is None and s_vals:
        score_range = (min(s_vals), max(s_vals))
    cmap = plt.cm.RdYlGn_r if lower_is_better else plt.cm.RdYlGn
    norm = Normalize(*score_range) if score_range else None

    for i in range(n - 1):
        ax_chain.annotate(
            "", xy=(0, ys[i + 1] + 0.25), xytext=(0, ys[i] - 0.25),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#6B7280"),
        )

    for c, y in zip(chain, ys):
        score = c["score"]
        face = cmap(norm(score)) if (score is not None and norm) else "white"
        ax_chain.scatter(
            [0], [y],
            c=[face] if score is not None else "white",
            edgecolor="black", linewidth=0.7, s=320, zorder=3,
        )
        # right-side label block
        score_str = f"val_bpb = {score:.4f}" if score is not None else "val_bpb = (none)"
        gen = c["generation"]
        meta = c["meta"]
        bits = [f"gen {gen}"]
        if meta.get("island_id") is not None:
            bits.append(f"island {meta['island_id']}")
        if meta.get("operator"):
            bits.append(meta["operator"])
        if meta.get("archive_status"):
            bits.append(meta["archive_status"])
        sub = "  •  ".join(bits)
        ax_chain.text(
            0.18, y + 0.05,
            f"{c['run_id']}  (iter {c['iteration']})",
            fontsize=10, va="center", ha="left", color="#1F2937",
        )
        ax_chain.text(
            0.18, y - 0.18,
            f"{score_str}\n{sub}",
            fontsize=8, va="center", ha="left", color="#6B7280",
        )

    ax_chain.set_xlim(-0.4, 1.4)
    ax_chain.set_ylim(-0.7, n - 0.3)
    ax_chain.set_xticks([]); ax_chain.set_yticks([])
    for s in ax_chain.spines.values():
        s.set_visible(False)
    ax_chain.set_title("Lineage", loc="left", fontsize=11)

    # baseline annotation
    ax_chain.text(
        0.0, -0.6,
        "↑ time / generation\nbaseline val_bpb = 0.992",
        fontsize=8, color="#9CA3AF", ha="left", va="top",
    )

    # --- right: idea text ---
    target = chain[-1]
    idea = target.get("idea") or "(no idea text recorded)"
    wrapped = _wrap_idea(idea, width=92)

    ax_idea.text(
        0.0, 1.0, wrapped,
        family="monospace", fontsize=7.5, va="top", ha="left",
        color="#1F2937",
    )
    ax_idea.set_xticks([]); ax_idea.set_yticks([])
    for s in ax_idea.spines.values():
        s.set_visible(False)
    ax_idea.set_xlim(0, 1); ax_idea.set_ylim(0, 1)
    head_score = f"{target['score']:.4f}" if target["score"] is not None else "—"
    ax_idea.set_title(
        f"Idea — {target['run_id']} (val_bpb {head_score})",
        loc="left", fontsize=11,
    )

    if title:
        fig.suptitle(title, fontsize=12, y=0.995)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment_id")
    p.add_argument("run_id")
    p.add_argument("iteration", type=int)
    p.add_argument("--out", default=None)
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--higher-is-better", action="store_true")
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)

    chain = fetch_chain(
        args.experiment_id, args.run_id, args.iteration,
        db_path=Path(args.db),
    )
    out = Path(args.out) if args.out else (
        RESEARCH_AGENT_ROOT / "analysis" / "experiments" / args.experiment_id
        / "figures" / f"idea_card_{args.run_id}_{args.iteration}.png"
    )
    render_card(
        chain, out,
        title=args.title or f"{args.experiment_id} — {args.run_id}@{args.iteration}",
        lower_is_better=not args.higher_is_better,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
