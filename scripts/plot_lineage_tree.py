#!/usr/bin/env python3
"""Plot a hereditary/evolution tree for a heuresis search run.

Examples:
  python scripts/plot_lineage_tree.py \
      --run runs/nanogpt/2026-04-28_145017_nanogpt-islands-final

  python scripts/plot_lineage_tree.py \
      --exp 2026-04-30_171603_nanogpt-map-elites-final-v2-r1 \
      --store-db runs/nanogpt/store.db --group-by none
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.utils.lineage_tree import (  # noqa: E402
    build_lineage_graph,
    load_lineage_nodes,
    plot_lineage_tree,
)

CANDIDATE_STORES = [
    REPO_ROOT / "runs/nanogpt/store.db",
    REPO_ROOT / "store.db",
    REPO_ROOT / "runs/_legacy/store.db",
]


def main() -> int:
    args = parse_args()
    run_dir = args.run.resolve() if args.run else None
    store_db = args.store_db or autodetect_store(run_dir, args.exp)
    exp_id = args.exp or resolve_experiment_id(store_db, run_dir)
    output = args.output or default_output_path(run_dir, exp_id)
    title = args.title or experiment_title(store_db, exp_id)

    nodes = load_lineage_nodes(
        store_db,
        exp_id,
        scored_only=args.scored_only,
    )
    if not nodes:
        raise SystemExit(f"No executor runs found for experiment {exp_id} in {store_db}")

    graph = build_lineage_graph(nodes)
    out = plot_lineage_tree(
        graph,
        output,
        title=title,
        group_by=args.group_by,
        lower_is_better=not args.higher_is_better,
        max_labels_per_group=args.max_labels,
    )
    print(
        f"wrote {out} "
        f"({len(graph.nodes)} nodes, {len(graph.edges)} edges, exp={exp_id})"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a generation-layered hereditary tree for an experiment."
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="Run directory, e.g. runs/nanogpt/<experiment_id>.",
    )
    parser.add_argument(
        "--exp",
        help="Experiment id. If omitted, inferred from --run and the store.",
    )
    parser.add_argument(
        "--store-db",
        type=Path,
        help="Path to store.db. Defaults to the first matching known store.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path. Defaults to <run>/figures/lineage_tree.png.",
    )
    parser.add_argument(
        "--title",
        help="Figure title. Defaults to '<experiment name> - Lineage Tree'.",
    )
    parser.add_argument(
        "--group-by",
        choices=["auto", "none", "island", "cell", "archive_status", "operator"],
        default="auto",
        type=str,
        help="Panel grouping. auto uses islands when island_id metadata exists.",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Use when larger scores are better. Default assumes lower is better.",
    )
    parser.add_argument(
        "--scored-only",
        action="store_true",
        help="Plot only rows with a non-null score.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=3,
        help="Number of best nodes to label per panel.",
    )
    args = parser.parse_args()
    if not args.run and not args.exp:
        parser.error("provide --run or --exp")
    return args


def autodetect_store(run_dir: Path | None, experiment_id: str | None) -> Path:
    for path in CANDIDATE_STORES:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            with sqlite3.connect(str(path)) as conn:
                if not _has_experiments_table(conn):
                    continue
                if experiment_id and _experiment_exists(conn, experiment_id):
                    return path
                if run_dir and _experiment_for_run(conn, run_dir):
                    return path
        except sqlite3.DatabaseError:
            continue
    tried = ", ".join(str(p) for p in CANDIDATE_STORES)
    raise SystemExit(f"Could not autodetect store.db. Tried: {tried}")


def resolve_experiment_id(store_db: Path, run_dir: Path | None) -> str:
    if run_dir is None:
        raise SystemExit("Pass --exp when --run is not provided.")
    with sqlite3.connect(str(store_db)) as conn:
        row = _experiment_for_run(conn, run_dir)
        if row:
            return row
        basename = run_dir.name
        if _experiment_exists(conn, basename):
            return basename
    raise SystemExit(f"Could not resolve experiment id for {run_dir}; pass --exp.")


def default_output_path(run_dir: Path | None, exp_id: str) -> Path:
    if run_dir is not None:
        return run_dir / "figures" / "lineage_tree.png"
    return REPO_ROOT / "analysis" / "experiments" / exp_id / "figures" / "lineage_tree.png"


def experiment_title(store_db: Path, exp_id: str) -> str:
    with sqlite3.connect(str(store_db)) as conn:
        row = conn.execute(
            "SELECT name, task FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
    if row and row[0]:
        task = f" ({row[1]})" if row[1] else ""
        return f"{row[0]}{task} - Lineage Tree"
    return f"{exp_id} - Lineage Tree"


def _has_experiments_table(conn: sqlite3.Connection) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='experiments'"
        ).fetchone()
    )


def _experiment_exists(conn: sqlite3.Connection, experiment_id: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ? LIMIT 1",
            (experiment_id,),
        ).fetchone()
    )


def _experiment_for_run(conn: sqlite3.Connection, run_dir: Path) -> str | None:
    run_dir_abs = str(run_dir.resolve())
    row = conn.execute(
        "SELECT experiment_id FROM experiments WHERE dir = ? OR experiment_id = ? LIMIT 1",
        (run_dir_abs, run_dir.name),
    ).fetchone()
    return row[0] if row else None


if __name__ == "__main__":
    raise SystemExit(main())
