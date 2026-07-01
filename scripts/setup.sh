#!/usr/bin/env bash
# Set up the project: preflight, auth, and task-specific data prep.
#
# Usage:
#   bash scripts/setup.sh              # basic setup (preflight + auth)
#   bash scripts/setup.sh nanogpt      # + download nanogpt training data
#
# Venvs are auto-created on first experiment run. Task venvs use task-local
# requirements files; the default sandbox venv uses the pyproject sandbox extra.
# This script handles data prep and authentication that can't be automated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENVS_DIR="$PROJECT_ROOT/venvs"
DATA_DIR="$PROJECT_ROOT/data"

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Step 1: Preflight checks
# ---------------------------------------------------------------------------

echo "=== Preflight Checks ==="
echo ""
bash "$SCRIPT_DIR/check.sh" || die "Preflight checks failed. Fix the issues above first."
echo ""

# ---------------------------------------------------------------------------
# Step 2: HuggingFace auth (needed for novelty review via hf papers)
# ---------------------------------------------------------------------------

echo "=== HuggingFace Auth ==="

HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
HF_STORED_TOKENS="$HOME/.cache/huggingface/stored_tokens"

check_hf_auth() {
    if [ -f "$HF_TOKEN_PATH" ]; then
        echo "  [OK] HF token found at $HF_TOKEN_PATH"
        return 0
    fi
    if [ -d "$HF_STORED_TOKENS" ] && [ "$(ls -A "$HF_STORED_TOKENS" 2>/dev/null)" ]; then
        echo "  [OK] HF stored tokens found at $HF_STORED_TOKENS"
        return 0
    fi
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "  [OK] HF_TOKEN env var is set"
        return 0
    fi
    return 1
}

if check_hf_auth; then
    echo "  HuggingFace authentication is configured."
else
    echo "  HuggingFace not authenticated."
    echo ""
    echo "  The novelty reviewer uses 'hf papers' to search for prior work."
    echo "  To authenticate, run:"
    echo ""
    echo "    hf auth login"
    echo ""
    echo "  [WARN] Skipping HF auth -- novelty review will not be available."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Ensure base venv exists
# ---------------------------------------------------------------------------

echo "=== Base Venv ==="
BASE_VENV="$VENVS_DIR/base"
create_base_venv() {
    echo "  Creating base venv from pyproject sandbox extra..."
    rm -rf "$BASE_VENV"
    mkdir -p "$BASE_VENV"
    uv venv "$BASE_VENV" --python python3.12
    uv pip install "$PROJECT_ROOT[sandbox]" --python "$BASE_VENV/bin/python"
    echo "  [OK] Base venv created: $BASE_VENV"
}
if [ -f "$BASE_VENV/bin/python" ]; then
    BASE_PY_VERSION=$("$BASE_VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    case "$BASE_PY_VERSION" in
        3.12|3.13)
            echo "  [OK] Base venv exists: $BASE_VENV (Python $BASE_PY_VERSION)"
            ;;
        *)
            echo "  [WARN] Base venv uses Python $BASE_PY_VERSION; rebuilding with Python 3.12"
            create_base_venv
            ;;
    esac
else
    create_base_venv
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Task-specific setup (data prep)
# ---------------------------------------------------------------------------

TASK_PATH="${1:-}"

if [ -z "$TASK_PATH" ]; then
    echo "=== Setup Complete (basic) ==="
    echo ""
    echo "Venvs are auto-created on first experiment run."
    echo "To set up task-specific data:"
    echo "  bash scripts/setup.sh nanogpt"
    exit 0
fi

TASK_TYPE="${TASK_PATH%%/*}"
SUBTASK="${TASK_PATH#*/}"
[ "$SUBTASK" = "$TASK_TYPE" ] && SUBTASK=""

case "$TASK_TYPE" in
    nanogpt)
        echo "=== NanoGPT Data Prep ==="
        CACHE_DIR="$HOME/.cache/autoresearch"

        # Need the nanogpt venv for prepare.py and for first experiment runs.
        # Create it even when data is already cached.
        NANOGPT_VENV="$VENVS_DIR/nanogpt"
        create_nanogpt_venv() {
            echo "  Creating nanogpt venv from requirements.txt..."
            rm -rf "$NANOGPT_VENV"
            mkdir -p "$NANOGPT_VENV"
            uv venv "$NANOGPT_VENV" --python python3.12
            uv pip install -r "$PROJECT_ROOT/src/heuresis/tasks/nanogpt/requirements.txt" --python "$NANOGPT_VENV/bin/python"
            echo "  [OK] Nanogpt venv created"
        }
        if [ ! -f "$NANOGPT_VENV/bin/python" ]; then
            create_nanogpt_venv
        else
            PY_VERSION=$("$NANOGPT_VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            case "$PY_VERSION" in
                3.12|3.13)
                    echo "  [OK] Nanogpt venv exists: $NANOGPT_VENV (Python $PY_VERSION)"
                    ;;
                *)
                    echo "  [WARN] Nanogpt venv uses Python $PY_VERSION; rebuilding with Python 3.12"
                    create_nanogpt_venv
                    ;;
            esac
        fi

        # Check if data already exists
        if [ -d "$CACHE_DIR/data" ] && [ -d "$CACHE_DIR/tokenizer" ]; then
            SHARD_COUNT=$(ls "$CACHE_DIR/data"/*.parquet 2>/dev/null | wc -l)
            if [ "$SHARD_COUNT" -gt 0 ]; then
                echo "  [OK] Data already downloaded: $SHARD_COUNT shards at $CACHE_DIR/data/"
                echo "  [OK] Tokenizer at $CACHE_DIR/tokenizer/"
                echo ""
                echo "  To re-download, remove $CACHE_DIR and re-run."
                exit 0
            fi
        fi

        PREPARE_PY="$PROJECT_ROOT/src/heuresis/tasks/nanogpt/prepare.py"
        NUM_SHARDS="${NUM_SHARDS:-10}"
        echo "  Downloading $NUM_SHARDS training shards + tokenizer..."
        echo "  (This may take a few minutes on first run)"
        echo ""
        "$NANOGPT_VENV/bin/python" "$PREPARE_PY" --num-shards "$NUM_SHARDS"
        echo ""
        echo "  [OK] NanoGPT data ready at $CACHE_DIR"
        ;;

    *)
        die "Unknown task: $TASK_TYPE (expected: nanogpt)"
        ;;
esac

echo ""
echo "=== Setup Complete ==="
