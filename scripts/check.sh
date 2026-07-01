#!/usr/bin/env bash
# Full preflight check for all project prerequisites.
#
# Usage:
#   bash scripts/check.sh              # check everything
#   bash scripts/check.sh --gpus 0,1   # also check specific GPU devices
#
# Checks:
#   - Python >= 3.11
#   - uv (package manager)
#   - bwrap (bubblewrap sandbox)
#   - taskset (CPU pinning, from util-linux)
#   - cp -al (hardlink support)
#   - coding agents (opencode, claude, agent, or codex -- at least one required)
#   - GPU devices (if --gpus specified)
#   - Filesystem: venvs/ and runs/ on same mount

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PASS=0
FAIL=0
WARN=0

pass() { echo "  [OK]   $*"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }
warn() { echo "  [WARN] $*"; WARN=$((WARN + 1)); }

GPU_IDS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPU_IDS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Quality Diversity Project - Preflight Check ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# --- Python ---
echo "--- Python ---"
if command -v python3.12 &>/dev/null; then
    pass "python3.12 found: $(python3.12 --version 2>&1)"
else
    PY_VER=$(python3 --version 2>&1 || echo "not found")
    if echo "$PY_VER" | grep -qE "3\.(1[2-9]|[2-9][0-9])"; then
        pass "python3 found: $PY_VER"
    else
        fail "Python >= 3.12 required, found: $PY_VER"
    fi
fi

# --- uv ---
echo ""
echo "--- Package Manager ---"
if command -v uv &>/dev/null; then
    pass "uv found: $(uv --version 2>&1)"
else
    fail "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# --- Sandbox binaries ---
echo ""
echo "--- Sandbox Binaries ---"
if command -v bwrap &>/dev/null; then
    pass "bwrap found: $(which bwrap)"
    # Quick functional test using the same mount pattern as the real sandbox
    BWRAP_TEST=$(bwrap \
        --ro-bind /usr /usr \
        --ro-bind /etc /etc \
        --symlink usr/bin /bin \
        --symlink usr/sbin /sbin \
        --symlink usr/lib /lib \
        --symlink usr/lib64 /lib64 \
        --proc /proc \
        --dev /dev \
        --tmpfs /tmp \
        --unshare-pid \
        --die-with-parent \
        -- echo bwrap_ok 2>&1) || true
    if [ "$BWRAP_TEST" = "bwrap_ok" ]; then
        pass "bwrap functional test passed"
    else
        fail "bwrap exists but failed to run: $BWRAP_TEST"
    fi
else
    fail "bwrap (bubblewrap) not found. Install: dnf install bubblewrap / apt install bubblewrap"
fi

if command -v taskset &>/dev/null; then
    pass "taskset found: $(which taskset)"
else
    fail "taskset not found (part of util-linux)"
fi

# --- Hardlink support ---
echo ""
echo "--- Hardlink Support ---"
TEST_DIR="$PROJECT_ROOT/.preflight_test_$$"
mkdir -p "$TEST_DIR"
echo "test" > "$TEST_DIR/src"
if cp -al "$TEST_DIR/src" "$TEST_DIR/dst" 2>/dev/null; then
    SRC_INO=$(stat -c %i "$TEST_DIR/src" 2>/dev/null || stat -f %i "$TEST_DIR/src")
    DST_INO=$(stat -c %i "$TEST_DIR/dst" 2>/dev/null || stat -f %i "$TEST_DIR/dst")
    if [ "$SRC_INO" = "$DST_INO" ]; then
        pass "cp -al creates real hardlinks (same inode)"
    else
        fail "cp -al copied instead of hardlinking (different inodes)"
    fi
else
    fail "cp -al not supported on this filesystem"
fi
rm -rf "$TEST_DIR"

# --- Filesystem check ---
echo ""
echo "--- Filesystem ---"
VENVS_DIR="$PROJECT_ROOT/venvs"
RUNS_DIR="$PROJECT_ROOT/runs"
mkdir -p "$VENVS_DIR" "$RUNS_DIR"
VENVS_DEV=$(stat -c %d "$VENVS_DIR" 2>/dev/null || stat -f %d "$VENVS_DIR")
RUNS_DEV=$(stat -c %d "$RUNS_DIR" 2>/dev/null || stat -f %d "$RUNS_DIR")
if [ "$VENVS_DEV" = "$RUNS_DEV" ]; then
    pass "venvs/ and runs/ on same filesystem (hardlinks will work)"
else
    fail "venvs/ and runs/ on different filesystems! Hardlinks won't work."
fi

# --- Coding Agents ---
echo ""
echo "--- Coding Agents ---"
AGENT_FOUND=0
if command -v opencode &>/dev/null; then
    pass "opencode found: $(which opencode)"
    AGENT_FOUND=1
fi
if command -v claude &>/dev/null; then
    pass "claude (Claude Code) found: $(which claude)"
    AGENT_FOUND=1
fi
if command -v agent &>/dev/null; then
    pass "agent (Cursor Agent) found: $(which agent)"
    AGENT_FOUND=1
fi
if command -v codex &>/dev/null; then
    pass "codex (OpenAI Codex CLI) found: $(which codex)"
    AGENT_FOUND=1
fi
if [ "$AGENT_FOUND" -eq 0 ]; then
    fail "No coding agent found. Install at least one of:"
    echo "         opencode  -> curl -fsSL https://opencode.ai/install | bash"
    echo "         claude    -> https://docs.anthropic.com/en/docs/claude-code"
    echo "         agent     -> https://cursor.com/docs/cli/overview"
    echo "         codex     -> curl -fsSL https://github.com/openai/codex/releases/latest/download/install.sh | bash"
fi

# --- Memory primitives (sqlite-vec + embeddings) ---
echo ""
echo "--- Memory Primitives ---"
PY_BIN="$(command -v python3.12 || command -v python3.11 || command -v python3)"
if [ -n "$PY_BIN" ]; then
    if command -v uv &>/dev/null; then
        # sqlite-vec loads as a SQLite extension via connection.enable_load_extension().
        # Probe the uv-managed project environment, not host Python, because uv sync
        # installs sqlite-vec into .venv.
        if uv run python -c "import sqlite3, sys; c=sqlite3.connect(':memory:'); c.enable_load_extension(True); sys.exit(0)" 2>/dev/null; then
            pass "sqlite3 supports enable_load_extension in uv environment (sqlite-vec will load)"
        else
            warn "uv environment sqlite3 was built without enable_load_extension; MemoryStore will fail on init"
        fi
        if uv run python -c "import sqlite_vec" 2>/dev/null; then
            pass "sqlite-vec Python package importable in uv environment"
        else
            warn "sqlite-vec not installed in uv environment (run: uv sync)"
        fi
    else
        warn "uv not found; skipping uv environment memory checks"
    fi
fi
if [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${GOOGLE_GENERATIVE_AI_API_KEY:-}" ] || [ -n "${GEMINI_API_KEYS:-}" ]; then
    pass "Gemini API key set (embeddings enabled)"
else
    warn "Gemini API key not set — set GEMINI_API_KEYS, GEMINI_API_KEY, or GOOGLE_GENERATIVE_AI_API_KEY (see .env.example)"
fi

# --- HuggingFace Auth ---
echo ""
echo "--- HuggingFace Auth ---"
HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
HF_STORED_TOKENS="$HOME/.cache/huggingface/stored_tokens"
if [ -f "$HF_TOKEN_PATH" ]; then
    pass "HF token found at $HF_TOKEN_PATH"
elif [ -d "$HF_STORED_TOKENS" ] && [ "$(ls -A "$HF_STORED_TOKENS" 2>/dev/null)" ]; then
    pass "HF stored tokens found"
elif [ -n "${HF_TOKEN:-}" ]; then
    pass "HF_TOKEN env var is set"
else
    warn "No HuggingFace auth found (needed for novelty review). Run: hf auth login"
fi

# --- GPU devices ---
echo ""
echo "--- GPU Devices ---"
GPU_COUNT=$(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -gt 0 ]; then
    pass "$GPU_COUNT GPU(s) detected: $(ls /dev/nvidia[0-9]* 2>/dev/null | tr '\n' ' ')"
    if [ -n "$GPU_IDS" ]; then
        IFS=',' read -ra GPUS <<< "$GPU_IDS"
        for GID in "${GPUS[@]}"; do
            for DEV in "/dev/nvidia$GID" "/dev/dri/card$GID" "/dev/dri/renderD$((128 + GID))"; do
                if [ -e "$DEV" ]; then
                    pass "$DEV exists"
                else
                    warn "$DEV missing"
                fi
            done
        done
        for DEV in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
            if [ -e "$DEV" ]; then
                pass "$DEV exists"
            else
                warn "$DEV missing"
            fi
        done
    fi
else
    warn "No NVIDIA GPUs detected (not required for CPU-only tasks)"
fi

# --- Existing venvs ---
echo ""
echo "--- Task Venvs ---"
VENVS_DIR="$PROJECT_ROOT/venvs"
if [ -d "$VENVS_DIR" ] && ls -1d "$VENVS_DIR"/*/ &>/dev/null 2>&1; then
    for VDIR in "$VENVS_DIR"/*/; do
        VNAME=$(basename "$VDIR")
        if [ -x "$VDIR/bin/python" ]; then
            PY_VERSION=$("$VDIR/bin/python" --version 2>&1)
            pass "venvs/$VNAME ready ($PY_VERSION)"
        elif ls -1d "$VDIR"*/bin/python &>/dev/null 2>&1; then
            for PYTHON_BIN in "$VDIR"*/bin/python; do
                SUBDIR=$(basename "$(dirname "$(dirname "$PYTHON_BIN")")")
                PY_VERSION=$("$PYTHON_BIN" --version 2>&1)
                pass "venvs/$VNAME/$SUBDIR ready ($PY_VERSION)"
            done
        elif [ -f "$VDIR/requirements.txt" ]; then
            warn "venvs/$VNAME has requirements.txt but is not built yet"
        else
            fail "venvs/$VNAME exists but has no bin/python"
        fi
    done
else
    echo "  (none yet — venvs auto-created on first run)"
fi

# --- Summary ---
echo ""
echo "==========================================="
echo "  PASS: $PASS  |  FAIL: $FAIL  |  WARN: $WARN"
echo "==========================================="
if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Fix the FAIL items above before running experiments."
    exit 1
else
    echo ""
    echo "All critical checks passed. Ready to go."
    exit 0
fi
