"""Pre-built Tool instances for common workspace tools."""

from pathlib import Path

from heuresis.tool import Tool

_TOOLS_DIR = Path(__file__).parent

GRADE = Tool(
    name="grade",
    binary=_TOOLS_DIR / "grade.py",
    docs=(
        "Run `grade <file>` to submit your solution for scoring. "
        "Defaults to `submission.csv` if no file is given. "
        "Prints JSON: {score, valid, details}."
    ),
)

MEMORY = Tool(
    name="memory",
    binary=_TOOLS_DIR / "memory.py",
    docs=(
        "Shared campaign memory. `memory` is a preinstalled shell command "
        "(on PATH, like `ls` or `cat`). Invoke it from bash — do NOT "
        "write Python wrappers and do NOT try to open sockets yourself.\n"
        "  memory search <query> [--table experiments|learnings] [--k 5]\n"
        "  memory read <sql>   # SELECT/WITH over memory_experiments_v, memory_learnings_v\n"
        "  memory append <content> [--tags t1,t2] [--related id1,id2]\n"
        "Each prints one JSON line. Example:\n"
        "  memory search \"CMA-ES restarts\" --table experiments --k 3"
    ),
    system_install=True,
)
