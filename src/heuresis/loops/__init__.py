from heuresis.loops.curiosity import run_curiosity
from heuresis.loops.curiosity_plus import run_curiosity_plus
from heuresis.loops.go_explore import run_go_explore
from heuresis.loops.islands import run_islands
from heuresis.loops.linear import run_linear
from heuresis.loops.map_elites import run_map_elites
from heuresis.loops.omni_epic import run_omni_epic
from heuresis.tasks.adapter import TaskAdapter, get_task_adapter

# Strategy name -> loop entry point. The CLI (`heuresis <task> <strategy>`)
# dispatches through this, so a task x strategy pairing needs no per-pairing
# script — only an entry here and a launch config under configs/experiments/.
LOOPS = {
    "linear": run_linear,
    "islands": run_islands,
    "map_elites": run_map_elites,
    "go_explore": run_go_explore,
    "omni_epic": run_omni_epic,
    "curiosity": run_curiosity,
    "curiosity_plus": run_curiosity_plus,
}

__all__ = [
    "TaskAdapter", "LOOPS", "run_linear", "run_islands", "run_map_elites",
    "run_go_explore", "run_omni_epic", "run_curiosity", "run_curiosity_plus",
    "get_task_adapter",
]
