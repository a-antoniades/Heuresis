"""Heuresis: composable primitives for sandboxed agent experiments."""

from heuresis.env import load_environment
from heuresis.tool import Tool
from heuresis.workspace import Mount, Workspace
from heuresis.harness import Harness, RunFuture
from heuresis.models import RunResult, RunRecord
from heuresis.store import ResultStore
from heuresis.grading import GradingServer
from heuresis.judge import HackerJudge, HackerVerdict
from heuresis.novelty import NoveltyAssessment, NoveltyReviewer
from heuresis.parsing import parse_workspace
from heuresis.experiment import (
    ExperimentLoop,
    ExperimentState,
    Settings,
    build_harnesses,
    execute,
    executor_files,
    ideate,
    iterate_until_valid,
    judge_and_maybe_regrade,
    next_run_index,
    novelty_retry_loop,
    parallel_ideators,
    record_run,
    regenerate,
    reserve_gpus,
    resume_or_new,
)
from heuresis.experiment_cli import (
    BBOBTaskConfig,
    CellConfig,
    CuriosityConfig,
    CuriosityPlusConfig,
    DiscoGenTaskConfig,
    EmptyConfig,
    ExperimentDefinition,
    IslandConfig,
    NanoGPTTaskConfig,
    OmniEpicConfig,
    ParsedExperiment,
    build_parser,
    make_definition,
    parse_experiment,
)

load_environment()

__all__ = [
    "GradingServer",
    "HackerJudge",
    "HackerVerdict",
    "Harness",
    "Mount",
    "NoveltyAssessment",
    "NoveltyReviewer",
    "ResultStore",
    "RunFuture",
    "RunRecord",
    "RunResult",
    "Tool",
    "Workspace",
    "load_environment",
    "parse_workspace",
    # experiment helpers
    "ExperimentLoop",
    "ExperimentState",
    "Settings",
    "BBOBTaskConfig",
    "CellConfig",
    "CuriosityConfig",
    "CuriosityPlusConfig",
    "DiscoGenTaskConfig",
    "EmptyConfig",
    "ExperimentDefinition",
    "IslandConfig",
    "NanoGPTTaskConfig",
    "OmniEpicConfig",
    "ParsedExperiment",
    "build_harnesses",
    "build_parser",
    "execute",
    "executor_files",
    "ideate",
    "iterate_until_valid",
    "judge_and_maybe_regrade",
    "next_run_index",
    "novelty_retry_loop",
    "parallel_ideators",
    "make_definition",
    "parse_experiment",
    "record_run",
    "regenerate",
    "reserve_gpus",
    "resume_or_new",
]
