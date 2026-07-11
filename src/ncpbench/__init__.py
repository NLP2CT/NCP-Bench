"""NCP-Bench dataset construction and evaluation framework."""

from ncpbench.construction import (
    ConstructionFormatError,
    build_dataset,
    load_specification_sources,
)
from ncpbench.narrator import Narrator, NarratorRequest, NarratorResponse, OpeningRequest
from ncpbench.dataset import (
    Dataset,
    DatasetFormatError,
    StorySpec,
    load_dataset,
    load_spec,
    save_spec,
)
from ncpbench.input_conditions import (
    AdversarialInputCondition,
    NaturalInputCondition,
    PlayerInputCondition,
    create_input_condition,
)
from ncpbench.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    MetricInputError,
    aggregate_metrics,
    compute_episode_metrics,
    survival_curve,
)
from ncpbench.opening import OpeningEvaluationError, OpeningEvaluator
from ncpbench.results import EpisodeResult, episode_trace_to_result
from ncpbench.specification import (
    SpecificationGenerationError,
    SpecificationGenerator,
    SpecificationSource,
)
from ncpbench.runner import (
    EpisodeCheckpoint,
    EpisodeRunner,
    EpisodeSession,
    EpisodeState,
    EpisodeTermination,
    EpisodeTransition,
    EpisodeTrace,
    EpisodeTurnResult,
    initialize_episode_state,
)

__all__ = [
    "Dataset",
    "DatasetFormatError",
    "ConstructionFormatError",
    "AggregateMetrics",
    "EpisodeMetrics",
    "EpisodeResult",
    "EpisodeCheckpoint",
    "EpisodeRunner",
    "EpisodeSession",
    "EpisodeState",
    "EpisodeTermination",
    "EpisodeTransition",
    "EpisodeTrace",
    "EpisodeTurnResult",
    "Narrator",
    "NarratorRequest",
    "NarratorResponse",
    "OpeningRequest",
    "MetricInputError",
    "NaturalInputCondition",
    "AdversarialInputCondition",
    "OpeningEvaluator",
    "OpeningEvaluationError",
    "PlayerInputCondition",
    "StorySpec",
    "SpecificationGenerationError",
    "SpecificationGenerator",
    "SpecificationSource",
    "create_input_condition",
    "aggregate_metrics",
    "build_dataset",
    "compute_episode_metrics",
    "episode_trace_to_result",
    "initialize_episode_state",
    "load_dataset",
    "load_spec",
    "load_specification_sources",
    "save_spec",
    "survival_curve",
]
