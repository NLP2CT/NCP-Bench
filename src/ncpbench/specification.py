"""Generate one benchmark story specification from one curated synopsis."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal, Protocol, cast

from ncpbench.dataset import StorySpec
from ncpbench.model_output import (
    ModelCallError,
    ModelOutputError,
    call_with_retries,
    parse_json_object,
)
from ncpbench.models import Commitment, Fact, TrajectoryNode


ChatMessage = Mapping[str, str]


class SpecificationClient(Protocol):
    """Injected model client for the three published extraction stages."""

    def complete(self, messages: Sequence[ChatMessage], *, stage: str) -> str:
        """Return one raw model response."""


class SpecificationGenerationError(RuntimeError):
    """Raised when a generation stage does not return valid JSON."""


@dataclass(frozen=True)
class SpecificationSource:
    """Curated metadata and source text for one benchmark story."""

    id: str
    title: str
    genres: tuple[str, ...]
    player_role: str
    synopsis: str


@dataclass(frozen=True)
class SpecificationPrompts:
    trajectory: str
    commitments: str
    initial_facts: str


def load_specification_prompts() -> SpecificationPrompts:
    """Load the fixed prompts used by the construction pipeline."""

    prompt_dir = files("ncpbench.prompts")
    return SpecificationPrompts(
        trajectory=prompt_dir.joinpath("specification_trajectory.txt").read_text(
            encoding="utf-8"
        ),
        commitments=prompt_dir.joinpath("specification_commitments.txt").read_text(
            encoding="utf-8"
        ),
        initial_facts=prompt_dir.joinpath("specification_initial_facts.txt").read_text(
            encoding="utf-8"
        ),
    )


class SpecificationGenerator:
    """Run the paper's trajectory, commitment, and initial-fact stages."""

    def __init__(
        self,
        client: SpecificationClient,
        prompts: SpecificationPrompts | None = None,
    ) -> None:
        self._client = client
        self._prompts = prompts or load_specification_prompts()

    def generate(self, source: SpecificationSource) -> StorySpec:
        trajectory_raw = self._request_object_list(
            self._prompts.trajectory.format(
                synopsis=source.synopsis,
                player_role=source.player_role,
            ),
            stage="trajectory_extraction",
            field="trajectory",
        )

        commitments_raw = self._request_object_list(
            self._prompts.commitments.format(
                synopsis=source.synopsis,
                player_role=source.player_role,
                reference_trajectory=_pretty_json(trajectory_raw),
            ),
            stage="commitment_extraction",
            field="commitments",
        )

        facts_raw = self._request_object_list(
            self._prompts.initial_facts.format(
                synopsis=source.synopsis,
                player_role=source.player_role,
                reference_trajectory=_pretty_json(trajectory_raw),
                commitments=_pretty_json(commitments_raw),
            ),
            stage="initial_facts_extraction",
            field="facts",
        )

        return StorySpec(
            id=source.id,
            title=source.title,
            genres=source.genres,
            player_role=source.player_role,
            synopsis=source.synopsis,
            initial_facts=_build_facts(facts_raw),
            commitments=_build_commitments(commitments_raw),
            trajectory=_build_trajectory(trajectory_raw),
        )

    def _request_object_list(
        self, prompt: str, *, stage: str, field: str
    ) -> list[Mapping[str, object]]:
        messages = ({"role": "user", "content": prompt},)
        try:
            return call_with_retries(
                lambda: self._client.complete(messages, stage=stage),
                lambda raw: _parse_object_list(raw, field),
                stage=stage,
            )
        except ModelCallError as exc:
            raise SpecificationGenerationError(str(exc)) from exc


def _parse_object_list(raw: str, field: str) -> list[Mapping[str, object]]:
    return _required_object_list(parse_json_object(raw), field)


def _required_object_list(
    payload: Mapping[str, object], field: str
) -> list[Mapping[str, object]]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise ModelOutputError(f"Field {field!r} must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ModelOutputError(f"{field}[{index}] must be an object")
    return cast(list[Mapping[str, object]], value)


def _build_trajectory(
    raw_nodes: Sequence[Mapping[str, object]],
) -> tuple[TrajectoryNode, ...]:
    result: list[TrajectoryNode] = []
    seen_ids: set[str] = set()
    for index, node in enumerate(raw_nodes):
        node_id = _text(node.get("id") or f"s_{index}")
        description = _text(node.get("description"))
        trigger_event = _text(node.get("trigger_event"))
        key_delta = _text(node.get("key_delta"))
        if not description and not trigger_event and not key_delta:
            continue
        normalized = _normalize_identifier(node_id, default_prefix="") or f"s_{index}"
        result.append(
            TrajectoryNode(
                id=_unique_identifier(normalized, seen_ids),
                description=description,
                trigger_event=trigger_event,
                key_delta=key_delta,
            )
        )
    return tuple(result)


def _build_commitments(raw_items: Sequence[Mapping[str, object]]) -> tuple[Commitment, ...]:
    result: list[Commitment] = []
    seen_ids: set[str] = set()
    valid_kinds = {"invariant", "achievement", "ordering"}
    for item in raw_items:
        description = _text(item.get("description"))
        satisfaction = _text(item.get("satisfaction_condition"))
        violation = _text(item.get("violation_condition"))
        if not description or not satisfaction or not violation:
            continue
        normalized = _normalize_identifier(
            _text(item.get("id")) or description,
            default_prefix="c_",
        )
        if not normalized:
            continue
        raw_kind = _text(item.get("type")).lower()
        kind = raw_kind if raw_kind in valid_kinds else "invariant"
        result.append(
            Commitment(
                id=_unique_identifier(normalized, seen_ids),
                kind=cast(Literal["invariant", "achievement", "ordering"], kind),
                description=description,
                satisfaction_condition=satisfaction,
                violation_condition=violation,
            )
        )
    return tuple(result)


def _build_facts(raw_facts: Sequence[Mapping[str, object]]) -> tuple[Fact, ...]:
    result: list[Fact] = []
    seen_ids: set[str] = set()
    for index, fact in enumerate(raw_facts):
        content = _text(fact.get("content"))
        if not content:
            continue
        normalized = _normalize_identifier(
            _text(fact.get("id")) or f"f_{index}",
            default_prefix="f_",
        )
        result.append(
            Fact(_unique_identifier(normalized or f"f_{index}", seen_ids), content)
        )
    return tuple(result)


def _text(value: object) -> str:
    return cast(str, value or "").strip()


def _normalize_identifier(raw_id: str, *, default_prefix: str) -> str:
    candidate = re.sub(r"[^\w\s]", "", raw_id).lower().replace(" ", "_") or "unknown"
    candidate = re.sub(r"[^a-z0-9_]", "", candidate)
    if not candidate:
        return ""
    if default_prefix and not candidate.startswith(default_prefix):
        candidate = f"{default_prefix}{candidate}"
    return candidate


def _unique_identifier(candidate: str, seen: set[str]) -> str:
    if candidate not in seen:
        seen.add(candidate)
        return candidate
    counter = 2
    while f"{candidate}_{counter}" in seen:
        counter += 1
    unique = f"{candidate}_{counter}"
    seen.add(unique)
    return unique


def _pretty_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


__all__ = [
    "SpecificationClient",
    "SpecificationGenerationError",
    "SpecificationGenerator",
    "SpecificationPrompts",
    "SpecificationSource",
    "load_specification_prompts",
]
