"""Read NCP-Bench data without coupling it to the package wheel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence, cast

import yaml

from ncpbench.models import Commitment, Fact, TrajectoryNode


class DatasetFormatError(ValueError):
    """Raised when a benchmark dataset does not satisfy the published schema."""


@dataclass(frozen=True)
class StorySpec:
    """One immutable benchmark story specification ready for episode execution."""

    id: str
    title: str
    genres: tuple[str, ...]
    player_role: str
    synopsis: str
    initial_facts: tuple[Fact, ...]
    commitments: tuple[Commitment, ...]
    trajectory: tuple[TrajectoryNode, ...]


@dataclass(frozen=True)
class Dataset:
    """The dataset root and its ordered, immutable specification IDs."""

    root: Path
    spec_ids: tuple[str, ...]

    def load_spec(self, spec_id: str) -> StorySpec:
        if spec_id not in self.spec_ids:
            raise KeyError(f"Unknown benchmark spec {spec_id!r}")
        return _load_story_spec(self.root, spec_id)


def load_dataset(root: str | Path) -> Dataset:
    """Load a dataset index from ``root/index.yaml`` without reading every spec."""

    dataset_root = Path(root).expanduser().resolve()
    index_path = dataset_root / "index.yaml"
    if not index_path.is_file():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")

    raw_index = _yaml_mapping(index_path)
    raw_entries = _required_list(raw_index, "specs", index_path)
    spec_ids: list[str] = []
    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, str) or not raw_entry.strip():
            raise DatasetFormatError(
                f"specs[{index}] must be a non-empty string in {index_path}"
            )
        spec_id = raw_entry.strip()
        if spec_id in seen_ids:
            raise DatasetFormatError(f"Duplicate spec id {spec_id!r} in {index_path}")
        if spec_id in {".", ".."} or Path(spec_id).name != spec_id:
            raise DatasetFormatError(f"Invalid spec id {spec_id!r} in {index_path}")
        seen_ids.add(spec_id)
        spec_ids.append(spec_id)

    return Dataset(root=dataset_root, spec_ids=tuple(spec_ids))


def load_spec(root: str | Path, spec_id: str) -> StorySpec:
    """Load one specification by its declared ID."""

    return load_dataset(root).load_spec(spec_id)


def save_spec(spec: StorySpec, path: str | Path) -> None:
    """Write one story specification in the published YAML schema."""

    output_path = Path(path).expanduser()
    if output_path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Spec path must point to YAML: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            _story_spec_mapping(spec),
            stream,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def _story_spec_mapping(spec: StorySpec) -> dict[str, object]:
    return {
        "meta": {
            "id": spec.id,
            "title": spec.title,
            "genres": list(spec.genres),
            "player_role": spec.player_role,
        },
        "synopsis": spec.synopsis,
        "initial_facts": [
            {"id": fact.id, "content": fact.text, "negated": not fact.active}
            for fact in spec.initial_facts
        ],
        "commitments": [
            {
                "id": commitment.id,
                "type": commitment.kind,
                "description": commitment.description,
                "satisfaction_condition": commitment.satisfaction_condition,
                "violation_condition": commitment.violation_condition,
            }
            for commitment in spec.commitments
        ],
        "trajectory": [
            {
                "id": node.id,
                "description": node.description,
                "trigger_event": node.trigger_event,
                "key_delta": node.key_delta,
            }
            for node in spec.trajectory
        ],
    }


def _load_story_spec(dataset_root: Path, spec_id: str) -> StorySpec:
    relative_path = Path("specs") / f"{spec_id}.yaml"
    path = (dataset_root / relative_path).resolve()
    if not path.is_relative_to(dataset_root):
        raise DatasetFormatError(f"Spec path escapes dataset root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Spec file not found: {path}")

    raw = _yaml_mapping(path)
    meta = _required_mapping(raw.get("meta"), "meta", path)
    declared_id = _required_string(meta, "id", path)
    if declared_id != spec_id:
        raise DatasetFormatError(
            f"Index id {spec_id!r} does not match meta.id in {path}"
        )

    return StorySpec(
        id=declared_id,
        title=_required_string(meta, "title", path),
        genres=_string_tuple(_required_list(meta, "genres", path), "meta.genres", path),
        player_role=_required_string(meta, "player_role", path),
        synopsis=_required_string(raw, "synopsis", path),
        initial_facts=_parse_facts(_required_list(raw, "initial_facts", path), path),
        commitments=_parse_commitments(_required_list(raw, "commitments", path), path),
        trajectory=_parse_trajectory(_required_list(raw, "trajectory", path), path),
    )


def _parse_facts(raw_facts: Sequence[object], path: Path) -> tuple[Fact, ...]:
    facts: list[Fact] = []
    seen_ids: set[str] = set()
    for index, raw_fact in enumerate(raw_facts):
        fact = _required_mapping(raw_fact, f"initial_facts[{index}]", path)
        fact_id = _required_string(fact, "id", path)
        if fact_id in seen_ids:
            raise DatasetFormatError(f"Duplicate fact id {fact_id!r} in {path}")
        negated = fact.get("negated", False)
        if not isinstance(negated, bool):
            raise DatasetFormatError(f"initial_facts[{index}].negated must be boolean in {path}")
        facts.append(Fact(fact_id, _required_string(fact, "content", path), active=not negated))
        seen_ids.add(fact_id)
    return tuple(facts)


def _parse_commitments(raw_commitments: Sequence[object], path: Path) -> tuple[Commitment, ...]:
    commitments: list[Commitment] = []
    seen_ids: set[str] = set()
    valid_kinds = {"invariant", "achievement", "ordering"}
    for index, raw_commitment in enumerate(raw_commitments):
        commitment = _required_mapping(raw_commitment, f"commitments[{index}]", path)
        commitment_id = _required_string(commitment, "id", path)
        kind = _required_string(commitment, "type", path)
        if commitment_id in seen_ids:
            raise DatasetFormatError(f"Duplicate commitment id {commitment_id!r} in {path}")
        if kind not in valid_kinds:
            raise DatasetFormatError(f"Unsupported commitment type {kind!r} in {path}")
        commitments.append(
            Commitment(
                id=commitment_id,
                kind=cast(Literal["invariant", "achievement", "ordering"], kind),
                description=_required_string(commitment, "description", path),
                satisfaction_condition=_required_string(commitment, "satisfaction_condition", path),
                violation_condition=_required_string(commitment, "violation_condition", path),
            )
        )
        seen_ids.add(commitment_id)
    return tuple(commitments)


def _parse_trajectory(raw_trajectory: Sequence[object], path: Path) -> tuple[TrajectoryNode, ...]:
    trajectory: list[TrajectoryNode] = []
    seen_ids: set[str] = set()
    for index, raw_node in enumerate(raw_trajectory):
        node = _required_mapping(raw_node, f"trajectory[{index}]", path)
        node_id = _required_string(node, "id", path)
        if node_id in seen_ids:
            raise DatasetFormatError(f"Duplicate trajectory id {node_id!r} in {path}")
        trajectory.append(
            TrajectoryNode(
                id=node_id,
                description=_required_string(node, "description", path),
                trigger_event=_required_string(node, "trigger_event", path),
                key_delta=_required_string(node, "key_delta", path),
            )
        )
        seen_ids.add(node_id)
    return tuple(trajectory)


def _yaml_mapping(path: Path) -> Mapping[str, object]:
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise DatasetFormatError(f"Invalid YAML in {path}: {exc}") from exc
    return _required_mapping(raw, "top-level document", path)


def _required_mapping(value: object, label: str, path: Path) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetFormatError(f"{label} must be a mapping in {path}")
    return value


def _required_list(value: Mapping[str, object], key: str, path: Path) -> Sequence[object]:
    raw = value.get(key)
    if not isinstance(raw, list):
        raise DatasetFormatError(f"{key} must be a list in {path}")
    return raw


def _required_string(value: Mapping[str, object], key: str, path: Path) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise DatasetFormatError(f"{key} must be a non-empty string in {path}")
    return raw


def _string_tuple(values: Sequence[object], label: str, path: Path) -> tuple[str, ...]:
    result = tuple(value for value in values if isinstance(value, str) and value.strip())
    if len(result) != len(values):
        raise DatasetFormatError(f"{label} must contain only non-empty strings in {path}")
    return result
