"""Build a complete dataset from a curated story-source manifest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from ncpbench.dataset import Dataset, load_dataset, save_spec
from ncpbench.specification import SpecificationGenerator, SpecificationSource


class ConstructionFormatError(ValueError):
    """Raised when a construction-source manifest is invalid."""


def load_specification_sources(path: str | Path) -> tuple[SpecificationSource, ...]:
    """Load the curated inputs required to generate benchmark specifications."""

    source_path = Path(path).expanduser()
    try:
        with source_path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConstructionFormatError(f"Invalid YAML in {source_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ConstructionFormatError(
            f"Construction source must be a top-level list in {source_path}"
        )

    sources: list[SpecificationSource] = []
    for index, value in enumerate(raw):
        item = _required_mapping(value, f"entry {index}", source_path)
        sources.append(
            SpecificationSource(
                id=_required_string(item, "id", index, source_path),
                title=_required_string(item, "title", index, source_path),
                genres=_required_genres(item, index, source_path),
                player_role=_required_string(item, "player_role", index, source_path),
                synopsis=_required_string(item, "synopsis", index, source_path),
            )
        )
    _validate_sources(sources)
    return tuple(sources)


def build_dataset(
    sources: Sequence[SpecificationSource],
    generator: SpecificationGenerator,
    output_dir: str | Path,
) -> Dataset:
    """Generate every source in order and return the loadable dataset."""

    ordered_sources = tuple(sources)
    _validate_sources(ordered_sources)
    root = Path(output_dir).expanduser()
    spec_ids: list[str] = []
    for source in ordered_sources:
        spec = generator.generate(source)
        relative_path = Path("specs") / f"{source.id}.yaml"
        save_spec(spec, root / relative_path)
        spec_ids.append(source.id)

    root.mkdir(parents=True, exist_ok=True)
    with (root / "index.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {"specs": spec_ids},
            stream,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    dataset = load_dataset(root)
    for source in ordered_sources:
        dataset.load_spec(source.id)
    return dataset


def _validate_sources(sources: Sequence[SpecificationSource]) -> None:
    if not sources:
        raise ConstructionFormatError("Construction source must contain at least one entry")
    seen_ids: set[str] = set()
    for source in sources:
        if (
            not source.id
            or source.id in {".", ".."}
            or Path(source.id).name != source.id
        ):
            raise ConstructionFormatError(f"Invalid specification id: {source.id!r}")
        if source.id in seen_ids:
            raise ConstructionFormatError(f"Duplicate specification id: {source.id!r}")
        for field in ("title", "player_role", "synopsis"):
            value = getattr(source, field)
            if not isinstance(value, str) or not value.strip():
                raise ConstructionFormatError(
                    f"Specification {source.id!r} has invalid {field}"
                )
        if not source.genres or any(
            not isinstance(genre, str) or not genre.strip() for genre in source.genres
        ):
            raise ConstructionFormatError(
                f"Specification {source.id!r} has invalid genres"
            )
        seen_ids.add(source.id)


def _required_mapping(value: object, label: str, path: Path) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConstructionFormatError(f"{label} must be a mapping in {path}")
    return value


def _required_string(
    item: Mapping[str, object], key: str, index: int, path: Path
) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConstructionFormatError(
            f"entry {index}.{key} must be a non-empty string in {path}"
        )
    return value


def _required_genres(
    item: Mapping[str, object], index: int, path: Path
) -> tuple[str, ...]:
    value = item.get("genres")
    if not isinstance(value, list) or not value:
        raise ConstructionFormatError(
            f"entry {index}.genres must be a non-empty list in {path}"
        )
    genres = tuple(genre for genre in value if isinstance(genre, str) and genre.strip())
    if len(genres) != len(value):
        raise ConstructionFormatError(
            f"entry {index}.genres must contain only non-empty strings in {path}"
        )
    return genres


__all__ = [
    "ConstructionFormatError",
    "build_dataset",
    "load_specification_sources",
]
