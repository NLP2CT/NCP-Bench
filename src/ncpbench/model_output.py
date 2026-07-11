"""Validated model calls with bounded request-level retries."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")
MAX_RETRIES = 2


class ModelOutputError(ValueError):
    """Raised when a model response violates a stage's output contract."""


class ModelCallError(RuntimeError):
    """Raised after one model call fails its retry budget."""


def call_with_retries(
    call: Callable[[], str],
    validate: Callable[[str], T],
    *,
    stage: str,
    max_retries: int = MAX_RETRIES,
) -> T:
    """Retry only ``call`` until its response passes ``validate``."""

    last_error: Exception | None = None
    attempts = max_retries + 1
    for _ in range(attempts):
        try:
            raw = call()
        except Exception as exc:
            # Provider SDKs expose transport and HTTP failures through their own
            # exception hierarchies, so the call boundary is the only common type.
            last_error = exc
            continue
        try:
            return validate(raw)
        except ModelOutputError as exc:
            last_error = exc

    raise ModelCallError(
        f"{stage} failed after {attempts} attempts: {last_error}"
    ) from last_error


def parse_json_object(raw: str) -> dict[str, object]:
    """Parse the exact JSON-object contract used by structured model stages."""

    if not isinstance(raw, str) or not raw.strip():
        raise ModelOutputError("response is empty")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelOutputError("response must be a JSON object")
    return payload


def require_text(raw: str) -> str:
    """Validate a non-empty plain-text model response."""

    if not isinstance(raw, str) or not raw.strip():
        raise ModelOutputError("response must contain non-empty text")
    return raw


def parse_text_object(raw: str) -> str:
    """Parse a JSON object containing exactly one non-empty ``text`` value."""

    payload = parse_json_object(raw)
    if set(payload) != {"text"}:
        raise ModelOutputError("response must contain only the text field")
    text = payload["text"]
    if not isinstance(text, str) or not text.strip():
        raise ModelOutputError("text must be a non-empty string")
    return text.strip()
