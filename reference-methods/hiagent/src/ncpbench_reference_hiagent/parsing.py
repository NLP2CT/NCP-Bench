from __future__ import annotations

import re

from ncpbench.model_output import ModelOutputError


_RE_RETRIEVE = re.compile(
    r"retrieve\(\s*((?:s?_?\d+)(?:\s*,\s*(?:s?_?\d+))*)\s*\)",
    re.IGNORECASE,
)


def extract_retrieve_ids(text: str) -> list[int]:
    if not text:
        return []
    match = _RE_RETRIEVE.search(text)
    if not match:
        return []
    raw = match.group(1)
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        normalized = re.sub(r"^[a-zA-Z]+_?", "", part)
        try:
            values.append(int(normalized))
        except ValueError:
            continue
    return values


def parse_control_output(response: str) -> tuple[str | None, str]:
    """Parse one exact HiAgent control response without changing method state."""

    if not isinstance(response, str):
        raise ModelOutputError("HiAgent response must be text")
    text = response.strip()
    if not text:
        raise ModelOutputError("HiAgent response is empty")

    lines = text.splitlines()
    first = lines[0].strip()
    if first.startswith("Action:"):
        action = "\n".join((first[len("Action:") :].strip(), *lines[1:])).strip()
        if not action:
            raise ModelOutputError("Action must contain non-empty text")
        return None, action

    if not first.startswith("Subgoal:"):
        raise ModelOutputError("HiAgent response must start with Action: or Subgoal:")
    subgoal = first[len("Subgoal:") :].strip()
    if not subgoal:
        raise ModelOutputError("Subgoal must contain non-empty text")
    if len(lines) < 2 or not lines[1].strip().startswith("Action:"):
        raise ModelOutputError("Subgoal must be followed by Action:")
    action = "\n".join(
        (lines[1].strip()[len("Action:") :].strip(), *lines[2:])
    ).strip()
    if not action:
        raise ModelOutputError("Action must contain non-empty text")
    return subgoal, action
