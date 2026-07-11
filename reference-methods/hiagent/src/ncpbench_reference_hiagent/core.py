from __future__ import annotations

from collections.abc import Mapping
from typing import List, Tuple

import tiktoken

from ncpbench_reference_hiagent.client import TextGenerationClient
from ncpbench_reference_hiagent.config import HiAgentRuntimeConfig
from ncpbench.model_output import ModelCallError, call_with_retries
from ncpbench_reference_hiagent.parsing import extract_retrieve_ids, parse_control_output
from ncpbench_reference_hiagent.hiagent_prompts import HIAGENT_SUFFIX
from ncpbench_reference_hiagent.summarizer import TrajectorySummarizer

MemoryPair = Tuple[str, str]
MemoryChunk = List[MemoryPair]


class HiAgentCore:
    """Core HiAgent policy adapted to a generic text environment."""

    def __init__(
        self,
        *,
        client: TextGenerationClient,
        config: HiAgentRuntimeConfig,
        summarizer: TrajectorySummarizer,
    ) -> None:
        self.client = client
        self.config = config
        self.summarizer = summarizer
        self.instruction = HIAGENT_SUFFIX
        self.init_obs: str | None = None
        self.memory: List[MemoryChunk] = []
        self.subgoal_idx: List[int] = []
        self._encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def reset(self, init_obs: str) -> None:
        self.init_obs = init_obs
        self.memory = [[("Observation", init_obs)]]
        self.subgoal_idx = []

    def update(self, action: str, state: str) -> None:
        self.memory.append([("Action", action), ("Observation", state)])

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "init_obs": self.init_obs,
            "memory": self.memory,
            "subgoal_idx": self.subgoal_idx,
        }

    def restore_state(self, state: Mapping[str, object]) -> None:
        init_obs = state.get("init_obs")
        raw_memory = state.get("memory")
        raw_indices = state.get("subgoal_idx")
        if init_obs is not None and not isinstance(init_obs, str):
            raise ValueError("Invalid HiAgent checkpoint initial observation")
        if not isinstance(raw_memory, list) or not isinstance(raw_indices, list):
            raise ValueError("Invalid HiAgent checkpoint memory")
        memory: List[MemoryChunk] = []
        for chunk in raw_memory:
            if not isinstance(chunk, list):
                raise ValueError("Invalid HiAgent checkpoint memory chunk")
            parsed_chunk: MemoryChunk = []
            for pair in chunk:
                if (
                    not isinstance(pair, (list, tuple))
                    or len(pair) != 2
                    or not all(isinstance(value, str) for value in pair)
                ):
                    raise ValueError("Invalid HiAgent checkpoint memory pair")
                parsed_chunk.append((pair[0], pair[1]))
            memory.append(parsed_chunk)
        if any(not isinstance(value, int) for value in raw_indices):
            raise ValueError("Invalid HiAgent checkpoint retrieval indices")
        self.init_obs = init_obs
        self.memory = memory
        self.subgoal_idx = list(raw_indices)

    def run(self, *, turn_context: str, depth: int = 0) -> str:
        if depth > self.config.max_retrieve_depth:
            raise ModelCallError(
                "hiagent_reasoning exceeded the retrieve-depth limit"
            )

        prompt = self.make_prompt(turn_context=turn_context)
        messages = [
            {"role": "system", "content": self.config.system_message},
            {"role": "user", "content": prompt},
        ]
        stage = "method_response_generate" if depth == 0 else "hiagent_reasoning"
        subgoal, action = call_with_retries(
            lambda: self.client.complete(messages, stage=stage),
            parse_control_output,
            stage=stage,
        )
        if subgoal is not None:
            self.subgoal_idx = []
            self.memory.append([("Subgoal", subgoal)])

        if action.lower().startswith("retrieve("):
            numbers = extract_retrieve_ids(action)
            if numbers:
                self.subgoal_idx.extend(numbers)
                return self.run(turn_context=turn_context, depth=depth + 1)
            raise ModelCallError("hiagent_reasoning returned an invalid retrieve command")

        return action

    def make_prompt(self, *, turn_context: str) -> str:
        sections: List[str] = [self.instruction]
        sections.append(turn_context.strip())
        query = "\n\n".join([section for section in sections if section]) + "\n"

        history = self.memory[-self.config.memory_size :]
        prompt = query + "\n" + self._serialize_history(history)
        if history and history[-1] and history[-1][0][0] == "Subgoal":
            prompt += "\nAction: "

        while (
            self._estimate_tokens(self.config.system_message, prompt)
            > self.config.max_input_tokens
            and len(history) > 1
        ):
            history = history[1:]
            prompt = query + "\n" + self._serialize_history(history)
            if history and history[-1] and history[-1][0][0] == "Subgoal":
                prompt += "\nAction: "

        return prompt

    def _serialize_history(self, history: List[MemoryChunk]) -> str:
        if not history:
            return ""

        subgoal_indices = [idx for idx, chunk in enumerate(history) if chunk and chunk[0][0] == "Subgoal"]
        if len(subgoal_indices) <= 1:
            return self._vanilla_serialize_history(history)

        keep_indices = {value - 1 for value in self.subgoal_idx if value > 0}
        final_subgoal_index = subgoal_indices[-1]
        new_history: List[MemoryChunk] = list(history[: subgoal_indices[0]])

        for i in range(0, len(subgoal_indices) - 1):
            begin = subgoal_indices[i]
            end = subgoal_indices[i + 1]
            if i in keep_indices:
                new_history.extend(history[begin:end])
                continue

            subgoal = history[begin][0]
            trajectory = history[begin + 1 : end]
            trajectory = [
                pair for pair in trajectory if not (pair and pair[0][0] == "Action" and "check valid" in pair[0][1])
            ]
            summary = self.summarizer.generate_summary([trajectory], [subgoal])[0]

            numbered_subgoal = (f"{i + 1} {subgoal[0]}", subgoal[1])
            new_history.append([numbered_subgoal, ("Observation", summary)])

        final_subgoal = history[final_subgoal_index][0]
        numbered_final_subgoal = (f"{len(subgoal_indices)} {final_subgoal[0]}", final_subgoal[1])
        new_history.append([numbered_final_subgoal])
        new_history.extend(history[final_subgoal_index + 1 :])

        return self._vanilla_serialize_history(new_history)

    def _vanilla_serialize_history(self, history: List[MemoryChunk]) -> str:
        lines: List[str] = []
        for chunk in history:
            for role, text in chunk:
                if text is None:
                    continue
                lines.append(f"{role}: {text}")
        return "\n".join(lines)

    def _estimate_tokens(self, system_message: str, user_message: str) -> int:
        return len(self._encoding.encode(f"{system_message}\n{user_message}"))
