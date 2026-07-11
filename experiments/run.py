"""Run NCP-Bench experiments with OpenAI-compatible chat APIs."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ncpbench import (
    EpisodeRunner,
    Narrator,
    OpeningEvaluator,
    create_input_condition,
    episode_trace_to_result,
    load_dataset,
)
from ncpbench.checkpoint import checkpoint_from_mapping, checkpoint_to_mapping
from ncpbench.evaluator import CommitmentChecker, TrajectoryChecker, TurnEvaluator


class OpenAITextClient:
    """Text client for one model endpoint and one structured-output policy."""

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str,
        base_url: str | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        structured_stages: set[str] | None,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env!r} is not set")
        client_args: dict[str, object] = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url
        self._client = OpenAI(**client_args)
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._structured_stages = structured_stages

    def complete(
        self, messages: Sequence[Mapping[str, str]], *, stage: str
    ) -> str:
        structured = (
            self._structured_stages is None or stage in self._structured_stages
        )
        return self._generate(messages, structured=structured)

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._generate(messages, structured=False)

    def _generate(
        self, messages: Sequence[Mapping[str, str]], *, structured: bool
    ) -> str:
        request: dict[str, object] = {
            "model": self._model,
            "messages": [dict(message) for message in messages],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
        }
        if structured:
            request["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**request)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"Model {self._model!r} returned an empty response")
        return content


def parse_args() -> argparse.Namespace:
    shared_model = os.getenv("NCPBENCH_MODEL", "gpt-4o-mini")
    shared_base_url = os.getenv("NCPBENCH_BASE_URL")
    shared_api_key_env = os.getenv("NCPBENCH_API_KEY_ENV", "OPENAI_API_KEY")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--spec",
        action="append",
        dest="spec_ids",
        metavar="SPEC_ID",
        help="run one specification; repeat to select multiple specifications",
    )
    selection.add_argument(
        "--all", action="store_true", help="run every specification in index order"
    )
    parser.add_argument("--method", choices=("baseline", "hiagent"), default="baseline")
    parser.add_argument(
        "--condition", choices=("natural", "adversarial"), default="adversarial"
    )
    parser.add_argument(
        "--narrator-model",
        default=os.getenv("NCPBENCH_NARRATOR_MODEL", shared_model),
    )
    parser.add_argument(
        "--player-model", default=os.getenv("NCPBENCH_PLAYER_MODEL", shared_model)
    )
    parser.add_argument(
        "--auditor-model", default=os.getenv("NCPBENCH_AUDITOR_MODEL", shared_model)
    )
    parser.add_argument(
        "--narrator-base-url",
        default=os.getenv("NCPBENCH_NARRATOR_BASE_URL", shared_base_url),
    )
    parser.add_argument(
        "--player-base-url",
        default=os.getenv("NCPBENCH_PLAYER_BASE_URL", shared_base_url),
    )
    parser.add_argument(
        "--auditor-base-url",
        default=os.getenv("NCPBENCH_AUDITOR_BASE_URL", shared_base_url),
    )
    parser.add_argument(
        "--narrator-api-key-env",
        default=os.getenv("NCPBENCH_NARRATOR_API_KEY_ENV", shared_api_key_env),
    )
    parser.add_argument(
        "--player-api-key-env",
        default=os.getenv("NCPBENCH_PLAYER_API_KEY_ENV", shared_api_key_env),
    )
    parser.add_argument(
        "--auditor-api-key-env",
        default=os.getenv("NCPBENCH_AUDITOR_API_KEY_ENV", shared_api_key_env),
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8092)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_client(
    role: str,
    args: argparse.Namespace,
    *,
    structured_stages: set[str] | None,
) -> OpenAITextClient:
    return OpenAITextClient(
        getattr(args, f"{role}_model"),
        api_key_env=getattr(args, f"{role}_api_key_env"),
        base_url=getattr(args, f"{role}_base_url"),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        structured_stages=structured_stages,
    )


def make_narrator(args: argparse.Namespace) -> Narrator:
    if args.method == "baseline":
        from ncpbench_reference_baseline import BaselineNarrator

        return BaselineNarrator(
            make_client(
                "narrator",
                args,
                structured_stages={"opening", "method_response_generate"},
            )
        )

    from ncpbench_reference_hiagent import HiAgentNarrator

    return HiAgentNarrator(
        make_client("narrator", args, structured_stages={"opening"})
    )


def selected_spec_ids(
    args: argparse.Namespace, available_ids: tuple[str, ...]
) -> tuple[str, ...]:
    if args.all:
        return available_ids
    requested = tuple(dict.fromkeys(args.spec_ids or ()))
    unknown = sorted(set(requested) - set(available_ids))
    if unknown:
        raise ValueError(f"Unknown specification IDs: {', '.join(unknown)}")
    return requested


def checkpoint_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "method": args.method,
        "condition": args.condition,
        "narrator_model": args.narrator_model,
        "player_model": args.player_model,
        "auditor_model": args.auditor_model,
        "narrator_base_url": args.narrator_base_url,
        "player_base_url": args.player_base_url,
        "auditor_base_url": args.auditor_base_url,
        "narrator_api_key_env": args.narrator_api_key_env,
        "player_api_key_env": args.player_api_key_env,
        "auditor_api_key_env": args.auditor_api_key_env,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_turns": args.max_turns,
    }


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    args = parse_args()
    dataset = load_dataset(args.dataset)
    spec_ids = selected_spec_ids(args, dataset.spec_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending_spec_ids: list[str] = []
    for spec_id in spec_ids:
        output = args.output_dir / f"{spec_id}.json"
        if output.exists() and not args.overwrite:
            print(f"{spec_id}: skipped existing {output}")
        else:
            pending_spec_ids.append(spec_id)

    if not pending_spec_ids:
        return

    auditor = make_client("auditor", args, structured_stages=None)
    player = make_client("player", args, structured_stages=set())
    narrator = make_narrator(args)
    runner = EpisodeRunner(
        TurnEvaluator(auditor),
        TrajectoryChecker(auditor),
        CommitmentChecker(auditor),
        OpeningEvaluator(auditor),
    )

    for spec_id in pending_spec_ids:
        output = args.output_dir / f"{spec_id}.json"
        checkpoint_path = args.output_dir / f"{spec_id}.checkpoint.json"
        spec = dataset.load_spec(spec_id)
        configuration = checkpoint_configuration(args)
        checkpoint = None
        if args.overwrite:
            checkpoint_path.unlink(missing_ok=True)
        elif checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if saved.get("configuration") != configuration:
                raise RuntimeError(
                    f"Checkpoint configuration does not match: {checkpoint_path}"
                )
            checkpoint = checkpoint_from_mapping(saved["episode"], spec)
            print(
                f"{spec_id}: resuming after {len(checkpoint.turns)} committed turns"
            )

        def save_checkpoint(value) -> None:
            temporary_checkpoint = checkpoint_path.with_suffix(".json.tmp")
            temporary_checkpoint.write_text(
                json.dumps(
                    {
                        "configuration": configuration,
                        "episode": checkpoint_to_mapping(value),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary_checkpoint.replace(checkpoint_path)

        trace = runner.run(
            narrator,
            spec,
            create_input_condition(args.condition, player),
            max_turns=args.max_turns,
            checkpoint=checkpoint,
            on_checkpoint=save_checkpoint,
        )
        result = episode_trace_to_result(trace)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        checkpoint_path.unlink(missing_ok=True)
        print(
            f"{spec.id}: termination={result['termination']}, "
            f"turns={len(result['turns'])}, output={output}"
        )


if __name__ == "__main__":
    main()
