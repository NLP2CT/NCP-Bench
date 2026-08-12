# NCP-Bench

<p align="center">
  <a href="https://arxiv.org/abs/2608.08160">Paper</a> &nbsp;|&nbsp;
  <a href="https://openreview.net/forum?id=JoJUWsQBp0&noteId=cIuEdqlTup">OpenReview</a> &nbsp;|&nbsp;
  <a href="https://icml.cc/virtual/2026/poster/64786">ICML 2026</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/papers/2608.08160">Hugging Face Daily Papers</a> &nbsp;|&nbsp;
  <a href="README.zh-CN.md">中文</a>
</p>

NCP-Bench is a benchmark for evaluating **Narrative Commitment Preservation (NCP)** in interactive narratives. It tests whether a narrator agent can respond to free-form player actions while preserving established world facts, narrative commitments, and the order of a reference trajectory.

The repository contains the 100 benchmark environments, fixed evaluation prompts, a method-neutral episode runner, and the Baseline and HiAgent reference methods reported in the paper.

## Overview

Interactive narrators must accommodate player agency without silently rewriting established events or bypassing required plot developments. NCP-Bench converts each movie synopsis into a stateful environment with an initial fact ledger, a set of narrative commitments, and an ordered reference trajectory. After every narrator response, a fixed evaluator audits fact, commitment, and player-input conflicts before updating the state for the next turn.

<p align="center">
  <img src="assets/overview.png" alt="NCP-Bench overview: data construction, interaction history, and evaluation framework" width="100%">
</p>

## Dataset

NCP-Bench contains 100 curated movie-level narrative environments spanning 18 genres. Each YAML specification contains story metadata, a synopsis, initial facts, commitments, and trajectory nodes. Across the dataset, there are 1,660 initial facts, 1,222 commitments, and 1,511 trajectory nodes.

| Component | Total | Per environment (mean) | Range |
| --- | ---: | ---: | ---: |
| Initial facts | 1,660 | 16.60 | 8-31 |
| Commitments | 1,222 | 12.22 | 5-24 |
| Trajectory nodes | 1,511 | 15.11 | 6-28 |

### Data Format

[`dataset/specs/movie52.yaml`](dataset/specs/movie52.yaml) is the environment for *Iron Man*. It contains 16 initial facts, 12 commitments, and 13 trajectory nodes. The excerpt below shows the published schema.

```yaml
meta:
  id: movie52
  title: "Iron Man"
  genres: ["Action", "Sci-Fi"]
  player_role: Tony Stark

initial_facts:
  - content: Tony Stark is located at a remote military demonstration site in Afghanistan.
    id: f_0
    negated: false
  - content: Tony Stark is uninjured and not yet captured by any hostile party.
    id: f_4
    negated: false

commitments:
  - description: Stark's critical wounding and capture by the Ten Rings (s_0) must occur
      before any captivity or surgery events; ensures no premature knowledge or access
      to captivity phase.
    id: c_0
    satisfaction_condition: Event s_0 (Stark wounded and captured) occurs before s_1
      (captive surgery and imprisonment workshop).
    type: ordering

trajectory:
  - description: Tony Stark and Lieutenant Colonel James Rhodes are in Afghanistan at
      a remote military demonstration site of the new Jericho missile, representing
      Stark Industries. The environment is tense but controlled, with military personnel
      and local observers present.
    id: s_0
    key_delta: Tony Stark is injured and taken prisoner by the Ten Rings, shifting from
      a public demonstration to captivity.
    trigger_event: Tony Stark is critically wounded in an ambush by the terrorist group
      the Ten Rings and captured.
```

## Main Results

We evaluate six state-of-the-art LLM narrators with Gemini-2.5-Flash as the auditor under adversarial player inputs. Long-horizon commitment preservation remains difficult: GPT-5.2 has the highest average interaction length (32.92 turns), but its survival rate is only 42% after 20 turns. Fact conflicts are the most frequent failure mode, occurring in 40%-68% of runs across models.

<p align="center">
  <img src="assets/survival-rate.png" alt="Survival rate across interaction turns for six LLMs on NCP-Bench" width="72%">
</p>

| Model | Avg. turns | Trajectory (%) | Satisfied commitments (%) | Fact conflicts (%) | Commitment conflicts (%) | Player-input conflicts (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.2 | **32.92** | 9.94 | 11.22 | **40.0** | **24.0** | 31.0 |
| GPT-4o-mini | 24.80 | 10.57 | 10.90 | 64.0 | 32.0 | **11.0** |
| Qwen3-235B-A22B | 16.76 | 13.16 | 10.60 | 68.0 | 33.0 | 26.0 |
| DeepSeek-V3.2 | 15.88 | **15.40** | **13.42** | 55.0 | 32.0 | 23.0 |
| Grok-4.1-Fast | 7.87 | 12.07 | 10.37 | 65.0 | 47.0 | 12.0 |
| Kimi-K2.5 | 2.88 | 10.26 | 5.18 | 66.0 | 54.0 | **11.0** |

A hierarchical-memory HiAgent baseline, built on GPT-4o-mini and evaluated with the same GPT-5.4-mini auditor, extends the average interaction length from 22.16 to 30.05 turns and reduces commitment conflicts from 26% to 4%. It also raises player-input conflicts from 13% to 38%, and no run satisfies every achievement commitment.

| Method | Avg. turns | Trajectory (%) | Satisfied commitments (%) | Fact conflicts (%) | Commitment conflicts (%) | Player-input conflicts (%) | Conflict-free runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4o-mini | 22.16 | 16.64 | 10.03 | 60.0 | 26.0 | 13.0 | 5 |
| HiAgent | 30.05 | 15.68 | 9.14 | 59.0 | 4.0 | 38.0 | 3 |

## Quick Start

NCP-Bench requires Python 3.10 or later. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiment]"
pip install -e reference-methods/baseline
cp .env.example .env
```

Add an API key to `.env`, then run one benchmark turn. The experiment runner loads this file automatically:

```bash
python experiments/run.py \
  --spec movie52 \
  --max-turns 1 \
  --max-tokens 8092 \
  --output-dir runs/quickstart
```

The run executes the paper Baseline through opening generation, player-input generation, narrator response, conflict and fact evaluation, trajectory and commitment checks, and state transition. It writes the result to `runs/quickstart/movie52.json`.

Every model output is validated before use. An invalid response retries only that model call up to two times; the turn state is not advanced. The run stops if all three attempts fail.

`.env.example` configures one shared OpenAI-compatible text endpoint for the narrator, player, and auditor. The three roles can also use independent endpoints, API keys, and models through role-specific `NCPBENCH_*` variables or command-line options. For example:

```bash
python experiments/run.py \
  --spec movie52 \
  --condition adversarial \
  --narrator-model gpt-4o-mini \
  --player-model gpt-4o-mini \
  --auditor-model gpt-4o-mini \
  --max-turns 20 \
  --output-dir runs/gpt-4o-mini
```

The paper experiments use `temperature=0.6`, `top-p=0.95`, a maximum output length of 8092 tokens, and at most 100 interaction turns. These are the runner defaults; the quick-start command limits the run to one turn and an 8092-token output budget. Use `--all` instead of `--spec` to run the complete dataset. Existing result files are skipped unless `--overwrite` is passed.

After the opening and each committed turn, the runner atomically updates a `*.checkpoint.json` file in the output directory. Rerunning the same command resumes from that boundary; a completed result removes its checkpoint. The narrator, player, and auditor may use different OpenAI-compatible endpoints and API-key environment variables through role-specific `NCPBENCH_*` variables or command-line options. Closed model APIs are nondeterministic, so reruns are not expected to be bitwise identical.

## Use the Dataset

The dataset is kept outside the Python wheel and loaded from an explicit path:

```python
from ncpbench import load_dataset

dataset = load_dataset("dataset")
spec = dataset.load_spec("movie52")

print(spec.title)
print(len(spec.initial_facts), len(spec.commitments), len(spec.trajectory))
```

`dataset/index.yaml` fixes the evaluation order. The first trajectory node is current at episode start, but no node is complete. At the end of every successful turn, the trajectory auditor checks the current node; when both its trigger and key delta occur, it marks that node complete and advances to the next one. Trajectory progress is therefore the number of completed nodes divided by the total number of nodes.

## Evaluate a Narrator

Any method under test implements the public `Narrator` interface:

```python
from ncpbench import Narrator, NarratorRequest, NarratorResponse, OpeningRequest


class MyNarrator(Narrator):
    def open(self, request: OpeningRequest) -> NarratorResponse:
        return NarratorResponse(text="Your opening narrative")

    def respond(self, request: NarratorRequest) -> NarratorResponse:
        return NarratorResponse(text="Your next narrative response")
```

A narrator receives the visible interaction history and benchmark story state. It never receives evaluator prompts, conflict decisions, or evaluator-model outputs. `experiments/run.py` shows how to connect a narrator to `EpisodeRunner`.

The paper reference methods are independent optional packages:

```bash
pip install -e reference-methods/baseline
pip install -e reference-methods/hiagent
```

Both use the same `Narrator` interface and receive no method-specific handling from the benchmark runner. Select HiAgent with `--method hiagent` after installing its package.

## Build Specifications

`SpecificationGenerator` implements the paper's three dependent construction calls: trajectory extraction, commitment extraction, and initial-fact extraction.

```python
from ncpbench import SpecificationGenerator, SpecificationSource

source = SpecificationSource(
    id="movie52",
    title="Iron Man",
    genres=("Action", "Sci-Fi"),
    player_role="Tony Stark",
    synopsis="...",
)
spec = SpecificationGenerator(client).generate(source)
```

The injected client implements `complete(messages, *, stage) -> str`. For a curated list of sources, `build_dataset(...)` writes the ordered index and one YAML specification per story.

## Repository Layout

```text
src/ncpbench/              benchmark package and fixed prompts
dataset/                   100 benchmark environments
reference-methods/         Baseline and HiAgent narrator packages
experiments/run.py         single- or multi-environment runner
tests/                     protocol, runner, dataset, and method tests
```

## Tests

Install the test dependencies and run:

```bash
pip install -e ".[test]"
pip install -e reference-methods/baseline
pip install -e reference-methods/hiagent
pytest -q
```

The tests use local fake clients and do not call model APIs.

## Citation

```bibtex
@inproceedings{ma2026can,
  title     = {Can LLM Agents Stick to the Script? A Benchmark for Long-Horizon Consistency in Interactive Narratives},
  author    = {Ma, Yingpeng and Yan, Jianhao and Shi, Bei and Kam, Ka Hou and
               Wang, Runnan and Liu, Xuebo and Chen, Yulong and Zhang, Yue and
               Wong, Derek F.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  address   = {Seoul, South Korea},
  year      = {2026},
  url       = {https://arxiv.org/abs/2608.08160}
}
```

The code is released under the [MIT License](LICENSE).
