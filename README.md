# NCP-Bench

<p align="center">
  <a href="https://arxiv.org/abs/2608.08160"><b>📄 arXiv</b></a> |
  <a href="https://openreview.net/forum?id=JoJUWsQBp0&noteId=cIuEdqlTup"><b>📝 OpenReview</b></a> |
  <a href="https://icml.cc/virtual/2026/poster/64786"><b>🎓 ICML</b></a> |
  <a href="https://huggingface.co/papers/2608.08160"><b>🤗 Daily Papers</b></a> |
  <a href="https://github.com/yingpengma/NCP-Bench"><b>💻 Code</b></a> |
  <a href="#dataset"><b>📦 Dataset</b></a>
</p>

<p align="center">
  <a href="#overview"><b>✨ Overview</b></a> |
  <a href="#main-results"><b>📊 Main Results</b></a> |
  <a href="#quick-start"><b>🚀 Quick Start</b></a> |
  <a href="#evaluate-a-narrator"><b>🧪 Evaluate a Narrator</b></a> |
  <a href="#citation"><b>📝 Citation</b></a>
</p>

> 中文文档: [README.zh-CN.md](README.zh-CN.md)

NCP-Bench is the official repository for the ICML 2026 paper **[Can LLM Agents Stick to the Script? A Benchmark for Long-Horizon Consistency in Interactive Narratives](https://arxiv.org/abs/2608.08160)**.

It evaluates **Narrative Commitment Preservation (NCP)** in interactive narratives: whether a narrator agent can respond to free-form player actions while preserving established world facts, narrative commitments, and the order of a reference trajectory. The repository contains the 100 benchmark environments, fixed evaluation prompts, a method-neutral episode runner, and the Baseline and HiAgent reference methods reported in the paper.

<a id="overview"></a>
## ✨ Overview

Interactive narrators must accommodate player agency without silently rewriting established events or bypassing required plot developments. NCP-Bench converts each movie synopsis into a stateful environment with an initial fact ledger, a set of narrative commitments, and an ordered reference trajectory. After every narrator response, a fixed evaluator audits fact, commitment, and player-input conflicts before updating the state for the next turn.

<p align="center">
  <img src="assets/overview.png" alt="NCP-Bench overview: data construction, interaction history, and evaluation framework" width="100%">
</p>

<a id="dataset"></a>
## 📦 Dataset

NCP-Bench contains 100 curated movie-level narrative environments spanning 18 genres. Each YAML specification contains story metadata, a synopsis, initial facts, commitments, and trajectory nodes. Across the dataset, there are 1,660 initial facts, 1,222 commitments, and 1,511 trajectory nodes.

| Component | Total | Per environment (mean) | Per-environment range |
| --- | ---: | ---: | ---: |
| Initial facts | 1,660 | 16.60 | 8 to 31 |
| Commitments | 1,222 | 12.22 | 5 to 24 |
| Trajectory nodes | 1,511 | 15.11 | 6 to 28 |

### 🎬 Data Format

[`dataset/specs/movie52.yaml`](dataset/specs/movie52.yaml) is the environment for *Iron Man*. It contains 16 initial facts, 12 commitments, and 13 trajectory nodes. The excerpt below shows the published schema.

```yaml
meta:
  id: movie52
  title: "Iron Man"
  genres: ["Action", "Sci-Fi"]
  player_role: Tony Stark

initial_facts:
  - content: Tony Stark has not started constructing any arc reactor or powered armor.
    id: f_9
    negated: false
  - content: Tony Stark does not possess knowledge of any secret plans or betrayals
      involving Obadiah Stane.
    id: f_10
    negated: false
  - content: No interaction with S.H.I.E.L.D., Phil Coulson, or Nick Fury has occurred.
    id: f_14
    negated: false

commitments:
  - description: Between s_1 and s_3, Stark and Yinsen's secrecy about building the
      arc reactor and armor must be maintained; terrorists must not know the project
      until s_3 trigger event.
    id: c_2
    satisfaction_condition: From s_1 until s_3, terrorists are unaware of Stark and
      Yinsen's secret construction activities.
    type: invariant
    violation_condition: Player informs terrorists or system prematurely reveals armor
      construction to terrorists before s_3 trigger event.

  - description: Stark must publicly declare himself Iron Man (s_11), marking a fundamental
      narrative milestone and identity acceptance.
    id: c_10
    satisfaction_condition: Player chooses or the system triggers the public announcement
      of Stark's identity as Iron Man at s_11.
    type: achievement
    violation_condition: Narrative ends or progresses beyond s_11 without Stark revealing
      his identity publicly, leaving contradiction or secrecy intact.

  - description: Introduction to the wider superhero world and Avengers Initiative by
      Nick Fury (s_12) must occur after Stark's public reveal as Iron Man (s_11).
    id: c_11
    satisfaction_condition: Node s_11 occurs completely before s_12 introduction and
      engagement with Fury.
    type: ordering
    violation_condition: Fury approaches Stark with Avengers Initiative before Stark's
      public identity declaration.

trajectory:
  - description: The Ten Rings terrorists discover Stark's secret activities and attack
      the workshop while Stark powers up his armor.
    id: s_3
    key_delta: Yinsen dies defending Stark; Stark activates his armored suit to fight
      the terrorists.
    trigger_event: Yinsen sacrifices himself to hold off attackers, allowing Stark's
      armor to fully power up.

  - description: Pepper Potts informs S.H.I.E.L.D. agent Phil Coulson of Stane's illegal
      activities. Stane ambushes Stark at home and steals his new arc reactor, paralyzing
      him with a sonic device.
    id: s_9
    key_delta: Stark regains the minimal power source necessary to animate his armor
      and continue the fight.
    trigger_event: Stark crawls to his lab to reinstall his original arc reactor and
      recover his strength.

  - description: The press dubs Stark's armored persona 'Iron Man'. S.H.I.E.L.D. provides
      Stark with a cover story to explain recent events and protect his identity.
    id: s_11
    key_delta: Stark publicly accepts his superhero identity, ending any illusion of
      secrecy.
    trigger_event: At a press conference, Stark begins the cover story but surprises
      all by declaring publicly that he is Iron Man.
```

<a id="main-results"></a>
## 📊 Main Results

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

A hierarchical-memory HiAgent baseline, built on GPT-4o-mini and evaluated with GPT-5.4-mini as the auditor, extends the average interaction length from 22.16 to 30.05 turns and reduces commitment conflicts from 26% to 4%. It also raises player-input conflicts from 13% to 38%, and no run satisfies every achievement commitment.

| Method | Avg. turns | Trajectory (%) | Satisfied commitments (%) | Fact conflicts (%) | Commitment conflicts (%) | Player-input conflicts (%) | Conflict-free runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4o-mini | 22.16 | 16.64 | 10.03 | 60.0 | 26.0 | 13.0 | 5 |
| HiAgent | 30.05 | 15.68 | 9.14 | 59.0 | 4.0 | 38.0 | 3 |

<a id="quick-start"></a>
## 🚀 Quick Start

NCP-Bench requires Python 3.10 or later. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiment]"
pip install -e reference-methods/baseline
cp .env.example .env
```

Configure a narrator model and a separate evaluation model in `.env`, then run one benchmark turn. The experiment runner loads this file automatically:

```bash
python experiments/run.py \
  --spec movie52 \
  --max-turns 1 \
  --max-tokens 8092 \
  --output-dir runs/quickstart
```

The run executes the paper Baseline through opening generation, player-input generation, narrator response, conflict and fact evaluation, trajectory and commitment checks, and state transition. It writes the result to `runs/quickstart/movie52.json`.

Every model output is validated before use. An invalid response retries only that model call up to two times; the turn state is not advanced. The run stops if all three attempts fail.

For reported evaluations, use a narrator model that differs from the player simulator and auditor so the model under test does not generate its own test inputs or judge its own outputs. The player and auditor may share an evaluation model, or all three roles may use separate models. Sharing one model across all roles remains available for quick smoke tests. Role-specific `NCPBENCH_*` variables and command-line options also allow different endpoints and API keys. For example:

```bash
python experiments/run.py \
  --spec movie52 \
  --condition adversarial \
  --narrator-model model-under-test \
  --player-model independent-evaluation-model \
  --auditor-model independent-evaluation-model \
  --max-turns 20 \
  --output-dir runs/model-under-test
```

The paper experiments use `temperature=0.6`, `top-p=0.95`, a maximum output length of 8092 tokens, and at most 100 interaction turns. These are the runner defaults; the quick-start command limits the run to one turn and an 8092-token output budget. Use `--all` instead of `--spec` to run the complete dataset. Existing result files are skipped unless `--overwrite` is passed.

After the opening and each committed turn, the runner atomically updates a `*.checkpoint.json` file in the output directory. Rerunning the same command resumes from that boundary; a completed result removes its checkpoint. Closed model APIs are nondeterministic, so reruns are not expected to be bitwise identical.

<a id="use-the-dataset"></a>
## 📚 Use the Dataset

The dataset is kept outside the Python wheel and loaded from an explicit path:

```python
from ncpbench import load_dataset

dataset = load_dataset("dataset")
spec = dataset.load_spec("movie52")

print(spec.title)
print(len(spec.initial_facts), len(spec.commitments), len(spec.trajectory))
```

`dataset/index.yaml` fixes the evaluation order. The first trajectory node is current at episode start, but no node is complete. At the end of every successful turn, the trajectory auditor checks the current node; when both its trigger and key delta occur, it marks that node complete and advances to the next one. Trajectory progress is therefore the number of completed nodes divided by the total number of nodes.

<a id="evaluate-a-narrator"></a>
## 🧪 Evaluate a Narrator

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

<a id="build-specifications"></a>
## 🛠️ Build Specifications

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

<a id="repository-layout"></a>
## 📂 Repository Layout

```text
src/ncpbench/              benchmark package and fixed prompts
dataset/                   100 benchmark environments
reference-methods/         Baseline and HiAgent narrator packages
experiments/run.py         single- or multi-environment runner
tests/                     protocol, runner, dataset, and method tests
```

<a id="tests"></a>
## ✅ Tests

Install the test dependencies and run:

```bash
pip install -e ".[test]"
pip install -e reference-methods/baseline
pip install -e reference-methods/hiagent
pytest -q
```

The tests use local fake clients and do not call model APIs.

<a id="citation"></a>
## 📝 Citation

```bibtex
@inproceedings{macan,
  title={Can LLM Agents Stick to the Script? Modeling Commitment in Interactive Narratives},
  author={Ma, Yingpeng and Yan, Jianhao and Shi, Bei and Kam, Ka Hou and Wang, Runnan and Liu, Xuebo and Chen, Yulong and Zhang, Yue and Wong, Derek F},
  booktitle={Forty-third International Conference on Machine Learning}
}
```

The code is released under the [MIT License](LICENSE).
