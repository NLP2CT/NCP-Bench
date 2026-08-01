# NCP-Bench

NCP-Bench is the dataset and evaluation framework for *Can LLM Agents Stick
to the Script? Modeling Commitment in Interactive Narratives*. It evaluates
whether a narrator agent preserves world facts, narrative commitments, and
trajectory order under free-form player interaction.

The repository contains the 100 benchmark environments, the fixed evaluation
prompts, the method-neutral episode runner, and the Baseline and HiAgent
reference methods used in the paper.

## Quick Start

NCP-Bench requires Python 3.10 or later. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiment]"
pip install -e reference-methods/baseline
cp .env.example .env
```

Add an API key to `.env`, then run one benchmark turn. The experiment runner
loads this file automatically:

```bash
python experiments/run.py \
  --spec movie00 \
  --max-turns 1 \
  --max-tokens 8092 \
  --output-dir runs/quickstart
```

The experiment runs the paper Baseline through the complete benchmark sequence:
opening generation, player-input generation, narrator response, conflict and
fact evaluation, trajectory and commitment checks, and state transition. The
result is written to `runs/quickstart/movie00.json`.

Every model output is validated before use. An invalid response retries only
that model call up to two times; the turn state is not advanced. The run stops
if all three attempts fail.

`.env.example` configures one shared OpenAI-compatible text endpoint for the
narrator, player, and auditor. Models and the input condition can also be
selected independently:

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

The paper experiments use `temperature=0.6`, `top-p=0.95`, a maximum output
length of 8092 tokens, and at most 100 interaction turns. These are the runner
defaults; the quick-start command limits the run to one turn and a 2048-token
output budget. Use `--all` instead of `--spec` to run the complete dataset.
Existing result files are skipped unless `--overwrite` is passed.
After the opening and each committed turn, the runner atomically updates a
`*.checkpoint.json` file in the output directory. Rerunning the same command
resumes from that boundary; a completed result removes its checkpoint.

The narrator, player, and auditor may use different OpenAI-compatible endpoints
and API-key environment variables through role-specific `NCPBENCH_*` variables
or command-line options. Run `python experiments/run.py --help` for the latter.
Closed model APIs are nondeterministic, so reruns are not expected to be
bitwise identical.

## Use The Dataset

The dataset is kept outside the Python wheel and loaded from an explicit path:

```python
from ncpbench import load_dataset

dataset = load_dataset("dataset")
spec = dataset.load_spec("movie00")

print(spec.title)
print(len(spec.initial_facts), len(spec.commitments), len(spec.trajectory))
```

`dataset/index.yaml` fixes the order of the 100 environments.
Each YAML specification contains the curated story metadata, synopsis, initial
facts, commitments, and reference trajectory used by the runner.

## Trajectory Progress

The first trajectory node is the current milestone when an episode starts, but
no node is completed yet. At the end of each successful turn, the trajectory
auditor checks the current node. When both its trigger and key delta have
occurred, that node is completed and the next node becomes current.
Accordingly, trajectory progress is the number of completed nodes divided by
the total number of nodes: it starts at 0 and reaches 1 only when every node
has completed.

## Evaluate A Narrator

Any method under test implements the public `Narrator` interface:

```python
from ncpbench import Narrator, NarratorRequest, NarratorResponse, OpeningRequest


class MyNarrator(Narrator):
    def open(self, request: OpeningRequest) -> NarratorResponse:
        return NarratorResponse(text="Your opening narrative")

    def respond(self, request: NarratorRequest) -> NarratorResponse:
        return NarratorResponse(text="Your next narrative response")
```

A narrator receives the visible interaction history and benchmark story state,
but never receives evaluator prompts, conflict decisions, or evaluator-model
outputs. `experiments/run.py` shows how to connect a narrator to
`EpisodeRunner`.

The paper reference methods are independent optional packages:

```bash
pip install -e reference-methods/baseline
pip install -e reference-methods/hiagent
```

Both use the same `Narrator` interface and receive no method-specific handling
from the benchmark runner. Select the latter with `--method hiagent` after
installing its package.

## Build Specifications

`SpecificationGenerator` implements the paper's three dependent construction
calls: trajectory extraction, commitment extraction, and initial-fact
extraction.

```python
from ncpbench import SpecificationGenerator, SpecificationSource

source = SpecificationSource(
    id="movie00",
    title="The Bourne Identity",
    genres=("Action", "Mystery"),
    player_role="Jason Bourne",
    synopsis="...",
)
spec = SpecificationGenerator(client).generate(source)
```

The injected client implements `complete(messages, *, stage) -> str`. For a
curated list of sources, `build_dataset(...)` writes the ordered index and one
YAML specification per story.

## Repository Layout

```text
src/ncpbench/              benchmark package and fixed prompts
dataset/                   100 benchmark environments
reference-methods/         Baseline and HiAgent narrator packages
experiments/run.py         thin single- or multi-environment runner
tests/                     protocol, runner, dataset, and method tests
```

The package, dataset, and reference methods are deliberately separate. The
core package does not select an API provider or contain method-specific runner
branches. Prompt-integrity tests pin the fixed protocol text by hash.

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
  title     = {Can LLM Agents Stick to the Script? Modeling Commitment in Interactive Narratives},
  author    = {Ma, Yingpeng and Yan, Jianhao and Shi, Bei and Kam, Ka Hou and
               Wang, Runnan and Liu, Xuebo and Chen, Yulong and Zhang, Yue and
               Wong, Derek F.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  address   = {Seoul, South Korea},
  year      = {2026},
  url       = {https://icml.cc/virtual/2026/poster/64786}
}
```

The code is released under the [MIT License](LICENSE).
