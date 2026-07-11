# Reference Methods

This directory contains the two narrator methods used in the NCP-Bench paper:
`baseline` and `hiagent`.

They are paper-reproduction artifacts, not components of the NCP-Bench
definition. They are not installed by `pip install ncpbench`, receive no
special handling from the benchmark runner, and use the same public `Narrator`
interface available to every external method.

Each directory is an independent optional Python project. Install only the
method whose paper result you want to reproduce:

```bash
pip install -e .
pip install -e reference-methods/baseline
pip install -e reference-methods/hiagent
```

Both methods receive a method-owned text-generation client and return only a
`NarratorResponse`. Fact updates, conflict checks, and all evaluator-model
calls are owned by NCP-Bench rather than by either method.
