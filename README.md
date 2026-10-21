# rag-boilerplate-dspy

A small project for learning DSPy syntax by classifying Reuters-21578 articles into 7 topics
(`earn`, `acq`, `crude`, `money-fx`, `grain`, `trade`, `interest`).

Motivation: in [fastapi-rag-boilerplate](https://github.com/fischerdm/fastapi-rag-boilerplate)
inferring these labels with an LLM turned out to be hard. Here we check how far DSPy's
modules and optimizers get.

## Setup

```bash
uv sync
cp .env.example .env   # add your API key
```

## Steps

| Script | DSPy concept |
|---|---|
| `common.py` | `dspy.Example`, `dspy.Signature` with a `Literal` output, metric, `dspy.configure` |
| `01_baseline.py` | `dspy.Predict`, `dspy.inspect_history`, `dspy.Evaluate` |
| `02_chain_of_thought.py` | `dspy.ChainOfThought` (adds a `reasoning` field) |
| `03_optimize.py` | `dspy.BootstrapFewShot` / `dspy.MIPROv2`, `save` / `load` |

```bash
uv run 01_baseline.py
```

Data: single-label articles only, sampled per topic (8 per topic for training,
up to 15 per topic for evaluation) since a uniform sample is ~80% `earn`/`acq`.
LM calls are cached by DSPy, so re-running a script is free.

Building a golden dataset is left for later.
