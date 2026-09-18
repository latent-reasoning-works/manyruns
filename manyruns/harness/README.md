# manyruns.harness

Human-in-the-loop DR workflow labeling tool. Run embeddings, inspect results, label topology, and feed data to Manyruns — all from one CLI.

## Install

```bash
uv sync --extra harness   # pulls manyLatents, numpy, matplotlib
```

## Run

```bash
# Display results (text table + save a PNG)
python -m manyruns.harness run --preset pca_umap --dataset swissroll

# Open the plot in your browser
python -m manyruns.harness run --preset pca_umap --dataset swissroll --viz browser

# Save the plot to a specific path
python -m manyruns.harness run --preset pca_umap --dataset swissroll --viz file --viz-output plot.png

# Use the real DR backend (requires manyLatents)
python -m manyruns.harness run --preset pca_umap --dataset swissroll --backend dr

# List available presets
python -m manyruns.harness presets
```

Invoke the harness as `python -m manyruns.harness <command>`. The examples below
use the short `manyruns <command>` form — that unified surface (folding the
harness under manyruns's top-level `manyruns` console script, which currently
runs `manyruns.app:main`) is the Phase-1 wiring step; until it lands, use
`python -m manyruns.harness`.

## Backends (the `run` DR path)

| Flag | Backend | Requires |
|------|---------|----------|
| `--backend mock` (default) | `MockBackend` — canned responses, no deps | nothing |
| `--backend dr` | `DRBackend` — real manyLatents execution | `uv sync --extra harness` |

## Chat — agentic REPL

`chat` (alias: `manyruns chat`) is an interactive loop where the model can
**call DR compute as a tool**, see the result, and answer — the "Hello
Manyruns, embed this and tell me the trustworthiness" experience. The agentic
loop lives in manyAgents; manyruns supplies the `run_dr_workflow` tool that wraps
the DR path. Install from PyPI with `uv sync --extra agents` in a checkout, or
`uv tool install --python 3.12 --force "manyruns[agents]"` for an installed tool.

```bash
# Local, free, on a laptop (no GPU): ollama serving Qwen
manyruns chat --agent ollama --model qwen3:30b

# Frontier model via the Anthropic API
manyruns chat --agent claude                 # uses ANTHROPIC_API_KEY (or prompts)
manyruns chat --agent claude --model claude-opus-4-8

# CI / smoke (no model, no network)
manyruns chat --agent mock
```

Then just talk to it:

```
> Run a umap_2d embedding on swissroll and tell me the trustworthiness.
🔧 run_dr_workflow(dataset='swissroll', preset='umap_2d')
↳ Ran UMAP(2) on 'swissroll'. (5000, 2). trustworthiness=0.9960
💬 Trustworthiness is 0.996 — the embedding preserves local structure very well.
> exit
```

History threads across turns; `exit`/`quit`/Ctrl-D ends the session.
`--max-steps` (default 8) bounds the tool loop per turn.

### Chat agents

| `--agent` | Backend | Requires |
|-----------|---------|----------|
| `ollama` (default) | local Qwen via ollama's OpenAI-compatible API | a running `ollama`; `uv sync --extra agents` |
| `claude` | Anthropic frontier models (`claude-opus-4-8` default) | an API key (below); `uv sync --extra agents` |
| `vllm` | GPU generation (cluster) | CUDA + `uv sync --extra agents` |
| `hf` | local Transformers on MPS | `uv sync --extra agents` |
| `mock` | canned — for CI/smoke | nothing |

### Providing an Anthropic API key

`--agent claude` needs a key. Three ways, in priority order:

```bash
# 1. Flag (one-off)
manyruns chat --agent claude --api-key sk-ant-...

# 2. Environment (persistent)
export ANTHROPIC_API_KEY=sk-ant-...
manyruns chat --agent claude

# 3. Neither → secure prompt (hidden input)
manyruns chat --agent claude
#   Anthropic API key: ********
```

The same loop runs on every backend — switching `--agent` from `ollama` to
`claude` is a config change, not a different code path. (Local Qwen is fine for
iterating on the harness; use a frontier model for real-data work, where
multi-step tool reasoning matters.)

## Prompt — single-shot

`prompt` sends one natural-language request to manyAgents and prints the reply
(no tool loop). Same `--agent` choices as chat, and the same `--api-key`
handling for `--agent claude` (flag / `ANTHROPIC_API_KEY` / secure prompt).

```bash
manyruns prompt "Summarize what a swiss roll dataset looks like" --agent ollama --model qwen3:30b
manyruns prompt "Describe a swiss roll" --agent claude   # uses ANTHROPIC_API_KEY, or prompts
```

## Stubs

`label` and `rlhf` are registered but not yet implemented (print a stub notice).

## The Backend contract

The harness routes every run through a single `Backend` interface (defined in `manyruns/harness/interface.py`). Any package — current or future — that implements `Backend.run(request: Request) -> Response` can be plugged in. `Request` carries the dataset name, resolved workflow steps, and requested metrics; `Response` carries the embedding array, metric scores, and execution metadata. The CLI never calls manyLatents, Manyruns, or any model directly — it only speaks to a `Backend`. This boundary is the seam where new inference engines, learned policies, and remote services attach without touching the CLI or labeling logic.
