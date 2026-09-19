<div align="center">

<pre>
    △  ∿  ◇  ·  ∴  ○  □         ◇
  ∿  ∴  ·  ∿  ∴  ·  ∿  ∴  ──▶ ∴ · ◈ ──▶  ◇(λ)
    ○  ∴  ·  □  ∿  ◇  △         ◈

          m a n y r u n s

  load. assume. run. repeat.
</pre>

[![license](https://img.shields.io/badge/license-MIT-F59E0B.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11–3.12-F59E0B.svg)](https://www.python.org)
[![uv](https://img.shields.io/badge/pkg-uv-F59E0B.svg)](https://docs.astral.sh/uv/)

</div>

---

**Manyruns is the harness** for the [Latent Reasoning Works](https://github.com/latent-reasoning-works)
geometry stack — the system that orchestrates a run, calls the tools, stores the artifacts, and
evaluates the result. The compute lives one layer down
([manylatents](https://github.com/latent-reasoning-works/manylatents)); this repo owns the
**workflow, the config layer, and the distribution**.

**Open beta.** The CLI and record schema can still change. Report problems in
[GitHub issues](https://github.com/latent-reasoning-works/manyruns/issues), with
`manyruns --version`, your platform, and a small reproduction.

**Supported platforms:** macOS and Linux, Python 3.11–3.12; Windows is unsupported because
a required single-cell dependency (`scikit-misc`) ships no Windows wheels.

Three ways to use it:

- **Just run `manyruns`** → an interactive app: it reads your data, tells you what it sees,
  offers the analyses it can *honestly* run, executes with a live trace, and explains the
  result in plain English. Pick a sample or drop your own files in the data folder.
  (See [the app](#the-interactive-app).)
- **Explore one dataset** → paste a path, get an embedding, a geometry read-out, and a
  written summary. `manyruns run <data>`.
- **Sweep many** → author a catalog of *recipes* (workflows) and *datasets*, run every
  recipe over every dataset, and rank which geometric metrics actually separate the
  workflows. The `experiment` harness.

New here? Start with the install below, then open `manyruns`.

Each working directory has one run ledger at `outputs/index.jsonl`, with artifacts under `outputs/<project>/`.

## install

**Quickstart — once published to PyPI.** Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv tool install --python 3.12 manyruns
manyruns
```

`uv tool upgrade manyruns` updates an installed tool in place (keeping its extras); `sh packaging/install.sh` reinstalls the base tool from scratch, which drops any extras you added.

If you do not have uv, use [uv's official installer](https://docs.astral.sh/uv/getting-started/installation/)
or follow the [setup details](#setup-details) below, then run the tool install command above.

Alternatively, install from the public repository (requires [Git](https://git-scm.com/downloads)):

```sh
uv tool install --python 3.12 --force git+https://github.com/latent-reasoning-works/manyruns
manyruns
```

### Setup details

If `uv` is absent on macOS or Linux, use its official installer:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a **new terminal** or activate the PATH printed by the installer. If installation
succeeds but `manyruns` is not found, run `uv tool update-shell` and open a new terminal.
`uv tool dir --bin` shows where the commands were installed. Installation and metric
computation time depend on your machine, data, and dependencies.

`uv tool install` creates an isolated tool environment with globally invokable commands.
`uv sync` prepares the current developer checkout; run its commands through `uv run`.
For development, use `uv sync --extra harness --extra dev`; include **all already used
extras**, such as `--extra agents`, on subsequent syncs. See [CONTRIBUTING.md](CONTRIBUTING.md).

From a checkout, `make install` installs its **committed HEAD** through a local Git URL;
uncommitted edits are excluded. `make install-public` repairs an installed tool from the
public repository. `make demo` runs swissroll with the manylatents engine and the embed
recipe. `make test` and `make lint` use the checkout's development environment.

`sh packaging/install.sh` also installs from the public repository. It bootstraps uv if
absent, then checks `manyruns --version` or prints PATH advice. No organisation access
or GitHub login is required.

**The compute is a base dependency.** The metadata requests
`manylatents-omics[singlecell]`, including scanpy, because bare omics does not supply the
single-cell runtime. Python 3.12 is recommended within the supported range.

A Git tool install resolves the declared dependencies; it does not inherit this checkout's
`tool.uv.sources` or `uv.lock`. `manyruns --version` reports the installed version and Git
revision when available; Git provenance is stored in the tool's `direct_url.json`.
The release workflow verifies the built wheel and a wheel rebuilt from the sdist in clean
environments before publishing.

| install | pulls | unlocks |
|---|---|---|
| *(base)* | phate, manylatents, manylatents-omics[singlecell], scanpy, scikit-learn, scipy, statsmodels, matplotlib, anndata, textual-image | 11 bundled recipes, the metric suite, terminal figures (`--engine manylatents`); available analyses depend on the data |
| `[harness]` | manylatents[omics], scanpy, wandb | the acquire shim + the metric sweep |
| `[agents]` | manyagents>=0.0.1, anthropic (from PyPI) | the agentic **chat / prompt** stream |
| `[dev]` | pytest, ruff | the test suite |

`[real]`, `[datasets]` and `[engine]` still parse and carry nothing — deprecated no-op
aliases. To install the harness extra from PyPI once published:

```sh
uv tool install --python 3.12 --force "manyruns[harness]"
```

The agents extra installs from PyPI too, with no repository access required:

```sh
uv tool install --python 3.12 --force "manyruns[agents]"
```

## the model

Three things, three axes — kept apart on purpose (see [CLAUDE.md](CLAUDE.md)):

- **recipe** — *what runs*: a named, ordered list of steps. Each step declares a `group`
  (`latent` embedding · `lightning` trained step · `analysis` in-process read-out) and feeds
  its embedding to the next. Recipes live in `manyruns/configs/recipe/*.yaml`.
- **dataset** — *what it runs on*: a handle (a manylatents name or a file path) plus a
  **ground-truth `shape`** (`manifold` · `clusters` · `time-course` · `case-control` · …).
  In `manyruns/configs/dataset/*.yaml`.
- **engine** — *who computes*: `manylatents` (the real DR/trajectory engine) · `mock` (fake,
  no deps, never reads your data). A backend swap for one verb — run a recipe → a **g-vector**
  (a dict of named geometric scalars). It is *not* how you add a new capability: a new tool (a
  metric, a simulation) is a new step `group`, never a new engine. Two names are gone from this
  list: `real` (it computed the same PHATE from the same PyPI package and could not run half
  the catalogue) and the learner's (manyruns holds no track to a learner — see the architecture
  below).

*Where* it runs (laptop / GPU / SLURM) is a separate concern the dispatcher owns.

## the interactive app

Run `manyruns` with no arguments to open the data roster. Pick a dataset, inspect what it
contains, then choose an analysis.

- **Drop folder.** The destination is `$MANYRUNS_DATA_DIR` when set, otherwise an
  existing `./data`, otherwise `~/.manyruns/data`. Launch creates the chosen folder.
  Without an environment override, both existing `./data` and `~/.manyruns/data` folders
  are read; the chosen destination wins a filename clash. An override reads only that folder.
- **Live roster.** Drop a file in while the app is open. The roster polls once a second and
  waits for two stable observations before inspecting a new or changed copy. `ctrl+r`
  requests an immediate rescan with the same settling rule; returning to the roster also
  refreshes it. Your filter and selection are preserved when other files arrive.
- **Samples.** No .h5ad files are bundled in Git or the wheel. Generators such as `swissroll`
  and `synthetic_timecourse` create data locally. An absent `pbmc3k` row stays visible:
  its first explicit selection offers **Fetch / Back**, with size (about 5.6 MB) and
  destination. A CLI run of the absent sample also explicitly fetches it. The download
  checks the YAML's `sha256` and byte count in a temporary file before publishing it into
  the drop folder. A mismatch refuses and removes the partial download. Later uses read
  the local file and work offline. Offline first use gives the filename, URL and destination:
  download `pbmc3k_raw.h5ad` from
  <https://exampledata.scverse.org/scanpy/pbmc3k_raw.h5ad> and place it in the chosen drop
  folder. `manyruns check` validates declarations, never downloads, and does not tell you
  whether a sample file is present. The roster marks missing samples and offers **Fetch**
  when a download is available.
- **Available analyses.** The menu offers what the data can support. A trajectory on data
  with no time axis is refused with an explanation. PCA component requests above a named
  generator's feature count are clamped to the available dimensions and reported in the run.
- **No LLM required.** Intent is rule-based; a one-call Anthropic router is an optional
  upgrade behind the `[agents]` extra.

**Repeated attempts.** Press `esc` to open **Run again?**, then choose **Run again** to
return to the ledger with the same recipe selected. Choose the analysis again to start a
fresh run. If you have an answer draft, the first Escape clears it; press Escape again to
leave. Leaving an unfinished run closes tuning: an in-flight step or measurement may finish,
but later steps and final metrics that have not started are skipped. Once that work returns,
the run is recorded as incomplete. Measurements already underway are retained.

A new attempt using the same output directory waits until the previous run finishes.
The app stays responsive and shows
`waiting for the previous run to finish · esc cancels this start`.
Escape cancels the queued start; quitting cancels it too. During normal finalization, the
run screen shows `measuring geometry… final metrics are running; duration depends on dataset and machine`
with elapsed time until the run finishes.

Prefer one command? The `run` / `check` / sweep verbs below use the same tools.

## run

```bash
manyruns run swissroll --engine manylatents --project demo --smoke
#   → outputs/demo/summary.md  +  outputs/demo/plots/*.png

manyruns run /path/to/counts.h5ad --project run1   # your file; the engine is the default
manyruns run swissroll --engine mock --project smoke             # $0, no stack
```

The data arg accepts **a path or a name** (auto-detected). The recipe is **auto-selected
from the data's shape** — a time axis → `cflows` (PHATE → MIOFlow → Granger), two conditions
→ `contrast` (separation + composition), else `embed`. Force one with `--recipe <name>`;
`--smoke` runs MIOFlow as a single-batch check instead of the 50-epoch train.

Choose a metadata column for display with repeatable `--color-by`:

```sh
manyruns run synthetic_timecourse --recipe embed --color-by timepoint
manyruns run /path/to/annotated.h5ad --recipe embed --color-by cell_type --color-by score
```

The first example generates a time course locally with a `timepoint` column; accept the
normal project-name default/prompt or add `--project NAME`. The second uses columns in your
own annotated file. Each chosen column gets its own scatter PNG. Colour choice changes
only the display, leaving analysis labels and results unchanged. Raw pbmc3k has no obs metadata
to choose from.

In the TUI, F7/F8 navigate the run's figure strip; `f` opens the full-screen viewer, where
left/right arrows navigate. Press `c` on either the run screen or the full-screen viewer
to pick a metadata column by click or Enter. Recolouring needs the dataset source and a
saved figure spec; while the run is active, both screens say
`finish this run before recolouring`. Each choice writes a new view, preserves the original,
and stays in the run's figure strip after returning from the viewer. Missing or unreadable
sources and datasets without metadata show an explanation.

Press `s` on either screen to save the selected figure, including a colour view, to
`figures/` under the directory where you launched the app. With a saved figure spec,
it exports SVG, PDF and PNG at 300 dpi. When typing an answer or an ask, letter keys stay
in the text field.

Set the recorded run seed with `--seed 7`. For a MIOFlow step, manyruns passes that seed
to `MIOFlowODEFunc(init_seed=7)` **at network construction**, to the `MIOFlow` wrapper,
and to the experiment. The wrapper preserves an existing network's weights; its seed
alone cannot initialize them retroactively. Both constructor parameters describe the same
weight-initialization choice. `params: {init_seed: 7}` is refused with directions to
`--seed`, so the seed used stays consistent with the run's lineage.

`manyruns init … && manyruns explore` splits setup from the run and can step the recipe
interactively (one action at a time through `Session.step` — the same seam a
policy will drive).

## author & check

Recipes and datasets are **directories of YAML** — the directory *is* the registry, so
adding one is adding a file. Validate before you commit:

```bash
manyruns check
#   recipes    ok    cflows     latent:phate → lightning:mioflow → analysis:granger
#   datasets   ok    swissroll  manylatents:swissroll  shape=manifold
#   coverage   WARN  contrast   no dataset can exercise it — needs a shape providing ['conditions']
```

`check` validates every file **and** reports which recipes the current datasets can actually
exercise — a recipe no dataset can run is flagged (not silently pruned at sweep time). It
exits non-zero on a real error, so CI gates on it. This whole loop is **stackless** (mock).

Keep experimental variants out of the shipped app with `$MANYRUNS_RECIPE_DIR` /
`$MANYRUNS_DATASET_DIR` overlays — the wheel ships the blessed few; your sweep's many arms
live in your own directory.

## the experiment

The harness runs every recipe over every dataset, collects the g-vectors into one table, and
ranks which metrics separate the workflows — the pruned metric set *is* the g-vector
definition, learned from data. It plans on the canonical runner, prunes cells the data can't
support, and parallelises with **processes**.

```python
from manyruns import experiment as ex
from manyruns.catalog import load_recipes, load_datasets

def main():
    runs = ex.plan(load_recipes(), load_datasets(), seeds=(1, 2))
    rows = ex.to_rows(ex.run_all(runs, suite=["anisotropy", "lid", "trustworthiness"],
                                 engine="manylatents", workers=4))
    ex.write_table(rows, "outputs/sweep/table.csv")
    print(ex.summarize(rows))

if __name__ == "__main__":            # required — the pool spawns processes
    main()
```

The sweep spec and the recipe/dataset authoring briefs are pending a rewrite; the shipped
recipes under `manyruns/configs/recipe/` are the working reference.

## architecture

```
manyruns (this repo)   the harness — distribution + workflow + the config layer it drives
    │  drives
    ▼
manylatents · manyagents · shop   DR algorithms, metrics, orchestration, cluster infra


the learner             a separate process — and it is NOT in the chain above
    ╎ observes ▶  manyruns's artifacts (index.jsonl, g-vectors, decision rows)
    ╎ drives   ▶  manyruns's CLI, as any other user would
```

**manyruns names no learner** — no dependency, no import, no `--engine` value, no
config. That is deliberate and it is the architecture: manyruns is the **environment** (what
is *legal*, and the *record* of what happened); the learner is what decides what is *good*.
The learner observes the environment and drives it from outside; the environment never learns
it is being driven, which is what keeps the record trustworthy — there is no privileged caller.

Manyruns is not "thin" — it is a **different kind of thing** from a learner. It *owns*
orchestration, the recipe/dataset catalog, provenance, and the tools that exist nowhere
downstream (`separation`, `composition`). It *delegates* the heavy compute (DR, metrics) and
does not train at all: the harness workflow is `run → label → store`, and it stops at the
store. The rule is **no *duplicated* compute and no *training*** — not "no compute"
(the base metadata includes the biology runtime). The full
framing and the layering rules are in [CLAUDE.md](CLAUDE.md).

## docs

Start with the contribution guide for development, or the environment contract for the
boundary between this application and its callers.

| | |
|---|---|
| [CLAUDE.md](CLAUDE.md) | layering rules · the engine/tool/substrate guardrail · gotchas |
| [CONTRIBUTING.md](CONTRIBUTING.md) | development, tests, and releasing |
| [CHANGELOG.md](CHANGELOG.md) | user-visible changes |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | community standards and reporting |
| [docs/environment-contract.md](docs/environment-contract.md) | the environment contract |
| [docs/sidecars/](docs/sidecars/) | out-of-process sidecar tools: the pyrovelocity brief, its fixture generator and recipe |

Source comments keep their reasoning self-contained; cited documentation paths are checked
for existence by the test suite.

## licensing and sample data

manyruns is [MIT licensed](LICENSE). Its dependencies `phate`, `graphtools`, `tasklogger`,
`leidenalg`, and `igraph` are GPL licensed; they are imported rather than bundled in the
manyruns distribution.

The `pbmc3k` sample is **10x Genomics, 3k PBMCs from a Healthy Donor**, Cell Ranger 1.1.0
(2016): [original dataset](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0).
The app fetches the [Scanpy H5AD copy](https://exampledata.scverse.org/scanpy/pbmc3k_raw.h5ad)
on explicit first use. The data is not bundled with manyruns; the MIT license covers this
project's code, not third-party data.

## status

**0.1.0 is the first public open beta.** The release is published as `manyruns` on PyPI
through GitHub Actions trusted publishing. See the [changelog](CHANGELOG.md) for changes;
CLI and record-schema compatibility are still settling during beta.

The consolidated substrate: one step loop and one g-vector schema across every engine, the
two YAML registries, `manyruns check` with coverage, the precondition calculus, and the
metric-separation harness — all runnable on the mock with no private stack, and verified
end-to-end on the real engine. Not yet done: a real `case-control` cohort (so `contrast` is
exercisable — `manyruns check` names it), the sweep on real data (the mock's numbers are
degenerate by construction), and one upstream manylatents fix (MIOFlow's `time`-key) that
lets the trajectory step drop its local shim.

---

<p align="center">
<sub>MIT License · <a href="https://github.com/latent-reasoning-works">Latent Reasoning Works</a></sub>
</p>
