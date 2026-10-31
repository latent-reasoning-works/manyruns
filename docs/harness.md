# Driving manyruns from a harness

A harness is any program that decides what manyruns runs next: a learner, a lab controller, a
script, or a person at a review screen. manyruns names none of them. This page states the one
way that every harness uses manyruns, in two modes: through the CLI or as a library.

The labeling package `manyruns.harness` is a different thing with a similar name. This page is
not about it.

## The intended use

Every harness follows the same four steps:

1. Get the action document: the legal action space of one dataset on one surface.
2. Choose: a policy reads the action document and returns a choice document.
3. Let manyruns run the choice. manyruns checks it first and refuses an illegal or stale choice
   before it writes anything.
4. Let manyruns' own code write the record.

Step 4 is the rule. A harness never writes `outputs/index.jsonl` or `outputs/decisions.jsonl`
itself. manyruns writes the decision row before the run, and puts its `decision_id` on the run
row.

A policy is any object with `name`, `chooser()` and
`choose(action_document, state, history) -> choice document`. It never decides legality and
never writes a record.

## The action space wrapper

`manyruns.action_space.ActionSpace` gives both modes one interface:

```python
from manyruns.action_space import ActionSpace
space = ActionSpace("swissroll", mode="library", store=".", engine="mock")
doc = space.offer("ledger", gate=True)
result = space.act(policy.choose(doc, state, history))
```

| Part | What it does |
|---|---|
| `ActionSpace(dataset, *, mode="library", store=".", engine=None, seed=42, device=None)` | `mode` is `"library"` (manyruns runs in this process) or `"cli"` (manyruns runs as `python -m manyruns` in a child process, with the store as its working directory). `store` is the folder whose `outputs/` holds the record; it must exist. |
| `offer(surface="ledger", *, gate=True, recipe=None, step=None)` | Returns the action document of `ledger`, `tune` or `ladder`. Both modes return equal documents. |
| `gate` | The geometry mask. With `gate=True`, the legal set holds only the recipes that the measured geometry of the dataset supports. The evidence is the probe run: the first complete run of the dataset and seed in the store. With `gate=False`, only the data preconditions limit it. The document records `gate`, so its digest covers it. |
| `act(choice, *, extra=None)` | Runs one choice document and returns the result document `{"schema_version": 1, "decision_id", "run_id", "exit_code"}`. `exit_code` is 1 when the run itself failed; the decision row is still on disk. |
| `extra` | An optional JSON object for the run row. `act` puts it in the choice document as the key `extra`, and `run --choice` writes it as the run row's `extra` key. manyruns reads none of its keys. |
| `Refused` | The exception of a refusal. `exit_code` is 2 for a usage error or an illegal or stale choice, and 1 when the action document cannot be loaded. `message` holds the reason. Nothing was written. |
| `review(choice, *, ask, human_id)` | Shows a filler's choice to a person. `ask` is a callable that gets a copy of the choice and returns `"accept"` or a replacement `chosen`. The result is the reviewed choice: `chooser` is `human:<human_id>`, `bound` is `"none"`, and the key `proposed` holds the filler's choice. `review` checks no legality, reads no file and calls no `act`. |

**Model fillers.** `as_tools(doc)` returns the legal actions of one action document as tool
schemas: one `choose_recipe` tool on `ledger`, and one `tune_<step>` tool per tunable step on
`tune`, each with the floor, the cap and the domain of its knobs as bounds. Each tool is a dict
with the keys `name`, `description` and `parameters`, without a callable; the filler adds its
own. `choice_from_tool_call(doc, name, arguments, *, chooser, bound)` turns one tool call into
a choice document. A model filler names itself with the chooser `llm:<model>`.

The provider, the API keys and the retries belong to the agent package that runs the filler.
manyruns reads no key and calls no model.

`act` changes the working directory of the whole process in library mode, so a threaded
caller must not call `act` from two threads at once. In CLI mode the child runs `app.main`,
which loads `<store>/.env`; library mode does not. Keep `.env` out of a store when the two
modes must agree.

## The choice document

A choice document is one JSON object with seven required keys and four optional keys:

| Key | Required | Content |
|---|---|---|
| `schema_version` | yes | `1` |
| `surface` | yes | `ledger` or `tune`. `ladder` is read only. |
| `dataset` | yes | A catalog dataset name. |
| `offer_sha256` | yes | The digest of the action document that the policy chose on. |
| `chosen` | yes | `{"recipe": R}` on `ledger`; `{"recipe": R, "step": S, "params": P}` on `tune`. |
| `chooser` | yes | Who chose: `human:<pseudonym>`, `prior:<id>`, `learner:<id>` or `llm:<model>`. |
| `bound` | yes | The tier that enforced the legal set: `logits`, `schema`, `prompt` or `none`. |
| `scores` | no | The policy's ranked actions: `[{"action": {...}, "score": number}]`, finite, not increasing. |
| `meta` | no | The filler's own notes. |
| `proposed` | no | The filler's choice that a person reviewed. `review` writes it. |
| `extra` | no | A JSON object for the run row. The caller of `act` writes it. |

`offer_sha256` is the SHA-256 hex digest of
`json.dumps(doc, sort_keys=True, separators=(",", ":"))`, UTF-8
(`manyruns.action_space.offer_sha256`). manyruns recomputes the action document with the gate
on and with the gate off, and judges the choice against the document whose digest matches. A
choice that matches no document is stale, and is refused.

**The legal set.** On `ledger`, the legal set is the recipes of the offer whose `can_run` is
true. On `tune`, the recipe must be legal on the ledger under the same gate, the step must be a
tunable step of the recipe, and each value may be anywhere inside the knob's floor, cap and
domain. A non-finite value, a value of the wrong kind (a float for an integer knob) and a value
outside the domain are refused.

**The proposed block.** A reviewed choice has `proposed` with the keys `chooser`, `bound`,
`chosen` and `offer_sha256`, and optionally `scores` and `meta`. Its own `chooser` must start
with `human:`. `proposed.offer_sha256` must equal `offer_sha256`, and `proposed.chosen` must
be legal on the same document as `chosen`. On the decision row, an accept is
`row["chosen"] == row["proposed"]["chosen"]`, and an override is
`row["chosen"] != row["proposed"]["chosen"]`. No row stores which one it is.

**The meta rule.** `meta` is an object. Each key is a string of 1 to 64 characters. Each value
is `null`, a boolean, an integer, a finite number, a string, or a list of at most 16 such
values. Its compact JSON is at most `META_MAX_BYTES = 2048` bytes. manyruns checks it and does
not interpret it. The top-level `scores` and `meta` are checked and not written; inside
`proposed` they are written on the decision row.

**The extra rule.** `extra` is an object. Each top-level key is a string of 1 to 64 characters
and is not `verdict` or `proposer`, because manyruns' own readers read those two keys of a run
row's `extra`. The values are any JSON with finite numbers. Its compact JSON is at most
`EXTRA_MAX_BYTES = 8192` bytes.

**Refusals.** Every check runs before the first write. An illegal, stale or malformed choice
exits 2 (`Refused.exit_code == 2`) and writes no file.

## CLI mode

| Verb | What it does |
|---|---|
| `manyruns actions DATASET --surface S [--gate on\|off] [--seed N]` | Prints the action document as one JSON object. |
| `manyruns run --choice FILE [--engine E] [--seed N] [--device D]` | Checks the choice document in `FILE`, writes the decision row, runs, and writes the run row. |

`run --choice` refuses the flags that name or change what runs (`DATA`, `--dataset`, `--data`,
`--recipe`, `--modality`, `--transform`, `--data-set`, `--time-key`, `--color-by` and `--ask`)
with exit 2. It prints two lines, each at column 0:

- `decision <decision_id> recorded in <path>`, before the run starts;
- `run <run_id> recorded in <index>`, after it.

A worked example:

```
manyruns actions swissroll > offer.json
# the policy reads offer.json and writes choice.json, with offer_sha256 of offer.json
manyruns run --choice choice.json --engine mock
# stdout: "decision <id> recorded in ..." then "run <id> recorded in ..."
```

`python -m manyruns` is the same program as `manyruns`.

## Library mode

Import these. Other names may change without notice.

| Name | What it is |
|---|---|
| `manyruns.action_space.ActionSpace` | The wrapper: `offer` and `act`, in either mode. |
| `manyruns.action_space.offer_sha256` | The digest of an action document. |
| `manyruns.action_space.check_choice` | Judges a choice document; returns the matched action document. |
| `manyruns.action_space.Refused` | The exception of a refusal. |
| `manyruns.action_space.as_tools` | The legal actions of a document as tool schemas. |
| `manyruns.action_space.choice_from_tool_call` | One tool call as a choice document. |
| `manyruns.action_space.review` | A person's review of a filler's choice. |
| `manyruns.actions.enumerate_actions` | The action document of one surface. |
| `manyruns.decisions.BOUNDS` | The tiers that can enforce a legal set. |
| `manyruns.session.Session` | The step-by-step session of the interactive loop. |
| `manyruns.tui.state.catalog_entry` | One dataset declaration, resolved without reading the data. |
| `manyruns.tui.state.ledger` | The ledger rows of an observation. |
| `manyruns.tui.state.LedgerRow` | One ledger row; `as_offer()` is the offer entry. |
| `manyruns.narrate.blocked` | Why a recipe cannot run on this data. |
| `manyruns.narrate.unavailable` | Why a recipe cannot run anywhere now, whatever the data. |
| `manyruns.catalog.load_recipe` | One recipe by name. |
| `manyruns.catalog.load_dataset` | One dataset declaration by name. |
| `manyruns.store.append` | The run-row writer (manyruns calls it; a harness does not). |
| `manyruns.store.read` | The run rows of a store. |
| `manyruns.decisions.append` | The decision-row writer (manyruns calls it; a harness does not). |
| `manyruns.decisions.read` | The decision rows of a store. |
| `manyruns.decisions.examples` | The decision rows as training pairs. |
| `manyruns.score.RUNG` | The rung of each structure claim on the ladder. |
| `manyruns.score.recipe_rung` | The rung that a recipe claims. |

The first seven names are the interface of a harness. The other names stay public for a harness
that needs the lower parts.

## The record

The record is two files under `<store>/outputs/`, one JSON object per line:
`index.jsonl` (one run row per run) and `decisions.jsonl` (one decision row per committed
choice). A reader skips a line that does not parse, as `decisions.read` does.

The join key is `decision_id`: the decision row is written before the run, so the run row
names its decision, and the decision row's `run_id` is `null`.

A decision row that `run --choice` wrote has the key `gate` after `chosen`, and the key
`proposed` after `gate` when the choice was reviewed. A row without `gate` reads as missing,
not as `false`. A row without `proposed` reads as missing, not as an accept. An accept is
`row["chosen"] == row["proposed"]["chosen"]`. Inside `proposed`, `chosen` has the same form as
the row's `chosen`: the recipe on `ledger`, and the attempt label (for example
`phate knn=7 n_components=3`) on `tune`.

A run row has the key `extra` when the choice carried one, and manyruns does not read its keys.

## Who is a harness

Anything that drives manyruns from outside, through the CLI or as a library, like any user.
manyruns names no harness and no learner: no dependency, no import, no engine value, no
config and no CLI path. The arrow points from the harness to manyruns only; manyruns defines
what is legal and records what happened, and the harness decides what to do. See
[the environment contract](environment-contract.md), §3.5.
