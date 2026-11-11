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
| `offer(surface="ledger", *, gate=None, recipe=None, step=None, run=None)` | Returns the action document of `ledger`, `tune`, `derive` or `ladder`. Both modes return equal documents. |
| `gate` | The geometry mask. With `gate=True`, the legal set holds only the recipes that the measured geometry of the dataset supports. The evidence is the g-vector of the probe run. See [The probe run](#the-probe-run). With `gate=False`, only the data preconditions limit it. The document records `gate`, so its digest covers it. |
| `act(choice, *, extra=None)` | Runs one choice document and returns the result document `{"schema_version": 1, "decision_id", "run_id", "exit_code"}`. `exit_code` is 1 when the run itself failed; the decision row is still on disk. |
| `extra` | An optional JSON object for the run row. `act` puts it in the choice document as the key `extra`, and `run --choice` writes it as the run row's `extra` key. manyruns reads none of its keys. |
| `Refused` | The exception of a refusal. `exit_code` is 2 for a usage error or an illegal or stale choice, and 1 when the action document cannot be loaded. `message` holds the reason. Nothing was written. |
| `review(choice, *, ask, human_id)` | Shows a filler's choice to a person. `ask` is a callable that gets a copy of the choice and returns `"accept"` or a replacement `chosen`. The result is the reviewed choice: `chooser` is `human:<human_id>`, `bound` is `"none"`, and the key `proposed` holds the filler's choice. `review` checks no legality, reads no file and calls no `act`. |
| `check_proposed(proposed, offer, *, store=".", seed=42)` | Checks a filler's proposed block on its own, against the action document `offer` that it answers, before any person reviews it. It applies the same rules as `proposed` inside a reviewed choice: the keys, `chooser`, `bound`, `scores`, `meta`, `offer_sha256` equal to the digest of `offer`, and the legality of `chosen` on `offer`. It raises `Refused` with exit code 2, and writes nothing. Pass the `store` and `seed` that `offer` was built with. |

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

### The probe run

The probe run is the run whose g-vector the gate reads. It is the first row in
`outputs/index.jsonl`, in file order, that has these properties:

- It is a run of the dataset.
- Its `ok` is `true`.
- Its `cancelled` is not `true`, and its `not_run` list is empty.
- No step record has the outcome `"error"`.

A step that skipped does not disqualify a run. For example, `normalize` and `transform` skip
a file whose values are already log-normalised. That run has `complete: false`, but it is a
probe run. The gate still refuses each claim whose measurement is not in the g-vector, with
the reason `<key>:not_measured`.

manyruns looks first for a probe run at the requested seed. If there is none, it uses the
first probe run of the dataset at any seed. The document names the probe run in `evidence`:

```json
{"run_id": "aaaaaaaaaaaa", "recipe": "embed", "seed": 42, "requested_seed": 1}
```

`seed` is the seed of the probe run. `requested_seed` is present only when the probe run came
from a different seed than the request. If a probe run at the requested seed is written later,
the next document names that run, and a choice made on the older document is stale. With no
probe run at any seed, `evidence` is `null` and the gate refuses every claim that has a rule.

## The choice document

A choice document is one JSON object with seven required keys and eight optional keys:

| Key | Required | Content |
|---|---|---|
| `schema_version` | yes | `1` |
| `surface` | yes | `ledger`, `tune` or `derive`. `ladder` is read only. |
| `dataset` | yes | A catalog dataset name. |
| `offer_sha256` | yes | The digest of the action document that the policy chose on. |
| `chosen` | yes | `{"recipe": R}` on `ledger`; `{"recipe": R, "step": S, "params": P}` on `tune`; `{"connector": NAME, "params": P}` on `derive`. |
| `chooser` | yes | Who chose: `human:<pseudonym>`, `prior:<id>`, `learner:<id>` or `llm:<model>`. |
| `bound` | yes | The tier that enforced the legal set: `logits`, `schema`, `prompt` or `none`. |
| `scores` | no | The policy's ranked actions: `[{"action": {...}, "score": number}]`, finite, not increasing. |
| `meta` | no | The filler's own notes. |
| `proposed` | no | The filler's choice that a person reviewed. `review` writes it. |
| `extra` | no | A JSON object for the run row. The caller of `act` writes it. |
| `time_key` | no | The dataset column that holds the time of each cell. |
| `condition_key` | no | The dataset column that holds the condition (case/control label) of each cell. |
| `sample_key` | no | The dataset column that holds the sample (donor or replicate) of each cell. |
| `batch_key` | no | The dataset column that holds the technical batch of each cell. |

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

**The column keys.** `time_key`, `condition_key`, `sample_key` and `batch_key` sit at the top
level, next to `dataset`, because `dataset` is a catalog name (a string), not an object. Each is
a non-empty string that is a column of the dataset, and no two keys name the same column. manyruns reads the
dataset's column names to check them; a dataset whose data the engine generates has no columns,
so a choice on it takes neither key. The run uses a given key before it finds a column by name:
the time key first, then the condition key, then name detection. `sample_key` and `batch_key`
are never detected: a tool that asks for a `sample` or `batch` role (`input.obs`) is skipped
when its key is absent. The keys are not part of the
action document, so they do not change `offer_sha256`. `review` keeps them.

**Refusals.** Every check runs before the first write. An illegal, stale or malformed choice
exits 2 (`Refused.exit_code == 2`) and writes no file.

## CLI mode

| Verb | What it does |
|---|---|
| `manyruns actions DATASET --surface S [--gate on\|off] [--seed N]` | Prints the action document as one JSON object. |
| `manyruns run --choice FILE [--engine E] [--seed N] [--device D]` | Checks the choice document in `FILE`, writes the decision row, runs, and writes the run row. |

`run --choice` refuses the flags that name or change what runs (`DATA`, `--dataset`, `--data`,
`--recipe`, `--modality`, `--transform`, `--data-set`, `--time-key`, `--condition-key`,
`--sample-key`, `--batch-key`, `--color-by` and `--ask`) with exit 2. The choice document
carries the four column keys instead. It prints two lines, each at column 0:

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

A run row has the keys `time_key` and `condition_key`: the column whose values the run used as
its time labels or its condition labels. At most one is a name. Both are `null` when the run
read no such column, for example a run with `--engine mock`, which reads no data.

A run row has the keys `sample_key` and `batch_key`: the column the run read as each cell's
sample or batch, or `null` when it read none.

A run row has the keys `cancelled` and `not_run`. `cancelled` is `true` when the run was
cancelled, abandoned, or had a tune attempt discarded. `not_run` lists the declared steps that
the run never issued, each as `{"index", "name", "reason"}`. The probe rule reads both. A row
written before these keys existed has neither key and reads as not cancelled.

## Adding a tool

A tool manifest is the configuration that declares a tool's inputs, outputs and parameters.
Add the manifest and an adapter that implements those declarations. An adapter translates
the request into the tool's call and writes its result; it must not import manyruns.

| Manifest key | What to declare |
|---|---|
| `input.kind` | `state_matrix` for the current matrix, `state_frame` for the current cells and genes in an `.h5ad`, or `source_file` for the original `.h5ad`. A `source_file` tool is refused after cells have been removed; put it before the filter. |
| `input.obs` | The observation roles required by a `state_frame` tool: `sample`, `batch` or `condition`. |
| `output.tables` | The table names and required columns, with `output.schema: result-v1`. |
| `output.mask` | The observation column containing the boolean mask that selects cells to keep. A mask output cannot also declare fitted outputs or tables. |
| `declaration` | The parameter declaration checked by manykinds. Its method must match the tool name, its group must be `tool`, and manifest parameter defaults must agree with it. |

On the laptop, create the tool's dependency lock after adding its manifest. Replace `TOOL`
with the manifest name:

```sh
uv run --no-sync manyruns tools lock TOOL
```

The conformance suite in `tests/test_tool_conformance.py` is the check every new tool must
pass. It validates every bundled manifest and uses local stub adapters to check matrix,
frame, table and mask behavior after filtering. It also checks that a source-file tool runs
before filtering and is refused after filtering removes cells.

On the laptop, run the conformance suite:

```sh
uv run --no-sync pytest tests/test_tool_conformance.py -q
```

## Adding a connector

A connector is a tool that reads tables and writes tables after a finished run.
Its manifest declares non-empty `output.tables` and empty `produces`. Its adapter runs
in the tool's own locked environment and must not import manyruns.

| Manifest key | Meaning |
|---|---|
| `input.tables` | Table names mapped to required columns. The executor binds the parent's latest table of each name. `selection` is reserved for a table with only `cell` and `group` columns. |
| `input.kind: none` | Read only tables. The request has `input: null`; it still carries the parent's cell count as `n_cells`. |
| `input.gene_id: true` | On a `state_frame`, export Ensembl identifiers as `var.gene_id`. Read them from the source file, or use the prepared `gene_ids` mapping. |
| `resource` | A map from resource name to `release`, `url`, `licence`, `commercial_use`, `prepare` and optional `key`. The `key` names a step parameter, such as `disease`. |
| `network: none` | The adapter uses prepared local resources. Version 1 accepts only `none`. Downloading belongs in an explicit resource build. |

A resource build runs the prepare script with `--url`, `--release`, `--out` and an
optional `--key`. The script writes a tab-separated file. The builder saves its SHA-256
digest, which identifies the prepared bytes, beside it under
`~/.cache/manyruns/resources/`. `MANYRUNS_RESOURCE_HOME` overrides that location.
The executor refuses a missing resource or a digest mismatch and prints the build command.

On the laptop, build a keyed resource after installing its tool environment. Replace
`NAME`, `KEY` and `VALUE` with the resource name, parameter name and value:

```sh
manyruns tools build-resource NAME --param KEY=VALUE
```

The bundled connectors have these outputs:

| Connector | Output and interpretation |
|---|---|
| `pseudobulk_de` | `genes`: counts summed by sample, then a condition comparison with design `~condition`. Adjusted p-values require at least two samples per level with at least `min_cells_per_sample` selected cells. Otherwise the score is the log2 ratio of mean counts per million, with one added to each mean. For `contrast: groups`, the design is `~sample + group`, the score is the Wald statistic, and adjusted p-values are always empty because the groups came from the same data. |
| `target_evidence` | `gene_evidence`: the top `top_n` genes by absolute score joined by `gene_id` to the pinned association resource for `disease`. Association files do not supply gene symbols, so the prepare script leaves `symbol` empty. |
| `sample_score` | `sample_scores`: one row per sample. `n_cells` is the total number of cells in that sample; `fraction` is the selected count divided by that total. Genes are ranked by the mean difference in log1p counts per million between selected cells and the rest. `score` averages those genes over all cells in the sample. A Mann–Whitney U test compares fractions between two conditions, using samples as replicates. Its p-value is empty when either condition has fewer than two samples. |

`pseudobulk_de` requires unnormalized, non-negative integer counts. Its default `levels`
are `[case, control]`. For a custom pair, supply JSON text, such as
`"levels": "[\"treated\", \"baseline\"]"`. The installed parameter declaration format has
no list type. `groups: null` uses the two sorted values in the selection; a custom `groups`
pair uses the same JSON text format. The first value is the numerator of the comparison.
The selection must contain exactly two non-empty group values for a group comparison.

The gene mapping uses a dated human HGNC archive. Other organisms need identifiers already
present in the source file. Neither a derived row nor these descriptive sample scores are
probe evidence for the geometry gate.

The derive loop uses the cells recorded at a persisted parent step. By default it uses the
last embedding. Set `parent_index` to select another persisted step. The original source
file and recorded cell identifiers must still be available. A user file can be named by its
recorded source path in place of a catalog dataset name.

Each derive offer entry has `connector`, `can_run`, `blocked`, `params`, `declaration`,
`binds`, `selection`, `roles`, `role_keys` and `resources`. The `resources` list has one
item per resource in the connector's manifest, sorted by `name`. It is empty when the
manifest declares no resources. An offer reads local files and never builds a resource.

| Resource field | Meaning |
| --- | --- |
| `name`, `release` | The resource name and pinned release declared by the connector. |
| `key_param` | The manifest's `key`, which names the step parameter that selects a resource key, or `null` for a keyless resource. |
| `required` | `always`, or `if_no_gene_ids` for the `gene_ids` mapping. The executor reads that mapping only when the source has no Ensembl column. Offers retain this condition without opening the source dataset. |
| `built`, `status` | Keyless resources only. `built` is true only for `status: ready`. Other statuses are `not-built` and `mismatch`. A mismatch means that the file differs from its SHA-256 sidecar. |
| `built_keys` | Keyed resources only. A sorted list of keys whose files and sidecars resolve as `ready` in this release. |
| `build` | For a keyless resource, the exact resolver fix command. For a keyed resource, the build command with a key placeholder, such as `manyruns tools build-resource opentargets --param disease=<disease>`. Replace the placeholder with the chosen key before running it. |

A keyless resource with `required: always` and a status other than `ready` makes
`can_run` false and adds `resource_not_built:<name>:<release>` to `blocked`.
A keyed resource does not block the offer because its key is chosen later.
Choice checks merge the chosen parameters over the manifest defaults. An unavailable
key refuses the choice with exit code 2 and a message starting with
`resource_not_built:<name>:<release>:<key>`. A mismatch refuses with
`resource_mismatch:<name>:<release>:<key>`, or `resource_mismatch:<name>:<release>`
for a keyless resource. A missing required keyless resource refuses with
`resource_not_built:<name>:<release>`. Conditional `gene_ids` resources do not block
the offer or choice; the executor checks the mapping if the source needs it.

The offer digest includes all resource fields. A successful build that changes resource
readiness makes a choice based on the old offer stale. A caller can show the resource
and build command to its user for consent, build explicitly, then read a fresh offer
before choosing. It must compute `offer_sha256` from that fresh offer.

On the laptop, in the directory containing `outputs/index.jsonl`, get an offer. Replace
`DATASET` and `PARENT_RUN` with the dataset and parent run identifiers:

```sh
manyruns actions DATASET --surface derive --run PARENT_RUN > offer.json
```

Save the selected cell identifiers and group values as `selection.tsv`. Do not put expression
values in this file. On the laptop, create a choice from that offer and selection:

```sh
python - <<'PY'
import hashlib
import json
from pathlib import Path
from manyruns.action_space import offer_sha256

offer = json.loads(Path('offer.json').read_text())
choice = {
    'schema_version': 1,
    'surface': 'derive',
    'dataset': offer['dataset']['name'],
    'parent_run': offer['parent']['run_id'],
    'offer_sha256': offer_sha256(offer),
    'selection_sha256': hashlib.sha256(Path('selection.tsv').read_bytes()).hexdigest(),
    'chosen': {'connector': 'pseudobulk_de', 'params': {'contrast': 'condition'}},
    'chooser': 'human:reviewer',
    'bound': 'none',
}
Path('choice.json').write_text(json.dumps(choice, indent=2))
PY
```

If the offer has a null role key, add the corresponding column name to the choice, such as
`sample_key` or `condition_key`. On the laptop, run the checked choice:

```sh
manyruns run --choice choice.json --selection selection.tsv
```

This writes a decision with `surface: derive` and a new run with
`recipe: derive:pseudobulk_de`, joined by `decision_id`. The new run inherits the parent's
engine and seed unless `--seed` is explicit. It saves `00-derive_selection.tsv` and
`00-pseudobulk_de_genes.tsv` in its own state directory and measures no geometry.

To chain evidence, get another offer using the derived run as `PARENT_RUN`. Choose
`target_evidence` with a built `disease` resource. Omit `selection_sha256` and `--selection`
because that connector reads the parent's `genes` table. The chain retains the original
cell identities. `as_tools` and `choice_from_tool_call` support only `ledger` and `tune`.

Dependency locks are platform-specific. Before installing a connector, create its lock with
`manyruns tools lock TOOL` on each supported platform. These three connectors do not yet
ship locks. The tests use explicit interpreter overrides and do not verify locked installation.

## Who is a harness

Anything that drives manyruns from outside, through the CLI or as a library, like any user.
manyruns names no harness and no learner: no dependency, no import, no engine value, no
config and no CLI path. The arrow points from the harness to manyruns only; manyruns defines
what is legal and records what happened, and the harness decides what to do. See
[the environment contract](environment-contract.md), §3.5.
