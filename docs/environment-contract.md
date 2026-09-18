---
note:        Spec
title:       The environment contract, and where the learner ends
subtitle:    manyruns decides what is LEGAL and records what HAPPENED; the learner decides what is GOOD and learns from it. Most of that line is already drawn in the code and nowhere written down — so a decision row cannot say who chose it or under what constraint, and the first model tier to ship will make human and machine choices indistinguishable in the one file the moat is built on.
author:      César M. Valdez
date:        2026-08-19
status:      Spec — NOTHING IMPLEMENTED. §2 is a three-field change to `decisions.append`;
             §3 is a rule a reviewer can apply, not code. §3.5 was added 2026-08-19 and REVERSES
             this document's own §5, which is struck in place rather than edited away.
status_kind: open
---

> **Sourcing.** Every `file:line` re-derived by grep in this checkout on 2026-08-19 at `7d22cd5`.
> **Measured** means a command was run here and its output read; **Reasoned** is labelled where
> used. The learner is NOT installed in this checkout (its module lookup returned `None`), which is
> itself one of the findings rather than a gap in the method — see §3.1. Learner identifiers
> in this historical spec are now described by role.

## 0 · Why this exists

Two questions arrived together and turned out to be one:

1. What must a decision record carry to be worth training on later?
2. Where does manyruns end and the learner begin?

They are the same question because **the contract IS the boundary, stated in terms of what a row
must carry to cross it.** A row that cannot say who chose it cannot be pooled with rows chosen by
someone else, and pooling is the only thing the closed side does that the open side cannot.

## 1 · The finding that forces it

`decisions.append` (`decisions.py:56`) takes `offered, chosen, dataset, shape, topology, surface,
out_dir` and writes `decision_id, at, surface, dataset, shape, topology, offered, chosen`.

**Measured: there is no field for who chose, and none for how the choice was constrained.**

Today that is harmless — every row is a human at a prompt. It stops being harmless the moment a
model tier writes rows, which `manyruns/agents.py` and `tune.llm_override` now make possible:

- human and model choices become indistinguishable in one file;
- the learner trains on that file;
- the policy is trained partly on its own outputs, labelled as human choices.

That fails **silently**. No error, no bad run — a finetune that underperforms for reasons nobody
can reconstruct, because the contamination is not recoverable after the fact. The rows do not
carry the distinction, so it cannot be filtered later.

**This is the cheapest problem in this document and the only one that is not reversible.** Ship a
model tier without it and every row it writes degrades the asset.

## 2 · The decision-row contract

Three fields on every row.

### 2.1 `chooser` — who made this choice

`"human"`, or an identifier for the policy that did (`"llm:claude-opus-4-8"`,
`"learner:policy@<sha>"`, `"prior:suits"`). Not a boolean: "was it a model" is the question
today, "which model, which version" is the question the first time two policies disagree.

**A row whose chooser is a policy is not training data for that policy.** That is the whole
purpose of the field, and a trainer that cannot see it cannot enforce it.

### 2.2 `bound` — how the choice was constrained

Which controller tier actually enforced the task space:

| value | meaning | strength |
|---|---|---|
| `logits` | the sampler could not emit an illegal token | hard, ours |
| `schema` | a forced tool call over an enum, enforced by the provider | strong, borrowed |
| `prompt` | the legal set was described, not enforced | weak |
| `none` | a human chose from a rendered list | n/a — the UI was the bound |

This is not bookkeeping. Measured this week: manyruns sent a JSON Schema enum to its preferred
backend and the schema was **silently dropped in translation**, so a call the code called
"constrained" was free text. Nothing failed, because a caller-side check rejected bad answers. A
row that records `schema` when the schema never arrived is a lie the corpus cannot detect.

So `bound` must be reported by **the code path that actually enforced it**, never assumed from
configuration — and a backend that cannot confirm its tier reports `prompt`.

### 2.3 `legal_id` — which task space this choice was made in

A hash of the sorted legal set (`vocab.unmet`-derived, `vocab.py:949`) plus the vocabulary version.

**Rows are only comparable if they were choices over the same kind of set, computed the same way.**
A Qwen-era row and a Claude-era row from different vocabularies are not two samples of one
decision; they are samples of two. Without an identity for the space, a corpus silently pools
them and the learner cannot tell.

`spec_id` already does exactly this for the analysis half (`runner.identity`), which is the
precedent: hash the thing that must be equal for two records to be about the same question.

### 2.4 What this buys, and it is the argument for doing it first

**It makes the provider question reversible.** Ship the frontier tier now, keep its rows separable
by `chooser` and honest by `bound`, and swap in a local model later without having poisoned what
was collected. Without these fields the provider decision is a one-way door: rows written before
the swap are unusable and unidentifiable.

Everything else in the LLM work — the router, capability probing, Qwen — is downstream of this and
can then be decided with data instead of guesses.

## 3 · Where the learner ends

### 3.1 The line

**manyruns decides what is LEGAL and records what HAPPENED. The learner decides what is GOOD and
learns from it.**

| | manyruns — open, observable | the learner — closed, the moat |
|---|---|---|
| task space (`vocab.unmet`, `vocab.py:949`) | **yes** | — |
| step executors, dispatch | yes | — |
| controller / bound enforcement | yes | — |
| **the checker** | **yes** | — |
| corpus WRITER (`decisions.append`) | yes | — |
| hand-written priors (`suits:`) | yes, inspectable | — |
| **reward — what counts as good** | **never** | yes |
| trainer, pooled corpus | — | yes |
| learned policy | — | yes — and it DRIVES manyruns, never the reverse (§3.5) |

**Measured, the line is already drawn in code and nowhere stated:** every import of the learner in
manyruns is lazy and inside a function (`serving.py:223`, `train.py:62`, `:70`, `:95`), and
The learner is not installed in this checkout while the suite is green. manyruns already runs
standalone. This section names a property the code has, so that it stops being an accident.

### 3.2 Two placements that are load-bearing and not obvious

**The checker is open.** It is the environment's contract, not the learner's advantage. A closed
checker would mean a user cannot verify that the product refuses correctly — and refusal-with-a-
reason is what this product actually promises. It must also validate *any* policy's answer
identically, including the learner's own, which it cannot do from behind a boundary the open side
cannot see.

**The task space is open, and that is what makes a corpus poolable at all.** Put `unmet` behind
the boundary and every provider or vocabulary change silently forks the corpus (§2.3).

### 3.3 The moat is the reward, not the weights

The g-vector is an **observation**, computed openly, and that is fine. What is secret is what
counts as a good outcome — the cost of the wrong route, silent-failure and regret over data-level
nulls, rather than agreement with a label.

Weights can be distilled. A reward definition backed by a corpus of real analyst choices cannot be
guessed.

**The rule this yields, which a reviewer can apply without judgement:** manyruns may carry
hand-written, inspectable priors; anything LEARNED lives in the learner. The moment manyruns
auto-tunes toward a metric it has absorbed the learner and leaked the moat into the public
repository.

### 3.4 The pattern is already the product's shape, three times over

An open rule tier that always works, plus an optional smarter tier behind a boundary:

- `narrate.interpret` → `intent` Tier B (one model call)
- rule-based `parse_overrides` → `tune.llm_override` (one model call)
- `suits:` (`tui/state.py:275`, "advice, never legality") → a learned policy (the learner)

The learner tier is the third instance, not a new architecture. Which is the strongest argument
that this boundary is right: it is the one the codebase keeps choosing on its own.

### 3.5 The direction: the learner observes manyruns, never the reverse

**manyruns has NO dependency on a learner, and no track to it. None.** Not an import, not a
protocol it implements, not an entry point, not a `Learner` interface, not an optional extra.
manyruns does not know the learner exists.

The learner **observes manyruns's full state and drives it through the CLI.** It runs the binary,
reads the artifacts, and issues the next command. In RL terms manyruns is the environment and
the learner is the agent — and an environment that imports its agent is not an environment.

**This supersedes the "injectable harness" shape and this document's own §5.** An injection seam is
still a track: it puts a learner-shaped hole in the open repository, versions an interface across
two repos, and tells any reader what the closed side expects. Struck rather than edited, because a
reader should see that the weaker design was considered:

> ~~"A learned policy is served, never shipped: manyruns calls out and degrades to `suits:` when
> the call is unavailable."~~ — manyruns calling out is the backwards direction. It never calls.

The interface is the one manyruns already has and already tests: **its CLI is the action space,
and its artifacts are the observation.** `outputs/<project>/` — `project.yaml`, `state/<run_id>/`,
`plots/`, `index.jsonl`, `decisions.jsonl` — is the full state, on disk, in formats this repository
documents. Nothing new has to be built for the learner to see everything.

**What this buys, and it is why the rule is absolute rather than a preference:**

* manyruns is genuinely standalone open source, with no vestigial hooks whose purpose a reader
  cannot infer from the open repository alone;
* there is no cross-repo protocol to version — the learner can drive any manyruns old enough to have
  the flags it uses, and a manyruns release cannot break a contract it does not know it has;
* the moat is entirely on the closed side. Nothing about it is inferable from what ships.

**Measured cost, stated rather than discovered later.** The rule is violated today:

| track | what it is |
|---|---|
| `app.ENGINES` carries a learner engine | manyruns calling the learner to run a recipe |
| `serving.LocalServer.SERVES` carries a learner backend | the backend for it |
| `manyruns/train.py`, 97 lines | a passthrough to the learner's run API |
| `harness/trainer.py` | drives the learner through that seam |
| `[engine]` extra names the learner | packaging |
| 20 test files name the learner | — |

All of it is the old direction: manyruns reaching into the learner. Under this rule it goes, and the
harness's workflow stops at **store** — `run → label → store` — with the learner picking up from the
store. That is the shape already stated elsewhere ("the learner would pick up env traces, keylog them,
train on them"), so `train.py` is a legacy of the direction this section reverses, not a feature
lost to it.

**A rented frontier model is NOT a track and stays.** `agents.py` calling Anthropic is manyruns
using a commodity, not manyruns reaching for the moat. The distinction is exact: a RENTED policy
may live in manyruns because it is not an asset; a LEARNED one may not, because the learner would
have to be reachable for it to run.

**One consequence for §2.** If the learner drives the CLI, manyruns cannot know who is driving — so
`chooser` (§2.1) must be **passed in**, not detected. The environment records what it is told and
labels it as told; attribution is the driver's responsibility, which is the only place it can
honestly live.

### 3.6 The soul is in the open repository

`manyruns#12` names it: *"data provenance + tagging is manyruns's soul, and a real part of our
moat."* Both halves are true and they sit on opposite sides of this boundary.

The lineage, the record, the refusal with its reason — those are manyruns's, inspectable, and they
are what makes a trace worth pooling at all. **The learner is not the soul. It is what learns from
one.** The moat is not that the honest part is hidden; it is that the honest part produces
something only the closed side can pool and score.

That is also the strongest argument for §3.5's absoluteness. A soul with a hook in it for someone
else's learner is not observable in the way this product promises.

## 4 · The trap this makes visible

`suits:` is manyruns's prior and it **contaminates its own corpus**. `decisions.py:133-139`
records the measurement: the ledger puts the recommendation first and the cursor starts there, so
`enter` selects it — and the first real decision the store recorded, from driving the shipped
binary, was `chosen == "embed"` with `embed` recommended, out of six legal moves.

> "If every row looks like that, the corpus teaches `narrate.offer`'s prior with a human's name on
> it, and a selector trained to high accuracy on it has learned to agree with the default."

`took_default` (`decisions.py:157`) carries the confound. **It is necessary and not sufficient**: a
corpus that is 90 % defaults has almost no signal however well it is labelled.

So the learner's first question is not "can we train a selector". It is: **do we have rows where a
human overrode the prior, and what do they have in common?**

**Reasoned, not measured, and worth testing early:** that makes the TUNE loop more valuable than
recipe selection. Tuning is where people disagree with the default repeatedly and say why by trying
something else — three attempts and an accept is a preference ordering, where a recipe pick is one
bit that is usually the default.

## 5 · What crosses, and when

The payload is the **state summary** — legal moves plus g-vector plus the three fields of §2 —
never the data. That is the federated shape already committed to: computation local, only geometry
and steps leave.

~~A learned policy is served, never shipped … manyruns calls out and degrades to `suits:`.~~
**STRUCK — see §3.5.** manyruns never calls out to the learner, because a call is a track. A learned
policy runs on the closed side and DRIVES manyruns through its CLI; manyruns's default is
`suits:` not as a fallback from a failed call, but because it is the only chooser it has ever
heard of.

The payload framing survives the correction and is unchanged: what leaves a machine is the state
summary, never the data.

**A rented frontier policy is strategically neutral** — it neither builds nor leaks the moat — so
it can live openly in manyruns with no tension. That is why it is safe to ship first, and it is an
argument about strategy rather than about engineering.

Any crossing is also a governance event (#12): a corpus leaving a machine carries what a user did
with their data, and consent for that is not implied by installing the product.

## 6 · Acceptance

1. Every row written by `decisions.append` carries `chooser`, `bound` and `legal_id`.
2. `bound` is reported by the code path that enforced it. A backend that cannot confirm its tier
   reports `prompt`, and a test proves a dropped schema cannot be recorded as `schema`.
3. Rows written by a policy are excluded from that policy's training set by construction —
   asserted in `decisions.examples`, not by convention.
4. Two rows from different vocabularies do not pool: `legal_id` differs, and a test shows it.
5. manyruns's suite stays green with the learner absent. No forbidden name may appear in
   any tracked file, including prose; the public-safety guard compares token hashes and never
   reports candidate text. §3.5 makes that the rule. The engine, serving and training seams
   listed above describe the historical state before their removal.
6. No function in manyruns computes a reward, asserted the only way a rule like this can be:
   named in `CLAUDE.md` so a reviewer can apply it.

## 7 · Open questions, and the measurement each needs

1. **Is `chooser` one field or two?** A model tier invoked by a human who then accepted its
   suggestion is neither purely human nor purely machine. Measure: how often does the tune loop's
   accept follow a model-proposed param, once the tier ships?
2. **Does `bound: schema` mean anything we can verify from inside manyruns?** Today, no — §2.2's
   defect was only visible by executing the provider's translator. Measure: can a backend report
   its own tier honestly, or must the seam probe it per call?
3. **Does removing the learner engine cost a user anything they have?** §3.5 deletes it as a
   track. Measure: how many recorded runs used it, and does the harness's `train` step have a
   caller outside the learner's own workflow?
4. **What is the minimum override rate for the corpus to be worth training on?** §4 says defaults
   dominate. Measure it on the rows that exist before building anything that consumes them.
