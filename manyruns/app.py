"""Manyruns application entry point — the Milestone 1 product flow.

Implements the seam of #9-14. Engine work is delegated (the learner ->
manylatents/manyagents); this file is the product flow + the serving seam.

  #9  package & distribute        → this installed console-script (`manyruns`)
  #10 accept a data folder        → detect bulk vs scRNA-seq, clear error on bad input
  #11 minimal project setup       → `manyruns init`: project name + data path, nothing else
  #12 auto-exploration            → `manyruns explore`: run the default recipe via serving
  #13 aggregate → data summary    → one markdown file at a predictable path
  #14 sample run on EB data       → the milestone acceptance test

Commands:
  manyruns init <folder> --project NAME    # #11 — set up a project, nothing else
  manyruns explore --project NAME          # #12/#13 — run the recipe, write the summary
  manyruns run <folder> --project NAME      # init + explore in one go (the default)
  manyruns open --project NAME             # reopen a saved project, no data source needed
  manyruns projects                        # list saved projects (outputs/*/project.yaml)

A bare `manyruns <folder> --project NAME` still works — it maps to `run`.
"""
from __future__ import annotations

import argparse
import contextlib
import os
from manyruns import env as _env
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Set before ANYTHING else in the process can import matplotlib — Lightning/scanpy/manylatents
# all import it for their own logging well before a recipe reaches its own plotting step, and
# whichever import happens first locks in the backend. A GUI-capable backend claimed that way
# has no event loop running under a CLI/REPL process, so `fig.savefig(...)` in
# `pipeline/io.py::_save_scatter` hangs rather than errors — measured on a real `cflows` run
# where the step finished (PHATE's own convergence log printed) but no plot ever appeared and
# the REPL never returned to its prompt. `setdefault` so an operator who wants a different
# backend (e.g. driving manyruns from a notebook) isn't overridden.
os.environ.setdefault("MPLBACKEND", "Agg")

from manyruns.serving import ModelServer, default_server

# #10 — minimal modality detection by folder contents
_SCRNA_SUFFIXES = {".h5ad", ".mtx", ".loom"}
_BULK_SUFFIXES = {".csv", ".tsv", ".txt"}

# The product owns the recipe choice (which named recipe runs for which modality);
# the recipe *content* lives in manyruns/configs/recipe/, the *compute* downstream.
# bulk maps to cflows for Milestone 1 — it should get its own recipe downstream.
# (`_RECIPE_FOR_MODALITY` lived here. It mapped scrna/bulk → cflows and was consulted only
#  when there was NO time axis and NO conditions — i.e. exactly when a trajectory is least
#  defensible. `_choose_recipe` now derives from `narrate.offer`, so modality no longer
#  selects a recipe and the table had no remaining reader.)

_SUBCOMMANDS = ("init", "explore", "run", "open", "projects", "check", "audit",
                "tools", "shell")


def _load_dotenv() -> None:
    """Best-effort: load a local .env so ANTHROPIC_API_KEY / WANDB_API_KEY are
    available without a manual export. Only the working directory is read; ancestor
    files are never searched. No-op if python-dotenv isn't installed."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    except Exception:  # noqa: BLE001 - dotenv is optional
        pass


def _normalize_argv(argv: Optional[list[str]]) -> list[str]:
    """`manyruns` (no args) → the interactive shell; `manyruns <folder> ...` → `run`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return ["shell"]
    if argv[0] in (*_SUBCOMMANDS, "-h", "--help", "--version"):
        return argv
    return ["run", *argv]


def _installed_build() -> str:
    """Installed provenance, including a git tool install's direct_url.json.

    The working tree may be unrelated to the installed tool. Do not substitute its HEAD
    for absent distribution provenance, or a version command would identify the wrong build.
    """
    import json
    from importlib import metadata

    try:
        dist = metadata.distribution("manyruns")
    except metadata.PackageNotFoundError:
        return "version unavailable · revision unavailable"
    revision = None
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        revision = direct.get("vcs_info", {}).get("commit_id")
    except (ValueError, TypeError, AttributeError, OSError):
        pass  # A wheel or damaged optional provenance must still report its version.
    return f"{dist.version} · revision {revision if revision else 'unavailable'}"


def _build_parser() -> argparse.ArgumentParser:
    from manyruns import commands

    parser = argparse.ArgumentParser(
        # NOT pinned. argparse defaults `prog` to `basename(sys.argv[0])`, so each installed
        # console-script reports the name it was actually invoked as — `manyruns` and the
        # `co-science` alias are one program, and a pinned `prog` would make `co-science
        # --help` print usage for a command the reader did not type.
        description="manyruns — geometric exploration of your data.",
        epilog=commands.render_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_installed_build()}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="set up a project, then step through it interactively")
    _add_setup_args(p_init)
    p_init.add_argument(
        "--batch", action="store_true", help="run the whole recipe at once instead of the REPL"
    )
    p_init.set_defaults(func=cmd_init)

    p_explore = sub.add_parser("explore", help="run the default recipe on an initialized project")
    p_explore.add_argument("--project", help="project name (locates outputs/<slug>/project.yaml)")
    _add_engine_args(p_explore)
    p_explore.set_defaults(func=cmd_explore)

    p_run = sub.add_parser("run", help="init + explore in one go (the default)")
    _add_setup_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_open = sub.add_parser(
        "open", help="reopen a saved project and step through it interactively — no data "
        "source needed, it's already on file"
    )
    p_open.add_argument("--project", help="project name (locates outputs/<slug>/project.yaml)")
    _add_engine_args(p_open)
    p_open.set_defaults(func=cmd_open)

    p_projects = sub.add_parser("projects", help="list saved projects (outputs/*/project.yaml)")
    p_projects.set_defaults(func=cmd_projects)

    p_check = sub.add_parser(
        "check", help="validate every recipe and dataset (run this before you commit)"
    )
    p_check.add_argument("--dataset-dir", help="validate datasets from this directory instead")
    p_check.set_defaults(func=cmd_check)

    p_audit = sub.add_parser(
        "audit", help="run the admission gate over the declared metric suite (non-zero on reject)"
    )
    p_audit.add_argument("--metrics", nargs="*", help="check these names instead of the suite")
    p_audit.set_defaults(func=cmd_audit)

    # `tools` is the ONLY place a tool environment is built, and that is the design rather than
    # ergonomics: a realized env is 2.4 GB (measured), so a recipe step must never create one. A
    # run REFUSES a missing tool and prints the command below; the operator runs it once.
    p_tools = sub.add_parser(
        "tools", help="external tools: what is declared, what is built, and how to build it"
    )
    p_tools.add_argument("action", nargs="?", default="list",
                         choices=("list", "install", "check"),
                         help="list: what is declared and its state · install: build a locked "
                              "environment · check: run each built tool's selftest")
    p_tools.add_argument("name", nargs="?", help="the tool (required for install)")
    p_tools.set_defaults(func=cmd_tools)

    p_shell = sub.add_parser(
        "shell", help="the interactive front door (the default when run with no arguments)"
    )
    _add_engine_args(p_shell)
    p_shell.set_defaults(func=cmd_shell)

    return parser


def _add_setup_args(p: argparse.ArgumentParser) -> None:
    """Args shared by `init` and `run` — the data source + engine selection."""
    p.add_argument(
        "data_folder",
        nargs="?",
        metavar="DATA",
        help="data source: a file/folder PATH or a dataset NAME (auto-detected) (#10)",
    )
    p.add_argument("--project", help="project name (#11). Default: derived from the data source.")
    p.add_argument("--dataset", help="alias for the DATA arg (a dataset name, or a path)")
    # `--data` has to be declared explicitly. Without it argparse sees only the *prefix*
    # `--data` and reports `ambiguous option: could match --dataset, --data-set` — which is
    # the first thing anyone types, and it reads like the flag doesn't exist. An exact
    # match wins over prefix matching, so declaring it resolves the ambiguity.
    p.add_argument("--data", dest="data_arg", metavar="PATH",
                   help="same as the positional DATA (explicit form)")
    p.add_argument("--modality", help="override modality for a named dataset (scrna/bulk)")
    p.add_argument(
        "--recipe",
        help=f"force a recipe, skipping auto-selection. Available: {', '.join(_recipe_names())}. "
        "Default: auto-selected from the data shape (time axis → cflows, conditions → contrast).",
    )
    _add_engine_args(p)


def _recipe_names() -> list[str]:
    """Recipes actually available, for `--help`. Discovered, not hardcoded — the list used to
    be a stale literal that omitted `contrast`. Never raises: `--help` must always render."""
    try:
        from manyruns.catalog import discover_recipes

        return discover_recipes() or ["(none found)"]
    except Exception:  # noqa: BLE001 - a broken registry must not break --help
        return ["(unavailable)"]


#: The three serving backends, in the order a human should consider them, each with the
#: module that has to import for it to work and what it actually does — in the user's terms,
#: not the architecture's. ONE home: the CLI's `--engine` choices and the shell's picker both
#: read this, so the list cannot drift between the two surfaces.
ENGINES: tuple[tuple[str, str, Optional[str]], ...] = (
    ("manylatents", "run it on the full engine (needs the private stack)", "manylatents"),
    ("mock", "demo only — invents numbers and never reads your data", None),
)

ENGINE_NAMES = tuple(name for name, _, _ in ENGINES)

#: Backends that exist for DEVELOPMENT and must never be offered to, or selected for, a user.
#: They stay in the package and leave the product surface: never offered by the picker, never
#: returned as a default, absent from `--help`. Passing one explicitly still works.
#:
#: `mock` invents a g-vector and never opens the file, which is exactly what makes it useful
#: for tests — CLAUDE.md requires it so the flow and the suite run with no private stack — and
#: exactly what makes it dangerous in a research flow: it produces a full-looking panel, a
#: finding, and a complete g-vector that is indistinguishable at a glance from a measured one.
#: A run on it is marked invented in its own record (see `_dev_engine_caveat`).
#:
#: `real` USED TO BE THE OTHER ONE, and it is now gone from this list because it is gone from
#: `ENGINES`: `--engine real` exits 2. The reason is not tidiness. It computed latent steps
#: in-process with public libraries and implemented two of them — `phate` and `mioflow`.
#: Measured against the bundled catalogue on this tree: `pca`, `umap`, `leiden`, `diffusionmap`
#: and `aa` every one reported "no in-process implementation", so several bundled recipes
#: (cluster, markers, archetypes, pseudotime) could not run. It existed because manylatents was
#: optional; it was the answer to "what can you do with no engine installed". There is no such
#: install now (manylatents is in `[project.dependencies]`), so offering it meant offering a
#: backend that silently could not run several bundled recipes while `manylatents`, present
#: in every install, runs all of it. It also selected NO distinct dependency: every library its executors
#: imported (phate, numpy, scipy, scikit-learn, matplotlib) is a hard requirement of manylatents.
#:
#: Its step loop survives, and NOT as a hidden engine — it is `runner.run_inproc` under
#: `vocab.INPROC`, which this module deliberately never mentions again. See that constant for
#: why it is kept: it is the only step loop a torch-free CI can run.
DEV_ENGINES = frozenset({"mock"})


def engine_available(name: str) -> bool:
    """Is this backend actually runnable here? (Its import target resolves.)"""
    import importlib.util

    for engine, _, module in ENGINES:
        if engine == name:
            return module is None or importlib.util.find_spec(module) is not None
    return False


def _default_engine() -> Optional[str]:
    """The best backend actually available here, or None when there is none.

    **A dev engine is never returned.** `mock` used to be the final fallback, so a base
    install — no extras — answered a real cohort with invented numbers and no indication
    that is what had happened.

    This used to be `manylatents if installed else mock`, which meant that on any install
    without the private stack — the shipped, documented configuration — pointing manyruns
    at a real cohort produced *invented numbers*, silently. It then fell back to `real`,
    which needed only public libraries.

    `real` IS NO LONGER A FALLBACK, because there is nothing left to fall back from:
    manylatents is a hard dependency, so it is present in every install. Falling back to an
    engine that cannot run several bundled recipes (measured: archetypes, cluster, markers,
    pseudotime all skip a latent step for want of an implementation) would trade a loud
    absence for a quiet incapacity. (the learner's engine was excluded from auto-selection here too, for
    needing engine-side config; it is not a backend at all now — §3.5.)

    None still means none — the caller says what to install rather than inventing numbers."""
    if engine_available("manylatents"):
        return "manylatents"
    # No compute backend. Returning `mock` here is what "silently invents numbers" looks like
    # in one line, so this returns None and the caller says what to install. A front door that
    # answers with fabricated geometry is worse than one that refuses.
    return None


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    """Backend selection + engine passthrough — shared by `init`/`run`/`explore`."""
    p.add_argument(
        "--engine",
        choices=ENGINE_NAMES,
        default=None,
        help="serving backend (default: the best one installed). "
        # dev engines are omitted: `--engine mock` still parses, so tests and development are
        # unaffected, but it is not advertised as something to reach for.
        + "; ".join(f"{n}: {d}" for n, d, _ in ENGINES if n not in DEV_ENGINES),
    )
    p.add_argument("--color-by", action="append", default=None, metavar="COLUMN",
                   help="display metadata column; repeat for one figure per column")
    p.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="compute device (default: auto). Resolved against the recipe: a float64 recipe "
        "(MIOFlow) never lands on MPS — it falls back to cpu/cuda with a note.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="random seed for the manylatents engine (default 42)"
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="manylatents engine: full training for lightning steps (now the default; kept "
        "for back-compat)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="manylatents engine: fast_dev_run plumbing check (single batch) instead of the "
        "default 50-epoch training",
    )
    p.add_argument(
        "--data-set",
        dest="data_kwargs",
        action="append",
        metavar="KEY=VALUE",
        help="dataset-generation kwarg for the manylatents engine (repeatable), "
        "e.g. --data-set gap_multiplier=2.5",
    )
    p.add_argument(
        "--time-key",
        help="AnnData .obs column with per-cell timepoints — MIOFlow's real time label "
        "(auto-detected from timepoint/day/sample/… if omitted); else pseudotime is used",
    )
    p.add_argument(
        "--transform",
        choices=("log1p", "sqrt"),
        default=None,
        help="SHIM: overrides the `transform` step's `method` param (sqrt = PHATE/MIOFlow EB "
             "convention). The transform is a declared recipe step now; this flag survives so "
             "existing commands keep working and is DELETED when the general "
             "`--set <step>.<param>=<value>` lands (backlog item 3).",
    )


def _wants_the_app(argv: list[str]) -> bool:
    """A bare `manyruns` at a real terminal — the Textual app (surface three of three).

    BARE, so the decision cannot be reached any other way: `manyruns run`, `manyruns shell`,
    `manyruns <folder>` and every flag keep the exact path they were on. `shell` in particular
    stays the rich console it names, which is what makes it the escape hatch when the app is not
    what you want.

    BOTH streams, where `_wants_the_console` below asks only about stdin, and the asymmetry is
    measured rather than stylistic. The rich shell writes through a `Console` that degrades on
    its own when its output is a pipe; a Textual app does not degrade at all. Measured on textual
    8.2.8 by running `ManyrunsApp().run()` with stdout redirected to a file: the redirect
    received **0 bytes** and the app painted anyway — 41,923 bytes, 1,937 escapes, `\\x1b[?1049h`
    and a full roster frame — into **stderr**, because `drivers/linux_driver.py` writes to
    `sys.__stderr__` unconditionally (line 58). So `manyruns | cat`, which is a TTY on stdin and
    a pipe on stdout, would hand the pipe an empty stream while taking over the user's screen.
    Checking stdout as well is what makes that impossible.
    """
    return not argv and sys.stdin.isatty() and sys.stdout.isatty()


def main(argv: Optional[list[str]] = None) -> int:
    _load_dotenv()
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(raw))
    # THE FRONT DOOR, and the one line where the three surfaces meet. argv is NOT rewritten:
    # `_normalize_argv([])` is still `["shell"]`, so the app is handed the same namespace the
    # rich console would have been handed for the same invocation (a bare one, so: the parser's
    # defaults). Only the function that receives it changes. Routing through a new subcommand
    # instead would have put a second name for the front door into `--help`.
    func = cmd_app if _wants_the_app(raw) else args.func
    try:
        return func(args, parser)
    except ColorSelectionError as e:
        parser.error(str(e))
    except (RuntimeError, ValueError) as e:
        # RuntimeError: e.g. the learner's engine without the private stack.
        # ValueError is this codebase's shape for "what you asked for is not something I can
        # do", and those messages are written FOR the user — `catalog` lists the known names,
        # the loader names the file and its byte count, `vocab` names the legal topologies.
        # Catching only RuntimeError meant every one of them reached the user as a traceback
        # through three libraries, with the sentence someone wrote for them on the last line.
        #
        # The cost, stated plainly: a ValueError from a genuine bug now prints one line
        # instead of a stack. `MANYRUNS_TRACEBACK=1` gets it back, and is what to reach for
        # when a message does not read like it was meant for you.
        if _env.is_set("TRACEBACK"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_init(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """#11 — minimal setup, then step through the exploration interactively (the reasoning
    trace: `phate` → `MIIOFlow`). `--batch` runs it all at once."""
    proj = _setup_project(args, parser)
    print(f"project initialized: {proj['path']}")
    if getattr(args, "batch", False):
        return _explore_project(
            proj, seed=args.seed, fast_dev_run=_smoke(args),
            data_kwargs=_parse_kwargs(args.data_kwargs),
            time_key=args.time_key, device=args.device, color_by=getattr(args, "color_by", None),
        )
    # Every engine with a step loop is steppable. The predicate here used to be
    # `in ("mock", "manylatents")` — the set from before `mock` left the product surface, and
    # the INVERSE of `_default_engine`'s preference order. Measured consequence: on a public
    # install (`real` is the default) `manyruns init` printed a refusal and exited 0, so the
    # interactive mode did not exist on the shipped configuration; on a private-stack install
    # it existed, was the default, and was the degraded loop. `runner.steppable` is the honest
    # question, and it has exactly one False answer — the learner's engine, which ran a whole recipe in
    # one downstream call and reports no per-step outcome to step through.
    from manyruns.pipeline import runner as _runner

    if _runner.steppable(proj.get("engine", "mock")):
        return interactive_session(_build_session(proj, args))
    print(f"(engine={proj.get('engine')} runs a whole recipe in one call and reports no "
          f"per-step outcome — use `manyruns run` or --batch)")
    return 0


def _align_resumed_inputs(loaded: dict, resume: dict) -> tuple[dict, list[str], bool]:
    """Join reloaded row channels and resumed facts to the artifact's sample axis.

    Each completed slot may come from a different run. The embedding establishes
    the resumed axis; pseudotime must join it independently by its own manifest IDs.
    Incompatible nameless slots are omitted; verified source-to-base joins stay strict.
    Legacy positional joins remain usable but never acquire a claim of verified IDs.
    """
    import numpy as np

    arrays = resume["arrays"]
    slots = [s for s in ("emb", "pseudotime") if s in arrays]
    if not slots:
        return loaded, [], True
    identities = resume.get("row_identity", {})
    notes = []

    def ids_for(slot):
        ids = (identities.get(slot) or {}).get("sample_ids")
        if ids is None:
            return None
        ids = np.asarray(ids, dtype=str)
        if ids.ndim != 1 or len(ids) != len(arrays[slot]) or len(set(ids)) != len(ids):
            raise ValueError(f"cannot resume {slot}: artifact sample IDs must be unique and match its rows")
        return ids

    def positions(source_ids, wanted, context):
        source_ids = np.asarray(source_ids, dtype=str)
        if source_ids.ndim != 1 or len(set(source_ids)) != len(source_ids):
            raise ValueError(f"cannot resume {context}: sample IDs must be unique")
        lookup = {value: i for i, value in enumerate(source_ids)}
        if any(value not in lookup for value in wanted):
            raise ValueError(f"cannot resume {context}: missing sample IDs; restore the original source or start a fresh run")
        return np.asarray([lookup[value] for value in wanted], dtype=int)

    def omit(slot, note):
        notes.append(f"omitted resumed {slot}: {note}")
        for key in ("arrays", "origin", "index", "row_identity"):
            resume.get(key, {}).pop(slot, None)
        resume["steps"] = [entry for entry in resume.get("steps", ()) if entry[1] != slot]

    # A filter can change the row count without leaving identities to join on.
    # Keep the project usable from its source, without guessing which rows survived.
    while slots:
        base = slots[0]
        if (ids_for(base) is not None or loaded.get("array") is None
                or loaded["array"].shape[0] == len(arrays[base])):
            break
        omit(base, "artifact has no sample IDs and its row count differs from the source; "
             "starting fresh from the source when no compatible slots remain")
        slots.pop(0)
    if not slots:
        return loaded, notes, True

    base = slots[0]
    wanted = ids_for(base)
    unverified = [base] if wanted is None else []
    loaded = dict(loaded)
    if wanted is not None:
        source_ids = loaded.get("obs_names")
        if source_ids is None:
            raise ValueError("cannot resume: source has no unique sample IDs to align with the artifact; start a fresh run")
        rows = positions(source_ids, wanted, "source")
        for key in ("array", "counts", "labels", "obs_names"):
            if loaded.get(key) is not None:
                loaded[key] = loaded[key][rows]
        loaded["layers"] = {k: v[rows] for k, v in (loaded.get("layers") or {}).items()}
        loaded["color"] = [{**c, "values": c["values"][rows]} for c in loaded.get("color") or ()]
    for slot in slots[1:]:
        try:
            ids = ids_for(slot)
            if wanted is not None and ids is not None:
                arrays[slot] = arrays[slot][positions(ids, wanted, slot)]
            elif len(arrays[slot]) != len(arrays[base]):
                raise ValueError("legacy artifact row counts differ")
        except ValueError as exc:
            omit(slot, f"incompatible with resumed {base} ({exc})")
            continue
        if ids is None:
            unverified.append(slot)
    notes.extend(f"cannot verify resumed {slot}: positional alignment is unverified; "
                 "source metadata will not be attached to derived coordinates"
                 for slot in unverified)
    return loaded, notes, not unverified


def _build_session(args_proj: dict, args: argparse.Namespace, *, resume: bool = True):
    """Build a `Session` from a project, loading the data every real engine needs.

    Also the resume point: if `out_dir/state/` holds a completed run from a PREVIOUS process
    (`session.resumable`), its last embedding/pseudotime are seeded into this session's state
    before step 0 — the gap `cmd_open`'s docstring flagged (a fresh session starts empty even
    when the project's prior run computed a real, possibly-stochastic embedding already sitting
    on disk). No prompt: `open` reopening the SAME project is "continue where I left off," not
    a new analysis, so silently doing that is the useful default rather than friction to opt
    into each time.

    `resume=False` IS FOR A CALLER WHOSE VERB IS NOT "CONTINUE", and there is exactly one:
    `tui/app.py`'s stepped front door. Its gesture is "pick data, pick a recipe, run" — a fresh
    analysis every time — where "continue where I left off" is a different verb (`open`). Two
    things break without the opt-out, and the second is the serious one:

      * the two front doors record different lineage for the same gesture. `explore_once` never
        resumes and writes `parent: None`; this would stamp a `parent` addressing a previous
        process's run.
      * `run_tune_loop` snapshots `baseline_state` at entry and `_reset()` restores it on
        cancel, so CANCELLING the embedding step would leave the RESUMED embedding sitting in
        `state` — and `session.close()` → `_finalize` then reports `n_embedded`, `final_dim` and
        the whole metric suite over an embedding this run never computed.

    Additive and keyword-only, so every existing CLI caller (`cmd_init`, `cmd_open`) keeps the
    resume it was written for without naming it."""
    from manyruns.session import Session, resumable

    engine = args_proj.get("engine", "mock")
    dataset = args_proj.get("dataset")
    data_folder = Path(args_proj["data_folder"]) if args_proj.get("data_folder") else None
    out_dir = Path("outputs") / _slug(args_proj["project"])

    data_kwargs, declared_shape = _dataset_contract(
        args_proj.get("dataset_name"), dataset,
        {**(args_proj.get("data_kwargs") or {}), **_parse_kwargs(getattr(args, "data_kwargs", None))})
    array, labels, label_kind = None, None, None
    _loaded = dict(_EMPTY_INPUTS)
    # `real` is loaded here too, and that is the other half of deleting the gate: it was
    # excluded because it could not be stepped, so leaving the load `manylatents`-only would
    # have opened a `real` prompt over `state["X"] = None` — every step declining for want of
    # data. `_load_inputs` is the one loader (`_explore_project` uses it as well); a fourth
    # copy here is how the CLI path once loaded the array and dropped the labels.
    if engine == "manylatents" and (dataset or data_folder is not None):
        if data_folder is not None:
            print(f"loading {data_folder} …")
        _loaded = _load_inputs(engine, data_folder, dataset, getattr(args, "time_key", None),
                               declared_layers(args_proj.get("recipe")),
                               **({"color_by": args.color_by} if getattr(args, "color_by", None) else {}))
        array, labels, label_kind = _loaded["array"], _loaded["labels"], _loaded["kind"]
    # The REPL never reaches `run_explorations` — the session drives the executors directly —
    # so the refusal has to be made here too, before the prompt opens. `runner._ml_lightning`
    # reads `label_kind` the same way (`labels if label_kind == "time" else None`), so a named
    # time-course dataset would hand MIOFlow `labels=None` exactly as the CLI path did.
    _require_declared_facts(engine, dataset, labels, label_kind)

    from manyruns import catalog

    # `resume and resumable(...)`, not `resumable(...)` then a guard on each of the three uses:
    # one `None` here makes the whole block below — the embedding, the `parent`, the
    # pseudotime seed and `session.resumed` — degrade together, which is the invariant. A
    # caller opting out must get a session indistinguishable from one whose `state/` is empty.
    resume = resumable(out_dir) if resume else None
    resume_notes = []
    resume_aligned = True
    if resume:
        _loaded, resume_notes, resume_aligned = _align_resumed_inputs(_loaded, resume)
        array, labels, label_kind = _loaded["array"], _loaded["labels"], _loaded["kind"]
    resumed_emb = resume["arrays"].get("emb") if resume else None
    # `parent` addresses (run_id, index) of the STEP whose embedding this is — exactly
    # `Session.branch`'s convention for "state that arrived from outside this lineage," which
    # is what a resumed session's start state actually is (two processes, not one lineage).
    # Keyed off `emb` specifically, same as `branch`: a run resumed with pseudotime only has no
    # embedding-producing step to name as parent. `origin["emb"]`, NOT `resume["run_id"]`:
    # `resumable` now folds over every completed run (not just the newest), so the embedding
    # and, say, a pseudotime can legitimately come from two DIFFERENT runs — `run_id` names
    # only the latest completion overall, `origin[slot]` names where that slot's value
    # actually came from, which is the run `parent` has to address.
    resumed_parent = (
        {"run_id": resume["origin"]["emb"], "index": resume["index"]["emb"]}
        if resume and "emb" in resume["arrays"] else None
    )

    _require_source_shape(engine, args_proj["recipe"], dataset, array, declared_shape, resumed_emb)
    session = Session(
        project=args_proj["project"],
        engine=engine,
        out_dir=out_dir,
        seed=args.seed,
        fast_dev_run=_smoke(args),
        modality=args_proj["modality"],
        recipe=args_proj["recipe"],
        array=array,
        labels=labels,
        counts=_loaded["counts"], genes=_loaded["genes"], layers=_loaded["layers"],
        color=_loaded.get("color"), sample_ids=_loaded.get("obs_names"), label_key=_loaded.get("label_key"),
        label_kind=label_kind,   # so a stepped manylatents run won't read a condition axis as time
        dataset=dataset,
        dataset_name=args_proj.get("dataset_name"),
        data_kwargs=data_kwargs,
        declared_shape=declared_shape,
        device=getattr(args, "device", None) or "cpu",
        # WHICH metrics matter is manyruns's call and computing them is the engine's
        # (CLAUDE.md). `_explore_project` has always forwarded this; the session never did.
        metrics=catalog.load_suite(),
        embedding=resumed_emb,
        parent=resumed_parent,
    )
    # `_new_state` has no `pseudotime=` parameter (only `emb` — see its docstring on why an
    # embedding needed one and pseudotime, so far, has had no caller that needed it as an
    # INPUT), so this slot is seeded directly rather than threaded through the constructor.
    if resume and "pseudotime" in resume["arrays"]:
        session.state["pseudotime"] = resume["arrays"]["pseudotime"]
    session.resumed = resume
    # Missing identities are unverified, not evidence of incompatible engine rows.
    session.ctx["metadata_aligned"] = True if resume_aligned else None
    if resume_notes:
        import warnings

        session.ctx["caveats"].extend(resume_notes)
        for note in resume_notes:
            warnings.warn(note, UserWarning, stacklevel=2)
    session.source = (dataset and f"dataset:{dataset}") or (data_folder or args_proj["project"])
    return session


def interactive_session(session, read=input, write=print) -> int:
    """The `manyruns>` REPL: issue one step at a time (phate / MIIOFlow / separation …).

    Commands + aliases come from `manyruns.commands` — the same list `--help` shows — so
    `resolve` maps any casing/alias (MIIOFlow, ?, q) to a canonical
    command. `read`/`write` are injectable so the loop is testable without a real terminal.

    A step line — `phate` or `phate knn=40` — doesn't just run once: it accepts/retries/
    cancels through `tune.run_tune_loop` before control returns here, so a rejected attempt
    never silently becomes "what ran." `accepted` tracks only the steps actually kept (not
    `session.performed()`, which would include every rejected retry too), purely so `quit`
    can offer to save them as a recipe."""
    from manyruns import commands, tune
    from manyruns.session import available_actions

    write(f"✓ initialized — modality {session.modality}, engine {session.engine}")
    if session.engine == "mock":
        write("  (mock engine: fast fake numbers, no plots — restart with "
              "`--engine manylatents` for real PHATE/MIOFlow + plots)")
    resumed = getattr(session, "resumed", None)
    if resumed:
        produced = ", ".join(f"{step}→{slot}" for step, slot, _ in resumed["steps"])
        write(f"  resumed from run {resumed['run_id'][:8]}: {produced} "
              "(a step that recomputes it overwrites this)")
    # What THIS session can dispatch, not a fixed four: a step whose group has no executor on
    # this engine would otherwise be offered and then decline.
    offered = ", ".join(available_actions(session.engine)) or "(none on this engine)"
    write(f"  steps: {offered}   meta: run · summary · trace · help · quit"
          "   (type `help` for the full alias list, or `<step> key=value ...` to override params)")
    declared = [s["name"] for s in (session.recipe.get("steps") or [])]
    accepted: list[dict] = []
    retried = False
    while True:
        try:
            line = read("manyruns> ").strip()
        except (EOFError, OSError, KeyboardInterrupt):  # Ctrl-D/Ctrl-C or no tty (tests)
            break
        if not line:
            continue
        first = line.split()[0]
        cmd = commands.resolve(first)
        name = cmd.name if cmd else None
        if name == "quit":
            break
        if name == "help":
            write(commands.render_help())
            continue
        if name == "trace":
            write(f"trace: {' → '.join(session.trace) if session.trace else '(none)'}")
            for step_name, st in session.status.items():
                write(f"  {step_name}: {st}")
            write(f"g-vector: {session.g}")
            continue
        if name == "run":
            for r in session.run_recipe():
                write(_fmt_step(r))
            continue
        if name == "summary":
            path = write_summary(
                session.project, getattr(session, "source", session.project),
                session.modality, session.results(), session.recipe,
            )
            write(f"✓ summary written: {path}")
            continue
        # a step action (phate/mioflow/separation), possibly carrying `key=value` overrides
        step = tune.parse_step_line(line, session.engine, recipe=session.recipe)
        if step is None:
            offered = ", ".join(available_actions(session.engine)) or "(none on this engine)"
            write(f"unknown action {first!r}; try {offered}")
            continue
        write(f"  … running {step['name']}")
        before = len(session.steps)
        accepted_step = tune.run_tune_loop(session, step, read, write)
        retried = retried or len(session.steps) - before > 1
        if accepted_step is None:
            continue
        accepted.append(accepted_step)
        if declared and accepted_step["name"] in declared:
            pos = declared.index(accepted_step["name"])
            if pos + 1 < len(declared):
                write(f"  (next: {declared[pos + 1]})")
    if retried and accepted:
        dest = read("save the accepted steps as a recipe? (path or blank to skip) > ").strip()
        # A bare `if dest:` took ANY non-blank answer as a filename — typing `no` wrote a
        # recipe named `no` to the cwd (measured: `no`/`save` both landed there this way,
        # neither an intended save). Blank and a declining word both mean "skip"; anything
        # else is a path, exactly as the prompt says.
        if dest and dest.lower() not in ("no", "n", "none", "skip"):
            import yaml

            recipe_name = Path(dest).stem
            with open(dest, "w") as f:
                yaml.safe_dump({"name": recipe_name, "steps": accepted}, f, sort_keys=False)
            write(f"✓ saved: {dest}")
    # Close the lineage on the way out. `finish_carry` is the only writer of `COMPLETE`, so
    # without this a session's state folder stays indistinguishable from an interrupted one
    # forever — and `artifacts.complete()` is the gate a resume menu has to read.
    results = session.close()
    # ...and then record it, for the same reason `shell._record` exists: `store.append` had
    # ONE product caller (`_explore_project`, the `run` / `init --batch` path), so an
    # interactive lineage left `state/<run_id>/` on disk with no index row pointing at it,
    # addressable only by a run id nobody ever saw. Both interactive doors write the row now.
    #
    # Gated on a lineage that actually ran a step: a prompt opened and closed without one
    # wrote no state, and `_finalize` calls a zero-step run `ok` (every step succeeded,
    # vacuously) — a row saying that is a record of nothing dressed as a success.
    if session.steps:
        from manyruns import store as _store

        try:
            index = _store.append(results, extra={
                "source": str(getattr(session, "source", session.project)),
                "modality": session.modality})
            write(f"run {results.get('run_id')} recorded in {index}")
        except OSError as e:  # noqa: BLE001 - a read-only cwd must not lose a finished session
            write(f"(run not recorded: {e})")
    return 0


def _fmt_step(r: dict) -> str:
    if r.get("ok"):
        return f"✓ {r['name']}: {r.get('detail', 'ok')}"
    return f"✗ {r.get('name', '?')}: {r.get('error', 'failed')}"


def cmd_shell(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """The interactive front door: read → offer → run → narrate, looped. See manyruns/shell.py.

    What a bare `manyruns` reaches when it is NOT at a terminal (a pipe, a script, CI), and what
    an explicit `manyruns shell` always reaches. Rich/questionary UI; rule-based intent, with an
    optional one-call LLM router when the `[agents]` extra + a key are present."""
    from manyruns import shell

    return shell.run_shell(args)


def cmd_app(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """The Textual app — what a bare `manyruns` reaches at a real terminal.

    Imported INSIDE the function, which is the same discipline every other command here follows
    and matters more for this one: `textual.app` costs 0.21 s to import and no other entry point
    needs it, so `manyruns run`,
    `manyruns check` and `--help` never pay it.

    THE FALLBACK IS NARROW ON PURPOSE. `textual` is a declared core dependency, so a supported
    install has it; the one case this catches is a checkout whose venv predates that declaration,
    where the alternative is a `ModuleNotFoundError` traceback in place of the front door. It
    catches `ImportError` only — a textual that imports and then misbehaves still raises, because
    a silent downgrade to the surface being replaced is how a broken app goes unnoticed. The
    reason is printed, not swallowed.
    """
    try:
        from manyruns.tui.app import run
    except ImportError as e:
        from manyruns import shell
        from manyruns.installation import ENVIRONMENT_REPAIR

        print(f"note: the interactive app is unavailable ({e}) — falling back to the console. "
              f"To restore it, {ENVIRONMENT_REPAIR}.", file=sys.stderr)
        return shell.run_shell(args)
    return run(args)


def _saved_project(
    args: argparse.Namespace, parser: argparse.ArgumentParser,
) -> tuple[str, dict]:
    """The saved project a command names, with the CLI's recipe overrides folded in.

    THE SECOND (and last) place a project dict is born on the command line. `explore` and `open`
    run a recipe they did not choose — it comes off `project.yaml` — so `--transform` has to be
    applied here, exactly as `_setup_project` applies it to the recipe `init`/`run` select. Two
    producers, two applications, and no consumer (`_explore_project`, `_build_session`) needs to
    know the flag exists. The alternative was a `transform=` parameter forwarded from four call
    sites, which is precisely how the flag came to be dropped on all four.

    Shared by both commands because they had the same four-line preamble copied out. `open`'s
    extra `--engine` override stays at its own call site: `cmd_explore` carries no such line, so
    folding it in here would change what `explore` does with a flag — a separate question from
    the one this function was factored out to answer.
    """
    project = args.project or input("project name: ").strip()
    proj = load_project(project)
    if proj is None:
        parser.error(f"no project found for '{project}' — run `manyruns init` first")
    saved_source = proj.get("data_folder")
    if proj.get("dataset_name") and saved_source and not Path(saved_source).exists():
        _fetch_cli_source(proj["dataset_name"], parser)
        from manyruns.shell import _resolve_dataset

        folder, dataset, _, _ = _resolve_dataset(proj["dataset_name"])
        proj = {**proj, "data_folder": str(folder) if folder is not None else None,
                "dataset": dataset}
    # Only when the flag was actually typed: a project.yaml that somehow carries no `recipe`
    # must keep failing where it failed before, not inside a shim.
    if getattr(args, "transform", None):
        proj = {**proj, "recipe": apply_transform_shim(proj["recipe"], args.transform)}
    return project, proj


def cmd_explore(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """#12/#13 — run the recipe on an already-initialized project, write the summary."""
    _project, proj = _saved_project(args, parser)
    return _explore_project(
        proj, seed=args.seed,
        fast_dev_run=_smoke(args), data_kwargs=_parse_kwargs(args.data_kwargs),
        time_key=args.time_key, device=args.device, color_by=getattr(args, "color_by", None),
    )


def cmd_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """init + explore in one go (reuses the in-process project; no file round-trip).

    A bare `manyruns run` at a terminal opens the interactive console instead. `run` is
    the verb people reach for, and routing it to a one-shot meant the console — where you
    choose the data, the question and how to run it — was reachable only by *omitting* the
    verb, which nobody guesses. Anything that says what to do (a path, `--dataset`,
    `--engine`, `--recipe`, …) keeps the one-shot, and so does a pipe or a script, so
    automation is unaffected."""
    if _wants_the_console(args):
        from manyruns import shell

        return shell.run_shell(args)
    proj = _setup_project(args, parser)
    return _explore_project(
        proj, seed=args.seed,
        fast_dev_run=_smoke(args), data_kwargs=_parse_kwargs(args.data_kwargs),
        time_key=args.time_key, device=args.device, color_by=getattr(args, "color_by", None),
    )


def cmd_open(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Reopen a project `init`/`run` already saved and step through it interactively.

    The whole point: the data source, modality and recipe are already on file in
    `project.yaml` (`write_project`) — `open` needs only the NAME, unlike `init`/`run` which
    both demand a data source. Mirrors the interactive half of `cmd_init` (`_build_session` +
    `interactive_session`), but never calls `_setup_project`/`write_project`, so reopening the
    same project twice never rewrites it — the file `init` wrote is the one `open` reads."""
    project, proj = _saved_project(args, parser)
    # `--engine` is offered on every command that takes `_add_engine_args`, `open` included —
    # honour it as an override of the saved engine rather than silently ignoring it, which is
    # what would happen if `_build_session` were left to read only `proj["engine"]`.
    if getattr(args, "engine", None):
        proj = {**proj, "engine": args.engine}
    from manyruns.pipeline import runner as _runner

    # Same predicate `cmd_init` gates on, for the same reason: the learner's engine ran a whole recipe
    # in one downstream call and reports no per-step outcome, so there is no REPL to open.
    if not _runner.steppable(proj.get("engine", "mock")):
        print(f"(engine={proj.get('engine')} runs a whole recipe in one call and reports no "
              f"per-step outcome — use `manyruns explore --project {project!r}` instead)")
        return 0
    # `load_project` returns the bare yaml contents (no `path` key — that's `_setup_project`'s
    # own return shape, not `project.yaml`'s), so the path is re-derived the same way
    # `load_project` located the file, from the project name.
    path = Path("outputs") / _slug(project) / "project.yaml"
    print(f"project opened: {path}")
    return interactive_session(_build_session(proj, args))


def cmd_projects(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """List every saved project — the roster `open`/`explore` read by name.

    Read-only: parses each `project.yaml` and prints it, never builds a `Session` and never
    touches the data source it names."""
    from omegaconf import OmegaConf

    root = Path("outputs")
    paths = sorted(root.glob("*/project.yaml")) if root.is_dir() else []
    if not paths:
        print(f"no saved projects ({root}/*/project.yaml not found — run `manyruns init` first)")
        return 0
    for path in paths:
        proj = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        source = (proj.get("dataset") and f"dataset:{proj['dataset']}") or proj.get("data_folder") or "?"
        recipe = (proj.get("recipe") or {}).get("name", "?")
        print(f"{proj.get('project', path.parent.name):<24} {source:<40} "
              f"engine={proj.get('engine', '?'):<12} recipe={recipe}")
    return 0


def cmd_audit(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """The admission gate, as a CI surface. Exits non-zero when a tool is REJECTED.

    Rejected means the tool returned a confident number for structurally wrong input —
    measured at ~16 of 38 callables across the registry, which is why running everything
    without this gate produces plausible nonsense at scale. `flagged` (unstable, or blind to
    the difference between structure and noise) is reported and does NOT fail the build: a
    number you know not to compare is still usable, as long as you were told."""
    from manyruns import admit as _admit

    verdicts = _admit.admit_metrics(getattr(args, "metrics", None) or None)
    if not verdicts:
        print("no metric suite declared — nothing to audit")
        return 0
    mark = {"admitted": "ok  ", "flagged": "WARN", "rejected": "FAIL", "deferred": "  ? "}
    for v in verdicts:
        checks = " ".join(f"{k}={'y' if p else 'n'}" for k, p in v.checks.items())
        print(f"{mark[v.verdict]} {v.name:<28} {checks}")
        for note in v.notes:
            print(f"       {note}")
    counts = {k: sum(1 for v in verdicts if v.verdict == k) for k in _admit.VERDICTS}
    print(f"\n{counts['admitted']} admitted · {counts['flagged']} flagged · "
          f"{counts['rejected']} rejected · {counts['deferred']} deferred")
    return 1 if counts["rejected"] else 0


def cmd_check(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Validate every recipe and dataset. The self-check for authors.

    Catches what a typo actually costs: a step group outside the vocabulary, a dataset shape
    that would invent a spurious class in the results table, a name that disagrees with its
    filename. Exits non-zero if anything is invalid, so CI can run it too."""
    from manyruns import catalog, vocab
    from manyruns.vocab import unmet

    bad = 0
    ddir = getattr(args, "dataset_dir", None)
    recipes: list[dict] = []   # the valid ones, for the coverage pass
    datasets: list[dict] = []

    print("recipes")
    names = catalog.discover_recipes()
    if not names:
        print("  (none found)")
    for n in names:
        try:
            cfg = catalog._read(catalog.recipe_dir(), n, "recipe", names)
            problems = catalog.check_recipe(cfg, n)
        except Exception as e:  # noqa: BLE001 - an unreadable file is one problem, not a crash
            cfg, problems = {}, [f"{type(e).__name__}: {e}"]
        if problems:
            bad += 1
            print(f"  FAIL  {n:<14} " + "; ".join(problems))
        else:
            recipes.append(cfg)
            steps = " → ".join(f"{s.get('group')}:{s.get('name')}" for s in cfg["steps"])
            print(f"  ok    {n:<14} {steps}")

    print("\ndatasets")
    dnames = catalog.discover_datasets(ddir)
    if not dnames:
        print(f"  (none found in {catalog.dataset_dir(ddir)})")
    for n in dnames:
        try:
            cfg = catalog._read(catalog.dataset_dir(ddir), n, "dataset", dnames)
            problems = catalog.check_dataset(cfg, n)
        except Exception as e:  # noqa: BLE001
            cfg, problems = {}, [f"{type(e).__name__}: {e}"]
        if problems:
            bad += 1
            print(f"  FAIL  {n:<14} " + "; ".join(problems))
        else:
            datasets.append(cfg)
            h = cfg["handle"]
            print(f"  ok    {n:<14} {h['kind']}:{h['ref']}  shape={cfg.get('shape')}")

    # coverage: which recipes the CURRENT datasets can actually exercise. A recipe with zero
    # legal cells is VALID but dead in a sweep (the precondition calculus prunes it), which is
    # invisible until you run one — so surface it here, at authoring time. It is a WARN, not a
    # FAIL: the recipe is fine, the DATA to exercise it is missing.
    #
    # ...unless it is not the data. A recipe blocked by a NARROWING is blocked on every shape
    # for a reason internal to its own step order, and telling that author to find "a shape
    # providing ['embedding']" is advice they cannot act on: no member of `vocab.SHAPES`
    # provides `embedding` at all (`SHAPE_PROVIDES` maps only `time` and `conditions`).
    # `vocab.invalidated` is the half of the refusal that says which case this is.
    warn = 0
    if recipes and datasets:
        print("\ncoverage (recipe × dataset legality)")
        # Iterates DATASETS, not the shapes deduped out of them, and the distinction is not
        # cosmetic: a dataset carries a `handle` as well as a `shape`, and since `genes` landed
        # (`vocab.DATASET_FACTS`) the handle is half of what makes a recipe legal. Reading the
        # shape alone would report `markers` unexercisable on an .h5ad that plainly has a gene
        # axis, and — worse — phrase it as "needs a shape providing ['genes']" when no member
        # of `SHAPES` provides `genes` at all. That is the unactionable advice `invalidated`
        # was added to stop, one fact over. Shapes are still what gets PRINTED, because that is
        # the readable summary; legality is computed per dataset.
        shapes = sorted({d.get("shape", "unknown") for d in datasets})
        for r in recipes:
            runs_on = [d for d in datasets
                       if not unmet(r, d.get("shape", "unknown"), handle=d.get("handle"))]
            if runs_on:
                where = ", ".join(sorted({d.get("shape", "unknown") for d in runs_on}))
                print(f"  ok    {r['name']:<14} runs on {where}")
            else:
                warn += 1
                probe = datasets[0]
                shape, handle = probe.get("shape", "unknown"), probe.get("handle")
                cleared = sorted(vocab.invalidated(r, shape, handle=handle))
                need = sorted(unmet(r, shape, handle=handle))
                # Three reasons, three fixes. A cleared fact is an ORDERING problem inside the
                # recipe; a frame fact nothing declares is a DATA problem no shape can solve
                # and naming a shape would send the author looking for one; anything else is
                # the original "no dataset of the right design" case.
                # `DATASET_FACTS`, NOT `frame_facts()`. The latter also holds `conditions` and
                # `time`, which SHAPES do provide — telling a `contrast` author that
                # `conditions` "comes from the data and not from any shape" is false, and
                # `case-control` is exactly the shape they should go find.
                absent = sorted(set(need) & set(vocab.DATASET_FACTS) - set(cleared))
                if cleared:
                    print(f"  WARN  {r['name']:<14} no dataset can exercise it — a later step "
                          f"clears {cleared}, which an earlier step produced; move the filter "
                          "before the step that needs it, or re-run that step after it")
                elif absent:
                    print(f"  WARN  {r['name']:<14} no dataset can exercise it — needs {absent}, "
                          f"which comes from the data itself and not from any shape; none of "
                          f"the {len(datasets)} declared datasets carries it")
                else:
                    print(f"  WARN  {r['name']:<14} no dataset can exercise it — needs a shape "
                          f"providing {need or '?'} (have: {', '.join(shapes)})")

    tail = "all valid" if not bad else f"{bad} INVALID"
    if warn:
        tail += f", {warn} unexercisable"
    print(f"\n{len(names)} recipes, {len(dnames)} datasets — {tail}")
    return 1 if bad else 0


# ── building blocks ──────────────────────────────────────────────────────────
def _ask(prompt: str, default: str) -> str:
    """Ask when there is a human to ask; otherwise take the default.

    A bare `input()` here made every non-interactive caller — a script, a CI job, a
    subprocess, `manyruns run data.h5ad` piped anywhere — die with a raw `EOFError` before
    it reached the work. Prompting is a convenience for a terminal, never a requirement."""
    import sys

    if not sys.stdin.isatty():
        return default
    try:
        answer = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:
        return default
    return answer or default


#: Args that mean "I already know what I want" — any of them keeps `run` a one-shot.
_RUN_INTENT_ARGS = ("data_folder", "data_arg", "dataset", "recipe", "engine", "project")


def _wants_the_console(args: argparse.Namespace) -> bool:
    """`manyruns run`, at a terminal, with nothing else said."""
    if not sys.stdin.isatty():
        return False
    return not any(getattr(args, name, None) for name in _RUN_INTENT_ARGS)


def _data_from_drop_folder() -> str:
    """No data argument → look in the drop folder, and if it is empty, say where it is.

    Requiring an explicit path for every run is the friction this removes: the folder is
    created for you, one file in it is simply used, several are offered, and an empty one
    prints an absolute path to drag files into rather than an error about a missing
    argument. `manyruns run` with no arguments should be a reasonable thing to type."""
    from manyruns.shell import ensure_drop_folder, drop_folder_entries

    ddir = ensure_drop_folder()
    found = drop_folder_entries(ddir)
    here = ddir.resolve()

    if not found:
        raise SystemExit(
            f"No data yet. Drop a file or folder into:\n\n    {here}\n\n"
            "  (.h5ad, .h5, .csv, .tsv, .loom, .mtx, .npy — or a folder of per-sample "
            "sub-folders)\n"
            "  then run `manyruns` again. Or point at one directly: "
            "`manyruns run PATH`,\n  or try a built-in sample: `manyruns run --dataset swissroll`."
        )
    if len(found) == 1:
        print(f"using {found[0].name}  (from {here})")
        return str(found[0])

    listing = "\n".join(f"    {i}. {p.name}" for i, p in enumerate(found, 1))
    if not sys.stdin.isatty():
        raise SystemExit(
            f"{len(found)} things in {here} — name one:\n\n{listing}\n\n"
            "  e.g. `manyruns run " + found[0].name + "`"
        )
    print(f"{len(found)} things in {here}:\n{listing}")
    answer = _ask("which one", "1")
    try:
        return str(found[int(answer) - 1])
    except (ValueError, IndexError):
        raise SystemExit(f"no such choice: {answer!r}") from None


def _project_name_for(raw: str) -> str:
    """A project name derived from the data source — `cohort.h5ad` → `cohort`, `swissroll`
    → `swissroll`. Nobody should have to invent a name to look at a file."""
    stem = Path(str(raw)).stem or str(raw)
    return _slug(stem) or "project"


def _fetch_cli_source(name: str, parser: argparse.ArgumentParser) -> None:
    """Explicit CLI use authorizes recovery; catalog reads and resolution stay local."""
    from manyruns import catalog, datasetfetch, shell

    ds = catalog.load_dataset(name)
    lines = shell.where_the_file_should_be(name, ds)
    if not lines:
        return
    if not datasetfetch.can_fetch(ds):
        parser.error("\n".join(lines))
    handle = ds["handle"]
    target = shell.data_dir().resolve() / Path(handle["ref"]).name
    print(f"fetching {target.name}: {handle['bytes']} bytes from {handle['url']} to {target}")
    last_bucket = -1

    def progress(done, total):
        nonlocal last_bucket
        bucket = min(10, done * 10 // total)
        if bucket > last_bucket:
            last_bucket = bucket
            print(f"  {done}/{total} bytes ({done * 100 // total}%)", flush=True)

    try:
        datasetfetch.fetch_dataset(ds, progress=progress)
    except datasetfetch.DatasetFetchError as exc:
        parser.error(str(exc) + "\n" + "\n".join(lines))


def _dataset_contract(dataset_name: Optional[str], dataset: Optional[str],
                      overrides: Optional[dict] = None) -> tuple[dict, Optional[tuple[int, int]]]:
    """Resolve generator settings and their fit dimensions as one source contract.

    A legacy ref is useful only when it is itself a catalog name. Never reverse-map
    aliases: several declared experiments can share one engine generator.
    """
    from manyruns import catalog
    from manyruns.pipeline.bounds import declared_shape

    params = dict(overrides or {})
    name = dataset_name or dataset
    if not name:
        return params, None
    if name not in catalog.discover_datasets():
        if dataset_name:
            raise ValueError(f"unknown catalog dataset {dataset_name!r}")
        return params, None
    row = catalog.load_dataset(name)
    handle = row["handle"]
    if handle["kind"] != "manylatents":
        if dataset is not None:
            raise ValueError(f"catalog dataset {name!r} does not match engine ref {dataset!r}")
        return params, None
    if handle["ref"] != dataset:
        raise ValueError(f"catalog dataset {name!r} does not match engine ref {dataset!r}")
    pinned = row.get("params") or {}
    effective = {**pinned, **params}
    dims = row.get("dims")
    shape = (declared_shape([dims.get("n_samples"), dims.get("n_features")])
             if dims and effective == pinned else None)
    return effective, shape


def _require_source_shape(engine, recipe, dataset, array, shape, embedding=None):
    """Refuse a bounded named-source run when its input dimensions are unverified."""
    from manyruns.pipeline import bounds

    if (engine == "manylatents" and dataset
            and any(step.get("limits") for step in (recipe or {}).get("steps", []))
            and bounds.shape_of({"emb": embedding}, {"array": array, "declared_shape": shape}) is None):
        raise ValueError(
            f"cannot bound recipe parameters for {dataset!r}: generator settings have no "
            "verified fit dimensions. Use a matching catalog declaration or an explicit generated data file.")


def _setup_project(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """Prompt for the minimum (#11), detect modality (#10), pick the recipe, persist it.

    One data source (the DATA arg or --dataset) that is EITHER a filesystem path OR a
    named dataset — auto-detected by whether the path exists on disk."""
    engine = getattr(args, "engine", None) or _default_engine()
    if engine is None:
        from manyruns.installation import REINSTALL

        raise RuntimeError(
            f"no compute backend installed. Install one with: `{REINSTALL}`. "
            "The mock backend exists for development only and is never selected for you."
        )

    # Resolve the data source FIRST, so the project name can default from it.
    raw = (getattr(args, "dataset", None) or getattr(args, "data_folder", None)
           or getattr(args, "data_arg", None))
    if not raw:
        raw = _data_from_drop_folder()
    project = args.project or _ask("project name", _project_name_for(raw))

    dataset_name = None
    src = Path(raw)
    if src.exists():                                  # a file or folder → manyruns loads it
        data_folder, dataset = src, None
        try:
            modality = detect_modality(src)
        except ValueError as e:
            parser.error(str(e))
    elif _looks_like_path(raw):                       # meant as a path, but missing → clear error
        parser.error(f"data path not found: {raw}")
    else:
        # The catalog owns handles on both front doors. Only a manylatents handle is an
        # engine name; sending a file alias to Hydra loses the data before step one.
        from manyruns.shell import _resolve_dataset

        try:
            _fetch_cli_source(raw, parser)
            data_folder, dataset, modality, _ = _resolve_dataset(raw)
            dataset_name = raw
        except (FileNotFoundError, ValueError, KeyError) as exc:
            parser.error(
                f"cannot resolve {raw!r}: tried a local path and a catalog handle "
                f"(including manylatents generators). Supply an existing data path or "
                f"declare a dataset in MANYRUNS_DATASET_DIR. {exc}")
        if (data_folder is not None and not data_folder.exists()
                and not str(data_folder).startswith("synthetic:")):
            parser.error(f"catalog handle {raw!r} resolved to missing data {data_folder}; "
                         f"put the file there or in the data drop folder, or supply its existing path")

    recipe, rationale = select_analysis(modality, data_folder, override=getattr(args, "recipe", None))
    print(f"analysis: {recipe['name']} — {rationale}")
    # `--transform` reaches `init`/`run` HERE, at the one line where this command decides what
    # its recipe is — not at each of the four consumers downstream (`_explore_project` for
    # `run`/`--batch`, `_build_session` for the REPL, and the same pair again from
    # `explore`/`open`). Measured before this line existed: every argv path served
    # `transform.method=log1p` whatever `--transform` said, and printed nothing about the flag
    # it had dropped — argparse accepted it (`_add_engine_args`) and nothing between the
    # namespace and the recipe ever read it.
    #
    # BEFORE `write_project`, so `project.yaml` — the file `explore`/`open` replay from — is the
    # record of what actually ran. Shimming only the in-memory copy would make the same project
    # name preprocess one way today and another way tomorrow, with the file that exists to say
    # which asserting the one that did not happen.
    recipe = apply_transform_shim(recipe, getattr(args, "transform", None))
    data_kwargs, shape = _dataset_contract(dataset_name, dataset,
                                           _parse_kwargs(getattr(args, "data_kwargs", None)))
    _require_source_shape(engine, recipe, dataset, None, shape)
    if color_by := getattr(args, "color_by", None):
        # Refused display arguments must not replace the last saved data/recipe.
        _read_inputs(engine, data_folder, dataset, getattr(args, "time_key", None),
                     color_by=color_by)
    path = write_project(project, data_folder, modality, recipe, engine=engine, dataset=dataset,
                         dataset_name=dataset_name, data_kwargs=data_kwargs)
    return {
        "project": project,
        "data_folder": str(data_folder) if data_folder is not None else None,
        "dataset": dataset,
        "engine": engine,
        "modality": modality,
        "recipe": recipe,
        "dataset_name": dataset_name,
        "data_kwargs": data_kwargs,
        "path": str(path),
    }


def _explore_project(
    proj: dict,
    seed: int = 42,
    fast_dev_run: bool = True,
    data_kwargs: Optional[dict] = None,
    time_key: Optional[str] = None,
    device: Optional[str] = None,
    color_by: Optional[list[str]] = None,
) -> int:
    data_folder = Path(proj["data_folder"]) if proj.get("data_folder") else None
    engine = proj.get("engine", "mock")
    dataset = proj.get("dataset")
    data_kwargs, declared_shape = _dataset_contract(
        proj.get("dataset_name"), dataset, {**(proj.get("data_kwargs") or {}), **(data_kwargs or {})})
    out_dir = Path("outputs") / _slug(proj["project"])

    # Real engines load the data up front (#4). This delegates to `_load_inputs` rather
    # than repeating it: this function used to carry its own copy, which loaded the array
    # for the removed `engine=real` and dropped the labels — so the CLI reached separation/composition
    # with none even when it had just detected the conditions and *selected the contrast
    # recipe on that basis*. Two homes for one job, and only one of them was ever fixed.
    array, labels, label_kind = None, None, None
    # Bound BEFORE the branch: the payload below reads them unconditionally and this branch is
    # `manylatents`-only, so a `mock` run would reach the dict with them unbound.
    counts = genes = None
    layers: dict = {}
    loaded = dict(_EMPTY_INPUTS)
    if engine == "manylatents":
        if not dataset and data_folder is None:
            raise RuntimeError(
                f"engine={engine} needs a data path (file/folder) or --dataset NAME"
            )
        loaded = _load_inputs(engine, data_folder, dataset, time_key,
                              declared_layers(proj.get("recipe")), **({"color_by": color_by} if color_by else {}))
        array, labels, label_kind = loaded["array"], loaded["labels"], loaded["kind"]
        counts, genes, layers = loaded["counts"], loaded["genes"], loaded["layers"]

    from manyruns import catalog

    # #12 — run inference through the MODE ROUTER (modes.run) so the layers compose for real:
    # which-mode (infer) → engine dispatch → device resolution. modes._infer forwards to
    # run_explorations; the device is resolved (fp64/MPS) inside, not echoed.
    from manyruns import modes

    results = modes.run(
        "infer",
        {
            "data": data_folder, "modality": proj["modality"], "recipe": proj["recipe"],
            "engine": engine, "dataset": dataset,
            "dataset_name": proj.get("dataset_name"), "declared_shape": declared_shape,
            "array": array, "out_dir": out_dir, "seed": seed, "fast_dev_run": fast_dev_run,
            # `label_kind` was computed here and then dropped, so the one fact that stops a
            # trajectory being fitted to a case/control axis never left this function.
            "data_kwargs": data_kwargs, "labels": labels, "label_kind": label_kind,
            # The declared suite — WHICH metrics matter is manyruns's call (CLAUDE.md's
            # split); computing them is the engine's. `serving`/`modes` already forwarded a
            # `metrics` key, but nothing ever populated it, so every run silently fell back
            # to engine defaults and `configs/metrics/default.yaml` was inert.
            "metrics": catalog.load_suite(),
            # LOADED AT `_load_inputs` ABOVE AND PREVIOUSLY DROPPED HERE. `run_explorations`
            # re-reads the file only when the caller passed no array, and this caller always
            # passes one — so every `manyruns run` on a real `.h5ad` reached the engine with no
            # gene axis. Forwarding the pair is the whole fix.
            "counts": counts, "genes": genes,
            # DECLARED-ONLY (`declared_layers`): empty unless a step asked for a layer, so a
            # recipe that needs none pays nothing.
            "layers": layers,
            "color": loaded.get("color"), "obs_names": loaded.get("obs_names"),
            "label_key": loaded.get("label_key"),
        },
        device=device,
    )
    if results.get("device_note"):
        print(f"device: {results['device']} — {results['device_note']}")
    source = proj.get("dataset") and f"dataset:{proj['dataset']}" or data_folder
    summary = write_summary(
        proj["project"], source, proj["modality"], results, recipe=proj["recipe"]
    )  # #13
    # What actually ran, and then the finding. Until now this path printed only a device
    # note and a file path, so a user could not tell a run where every step succeeded from
    # one where every step failed — both wrote a summary and exited 0. The panel is a
    # reader over the step record; it triggers nothing.
    # Append to the run store. This is the FIRST product caller — `store.append` shipped
    # with five call sites, all in tests, so every run before this one was unrecorded once
    # its summary was written. A record with no history cannot answer "has this been run
    # before", "what else was run on this cohort", or "which run was this one's control" —
    # which is most of what "strict state of the data and the interventions" means.
    from manyruns import store as _store

    index = _store.append(results, extra={"source": str(source), "modality": proj["modality"]})

    from manyruns import shell as _shell

    # Print durable caveats literally on both terminal and piped runs. The plain
    # panel otherwise includes them itself, while the terminal panel omits them.
    _shell.render_result(_shell._default_console(), {**results, "caveats": []})
    for caveat in results.get("caveats") or ():
        print(caveat)
    print(f"summary written: {summary}")
    print(f"run {results.get('run_id')} recorded in {index}")
    from manyruns.outcomes import run_verdict

    code, verdict = run_verdict(results)
    print(verdict)
    if code:
        # The compact panel clips step details. A repair command must remain pasteable.
        for step in results.get("steps") or []:
            if step.get("outcome") == "error" or step.get("unsupported"):
                print(f"{step.get('name', '?')}: {step.get('detail', 'no reason recorded')}")
    return code


def explore_once(
    *,
    data_folder: Optional[Path],
    dataset: Optional[str],
    modality: str,
    recipe_name: str,
    engine: str,
    seed: int = 42,
    fast_dev_run: bool = True,
    time_key: Optional[str] = None,
    transform: Optional[str] = None,
    device: Optional[str] = None,
    out_dir: Optional[Path] = None,
    on_step: Any = None,
    dataset_name: Optional[str] = None,
    data_kwargs: Optional[dict] = None,
    declared_shape: Any = None,
    color_by: Optional[list[str]] = None,
    color: Any = None,
    obs_names: Any = None,
    label_key: Optional[str] = None,
) -> dict:
    """Run one named recipe on one data source and return the results dict (g_vector, status,
    final_dim, …) — no file I/O, no printing. The interactive shell's execute step for the
    non-steppable engines — a category that is currently EMPTY (it held `real` and the learner's engine,
    both removed), so this path runs today only for a name that is not an engine. It is kept
    because `_REPORTING_ENGINES` is a list and the next backend may not report per-step.
    Delegates to `run_explorations`, which owns loading and the label_kind threading, so this
    adds no fifth loader.

    `transform` NO LONGER REACHES THE LOADER — it is applied to the RESOLVED RECIPE, one line
    below, because the transform is a declared `prep` step now and not a load-time flag. This is
    the one seam both INTERACTIVE front doors pass through (`shell._run_once` and `tui/app.py`),
    which is why the shim is applied here rather than at each of them.

    THE ARGV DOORS DO NOT COME THROUGH HERE, and reading this docstring as if they did is what
    left them broken: `run`, `explore`, `init` (both halves) and `open` build or load a project
    dict and hand its recipe to `_explore_project` / `_build_session`, none of which touches
    this function. Measured while this was the only caller of the shim: `manyruns explore
    --transform sqrt` served `transform.method=log1p`, silently. Their seam is the pair of
    places a project dict is born — `_setup_project` and `_saved_project`.
    """
    recipe, _ = select_analysis(modality, data_folder, override=recipe_name)
    recipe = apply_transform_shim(recipe, transform)
    return run_explorations(
        data_folder, modality, recipe=recipe, engine=engine, dataset=dataset,
        seed=seed, fast_dev_run=fast_dev_run, time_key=time_key,
        device=device, out_dir=out_dir, on_step=on_step,
        dataset_name=dataset_name, data_kwargs=data_kwargs, declared_shape=declared_shape,
        color_by=color_by, color=color, obs_names=obs_names, label_key=label_key,
    )


# suffixes that mark a data source as an intended file path (vs a bare dataset name)
_PATH_SUFFIXES = _SCRNA_SUFFIXES | _BULK_SUFFIXES | {".py", ".npy"}


def _looks_like_path(raw: str) -> bool:
    """Heuristic: did the user mean a filesystem path (not a dataset name)? True if it has
    a path separator, a home/relative prefix, or a known data-file suffix — so a typo'd
    path errors clearly instead of being treated as a (missing) dataset name."""
    return "/" in raw or raw.startswith(("~", ".")) or Path(raw).suffix.lower() in _PATH_SUFFIXES


def detect_modality(data_path: Path) -> str:
    """#10 — differentiate bulk RNA-seq vs scRNA-seq from the data path.

    Accepts a folder (scan its files) or a single file (use its suffix). For a file with
    an unrecognized suffix (e.g. a `.py` generator or `.npy`) returns 'unknown' — the
    loader decides. Minimal heuristic; verified loaders (manylatents-omics) replace this."""
    if data_path.is_file():
        s = data_path.suffix.lower()
        if s in _SCRNA_SUFFIXES:
            return "scrna"
        if s in _BULK_SUFFIXES:
            return "bulk"
        return "unknown"
    # a directory of per-sample sub-folders (e.g. 10x timepoints T0/, T1/, …) → scRNA
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if any(any(d.glob("matrix.mtx*")) for d in subdirs):
        return "scrna"
    suffixes = {p.suffix.lower() for p in data_path.iterdir() if p.is_file()}
    if suffixes & _SCRNA_SUFFIXES:
        return "scrna"
    if suffixes & _BULK_SUFFIXES:
        return "bulk"
    raise ValueError(
        f"unsupported data folder: no recognized files in {data_path} "
        f"(expected one of {sorted(_SCRNA_SUFFIXES | _BULK_SUFFIXES)})"
    )


# The vocabulary lives in ONE place (manyruns.vocab) so the analysis selector and the
# loader can never disagree about what counts as a time axis — see that module for the three
# concrete failures the old fork produced.
from manyruns.vocab import CONDITION_KEYS as _CONDITION_KEYS  # noqa: E402
from manyruns.vocab import TIME_KEYS as _TIME_KEYS  # noqa: E402
from manyruns.vocab import find_group_key as _find_group_key  # noqa: E402


def _choose_recipe(
    *, has_time_axis: bool, conditions: Optional[list], modality: str = "unknown",
    override: Optional[str] = None,
) -> tuple[str, str]:
    """Pure decision: data shape → (recipe name, human rationale). Product-owned, testable.

    The point is to STOP always imposing a trajectory: only pick CFlows when there is a real
    ordering; case/control (unordered conditions) gets a plain embedding, not MIOFlow.

    The CHOICE is delegated to `narrate.offer` — the same ordered menu the interactive app
    shows — and this function only takes its head and writes the rationale. Previously there
    were two tables: `offer` returned `embed` for an unlabelled array while this returned the
    modality default, which was `cflows` for *every* modality (the fallback covered only
    `scrna`/`bulk` and defaulted to `cflows` besides). So `manyruns run` asserted a
    trajectory on data the interactive path declined to make any claim about — issue #26,
    and measurably the single largest source of over-claiming in the ladder eval.
    """
    from manyruns import narrate

    if override:
        return override, f"recipe forced to {override!r} (auto-selection skipped)"

    shape = narrate.shape_of(has_time_axis, conditions, modality)
    recipe = narrate.offer(narrate.Observation(
        shape=shape, modality=modality, conditions=conditions))[0]["recipe"]

    if shape == "time-course":
        why = "detected a time / ordering axis → trajectory (CFlows: phate→mioflow)"
    elif shape == "case-control":
        why = (f"detected conditions {sorted(conditions or [])} with no time axis → case/control "
               "contrast (PHATE + condition separation), not a trajectory")
    else:
        why = ("no time axis or conditions detected → structure-only embedding; asserting a "
               "trajectory here would invent an ordering the data does not carry")
    return recipe, why


# sub-folder names that read as a TIME axis (t0/, day3/, tp2/, 00/, …) — a real time course.
# Names like treated/, control/ do NOT match → they're conditions, not timepoints.
_TIME_DIR_RE = re.compile(r"^(t|tp|d|day|days|stage|week|timepoint|time)[-_ ]?\d+$|^\d+$", re.IGNORECASE)


def _sample_subdirs(data_path: Optional[Path]) -> list:
    """Sub-folders that look like a per-sample dir (10x mtx / .h5ad / .h5)."""
    if data_path is None or not Path(data_path).is_dir():
        return []
    return [
        d for d in Path(data_path).iterdir()
        if d.is_dir() and (any(d.glob("matrix.mtx*")) or list(d.glob("*.h5ad")) or list(d.glob("*.h5")))
    ]


def _has_timepoint_subdirs(data_path: Optional[Path]) -> bool:
    """≥2 sample sub-folders whose NAMES read as timepoints (T0/, day3/, 00/, …) — a time course.
    Per-condition folders (treated/, control/) are NOT temporal, so they fall through to conditions
    (that was the bug: any multi-sample folder was read as a trajectory)."""
    return sum(bool(_TIME_DIR_RE.match(d.name)) for d in _sample_subdirs(data_path)) >= 2


def _dir_conditions(data_path: Optional[Path]) -> Optional[list]:
    """≥2 sample sub-folders that are NOT a time course → their names ARE the conditions (the
    case/control layout: treated/, control/). Lets directory data reach the `contrast` recipe."""
    subs = _sample_subdirs(data_path)
    if len(subs) >= 2 and not all(_TIME_DIR_RE.match(d.name) for d in subs):
        return sorted(d.name for d in subs)
    return None


def _inspect_obs(path: Path) -> tuple[bool, Optional[list], Optional[tuple]]:
    """Best-effort `(has_time_axis, conditions, group)` from a single .h5ad's obs. Needs anndata;
    returns `(False, None, None)` if it's absent or the file is unreadable (e.g. a placeholder) —
    never raises.

    `group` is `(column_name, [category names])` or None — the unordered IDENTITY axis
    (`vocab.GROUP_KEYS`: `cell_type`, `branch`, `leiden`, …), the thing that COLOURS a plot.
    Names come back biggest-population-first (ties by name, so it is deterministic), because the
    sentence that renders them truncates and the honest six to show are the six that hold the
    most cells. The full list is carried, so `len()` is the true category count and the capping
    stays a rendering concern.

    **Gated on `not has_time and conditions is None`, matching `pipeline.loading.labels_of`'s
    precedence** (time → condition → group). That precedence is what decides which axis actually
    colours the run's plot, so reporting a group axis on a file that also carries a time or
    condition column would make `narrate.describe` promise a colouring the run will not produce.

    **THE GROUP AXIS IS READ FOR NARRATION ONLY.** It is deliberately not forwarded to
    `narrate.shape_of` — see the discard at `select_analysis`'s call site. Making it a third
    input there would move `pbmc3k_annotated.h5ad` and `tree8.h5ad` from shape `single` to
    `clusters`, and that is not the free rename it looks like:

    - Legality would NOT move, and that is structural rather than lucky: `vocab.SHAPE_PROVIDES`
      maps both `"single"` and `"clusters"` to `()`, so `vocab.unmet` seeds the same facts for
      either. Measured this session over all 11 discovered recipes × both shapes, with
      `provided=None` and with `provided={"genes"}`: byte-identical every time.
    - What WOULD move is advice and honesty. `narrate.offer`'s `recommended` gains a star on
      `cluster` (it declares `suits: [clusters]`), so `_choose_recipe`'s head — it takes
      `offer(...)[0]["recipe"]` — flips `embed` → `cluster`, a recipe that runs leiden and
      *discovers* groups while ignoring the eight curated ones, still carrying the `embed`
      rationale string. And `narrate.refusal` does not list `clusters` among the shapes it
      declines a trajectory on, so the app's headline honesty behaviour would go silent on
      exactly the two demo fixtures. `narrate.describe` has no `clusters` branch either and
      falls through to the *harder* denial ("No labels I recognize").

    So the shape stays `single` and only the sentence learns the labels. Moving the shape is a
    separate pass with its own list of things to fix first.
    """
    try:
        import anndata

        adata = anndata.read_h5ad(path, backed="r")
    except Exception:  # noqa: BLE001 - anndata absent, or not a real .h5ad
        return False, None, None
    try:
        cols = {c.lower(): c for c in adata.obs.columns}
        has_time = any(k in cols for k in _TIME_KEYS)
        conditions = None
        for k in _CONDITION_KEYS:
            if k in cols:
                vals = [str(v) for v in adata.obs[cols[k]].unique()]
                if len(vals) >= 2:
                    conditions = vals
                    break
        group = None
        if not has_time and conditions is None:
            # one read of the columns already open — `narrate.read_data` measures 361 ms cold on
            # a 5.9 MB .h5ad, and a second open would pay it twice for one string.
            gkey = _find_group_key(adata.obs.columns)
            if gkey is not None:
                counts: dict[str, int] = {}
                for v in adata.obs[gkey]:
                    s = str(v)
                    counts[s] = counts.get(s, 0) + 1
                if len(counts) >= 2:
                    group = (gkey, sorted(counts, key=lambda s: (-counts[s], s)))
        return has_time, conditions, group
    finally:
        if getattr(adata, "isbacked", False) and adata.file is not None:
            adata.file.close()


def select_analysis(
    modality: str, data_path: Optional[Path] = None, *, override: Optional[str] = None,
) -> tuple[dict, str]:
    """Auto-select the exploration recipe from the data shape (product-owned). Returns
    (recipe dict, rationale). Data-driven by default; `override` (from --recipe or the router)
    short-circuits detection but still reports why."""
    if override:
        name, why = _choose_recipe(has_time_axis=False, conditions=None, modality=modality,
                                   override=override)
        return load_recipe(name), why

    has_time = _has_timepoint_subdirs(data_path)
    conditions = None
    if not has_time and data_path is not None:
        p = Path(data_path)
        if p.is_dir():
            conditions = _dir_conditions(p)                    # per-condition folders → contrast
        elif p.suffix.lower() in {".h5ad", ".h5"}:
            # The group axis is DELIBERATELY DISCARDED here and not forwarded to
            # `_choose_recipe`. `narrate.shape_of` takes two facts, and a third would read
            # `pbmc3k_annotated.h5ad` as `clusters`, flipping the auto-selected recipe from
            # `embed` to `cluster` (measured) without changing one thing `vocab.unmet` permits.
            # It is a narration fact; `narrate.read_data` is where it is spent.
            has_time, conditions, _group = _inspect_obs(p)

    name, why = _choose_recipe(has_time_axis=has_time, conditions=conditions, modality=modality)
    return load_recipe(name), why


def select_recipe(modality: str) -> dict:
    """Back-compat: the modality default recipe, no data inspection. Prefer `select_analysis`."""
    return select_analysis(modality)[0]


def load_recipe(name: str) -> dict:
    """Load a recipe by name as the plain dict the pipeline consumes.

    Delegates to the catalog — locating, parsing AND validating a recipe happens in exactly
    one place, so the CLI cannot hand an invalid recipe to the runner."""
    from manyruns.catalog import load_recipe as _load

    return _load(name)


def apply_transform_shim(recipe: dict, transform: Optional[str]) -> dict:
    """`--transform sqrt` → the `transform` step's `method` param.

    The last survivor of a load-time flag for a step-level fact. `--transform` used to reach
    `loading._anndata_matrix`, which ran the undeclared preamble; that preamble is a declared
    `prep` step now, so the flag's only honest meaning is "set that step's param".

    **A no-op when the recipe declares no `transform` step, deliberately silent.** `qc` has no
    prep block at all, and `manyruns explore --recipe qc --transform sqrt` should not error over
    a flag that has a default in every other invocation. It is also why the flag's default moved
    to `None`: `"log1p"` as a default would have this function rewrite every recipe's declared
    method to log1p on every run, overriding a recipe that asked for `sqrt` with a value the user
    never typed.

    DELETED when the general `--set <step>.<param>=<value>` lands (backlog item 3), which is the
    real fix — one flag per param does not scale past the two this one covers.
    """
    if not transform:
        return recipe
    out = dict(recipe)
    out["steps"] = [
        {**s, "params": {**(s.get("params") or {}), "method": transform}}
        if s.get("name") == "transform" else s
        for s in (recipe.get("steps") or [])
    ]
    return out


def write_project(
    project: str,
    data_folder: Optional[Path],
    modality: str,
    recipe: dict,
    engine: str = "mock",
    dataset: Optional[str] = None,
    dataset_name: Optional[str] = None,
    data_kwargs: Optional[dict] = None,
) -> Path:
    """#11 — persist the minimal project so `explore` can pick it up later."""
    from omegaconf import OmegaConf

    data_kwargs, _ = _dataset_contract(dataset_name, dataset, data_kwargs)
    out_dir = Path("outputs") / _slug(project)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "project.yaml"
    OmegaConf.save(
        config={
            "project": project,
            "data_folder": str(data_folder) if data_folder is not None else None,
            "dataset": dataset,
            "dataset_name": dataset_name,
            "data_kwargs": data_kwargs,
            "engine": engine,
            "modality": modality,
            "recipe": recipe,
        },
        f=path,
    )
    return path


def load_project(project: str) -> Optional[dict]:
    """Load a project written by `init` (None if it doesn't exist yet).

    A project.yaml is the ONE place a removed engine name survives a release: `init --engine
    real` wrote `engine: real` into a file that outlives the flag. Checking it here — the only
    reader — means a stale project says so instead of quietly becoming something else. The
    REPL branch reaches `runner.steppable(proj["engine"])`, which answers True for the
    in-process substrate `vocab.INPROC`; without this a hand-written `engine: _inproc` opens a
    prompt on a step loop no product surface is supposed to reach."""
    from omegaconf import OmegaConf

    path = Path("outputs") / _slug(project) / "project.yaml"
    if not path.is_file():
        return None
    proj = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    engine = (proj or {}).get("engine")  # type: ignore[union-attr]
    if engine is not None and engine not in ENGINE_NAMES:
        raise ValueError(
            f"{path} names engine={engine!r}, which this build cannot run. "
            f"Available: {', '.join(ENGINE_NAMES)}. "
            "(`real` was removed — it selected no distinct dependency set; manylatents ships "
            "with manyruns and pulls the same phate. Edit `engine:` in that file.)"
        )
    return proj  # type: ignore[return-value]


def _smoke(args: argparse.Namespace) -> bool:
    """MIOFlow training length: full (50 epochs) is the default now — `--smoke` opts into the
    fast single-batch plumbing check. `--full` is still accepted and always means full."""
    return bool(getattr(args, "smoke", False)) and not getattr(args, "full", False)


def _parse_kwargs(pairs: Optional[list[str]]) -> dict:
    """Parse repeated KEY=VALUE flags into a dict, coercing int/float/bool values."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            continue
        key, raw = pair.split("=", 1)
        val: Any = raw.strip()
        for cast in (int, float):
            try:
                val = cast(raw)
                break
            except ValueError:
                pass
        if isinstance(val, str) and val.lower() in ("true", "false"):
            val = val.lower() == "true"
        out[key.strip()] = val
    return out


def _server_for(engine: str, device: str = "cpu") -> ModelServer:
    """Pick the serving backend for an engine name (mock is $0/default).

    `device` is the resolved compute device (from manyruns.devices) that reaches MIOFlow's
    Trainer on the manylatents path.

    It USED TO take `mode` and `overrides` too — Hydra passthrough for the learner path
    (`--mode trace`, `--set data=swissroll`). Both are gone with that path (the
    environment-contract spec, §3.5), along with the two CLI flags that fed them: manyruns
    holds no track to a learner, so there is no engine left to pass an override THROUGH to.

    REFUSES an engine this build cannot serve, instead of falling through to the mock. The
    final `return default_server()` used to answer ANY name — so a `project.yaml` written by
    an older `init --engine real`, or a request dict rebuilt from a stored record, reached
    `run_explorations` and came back with a full invented g-vector stamped `engine=mock`
    while the project still said `real`. Measured before this guard: `_server_for("real")`
    returned `LocalServer(engine="mock")` and the run exited 0. `LocalServer.predict` refuses
    too, but it never saw these calls — this is the frame the product actually goes through."""
    from manyruns.serving import LocalServer

    if engine not in LocalServer.SERVES:
        raise ValueError(
            f"engine={engine!r} is not a serving backend. "
            f"Available: {', '.join(sorted(LocalServer.SERVES))}. "
            "(`real` was removed: it selected no distinct dependency set — manylatents ships "
            "with manyruns and pulls the same phate. Re-run with --engine manylatents, or "
            "edit `engine:` in the project's project.yaml.)"
        )
    if engine == "manylatents":
        return LocalServer(engine=engine, device=device)
    return default_server()


#: Engines for which MANYRUNS owns the loading, and therefore the only ones whose missing
#: labels are manyruns's to refuse. `mock` reads no data at all ("invents numbers and never
#: reads your data"), so there is nothing to have failed to deliver. (the learner's engine was the other
#: exemption, for a different reason — it took the whole recipe downstream and loaded the named
#: dataset itself, so claiming it lacked the axis would have been a guess about someone else's
#: loader. It is no longer an engine; see the environment-contract spec, §3.5.)
_MANYRUNS_LOADS_FOR = frozenset({"manylatents"})

#: The loader's `label_kind` → the fact vocabulary `vocab.SHAPE_PROVIDES` is written in.
#: Two spellings for one idea ("condition" the label kind, "conditions" the data fact), which
#: is precisely the duplication `vocab.py` exists to prevent — the pair belongs NEXT TO
#: `SHAPE_PROVIDES`, in vocab. It is bridged here, explicitly and in one place, rather than
#: silently: this file may not edit vocab.py.
_FACT_FOR_LABEL_KIND = {"time": "time", "condition": "conditions"}

#: What is actually lost when a declared fact never reaches the run — stated per fact, because
#: the two costs are different and only one of them is loud.
_UNDELIVERED_FACT_COST = {
    "time": (
        "`vocab.unmet` reads the DECLARATION, not the run, so a trajectory recipe is still "
        "called legal; the trajectory step then falls back to a pseudotime computed from the "
        "embedding — a clock made of the geometry it is supposed to be independent of."
    ),
    "conditions": (
        "separation/composition would record `no condition labels` and return None, which the "
        "results table reads as a metric this recipe does not emit — the conflation "
        "`vocab.STEP_NEEDS` exists to remove."
    ),
}


def _require_declared_facts(
    engine: str, dataset: Optional[str], labels: Any, label_kind: Optional[str],
) -> None:
    """Refuse a NAMED dataset whose declared shape asserts a data fact the loader did not
    deliver. ONE home for that rule, in the shape of `pipeline.steps._require_embedding`.

    THE HOLE. `_load_inputs` routes through `pipeline.load_labeled` only when there is no
    named dataset, so for `--dataset <name>` labels and label_kind came back None
    unconditionally. Measured when `engine=real` still existed: `_load_inputs("real", None,
    "swissroll", …)` → `(SwissRollDataModule, None, None)`, and today
    `_load_inputs("manylatents", None, "swissroll", …)` → `(None, None, None)`; identical for
    `torus` and `dla_tree`. So a dataset declaring
    `shape: time-course` would have its time axis erased between the declaration the legality
    check reads and the run that consumes it — `unmet(cflows, "time-course")` is `frozenset()`,
    i.e. LEGAL, and `runner._ml_lightning` then computes `time_labels = state["labels"]` =
    None and `mioflow._run_mioflow_experiment` falls back to a diffusion pseudotime.

    What that produces, measured end to end (`--dataset swissroll --recipe cflows` on the
    in-process loop, seed 42, 5000×3): trace `latent:phate → lightning:mioflow`, status `phate: ok,
    mioflow: ok`, run `ok=True`, and `g_vector["pseudotime_range"] == [0.0, 1.0]` — a
    full-range trajectory whose clock came from the embedding, indistinguishable in the record
    from one driven by real collection times. Same failure mode as the step-order bug
    (commit df478cc), one level up: there the ORDER was unchecked, here the DATA is.

    WHY THIS REFUSES INSTEAD OF CARRYING THE LABELS. manylatents' named datasets cannot supply
    a time axis, measured against the engine actually installed here. Its dataset interface
    (`manylatents.data.capabilities.get_capabilities`) declares exactly four capabilities —
    `gt_dists`, `graph`, `labels`, `centers` — and no notion of time, and `get_labels()` returns
    an untyped integer per row whose meaning differs per dataset: `swissroll` gives 100
    distribution indices whose Spearman correlation with the roll's own arc coordinate `ts` is
    **-0.224** (the per-distribution means are drawn `rng.random(...)`, so the index is
    unordered by construction); `dla_tree` gives 20 branch ids, `gaussian_blob` 3 cluster ids,
    `archetypal` 4 archetype ids; `torus` and `saddlesurface` expose no labels at all. None of
    those is a clock, and nothing in the interface says which kind a label is — so carrying
    them through as `label_kind="time"` would fabricate exactly the thing this refuses.

    A POSTCONDITION, not an assumption: it compares what the declaration promised against what
    the loader actually returned. If a future loader does obtain per-row time for named
    datasets, this stops firing on its own rather than having to be remembered and deleted.

    SCOPE, measured over the catalog as it stands (13 datasets): 12 declare `manifold` or
    `clusters`, for both of which `vocab.dataset_provides` returns `frozenset()` — nothing is
    claimed, so nothing is refused and no existing run changes. The 13th,
    `synthetic_timecourse`, declares `shape: time-course` and DOES carry one, but reaches a run
    by a different route: its handle is `{kind: path, ref: synthetic:time-course}`, and
    `experiment._request` sends every non-manylatents handle as `data=` with `dataset=None`, so
    it loads through `pipeline.load_labeled`. Measured on that route: a (300, 5) matrix,
    `label_kind="time"`, timepoints {0.0, 0.5, 1.0} — delivered, and untouched by this guard.
    Asking for it as `--dataset synthetic_timecourse` is the case that IS refused, correctly:
    that route hands the name to manylatents, whose registry does not contain it.

    A NAMESPACE CAVEAT, stated rather than papered over. `--dataset` values are resolved
    against manylatents' registry (`pipeline.load_named_dataset`), while `discover_datasets()`
    lists manyruns's YAML names; the two overlap for `swissroll`/`torus`/`gaussian_blob` and
    diverge elsewhere (`tree_wide` handles to ref `dla_tree`). So the lookup below can MISS a
    declaration whose file name differs from its handle ref. It never invents one — an
    unmatched name simply asserts nothing — so the failure mode is under-refusal, not
    over-refusal. Unifying the two namespaces at the `--dataset` seam is a separate change.

    An INVALID declaration raises from `catalog.load_dataset`, which is that function's stated
    contract ("a bad file cannot flow into a run") and what `manyruns check` is for.
    """
    if not dataset or engine not in _MANYRUNS_LOADS_FOR:
        return
    from manyruns import catalog
    from manyruns.vocab import dataset_provides

    if dataset not in catalog.discover_datasets():
        return  # undeclared: it promises nothing, so there is nothing to break
    shape = catalog.load_dataset(dataset).get("shape")
    declared = dataset_provides(shape)
    if not declared:
        return

    fact = _FACT_FOR_LABEL_KIND.get(label_kind or "")
    delivered = frozenset({fact}) if (labels is not None and fact) else frozenset()
    missing = sorted(declared - delivered)
    if not missing:
        return
    names = ", ".join(f"`{m}`" for m in missing)
    cost = " ".join(_UNDELIVERED_FACT_COST[m] for m in missing if m in _UNDELIVERED_FACT_COST)
    raise ValueError(
        f"--dataset {dataset} declares shape: {shape}, which asserts {names}, but the loader "
        f"returned no {names} labels for it — a named dataset is loaded by the engine, so "
        f"manyruns never sees its rows and the declared fact cannot reach the run. "
        f"{cost} Refusing rather than running: a result computed without the fact its own "
        f"dataset declares is not distinguishable afterwards from one computed with it. "
        f"Either point at a file or folder that carries the axis (`--time-key` names the "
        f".obs column), or correct `shape:` in {dataset}.yaml."
    )


#: What the loader returns when there is nothing to load. A DICT rather than a positional
#: tuple, and the reason is measured rather than aesthetic: this shape has been widened twice
#: (3 -> 5, and 5 -> 7 on a branch), and every widening cost the same two mistakes — a call site
#: missed (11 tests died with `UnboundLocalError: cannot access local variable 'counts'`) and
#: four test fakes still returning the old arity, which keep passing while the real seam has
#: moved. A key that nobody reads is inert; a positional slot that nobody reads is a bug.
#:
#: `dict(_EMPTY_INPUTS)` at every return, never the constant itself — a shared mutable default
#: handed to a caller is the next silent failure along.
_EMPTY_INPUTS: dict = {"array": None, "labels": None, "kind": None,
                       "counts": None, "genes": None, "layers": {},
                       "color": None, "obs_names": None, "label_key": None}


def cmd_tools(args: Any, parser: Any = None) -> int:
    """`manyruns tools` — the operator's view of the external-tool layer.

    Prints STATE, not opinions: what is declared, whether this platform has a lock, whether the
    environment is realized, and the exact command for the gap. The same `toolchain.resolve`
    the runner calls, so what this shows is what a run will do.
    """
    from manyruns import toolchain as _tc

    action = getattr(args, "action", "list") or "list"
    names = _tc.discover_tools()
    if not names:
        print("no tools declared. A tool is a YAML in $MANYRUNS_TOOL_DIR or the bundled set.")
        return 0

    if action == "install":
        if not args.name:
            print("which tool? " + ", ".join(names))
            return 2
        try:
            out = _tc.build(args.name)
        except Exception as e:  # noqa: BLE001 - the message is the product here
            print(f"could not build {args.name}: {e}")
            return 1
        print(f"{args.name}: built {out['env']} (lock {out['digest']}) — selftest {out['selftest']}")
        return 0 if out["ok"] else 1

    rc = 0
    for name in names:
        r = _tc.resolve(name)
        cfg = r.get("cfg") or {}
        line = f"  {name:16} {str(cfg.get('version') or '?'):8} {r['status']}"
        if action == "check" and r["status"] in ("ready", "overridden"):
            res = _tc.selftest(cfg, r["python"])
            line += f"  selftest: {res['selftest']}"
            rc = rc or (0 if res["ok"] else 1)
        print(line)
        if r.get("fix"):
            print(f"      -> {r['fix']}")
    return rc


def _load_inputs(engine: str, data_folder: Optional[Path], dataset: Optional[str],
                 time_key: Optional[str], layers: "tuple | None" = None,
                 color_by: Optional[list[str]] = None) -> dict:
    """Load analysis, gene and display channels, then check declared analysis facts.

    Ordinary file/generated sources share one object read. Engine-owned generators are
    left to the engine; explicit colour choices refuse when their metadata is unavailable.
    ``color_by`` selects display channels only and never changes analysis labels.
    """
    loaded = _read_inputs(engine, data_folder, dataset, time_key, layers,
                          **({"color_by": color_by} if color_by else {}))
    _require_declared_facts(engine, dataset, loaded["labels"], loaded["kind"])
    return loaded


def declared_layers(recipe: Optional[dict]) -> tuple:
    """Which layers this recipe actually needs — the union of its steps' `layers:` declarations.

    DECLARED-ONLY IS THE WHOLE POLICY. An `.h5ad` may carry a dozen layers; loading all of them
    makes memory a property of the FILE rather than of the analysis, and no threshold picks
    itself. So a step says what it reads and the loader supplies that and nothing else — the
    same shape `metrics` already has, where manyruns declares which it wants and the engine
    computes those.

    Empty for every recipe that declares none, which is all of them today. That is the inert
    default: a run that asks for no layer loads no layer and costs nothing.
    """
    wanted: list = []
    for step in (recipe or {}).get("steps") or []:
        for name in (step.get("layers") or ()):
            if name not in wanted:
                wanted.append(str(name))
    return tuple(wanted)


class ColorSelectionError(ValueError):
    """A display choice the CLI reports as an argument error."""


def _read_inputs(engine: str, data_folder: Optional[Path], dataset: Optional[str],
                 time_key: Optional[str], layers: "tuple | None" = None,
                 color_by: Optional[list[str]] = None) -> dict:
    """Read analysis and display channels from one object whenever the source permits it.

    Directory timepoint labelling stays with load_labeled. Its separate object read is
    retained so folder-derived times and the loader's concatenated IDs remain compatible.
    """
    from manyruns import pipeline
    from manyruns.pipeline import loading, runner

    if engine == "manylatents" and not dataset and data_folder is not None:
        with pipeline.quiet():
            obj = pipeline.load_array(data_folder)
            label_key = loading.label_key_of(obj, time_key)[0]
            if Path(data_folder).is_dir():
                array, labels, kind = pipeline.load_labeled(data_folder, time_key=time_key)
                # A folder name is an analysis axis, not a column in the source object.
                own_labels, _ = loading.labels_of(obj, time_key)
                import numpy as np

                if labels is not None and (own_labels is None or not np.array_equal(labels, own_labels)):
                    label_key = None
            else:
                # Match load_labeled: keep the source frame sparse and in its original
                # dtype until a compute step actually needs a dense matrix.
                array = loading._anndata_matrix(obj) if loading._is_anndata(obj) else pipeline.as_matrix(obj)
                labels, kind = pipeline.labels_of(obj, time_key=time_key)
            counts, genes = pipeline.gene_axis(obj)
            loaded_layers = pipeline.layers_of(obj, layers)
            try:
                color = [loading.color_channel_of(obj, key) for key in color_by or ()]
            except ValueError as exc:
                raise ColorSelectionError(str(exc)) from exc
        names = runner.implicit_sample_ids(obj)
        return {"array": array, "labels": labels, "kind": kind,
                "counts": counts, "genes": genes, "layers": loaded_layers,
                "color": color, "obs_names": names, "label_key": label_key}
    if color_by:
        raise ColorSelectionError(f"cannot inspect metadata for {dataset or engine!r}; "
                                  "use an annotated data file to choose --color-by")
    return dict(_EMPTY_INPUTS)


#: Engines whose step loop actually calls the observer. `serving.LocalServer` forwards
#: `on_step` on the `real` and `manylatents` paths only (`_run_real` / `_run_manylatents`);
#: `mock` and the learner's engine execute the whole recipe in one call and never report a step, so a
#: panel opened for them would sit at "queued" for the entire run and then vanish — worse
#: than the silence it replaced.
_REPORTING_ENGINES = frozenset({"manylatents"})


def _live_step_panel(
    recipe: Optional[dict], engine: str, on_step: Any, source: str,
) -> tuple[Any, Any]:
    """The shell's live dashboard, for the paths that reach the engine without one.

    `shell.live_dashboard` is reused as-is; this only decides *whether* a live region may be
    opened. Returns `(live, on_step)`, or `(None, None)` when one would be wrong. Each guard
    is a specific way this breaks:

    - **an observer already exists** → `shell._run` opened its own dashboard and passed it
      down. Two Live regions on one stdout is two things owning the cursor, which shell.py
      (l. 507) records as how a TUI starts corrupting its own output. Nothing prompt-driven
      reaches here at all: the REPL and the stepped engines go through `session.py`, which
      drives `pipeline` directly, so no questionary prompt can be open while this blocks.
    - **stdout is not a terminal** → piped output stays byte-identical; rich would otherwise
      write a full repaint of the panel per update into the file. `is_terminal` is the check
      the rest of the repo uses (`shell._render_state`, `_emit_inline_image`, `_run`).
    - **the engine never reports a step**, or the recipe declares none → nothing to animate.

    Rich absent is covered by the terminal guard, not a second one: `shell._default_console`
    falls back to `_PlainConsole`, whose `is_terminal` is False."""
    steps = (recipe or {}).get("steps") or []
    if on_step is not None or engine not in _REPORTING_ENGINES or not steps:
        return None, None
    from manyruns import shell

    console = shell._default_console()
    if not getattr(console, "is_terminal", False):
        return None, None
    name = (recipe or {}).get("name") or "recipe"
    return shell.live_dashboard(console, recipe, f"[bold]{name}[/bold] · {engine} · {source}")


def run_explorations(
    data_folder: Optional[Path],
    modality: str,
    server: Optional[ModelServer] = None,
    recipe: Optional[dict] = None,
    engine: str = "mock",
    dataset: Optional[str] = None,
    array: Any = None,
    out_dir: Optional[Path] = None,
    seed: int = 42,
    fast_dev_run: bool = True,
    data_kwargs: Optional[dict] = None,
    labels: Any = None,
    device: Optional[str] = None,
    metrics: Any = None,
    time_key: Optional[str] = None,
    label_kind: Optional[str] = None,
    on_step: Any = None,
    counts: Any = None,
    genes: Any = None,
    layers: "dict | None" = None,
    dataset_name: Optional[str] = None,
    declared_shape: Any = None,
    color_by: Optional[list[str]] = None,
    color: Any = None,
    obs_names: Any = None,
    label_key: Optional[str] = None,
) -> dict:
    """#12 — run the exploration via the serving backend.

    The backend is swappable (`--engine` / Hydra `serving=`); defaults to the free
    in-process mock LocalServer. With a recipe, the run executes the product-owned
    pipeline (phate → mioflow) — the mock simulates it, and `engine=manylatents` chains
    manylatents.api.run per step over the loaded `array`.

    `device` is the requested compute device (cpu/cuda/mps, e.g. from the substrate
    dispatcher); it is *resolved* against the recipe's fp64 need before use, and the
    resolved device + rationale are returned — never a device the compute won't honor."""
    from manyruns.devices import recipe_requires_fp64, resolve_device

    from manyruns.pipeline import bounds
    from manyruns import catalog

    declared_shape = bounds.declared_shape(declared_shape)
    data_kwargs, pinned_shape = _dataset_contract(dataset_name, dataset, data_kwargs)
    if dataset_name or dataset in catalog.discover_datasets():
        declared_shape = pinned_shape
    resolved, device_note = resolve_device(device, requires_fp64=recipe_requires_fp64(recipe))
    server = server or _server_for(engine, device=resolved)
    # Load the array + labels here if the caller didn't pre-load them. The CLI still loads
    # up front (to print progress and to reuse the array across a session); the experiment
    # harness passes only a data path, so without this its runs reach separation/composition
    # with no condition labels — the exact gap the precondition calculus flagged.
    # MEASURED BUG, and this is the fix. These were LOCALS initialised to None, so a caller
    # that had already loaded the array — `cmd_run` does, to print progress — skipped the branch
    # below and reached the engine with `counts=genes=None`: the gene axis read three frames up
    # and dropped. On `manyruns run data/pbmc3k_raw.h5ad --recipe markers` that produced
    # `rank_genes_top: not measured — no gene axis`, which is exactly the "legal at plan time,
    # refused at run time" failure `_read_inputs` says it exists to prevent — live, on the one
    # real dataset this repo ships.
    #
    # PARAMETERS now, so a caller that loaded them passes them on. The branch below stays as the
    # fallback for a caller that did not (the experiment harness, which passes only a path).
    if array is None and labels is None and data_folder is not None:
        loaded = _load_inputs(engine, data_folder, dataset, time_key,
                              declared_layers(recipe), **({"color_by": color_by} if color_by else {}))
        array, labels, label_kind = loaded["array"], loaded["labels"], loaded["kind"]
        counts, genes, layers = loaded["counts"], loaded["genes"], loaded["layers"]
        color = loaded.get("color") if color is None else color
        obs_names = loaded.get("obs_names") if obs_names is None else obs_names
        label_key = loaded.get("label_key") if label_key is None else label_key
    elif color_by:
        raise ColorSelectionError("cannot inspect metadata for this preloaded matrix or generator; "
                                  "supply an annotated data file or explicit display channels")
    # `_load_inputs` is skipped above for a NAMED dataset (`data_folder is None`), and this is
    # the one function every path funnels into — `explore_once` (the shell), `modes._infer`
    # (the CLI's `run`/`explore`/`--batch`) and the experiment harness. Without this line
    # those three reach the engine with the declaration unchecked, which is the whole bug.
    # Idempotent: the callers that DID go through `_load_inputs` just pay one catalog read.
    _require_declared_facts(engine, dataset, labels, label_kind)
    _require_source_shape(engine, recipe, dataset, array, declared_shape)
    label = dataset or (f"{modality}:{data_folder.name}" if data_folder is not None else modality)
    # The command-line paths (`run`, `explore`, `init --batch`) arrive here with no observer:
    # they run app → `modes.run` → `_infer` → here, and `modes._infer` has no `on_step`
    # parameter, so the shell's dashboard could not reach them however hard app.py threaded it.
    # This is the one function every path funnels into, so opening the panel HERE covers all
    # three without a new channel through the mode router. `explore_once` (the shell) already
    # carries an observer and is left alone by the guard.
    live, observer = _live_step_panel(recipe, engine, on_step, str(label))
    if observer is not None:
        on_step = observer
    inputs: Any = label
    if recipe:
        inputs = {
            "data": label,
            "modality": modality,
            "recipe": recipe,
            "dataset": dataset,
            "dataset_name": dataset_name, "declared_shape": declared_shape,
            "color": color, "obs_names": obs_names, "label_key": label_key,
            "out_dir": str(out_dir) if out_dir is not None else ".",
            "seed": seed,
            "fast_dev_run": fast_dev_run,
            "data_kwargs": data_kwargs or {},
        }
        if array is not None:
            inputs["array"] = array
        if labels is not None:
            inputs["labels"] = labels
        # The gene axis, when the source carries one. Both or neither — `frame.new_frame` pairs
        # them, and a `counts` with no names is a matrix the gene steps still cannot use.
        if counts is not None and genes is not None:
            inputs["counts"] = counts
            inputs["genes"] = genes
        if layers:
            inputs["layers"] = layers
        if metrics:
            inputs["metrics"] = list(metrics)   # the declared suite → manylatents' registry
        if label_kind:
            # "time" | "condition" | "group"; each consumer allowlists the kind it wants
            inputs["label_kind"] = label_kind
        if on_step is not None:
            # a live observer for the dashboard. In-process only: it never crosses a process
            # boundary, is never serialised, and a run without one behaves identically.
            inputs["on_step"] = on_step
    # `with`, not start()/stop(): a run that raises must still hand the cursor back, or the
    # terminal is left with the panel's alternate-screen state and a hidden cursor.
    with live if live is not None else contextlib.nullcontext():
        result = server.predict(inputs)
    if isinstance(result, dict):  # honest: the device compute ACTUALLY used, + why (no echo)
        result["device"], result["device_note"] = resolved, device_note
    return result


def write_summary(
    project: str,
    source: Any,
    modality: str,
    results: dict[str, Any],
    recipe: Optional[dict] = None,
) -> Path:
    """#13 — aggregate exploration outputs into one markdown file at a predictable path.

    `source` is the data origin — a folder Path, a `dataset:<name>` label, or None."""
    out_dir = Path("outputs") / _slug(project)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.md"

    steps = (recipe or {}).get("steps") or []
    trace = results.get("trace", [])
    recipe_name = results.get("recipe") or (recipe or {}).get("name") or "(adaptive)"

    lines = [
        f"# {project}",
        "",
        "## overview",
        "",
        f"- data: `{source}`",
        f"- modality: **{modality}**",
        f"- recipe: **{recipe_name}**",
        f"- served by: {results.get('served_by')} ({results.get('engine')})",
        f"- exploration steps: {results.get('num_steps')}",
        "",
        "## recipe / steps",
        "",
    ]
    if steps:
        pipeline = " → ".join(s.get("name", "?") for s in steps)
        lines.append(f"- pipeline: {pipeline}")
        lines.append("")
        lines.append("| # | group | algorithm | params |")
        lines.append("| - | ---- | --------- | ------ |")
        for i, s in enumerate(steps, 1):
            lines.append(
                f"| {i} | {s.get('group')} | `{s.get('name')}` | `{s.get('params', {})}` |"
            )
    else:
        lines.append("_(adaptive run — steps emerged from the driver loop)_")
    if results.get("engine") == "mock":
        from manyruns.serving import MOCK_CAVEAT

        lines += ["", MOCK_CAVEAT]
    lines += [
        "",
        f"- trace: {' → '.join(trace) if trace else '(none)'}",
        "",
        "## g-vector",
        "",
        f"```\n{results.get('g_vector')}\n```",
        "",
    ]

    # per-step status (real engine reports which steps ran / errored)
    status = results.get("status")
    if status:
        lines += ["## step status", ""]
        lines += [f"- `{name}`: {state}" for name, state in status.items()]
        lines.append("")

    # plots — list real saved files, else note the mock has none
    lines += ["## plots", ""]
    plot_files = results.get("plots")
    if plot_files:
        lines += [f"- `{p}`" for p in plot_files]
    else:
        lines.append(
            f"Plots would be saved under `{plots_dir}/`. No plots generated on this backend "
            f"({results.get('engine')}) — use `--engine manylatents` for embeddings/plots."
        )
    lines.append("")

    summary.write_text("\n".join(lines))
    return summary


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", name.strip().lower()).strip("-") or "project"


if __name__ == "__main__":
    raise SystemExit(main())
