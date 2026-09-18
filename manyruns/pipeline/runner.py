"""The one step loop, the one state schema, the one g-vector assembler.

The in-process step loop (`run_inproc`, under `vocab.INPROC`) and `manylatents` differ only in
how a `latent`/`lightning` step is computed, so they share everything else here. Two loops
previously produced divergent g-vector key sets and an `n_samples` that meant different things
per engine.

The in-process half was an ENGINE named `real` when that divergence was measured. It is a test
substrate now — absent from `app.ENGINES` and from `serving.LocalServer.SERVES` (CLAUDE.md's
"engine, tool, substrate" split) — so the names below say `INPROC` where they used to say
`real`. Nothing about the argument changed with the name: one loop, one state schema, one
assembler, because the alternative was measured and it drifted.
"""
from __future__ import annotations

from manyruns.installation import REINSTALL

import datetime as _dt
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any

from manyruns import __version__
from manyruns import artifacts as _artifacts
from manyruns.measurements import annotate
from manyruns.pipeline import bounds as _bounds
from manyruns.pipeline import io as _io
from manyruns.pipeline import mioflow as _mioflow
from manyruns.pipeline import prep as _prep
from manyruns.pipeline import probes as _probes
from manyruns.pipeline import steps as _steps
from manyruns.pipeline import stubs as _stubs
from manyruns.pipeline import suite as _suite
from manyruns.pipeline.io import _numeric_quiet
from manyruns.pipeline.loading import as_matrix, dense as _dense
from manyruns.vocab import CATEGORICAL_KINDS, INPROC, STEP_GROUPS
from manyruns.watch import describe


class ComputeCancelled(Exception):
    """The caller abandoned a computation before its state could be committed."""


@dataclass(frozen=True)
class _RetainedModel:
    """Parent-owned reference to a model whose Python graph cannot cross a spawn pipe."""
    token: int


_worker_models: dict[int, Any] = {}


def _compute_worker(connection, log_path=None):
    """Spawn entry point. Only trusted, parent-created calls cross this private pipe.

    Ordinary pickle bytes deliberately avoid torch's multiprocessing reducers: returned
    tensors must remain usable after this worker exits, without a shared-storage server.
    Retained models stay here for subsequent step calls, selected by the parent's reference.
    Valid stderr also lets the engine create its own multiprocessing resource tracker.
    """
    import os
    import pickle
    import sys
    import traceback

    if os.name == 'posix':
        os.setsid()  # cancellation also reaches the engine's nested worker processes
    diagnostic_error = None
    try:
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        quiet = open(log_path or os.devnull, 'a', buffering=1)
    except OSError as exc:
        # Diagnostics are best-effort, just like persisted step arrays. Never let
        # an unwritable log stop compute or reconnect native output to the live TTY.
        quiet = open(os.devnull, 'a', buffering=1)
        diagnostic_error = (f'compute diagnostics unavailable at {log_path}: {exc}; '
                            'child output discarded')
    with quiet:
        # Preserve both Python and native diagnostics outside Textual's capture.
        import faulthandler

        # Libraries also use sys.__stdout__ and native fd 1/2. Leaving those on the
        # TTY lets a lazily imported image detector probe the parent's live terminal.
        os.dup2(quiet.fileno(), 1)
        os.dup2(quiet.fileno(), 2)
        sys.stdout = sys.stderr = quiet
        faulthandler.enable(file=quiet)

        def report(rec, steps):
            connection.send_bytes(pickle.dumps(('progress', (rec, steps)), protocol=5))

        try:
            if diagnostic_error is not None:
                connection.send_bytes(pickle.dumps(('diagnostic', diagnostic_error), protocol=5))
            while True:
                request = pickle.loads(connection.recv_bytes())
                if request is None:
                    return
                fn, args, kwargs, progress = request
                if progress:
                    kwargs['on_step'] = report
                try:
                    result = fn(*args, **kwargs)
                    response = pickle.dumps(('result', result), protocol=5)
                except BaseException:
                    response = pickle.dumps(('error', traceback.format_exc()), protocol=5)
                connection.send_bytes(response)
        except (EOFError, BrokenPipeError):
            pass
        except BaseException:
            traceback.print_exc(file=quiet)
            raise
        finally:
            connection.close()


class ComputeProcess:
    """One lazy spawn worker per run; all waits belong to the daemon run thread.

    The parent remains the session authority. Each call sends its current inputs (including
    tune rollback), and commits returned state only after a complete response. No metric is
    removed or deferred. UI cancellation signals immediately; close reaps off the UI thread.
    Callers must prewarm multiprocessing before Textual captures stderr (tui.app documents why).
    """

    def __init__(self, cancelled=lambda: False, *, log_path=None):
        from threading import Event, Lock

        self.cancelled = cancelled
        self._stopped = Event()
        self._process_lock = Lock()
        self.process = None
        self.connection = None
        self.log_path = Path(log_path) if log_path is not None else None
        self._diagnostic_error = None

    def _signal(self, *, kill=False):
        import os
        import signal

        # Serialize signalling with detachment: once close hands the process to
        # a reaper, no UI cancellation may still be using that process handle.
        with self._process_lock:
            process = self.process
            if process is None:
                return
            try:
                if process.pid is None:
                    return
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
                elif kill:
                    process.kill()
                else:
                    process.terminate()
            except (ProcessLookupError, PermissionError):
                # Startup can race setsid; managed environments can also deny a group
                # signal after its leader exits. The owned process is still cancellable.
                try:
                    if process.is_alive():
                        process.kill() if kill else process.terminate()
                except (ProcessLookupError, ValueError):
                    pass
            except ValueError:
                pass

    def cancel(self):
        self._stopped.set()
        self._signal()

    def _raise_if_cancelled(self):
        if self._stopped.is_set() or self.cancelled():
            self.cancel()
            raise ComputeCancelled('run abandoned: compute cancelled')

    def _disconnected(self):
        self._raise_if_cancelled()
        self.process.join(.2)
        code = self.process.exitcode
        detail = f'exit code {code}'
        if code is not None and code < 0:
            import signal

            detail += f' ({signal.Signals(-code).name})'
        if self._diagnostic_error is not None:
            detail += f'; {self._diagnostic_error}'
        elif self.log_path is not None:
            detail += f'; diagnostics: {self.log_path}'
        # Cancellation can arrive during poll/recv, or during the diagnostic join.
        # All worker-loss paths must resolve it before recording a failed attempt.
        self._raise_if_cancelled()
        return RuntimeError(f'compute process disconnected: {detail}')

    def call(self, fn, *args, on_step=None, **kwargs):
        import multiprocessing
        import pickle

        self._raise_if_cancelled()
        if self.process is None:
            context = multiprocessing.get_context('spawn')
            self.connection, child = context.Pipe()
            self.process = context.Process(target=_compute_worker, args=(child, self.log_path),
                                           name='manyruns-compute')
            # Not daemonic: manylatents can itself use multiprocessing for an algorithm.
            try:
                self.process.start()
            except OSError:
                # A failed spawn has no child to join. Release both handles so
                # finalization can report the failure from the retained state.
                self.connection.close()
                self.connection = None
                self.process.close()
                self.process = None
                raise
            finally:
                child.close()
        try:
            self.connection.send_bytes(pickle.dumps((fn, args, kwargs, on_step is not None),
                                                    protocol=5))
            while True:
                self._raise_if_cancelled()
                if not self.connection.poll(.05):
                    if not self.process.is_alive():
                        raise self._disconnected()
                    continue
                kind, value = pickle.loads(self.connection.recv_bytes())
                if kind == 'diagnostic':
                    import warnings

                    self._diagnostic_error = value
                    warnings.warn(value, RuntimeWarning, stacklevel=2)
                elif kind == 'progress':
                    on_step(*value)
                elif kind == 'error':
                    raise RuntimeError(value)
                else:
                    return value
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise self._disconnected() from exc

    @staticmethod
    def _reap(process):
        process.join()
        process.close()

    def close(self):
        """Bounded shutdown on the run thread, including abnormal child exits."""
        if self.connection is not None:
            try:
                import pickle

                self.connection.send_bytes(pickle.dumps(None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.connection.close()
            self.connection = None
        if self.process is not None and self.process.pid is not None:
            self.process.join(.2)
            if self.process.is_alive():
                self._signal()
                self.process.join(.2)
            # Reap nested engine workers too, including after their parent exited.
            self._signal(kill=True)
            self.process.join(.2)
            with self._process_lock:
                process, self.process = self.process, None
            if process.is_alive():
                from multiprocessing.process import _children
                from threading import Thread

                # SIGKILL has been sent, but reaping can outlast the run's wait
                # budget. Transfer sole ownership without closing a live handle.
                # multiprocessing's exit handler otherwise joins this child again,
                # defeating the daemon reaper and blocking interpreter shutdown.
                _children.discard(process)
                Thread(target=self._reap, args=(process,), daemon=True,
                       name='manyruns-compute-reaper').start()
            else:
                process.close()
        self._stopped.set()


def compute_step(step, state, g, *, dispatch, ctx, index, carry, steps, on_step):
    """The unchanged step loop, returning one coherent snapshot with aliases intact.

    MIOFlow's model retains mioflow._run_mioflow_experiment's local _TimeDM class,
    which pickle cannot carry. Keep the intact model here; the parent's selected token
    remains authoritative, including when tuning rolls back to an earlier model. Probes
    resolve that token only in this same worker. All models are released at run close.
    """
    model = state.get('model')
    if isinstance(model, _RetainedModel):
        state['model'] = _worker_models[model.token]
    apply_step(step, state, g, dispatch=dispatch, ctx=ctx, index=index, carry=carry,
               steps=steps, on_step=on_step)
    model = state.get('model')
    if model is not None:
        token = id(model)
        _worker_models[token] = model
        state['model'] = _RetainedModel(token)
    return state, g, ctx, carry, steps


def step_group(step: dict) -> str | None:
    """The group a step declares, or None if it declares none (recorded as a loud skip)."""
    return step.get("group") or None



# ── ONE step loop, ONE g-vector schema (shared by every engine) ──────────────
#
# The in-process loop and `manylatents` run the SAME recipe over the SAME step vocabulary; only
# how a `latent`/`lightning` step is *computed* differs. Two loops produced three divergent
# g-vector key sets and an `n_samples` that silently meant different things per engine
# (input rows here, embedding rows there — and MIOFlow subsamples, so they genuinely
# differ). That is fatal for any experiment that compares g-vectors across engines, so
# there is now one loop, one state schema, and one assembler.

#: The core g-vector keys, each with exactly ONE meaning:
#:   n_samples  — rows of the INPUT data. Omitted when the input is a named engine dataset
#:                whose shape we never see (honest absence beats a key that means two things).
#:   n_features — columns of the input data (same caveat).
#:   n_embedded — rows of the FINAL embedding. Differs from n_samples whenever a step
#:                subsamples — MIOFlow does, at sample_size=256.
#:   final_dim  — columns of the final embedding.
GVECTOR_CORE = ("n_samples", "n_features", "n_embedded", "final_dim")

#: The subset of the core that describes the run's SIZE rather than its geometry. Declared
#: here — the producer still owns the vocabulary — but declared SEPARATELY from
#: `GVECTOR_CORE`, because the two answer different questions and are only coextensive today.
#: `narrate` suppresses these four from the measurement list and shows them once as the
#: before/after arrows; that suppression is correct for a size descriptor and WRONG for a
#: geometry number.
#:
#: Keeping one name for both is a live hazard, not a hypothetical: extending `GVECTOR_CORE`
#: with an always-computed geometry key (such as the margin) made that
#: key vanish from the panel entirely with the full suite green (measured: `margin` present
#: in the g-vector at 0.42, absent from every panel row, 436 passed). A reader deriving the
#: size set from the core set cannot see the difference, because both sides of its assertion
#: come from the same tuple. Add a core key here only if the panel should stop reporting it
#: as a measurement.
GVECTOR_SIZE = ("n_samples", "n_features", "n_embedded", "final_dim")

#: The state every executor sees. Seeded identically by both engines so a step can rely on
#: it — the analysis steps fall back to `state["X"]`, which the manylatents runner never used
#: to set (a latent KeyError its emb-precheck happened to mask).
#:
#: A CLOSED vocabulary, and `counts`/`genes` are a deliberate SCHEMA CHANGE, not a convenience.
#: Justified one key at a time, because a state key is a promise every executor may read:
#:
#:   counts — the untransformed expression matrix. It cannot be derived from `X`, and THE REASON
#:            MOVED AT THE CUTOVER rather than going away. `loading._anndata_matrix` used to run
#:            the preamble itself and return `obsm["X_pca"]` — (2700, 50) from a (2700, 32738)
#:            file — so the gene axis was gone before this loop saw anything. It returns
#:            `adata.X` as stored now: measured on `data/pbmc3k_raw.h5ad`, a (2700, 32738) CSR,
#:            and at open `state["X"] is state["counts"]`, the same object.
#:            What separates them is the preamble, which is a DECLARED step now. Measured, the
#:            same file through `[normalize(1e4), transform(log1p)]` on `apply_step`: `X` is a
#:            dense (2700, 32738) ndarray with `loading.looks_like_counts` False, while `counts`
#:            is still the CSR `adata.X` with `looks_like_counts` True. A library size is a sum
#:            of counts and not of log-ratios, so once the preamble has run `counts` is the only
#:            slot a gene-level readout (`steps._step_qc`, `steps._step_rank_genes`) can read.
#:   genes  — the name of each COLUMN of `counts`. Without it a gene-level readout returns
#:            column indices, which is not an answer to "which genes"; and it cannot ride
#:            inside `counts` (an ndarray/CSR carries no column labels).
#:
#: Two keys rather than one `{"counts": …, "genes": …}` dict, on purpose. A dict would read as
#: one indivisible fact while still letting the two halves be set independently, and they are
#: guarded in two different places for two different axes — `loading.gene_axis` checks one name
#: per COLUMN, `run_inproc` checks one row per cell against `X`. It would also make "does this
#: run have genes?" a two-level question for every reader. `label_kind` and the mock's `vec`
#: are the standing evidence that
#: this tuple is documentation and nothing enforces it — both are written into `state` and
#: neither is declared here (`vec` deliberately, `label_kind` not) — so
#: `tests/test_gene_axis.py` now pins `_new_state`'s keys against it.
#:
#: NOT in `artifacts.PERSISTED`, and that is load-bearing rather than incidental: that tuple
#: excludes `X` because copying the scientist's data into an outputs folder is a data-handling
#: decision (`artifacts.py` module docstring — "geometry leaves, the person's data does not"),
#: and `counts` is MORE of the scientist's data than `X` is, not less. `_persist` iterates
#: `PERSISTED`, so nothing here reaches disk; `test_the_gene_axis_is_never_written_to_disk`
#: fails if that stops being true.
#:
#: `frame`/`rows`/`cols` are the second schema change, and they are what makes the seven keys
#: above safe once a step may REMOVE cells. `frame` is the data exactly as loaded (see
#: `pipeline/frame.py`); `rows`/`cols` are int64 index arrays into it, and every row-indexed
#: value is `frame[...][rows]` computed on demand rather than a copy something has to keep in
#: step. `counts`/`genes`/`labels` stay as top-level slots for every reader that predates the
#: frame (`steps._step_qc`, `steps._step_rank_genes`, `_finalize`), rewritten from the view by
#: `_apply_transition`; new readers take `frame.view(...)`, which is the only form that cannot
#: go stale. The frame is NOT persisted either, for the same reason `counts` is not — it is the
#: same bytes, and `_persist` still iterates `artifacts.PERSISTED` alone.
#:   model  — the TRAINED OBJECT a lightning step produced, not its coordinates. The one slot
#:            here that is CALLABLE: a transcript answers "what did the embedding look like",
#:            a model
#:            answers "what happens to THESE cells". In memory only — deliberately absent from
#:            `artifacts.PERSISTED`, because a LightningModule carries weights and pickling one
#:            into an outputs folder is a data-handling decision nobody has taken.
#:   layers — a SECOND MATRIX over the same cells and the same genes, `{name: matrix}`
#:            (spliced/unspliced counts, a protein panel). A frame slot rather than a derived
#:            one, and the distinction is the one `velocity` got wrong: a layer is raw counts
#:            per cell per gene, so subsetting rows and columns is exactly right and it SURVIVES
#:            a narrowing by construction. Carried and narrowed, never transformed — see
#:            `frame.new_frame`. Absent from `artifacts.PERSISTED` for the reason `counts` is:
#:            it is more of the scientist's data, not less.
_STATE_KEYS = ("X", "emb", "pseudotime", "labels", "label_key", "seed", "counts", "genes",
               "frame", "rows", "cols", "model", "layers", "velocity")


#: Re-exported from `steps`, which is where it has to be DEFINED (this module imports that
#: one at module scope, so the reverse import cannot exist) — but this is where the loop that
#: catches it lives, so the name stays reachable here for every existing caller.
_StepSkipped = _steps._StepSkipped
_StepUnsupported = _steps._StepUnsupported


def _new_state(X: Any = None, labels: Any = None, seed: Any = None, emb: Any = None,
               counts: Any = None, genes: Any = None, layers: "dict | None" = None) -> dict:
    """The state a run starts from. `emb` is a PARAMETER, not always None.

    It was hardcoded to None, so `state["emb"]` could only ever be written by a latent step —
    which made "an embedding exists" a statement about provenance rather than about the data.
    A caller holding a precomputed embedding, or a graph built by another tool with compatible
    dimensions, had no way to hand it in: the trajectory step would refuse, correctly by the
    letter of its rule and wrongly in substance.

    `counts`/`genes` are the gene axis (see `_STATE_KEYS`), and they are seeded HERE rather
    than in each runner so the two engines cannot disagree about whether a run has one — the
    divergence this module's header exists to record.

    THE FRAME IS SEEDED HERE FOR THE SAME REASON, and it is why all three state builders
    (`run_pipeline`, `run_manylatents`, `session.Session`) get one without saying so: a prep
    step declines on a state with no frame (`_run_prep_step`), so a builder that missed it
    would have every filter report `skipped` on that route alone.

    `rows`/`cols` are seeded from whichever of `counts` or `X` has two dimensions, `counts`
    first: on pbmc3k those are 2700 x 32738 and 2700 x 50, and the gene axis a column filter
    narrows exists only in the first. An embedding-only run seeds rows from `emb`,
    leaving the gene selection empty. A run with none (a named manylatents dataset, loaded
    inside the engine) gets empty selections — honest, and `_run_prep_step` declines on it
    rather than filtering a matrix this process never saw."""
    from manyruns.pipeline import frame as _frame

    shaped = counts if getattr(counts, "ndim", 0) == 2 else X
    n_rows, n_cols = (shaped.shape if getattr(shaped, "ndim", 0) == 2 else (0, 0))
    if getattr(shaped, "ndim", 0) != 2 and getattr(emb, "ndim", 0) == 2:
        n_rows = emb.shape[0]
    return {"X": X, "emb": emb, "pseudotime": None, "labels": labels, "label_key": None, "seed": seed,
            "counts": counts, "genes": genes, "model": None,
            "layers": dict(layers or {}),
            # SUPPLIED by a tool that ran elsewhere, never computed here. DERIVED, not frame:
            # a field is fitted over a neighbourhood, so a narrowing invalidates the rows that
            # remain — the same rule as `emb`, and the opposite of the `layers` it came from.
            "velocity": None,
            "frame": _frame.new_frame(counts=counts, genes=genes, labels=labels, layers=layers),
            "rows": _frame.full_selection(n_rows),
            "cols": _frame.full_selection(n_cols)}


# What a step attempt can come to is declared ONCE, in `watch.OUTCOMES` — the record's home,
# which this module already imports `describe` from. `STEP_OUTCOMES` stood here as a second
# copy of the identical 4-tuple; grepped across manyruns, the learner, manylatents and shop,
# both copies had zero readers, which is exactly why they could have drifted in silence.
# The literals the loop below writes are pinned to that tuple by
# `test_the_step_loop_only_ever_writes_a_declared_outcome`.


def _status_line(rec: dict) -> str:
    """A record → the legacy `status` string, byte-identical to what the old loop wrote."""
    outcome = rec["outcome"]
    if outcome == "ok":
        return "ok"
    if outcome == "reported":
        return rec["detail"]
    if outcome == "skipped":
        return f"skipped ({rec['detail']})"
    return f"error: {rec['detail']}"


def _live_suite() -> list:
    """The declared suite, or `[]` if it cannot be read.

    `_finalize` calls `catalog.load_suite()` bare and should — a missing suite there is a
    g-vector with no geometry in it, which is a real failure. Here it is an overlay, and an
    overlay that can abort a run is worse than no overlay: this is the same rule `_report`
    states for the dashboard, applied to the thing the dashboard displays."""
    try:
        from manyruns import catalog

        return catalog.load_suite()
    except Exception:  # noqa: BLE001 - a view never fails the run it is a view of
        return []


def _attach_geometry(rec: dict, emb: Any, declared: list, previous: dict) -> dict:
    """Give the step its own geometry delta. Returns what the NEXT step inherits.

    ``rec["geometry"] = {"lid": [from, to], ...}``, a two-element list per metric. `from` is
    None when there was no prior embedding: the first embedding step is an APPEARANCE, not a
    change, and a baseline of 0 would be an invention — `lid` did not go from 0 to 3.4, it
    did not exist. The key is omitted entirely when nothing could be measured, so a stackless
    run carries no empty dict for a reader to mistake for a measurement that came back empty.

    Deliberately NOT in the g-vector. The g-vector is the record and its key set is the
    comparability contract (`suite.measure` in `_finalize`, all twelve names, authoritative);
    this is a view, on a cheap subset, and mixing the two would make "which lid?" a real
    question. Returning `now` even when it is empty is the honest carry-forward: an embedding
    that changed and could not be measured leaves the next step's `from` as None rather than
    pairing it against a value two steps stale.

    No duration field, and that is not an oversight: `test_a_run_without_an_observer_is_
    unchanged` pins two runs of one recipe to identical step records modulo `seconds`, and a
    second clock — first call 1.8 s for the metric registry import, 0.008 s warm — breaks
    that determinism for a number that is the same for every step anyway. The cost is stated
    where costs belong, in `LIVE_SUBSET`'s measured table."""
    now = _suite.live(emb, declared)
    if now:
        rec["geometry"] = {name: [previous.get(name), value] for name, value in now.items()}
    return now


def _persist(rec: dict, state: dict, ctx: dict, index: int, name: str, written: dict,
             before: dict, before_extras: dict | None = None) -> None:
    """Write this step's derived state to disk and record where it went.

    A slot is written only when this step actually REPLACED it — object identity, the same
    test `_attach_geometry` uses — so a step that reads the embedding without changing it does
    not write a second identical copy under a different name. Checked per SLOT rather than on
    the embedding alone: `mioflow` in-process sets `pseudotime` and leaves `emb`
    untouched, so keying the whole write on the embedding dropped every ordering this product
    computes.

    A size skip is RECORDED rather than silent: `artifacts.save_array` returns None both when
    an array is too large and when the disk refused it, and those are different facts to a
    reader wondering why a run has no state folder."""
    out_dir, run_id = ctx.get("out_dir"), ctx.get("run_id")
    if out_dir is None:
        return
    for slot in _artifacts.PERSISTED:
        value = state.get(slot)
        if value is None or not hasattr(value, "shape") or value is before.get(slot):
            continue
        path = _artifacts.save_array(out_dir, run_id, index, name, slot, value)
        if path:
            rec.setdefault("artifacts", {})[slot] = path
            written[path] = {"step": name, "index": index, "slot": slot}
            if slot == "emb":
                # Rewind consults the chosen output, even if later steps lose alignment.
                rec["metadata_aligned"] = ctx.get("metadata_aligned", True)
            if ctx.get("sample_ids") is not None and ctx.get("metadata_aligned", True):
                # The annotation belongs to THIS artifact's row order, including any
                # filtering already applied. This assumes the executor preserves selected row
                # order; equal length cannot prove correspondence. The PCA integration
                # oracle checks that contract, not arbitrary executor outputs.
                # No raw expression matrix is copied.
                ids = ctx["sample_ids"][state["rows"]]
                labels = state.get("labels")
                if len(ids) == len(value) and (labels is None or len(labels) == len(ids)):
                    written[path]["row_identity"] = {
                        "sample_ids": ids.tolist(),
                        "labels": None if labels is None else list(map(str, labels)),
                        "label_kind": state.get("label_kind"),
                    }
        elif _artifacts.too_large(value):
            rec.setdefault("artifacts", {})[slot] = (
                f"not written: larger than {_artifacts.MAX_BYTES // (1024 * 1024)} MiB"
            )

    # ENGINE BYPRODUCTS, written on the same terms as manyruns's own slots — see
    # `_collect_extras`. Kept in a separate dict rather than added to `PERSISTED` because
    # `PERSISTED` is a DECLARATION about manyruns's state (and about what it refuses to
    # write: `X`, `labels`), while these are whatever the engine chose to hand back this run.
    # One is a fixed list with reasons; the other cannot be known ahead of the call.
    #
    # The size cap earns its keep here rather than on the embedding: an affinity is N×N, so
    # pbmc3k's 2,700 cells is 55.6 MiB and 5,000 cells is 190.7 MiB — over the 64 MiB cap,
    # which is then RECORDED as a stated absence rather than silently skipped.
    cap = _artifacts.EXTRAS_MAX_BYTES
    for slot, value in (state.get("extras") or {}).items():
        if value is (before_extras or {}).get(slot):
            continue
        path = _artifacts.save_array(out_dir, run_id, index, name, slot, value, max_bytes=cap)
        if path:
            rec.setdefault("artifacts", {})[slot] = path
            written[path] = {"step": name, "index": index, "slot": slot}
        elif _artifacts.too_large(value, cap):
            # The SHAPE is in the note, not just the refusal: "affinity was too big" and
            # "affinity was 2700x2700" are different amounts of help to a reader deciding
            # whether to raise the cap, and the shape is the whole reason it was too big.
            shape = "x".join(str(n) for n in getattr(value, "shape", ()) or ())
            rec.setdefault("artifacts", {})[slot] = (
                f"not written: {shape} is larger than {cap // (1024 * 1024)} MiB"
            )


#: Keys of an engine's return that manyruns takes for itself, and therefore does NOT treat as
#: a byproduct. Declared rather than inferred, because "everything else" is the rule below and
#: a rule with an undeclared exception list is a rule nobody can check.
#:
#:   embeddings — becomes `state["emb"]`; the whole point of a latent step.
#:   scores     — folded into the g-vector as `<step>.<metric>`, which is how the declared
#:                suite reaches the results table.
#:   label      — the caller's OWN labels, echoed back. manyruns already holds them in
#:                `state["labels"]`, and writing the engine's copy beside them would make the
#:                outputs folder hold two accounts of one input with nothing saying which won.
#:   metadata   — manylatents' provenance for its own run (`source`, `algorithm_type`,
#:                `data_shape`). manyruns has `run_id`/`spec_id`/`tool_version` for that
#:                question; a second provenance block invites a reader to trust the wrong one.
ENGINE_CLAIMED = ("embeddings", "scores", "label", "metadata")


def _collect_extras(name: str, out: Any, state: dict, g: dict) -> None:
    """Route whatever the engine returned BESIDES the embedding, instead of dropping it.

    **Measured, on a plain PHATE run through `api.run` today:** the returned dict is
    `embeddings, label, metadata, affinity, kernel` — and manyruns read exactly two of those
    keys (`runner.py`'s `out.get("embeddings")` and `out.get("scores")`). So every latent step
    this product has ever run has discarded two N×N matrices the engine had already computed.
    `manylatents.experiment` merges `algorithm.extra_outputs()` into the top level of its result
    (experiment.py:443), so this is not a new channel — it is one that was never read.

    It is not only affinity. `LatentModuleBase.extra_outputs` collects `trajectories`,
    `affinity`, `adjacency` and `kernel`, and `Cflows` — a LightningModule — adds
    `grn_edges`/`grn_weights`/`grn_node_ids`, a gene regulatory network decoded back to gene
    space. That last one is a FINDING, and it was going in the bin.

    The split follows what the thing IS, not what produced it:

      * a scalar goes to the **g-vector**, namespaced `<step>.<key>`, exactly as `scores` are —
        so it is comparable across runs by the same fixed-key rule everything else obeys;
      * anything with a shape goes to the **artifact channel**, which already has the atomic
        write, the manifest and the size cap. Namespaced the same way, so two steps that both
        emit `affinity` do not overwrite each other.

    Nothing here decides whether a byproduct is *interesting*. It decides that a byproduct
    which crossed the boundary is written down — the same argument `artifacts.py` makes about
    the embedding, one layer out.
    """
    if not isinstance(out, dict):
        return
    for key, value in out.items():
        if key in ENGINE_CLAIMED or value is None:
            continue
        if hasattr(value, "shape"):
            state.setdefault("extras", {})[f"{name}.{key}"] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            g[f"{name}.{key}"] = value


def new_carry() -> dict:
    """Open a lineage: the three RUN-scoped accumulators, made explicit.

    They were locals in `_run_steps`, which is fine for a `for` loop and impossible for a
    driver that applies one step per user action — a session has no loop body to hold a
    local in, so between actions the locals would be re-created and the accumulation lost.
    Each key is here because I measured what its absence costs: three `latent` steps
    (`a`, `b`, `a`) driven through `apply_step` on a 40x4 array, once with one carry and
    once with a fresh carry per step, everything else held identical.

      declared — the metric suite, read ONCE. A run-level constant; re-reading the YAML per
                 step costs more than the metrics do (`_live_suite`).
      geometry — the last measured values, carried forward so `rec["geometry"]` is a DELTA.
                 One carry: lid reads `[None, 1.0] [1.0, 2.0] [2.0, 3.0]`. Fresh carry per
                 step: `[None, 1.0] [None, 2.0] [None, 3.0]` — every step an appearance, no
                 change ever expressible, and nothing in the record says so.
      written  — the manifest accumulator, and `finish` WRITES IT WHOLE. One carry: the
                 COMPLETE manifest lists 3 files for the 3-file folder. Fresh carry per
                 step: 1 entry for that same 3-file folder — a manifest that lies.
    """
    return {"declared": _live_suite(), "geometry": {}, "written": {}}


def finish_carry(carry: dict, ctx: dict) -> str | None:
    """Close the lineage — the ONLY writer of the completion marker.

    A lineage that wrote nothing writes no marker either: an empty state folder and an
    interrupted one must not look alike. Deliberately NOT inside `apply_step`. Called per
    step it marks the folder complete while more steps are still coming, which destroys the
    one guarantee the marker exists to give — measured: after step 1 of 3,
    `artifacts.complete()` was already True with two steps left to run."""
    if carry["written"]:
        return _artifacts.finish(ctx.get("out_dir") or ".", ctx.get("run_id"), carry["written"])
    return None


def open_gvector(state: dict) -> dict:
    """The g-vector a lineage OPENS with: the size of the data that entered it.

    `n_samples`/`n_features` are the input's shape, and `GVECTOR_CORE` says so verbatim —
    "rows of the INPUT data". `_finalize` read them off `state["X"]` at the END, which was the
    same array until a `prep` filter could narrow it. Measured on `data/pbmc3k_raw.h5ad` with
    `[filter_cells(min_genes=500), phate, mioflow]` (218 cells dropped): the g-vector came back
    `{'n_samples': 2482, 'n_features': 50, 'n_embedded': 2482, 'final_dim': 3}` and the number
    2700 appeared nowhere in it. Since `n_embedded` is narrowed by the same amount, nothing in
    the record said cells had left at all — and `narrate` picks `⇢` over `→` exactly when
    `n_samples != n_embedded` (narrate.py:541-544), so the one surface built to show it printed
    `cells 2,482 → 2,482` and named the post-filter count as the count that entered.

    Seeded at open by all three lineage builders, for the same reason `new_carry` is: a fact
    about the run's start cannot be recovered once the loop has moved on. `_finalize` keeps its
    `setdefault` as the fallback for a caller that assembles a g-vector without opening one.

    Empty when the shape is unknown — a named manylatents dataset, loaded inside the engine
    (`run_manylatents` seeds `X=None` there). Honest absence beats a key that means two things,
    which is the rule `GVECTOR_CORE` already states for exactly this case.

    **`n_features` MOVED at the cutover, on every scRNA run, and that is a SCHEMA BREAK.** It
    is 32738 on pbmc3k where it was 50, because 50 was the undeclared preamble's PCA output and
    the preamble is gone (`loading._anndata_matrix`). The key has been wrong since it was
    introduced — `GVECTOR_CORE` defines it as the columns of the INPUT data, and a PCA output is
    not input — and `steps._step_qc`'s docstring named the defect ("`qc_n_genes` is the true
    width, and the g-vector's `n_features` is not") while declining to fix it there, because
    "changing a core key breaks comparability across every stored run". This is where that break
    is taken. A g-vector stored before this commit is not comparable to one stored after it on
    this key, and nothing can reconcile the two: the old number does not say how many genes the
    data had.
    """
    X = state.get("X")
    if getattr(X, "ndim", 0) != 2:
        return {}
    return {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}


#: State slots a step DERIVED, and which a narrowing therefore invalidates. The mirror of
#: `vocab.frame_facts` on the data side: `vocab` says the FACT `embedding` is gone, this says
#: the ARRAY is. Both have to be true or a recipe is refused at plan time while the array it
#: was refused over is still sitting in state.
#: `model` is here for the same reason `emb` is and it is the more important of the two: a flow
#: fitted over cells a filter has since dropped is not a model of the ones that remain, and
#: unlike a stale embedding it can be ASKED questions — so a probe would return a confident,
#: well-formed, wrong answer. `vocab.STEP_PRODUCES["mioflow"]` makes the same statement at plan
#: time; both have to hold or the two disagree about one recipe.
DERIVED_SLOTS = ("emb", "pseudotime", "model", "velocity")


def _apply_transition(out: dict, state: dict, rec: dict, ctx: dict) -> None:
    """A prep executor's result, applied to the state. The ONLY place a mask meets a selection.

    A `{"X": …}` result replaces the working matrix and narrows nothing, so nothing goes stale.
    A `{"mask": …, "axis": …}` result composes the selection, rewrites the frame-derived slots
    from the new view, subsets the working matrix, and clears every derived array.

    FIVE things move together on a narrowing, and every one of them fails SILENTLY when
    missed — which is the argument for doing it here once rather than in each executor:

      rows/cols     the selection itself; `frame.narrow` refuses a wrong-length mask.
      counts/genes/labels   re-read from the frame, never narrowed in place.
      X             the WORKING matrix, subset on the SAME axis the mask names. Subset rather
                    than cleared: at this point it is `normalize`/`transform` output and those
                    are per-cell, so a surviving cell's values do not depend on a dropped
                    one's. `emb` and `pseudotime` ARE fitted over all cells and are cleared.
      ctx["array"]  the run-scoped input, which is not decoration: `_ml_latent` embeds it
                    (`kwargs["input_data"]`) whenever no embedding exists yet, and
                    `bounds.shape_of` resolves a step's `limits:` against it BEFORE
                    `state["X"]`. Left alone, a filtered run fits the cells the filter
                    removed and reports `ok`.
      ctx["color"]  built ONCE at run open (`run_manylatents`) and never rebuilt, so a
                    narrowing that skipped it leaves `_save_scatter` to drop the colouring
                    on a length mismatch (`io.py:106`) — a grey plot and no explanation.
                    Rows only: a gene filter moves no cell, so the colouring still fits.

    THE WORKING MATRIX MOVES ON BOTH AXES, and the column half is not symmetry for its own
    sake. Measured with only the row half in place, `[embeds, hvg(keep 3 of 6), embeds]`
    through `run_pipeline`: the record said `dropped {'axis': 'cols', 'n': 3}` and
    `invalidated ['emb']`, and the matrix each `embeds` received was (12, 6) BOTH TIMES —
    `np.array_equal(before, after)` on the two embeddings, i.e. the refit was byte-identical
    to the one just discarded, `n_features` still 6, and `state["counts"]` (12, 3) beside a
    `state["X"]` of (12, 6). Two disagreeing accounts of the gene axis in one state, and the
    filter reached nothing that computes.

    SPEC §10.1, DECIDED HERE: a derived array whose length is not `len(rows)` is UNINDEXED and
    nothing indexes it. That is why `X` is subset only when it has exactly the entries the
    selection had on the narrowed axis, and why `emb`/`pseudotime` are cleared rather than
    masked. Measured on `cflows` over pbmc3k (2700 cells, run fe040e407533): MIOFlow's
    `sample_size=256` samples inside training only
    (`manylatents/algorithms/lightning/mioflow.py:180,234,248`) and `encode` integrates the
    whole input, so `01-mioflow_emb.npy` is (2700, 3) — the case the spec was written around
    does not exist. What IS shorter in that same run is PHATE's
    landmark affinity, 1811x1811 on 2700 cells, which lives in `state["extras"]` and is not a
    subset of cells at all: no sub-selection of `rows` could describe it, so the alternative
    the spec offered was never expressible for the one array that needed it.
    """
    import numpy as np

    from manyruns.pipeline import frame as _frame

    if "X" in out:
        new_x, rows = out["X"], state.get("rows")
        if getattr(new_x, "ndim", 0) == 2 and rows is not None and new_x.shape[0] != len(rows):
            # A transforming step that silently changed the cell count. Refused rather than
            # accepted, because from here on `X` and the selection would disagree and every
            # readout that joins them (labels against clusters, genes against cells) would be
            # off by an unrecorded amount. A step that means to drop cells returns a mask.
            raise ValueError(
                f"{rec.get('name')!r} returned a matrix with {new_x.shape[0]} rows for a "
                f"selection of {len(rows)} — a prep step that removes cells returns a mask")
        state["X"] = new_x
        if ctx.get("array") is not None:
            ctx["array"] = new_x
        return
    axis = "cols" if out.get("axis") == "cols" else "rows"
    at = 0 if axis == "rows" else 1
    before = int(state[axis].shape[0])
    mask = np.asarray(out["mask"], dtype=bool)
    holders = [(holder, key, holder.get(key)) for holder, key in ((state, "X"), (ctx, "array"))]
    if axis == "cols":
        # THE ONE CASE THE WIDTH GUARD CANNOT REPAIR, refused rather than half-applied. A row
        # mask and the working matrix always agree — both are one row per selected cell — but a
        # COLUMN mask is sized against the frame's gene axis, and `loading._anndata_matrix`
        # hands over `obsm["X_pca"]` (measured (2700, 50) from a (2700, 32738) file). So a
        # `filter_genes`/`hvg` declared after a reduction would narrow `counts`/`genes` while
        # the matrix every later step fits keeps all 32,738 columns — `dropped {'n': 30738}`
        # recorded for a narrowing nothing computes over, which is the silent-failure mode the
        # frame model exists to remove. Refused BEFORE the selection moves, so the state a
        # step leaves on `error` is the state it started from.
        for _holder, key, value in holders:
            if getattr(value, "ndim", 0) == 2 and value.shape[1] != before:
                raise ValueError(
                    f"{rec.get('name')!r} filters genes, but {key!r} is "
                    f"{value.shape[0]} x {value.shape[1]} against a selection of {before} "
                    f"genes — a gene filter has to run before the gene axis is reduced")
    state[axis] = _frame.narrow(state[axis], mask)
    remaining = int(state[axis].shape[0])
    # RECORDED even at zero, because "filter_cells dropped 0 cells" is a measurement — on
    # pbmc3k `min_genes=200` drops exactly 0 of 2700 — and an absent key cannot state it. What
    # is guarded on the count is the INVALIDATION below and `session.branch`'s refusal, both of
    # which used to fire on a filter that changed nothing.
    rec["dropped"] = {"axis": axis, "n": before - remaining, "remaining": remaining}

    view = _frame.view(state["frame"], state["rows"], state["cols"])
    state["counts"], state["genes"], state["labels"], state["layers"] = (
        view["counts"], view["genes"], view["labels"], view["layers"])
    state["label_key"] = ctx.get("label_key")
    for holder, key, value in holders:
        if getattr(value, "ndim", 0) == 2 and value.shape[at] == before:
            holder[key] = value[mask] if at == 0 else value[:, mask]
    if axis == "rows":
        ctx["color"] = _io._labels_to_numeric(state["labels"])
    # OUTSIDE the `axis == "rows"` guard, and it must stay there. `vocab.NARROWING_STEPS` is
    # AXIS-BLIND: it clears `embedding` on a column narrowing too, because the rule was decided
    # on FITTING rather than on length (a kNN graph over 32,738 genes is not the graph over the
    # 2,000 `hvg` kept). Clearing only on rows here would leave `unmet` refusing a recipe whose
    # `emb` is still populated — the plan-time/run-time disagreement `STEP_NEEDS`' mioflow note
    # exists to prevent.
    #
    # AND NOTHING IS CLEARED WHEN NOTHING WAS REMOVED, which is the other half of the same
    # agreement: `NARROWING_STEPS` is a DICT keyed by the param that makes a step narrow, so
    # `detect_doublets` with `remove: false` "flags without narrowing" (spec §5) and
    # `step_narrows` answers False for it. Measured before this guard, on
    # `[embeds, detect_doublets(remove=False), mioflow]`: plan time `unmet -> frozenset()` —
    # fully legal — and run time `invalidated: ['emb']` with mioflow recorded as
    # `skipped (mioflow needs an embedding to run on)`. `narrate.CLEARED_FACT` could not
    # explain it either, because `vocab.invalidated` is empty for a step that does not narrow.
    invalidated = ([k for k in DERIVED_SLOTS if state.get(k) is not None]
                   if remaining != before else [])
    for key in invalidated:
        state[key] = None
    if invalidated:
        rec["invalidated"] = invalidated


def apply_step(step: dict, state: dict, g: dict, *, dispatch: dict, ctx: dict,
               index: int, carry: dict, on_step: Any = None,
               steps: list | None = None) -> dict:
    """Apply ONE step — the unit the recipe loop and an interactive session both want.

    ``dispatch`` maps a step GROUP to an executor:

        executor(name, params, state, g, ctx) -> str | None

    which mutates ``state["emb"]``/``g``, returns an optional status override, and raises
    to fail (:class:`_StepSkipped` to decline).

    Mutates ``state`` and ``g`` (through the executor) and ``carry["geometry"]`` /
    ``carry["written"]``. Returns the step record.

    **Never raises.** `_StepSkipped` becomes ``outcome="skipped"`` and anything else becomes
    ``outcome="error"`` — one bad step must not kill the loop, and a session has no outer
    `try` for it to land in at all.

    **Step-scoped only.** Run identity, the full `_suite.measure`, `_artifacts.finish` (see
    `finish_carry`) and `_finalize` are deliberately absent: they close a lineage, and a
    lineage driven one action at a time is not closed after each action.

    ``step`` is a DICT, not a name, and that is the point of extracting this at all: the
    declared ``params`` reach the executor as ``rec["params"]``. An interaction surface whose
    unit is a bare string cannot carry them, and `_ml_latent`'s own comment records what that
    costs — ``final_dim`` stuck at the engine default of 2 whatever the recipe declares.

    ``index`` is the CALLER's, never derived here — a step does not know its own position in
    a lineage that is still being written. It addresses the artifact file
    (`artifacts.save_array` → ``NN-name_slot.npy``), so a driver that restarts it at 0 per
    action silently destroys data. Measured, three steps `a`, `b`, `a`, each assigning a new
    embedding: with the caller's monotonic 0/1/2 the folder holds
    ``00-a_emb.npy 01-b_emb.npy 02-a_emb.npy``; with `index` pinned at 0 it holds
    ``00-a_emb.npy 00-b_emb.npy``. The second `a` overwrote the first `a`'s embedding: both
    records still report `ok`, both `artifacts` paths still resolve, and loading the first
    one back returns the THIRD step's array. Nothing anywhere says the first is gone.

    ``steps`` is the lineage's record list, and ``rec`` joins it HERE, before the executor
    runs — not in the caller afterwards. That is what the observer contract promises:
    `on_step(rec, steps)` hands a live panel the list it renders (`shell.live_dashboard` →
    `_progress_panel`), which reads a step as `queued` unless a record for it is in that list.
    Measured over those same three steps, the `running` reports carry lists of length 1, 2, 3
    — each holding its own record and every earlier one. Leaving `steps` unset and appending
    the returned `rec` in the caller instead gives 1, 1, 1: the panel would show the running
    step correctly and re-render every FINISHED step as queued.

    The ordering below is the point. The old loop appended to `trace` *before* the group
    check, the dispatch lookup and the executor call, so the trace recorded what was
    ATTEMPTED while reading, to every consumer, as what EXECUTED. `outcome` and `seconds`
    are therefore written only after the executor returns or raises."""
    name = step.get("name")
    group = step_group(step)
    # params are copied: an executor that mutates its params dict must not rewrite the
    # recipe's own record of what it was asked to do.
    rec: dict[str, Any] = {
        "index": index, "name": name, "group": group,
        "params": dict(step.get("params") or {}),
        "constraints": _bounds.resolved_constraints(step, _bounds.shape_of(state, ctx)),
        "outcome": None, "detail": None, "seconds": 0.0,
    }
    lineage = steps if steps is not None else []
    lineage.append(rec)
    # `running` is a live state, not an outcome: a dashboard needs to show the step it
    # is ON, and the outcome vocabulary must keep meaning "how it ended". It is
    # overwritten below and never survives into a finished record.
    # `started` is a monotonic stamp the dashboard ticks from — a renderer that
    # recomputes elapsed at RENDER time keeps moving while this thread is blocked in
    # the fit, which is what makes a live view possible with no worker thread.
    rec["state"], rec["started"] = "running", time.monotonic()
    _report(on_step, rec, lineage)
    if group is None:
        rec["outcome"], rec["unsupported"] = "skipped", True
        rec["detail"] = f"step declares no 'group'; expected one of {STEP_GROUPS}"
        _settle(rec)
        _report(on_step, rec, lineage)
        return rec
    # THE `tool:` SEAM, ahead of the group lookup and only ahead of it. A step naming a tool
    # keeps its honest `group` — the calculus reads that, and nothing here changes what `unmet`
    # decides — but its IMPLEMENTATION is in another interpreter, so the group's engine executor
    # is the wrong one. `pipeline/external.py` argues why this is the substrate axis rather than
    # a new group or a new engine.
    if step.get("tool"):
        from manyruns.pipeline import external as _external

        fn = _external.run_external_step
    else:
        fn = dispatch.get(group)
    if fn is None:
        rec["outcome"], rec["unsupported"] = "skipped", True
        rec["detail"] = f"no {group!r} executor on this engine"
        _settle(rec)
        _report(on_step, rec, lineage)
        return rec
    # THE STEP'S OWN `limits:`, resolved against the matrix it will actually see. `n_components:
    # 10` is a scRNA default and it broke three of eight bundled recipes on a 150×3 array — the
    # precondition calculus is right to say nothing (the data is present, the recipe composes;
    # what is wrong is a number the recipe alone can cap). `_bounds` holds no table and knows no
    # method: a step with no `limits:` passes through untouched and the engine's own refusal is
    # what gets recorded. Written back into `rec["params"]`, so the record says what RAN rather
    # than what was asked, and echoed into the caveats so no surface shows a clamped number with
    # nothing beside it.
    rec["params"], bounded = _bounds.fit(step, rec["params"], _bounds.shape_of(state, ctx))
    if bounded:
        rec["bounded"] = bounded
        if isinstance(ctx.get("caveats"), list):
            ctx["caveats"].extend(bounded)
    # What the step INHERITED, so `produced`, `geometry` and `_persist` below each report a
    # change rather than a snapshot. `before_obj` is the embedding OBJECT, not its description:
    # `describe` collapses to shape+dtype, so a step that re-embeds to the same shape reads as
    # unchanged there. Identity is the honest test of "this step assigned a new embedding", it
    # is the ONE test all three now use, and it must be `is`, since `!=` on two arrays is
    # elementwise.
    before_keys, before_obj = set(g), state.get("emb")
    before_values = dict(g)
    # Figures are appended to a RUN-scoped list by `_io._save_scatter`, so until now nothing
    # recorded WHICH step drew WHICH figure. That is why the live panel could never show the
    # plot for the step that just ran — not a rendering gap, a record gap. Diffed the same way
    # `emitted` is, rather than threaded through every executor's signature.
    before_plots = len(ctx.get("plots") or ())
    before_slots = {k: state.get(k) for k in _artifacts.PERSISTED}
    # Diffed by identity like the slots above, so a step that only READS an extra another
    # step produced does not deposit a second copy of it under its own name.
    before_extras = dict(state.get("extras") or {})
    started = time.perf_counter()
    try:
        before_labels = state.get("labels")
        try:
            override = fn(name, rec["params"], state, g, ctx)
        finally:
            # Derived labels no longer name the source column. Keep provenance in
            # state so tune rollback restores it together with the labels themselves.
            if state.get("labels") is not before_labels:
                state["label_key"] = None
        # Applied INSIDE the `try` on purpose: a transition can be refused (a mask sized
        # against the wrong frame — `frame.narrow`; a matrix whose row count moved), and this
        # function never raises. Refusing it here makes it this step's `error` with the reason
        # in `detail`, which is what every other bad step already does.
        if isinstance(override, dict):
            _apply_transition(override, state, rec, ctx)
            # THE REPORT LANDS AFTER THE TRANSITION, and the ordering is the whole reason this
            # merge lives here rather than in `_run_prep_step` — see that function's docstring
            # for the measurement. `_apply_transition` refuses a transition it cannot apply
            # (a mask sized against the wrong frame; a column mask against a matrix with no
            # gene axis; a matrix whose row count moved), and it refuses BEFORE the selection
            # moves so the state a failed step leaves is the state it started from. The
            # g-vector is part of that state: a `filter_genes.dropped: N` for a narrowing that
            # was refused is a record of work that was rolled back, which is precisely what the
            # refusal exists to prevent one slot over. On `error` this line is never reached,
            # so `rec["emitted"]` comes back empty and the two agree.
            for key, value in (override.get("report") or {}).items():
                g[key] = value
                annotate(g, key, kind="setting" if key in _prep.REPORT_SETTINGS else "measurement",
                         stage=f"step {index + 1}: {name}")
    except _StepSkipped as e:
        rec["outcome"], rec["detail"] = "skipped", str(e)
        # The flag, set at the ONE place both kinds of decline converge. Present-and-True only
        # for a capability gap; a benign decline carries no key at all rather than `False`, so
        # a record written before this split reads as benign — which is what it was, since the
        # unsupported cases were the minority and a reader has no way to re-derive intent from
        # a detail string after the fact.
        if isinstance(e, _StepUnsupported):
            rec["unsupported"] = True
    except Exception as e:  # noqa: BLE001 - one bad step must not kill the run
        # Preserve the full engine exception. The 200-character cap belongs only to the
        # metric-suite note path, not ordinary step failures.
        rec["outcome"], rec["detail"] = "error", f"{type(e).__name__}: {e}"
    else:
        # WIDENED, not replaced. An executor still returns a status string (`reported`) or
        # None (`ok`); a DICT is the third form, and it is a `prep` executor's transition —
        # applied above, and `ok` because it ran.
        if isinstance(override, dict) or override is None or override == "ok":
            rec["outcome"] = "ok"
        else:
            rec["outcome"], rec["detail"] = "reported", str(override)
    finally:
        rec["seconds"] = round(time.perf_counter() - started, 6)
        # The OUTPUT half of the record, shared with `watch` so a wrapped tool and a
        # recipe step describe what they produced the same way. Until now a step could
        # report "ok" while contributing nothing — no embedding, no g-vector key — and
        # the record could not tell that apart from real work.
        #
        # IDENTITY, the same test `_persist` and `_attach_geometry` already use, and the same
        # one `before_obj` was computed for above. Comparing `describe` strings instead made
        # every re-embedding to the same shape read as "produced nothing" while `_persist`
        # wrote its array to disk from the same record: measured on the bundled `cflows` over
        # pbmc3k (run 60921b8e06b6), step 1 `mioflow` recorded `outcome='ok'`, `produced=None`
        # and `artifacts={'emb': '…/01-mioflow_emb.npy'}` holding a (2700, 3) array — the
        # trajectory step the recipe exists for, reading as contributing nothing, because
        # phate and mioflow both leave (2700, 3) float32. `watch.py:136,176` sets `produced`
        # unconditionally, so this is also the parity the shared field claims.
        rec["produced"] = (describe(state.get("emb"))
                           if state.get("emb") is not before_obj else None)
        # Every new readout belongs to this step unless its producer already stated a more
        # specific stage. Punctuation is not evidence that a number was a parameter.
        declared_settings = {f"{name}.{p}" for p in rec["params"]}
        prior_provenance = before_values.get("_provenance", {})
        for key in list(g):
            if key == "_provenance" or key.endswith("_note"):
                continue
            if key not in before_values or g[key] is not before_values[key]:
                provenance = g.get("_provenance", {}).get(key)
                if provenance is prior_provenance.get(key):
                    kind = "setting" if key in declared_settings else "measurement"
                    annotate(g, key, kind=kind, stage=f"step {index + 1}: {name}")
        rec["emitted"] = sorted(set(g) - before_keys - {"_provenance"})
        # A LIST, not one path: a step may draw more than one, and picking "the" figure here
        # would be a renderer's choice made in the record. Omitted entirely when the step drew
        # nothing, so an absent key and an empty list do not both have to mean the same thing.
        drew = list((ctx.get("plots") or ())[before_plots:])
        if drew:
            rec["plots"] = drew
        # Geometry goes on BEFORE the report, not after: `on_step` is how the live panel
        # learns a step is done, so a record that gains its delta afterwards is a record
        # the panel renders without one and never revisits. This is also why it sits
        # above `_settle` rather than beside `_finalize` — the whole point is that `lid`
        # exists while the run is still going, not once the last step has finished.
        if state.get("emb") is not before_obj:
            carry["geometry"] = _attach_geometry(rec, state.get("emb"), carry["declared"],
                                                 carry["geometry"])
        _persist(rec, state, ctx, index, name, carry["written"], before_slots, before_extras)
        _settle(rec)
        _report(on_step, rec, lineage)
    return rec


def _run_steps(recipe: dict, state: dict, g: dict, *, dispatch: dict, ctx: dict,
               on_step: Any = None) -> list:
    """THE recipe loop: one `apply_step` per declared step, over one lineage.

    Returns the positional STEP RECORD — one entry per *attempt*, carrying index, name,
    group, params, outcome, detail and duration. `trace` and `status` are DERIVED from it
    (`_finalize`), so there is one source of truth rather than three.

    Everything a step does now lives in `apply_step`; what is left here is the three things
    that are genuinely RUN-shaped and that `new_carry`/`finish_carry` name — a carry opened
    once, a monotonic `index`, and exactly one close. A recipe is the degenerate lineage: it
    is issued all at once."""
    steps: list[dict] = []
    carry = new_carry()
    for index, step in enumerate(recipe.get("steps", []) or []):
        apply_step(step, state, g, dispatch=dispatch, ctx=ctx, index=index,
                   carry=carry, on_step=on_step, steps=steps)
    finish_carry(carry, ctx)
    return steps


#: Outcomes that mean the executor never returned, so the step changed NOTHING — `apply_step`'s
#: two failure branches (`_StepSkipped` → `skipped`, any other exception → `error`) and the two
#: pre-dispatch declines above them (no group, no executor for this group). The complement is
#: `("ok", "reported")`, both of which ran; the partition is asserted against `watch.OUTCOMES`
#: in `tests/test_watch.py`, so a fifth outcome has to be classified rather than silently
#: counting as "it ran".
DID_NOT_RUN = ("skipped", "error")


def executed(recipe: dict, steps: "list | None") -> dict:
    """`recipe`, minus the steps that never reached the state — the run's account of itself.

    WHAT IT IS FOR: `vocab.noncanonical` folds a NARROWING off the step list alone, because at
    plan time a step list is all there is. Handed a DECLARED list after the fact, it therefore
    clears the run's step-produced facts for a filter that skipped or errored — and writes a
    provenance caveat that is false about the run it annotates. Measured on a session running
    `[phate, filter_cells, mioflow]` where `filter_cells` has no executor: the record carried
    "mioflow: embedding came from the caller, not from a latent step (e.g. phate)" while phate
    had assigned a fresh embedding that mioflow then read. That is the one channel a human
    reads to decide whether to trust a number, so it has to describe what happened.

    Positional, because the record is: `_run_steps` writes exactly one entry per declared step
    in order, and `session.Session.performed()` builds its step list from the records
    themselves. `zip` truncating on the shorter side is the honest degradation — a declared
    step with no record did not run either.

    NOT a filter on `_finalize(recipe=…)` or on `identity`: those want the sequence AS ISSUED,
    the same way a recipe's declared list contains steps that may skip. This is only for the
    questions asked ABOUT the execution.
    """
    declared = list(recipe.get("steps") or [])
    ran = [step for step, rec in zip(declared, list(steps or []))
           if rec.get("outcome") not in DID_NOT_RUN]
    return {**recipe, "steps": ran}


def _settle(rec: dict) -> None:
    """Drop the live fields. Called on EVERY exit from a step, not only the dispatched one.

    These lived in the `finally:` block, which the two `continue` paths never reach — so a
    step that declared no group, or whose group no engine implements, shipped with
    `state="running"` and the dashboard rendered it as running forever. The outcome was
    correct the whole time; only the live overlay lied, which is the kind of defect a test
    on the happy path cannot see."""
    rec.pop("state", None)
    rec.pop("started", None)


def _report(on_step: Any, rec: dict, steps: list) -> None:
    """Tell a watcher what just changed, without letting it break the run.

    A dashboard is an observer: a renderer that raises — a closed terminal, a bad width —
    must not take the compute down with it. The run is the thing being protected here, not
    the display."""
    if on_step is None:
        return
    try:
        on_step(rec, steps)
    except Exception:  # noqa: BLE001 - an observer never fails the thing it observes
        pass


def identity(recipe: dict, *, engine: str, seed: Any, dataset: Any,
             dataset_name: str | None = None, data_kwargs: dict | None = None) -> tuple[str, str]:
    """`(run_id, spec_id)` — the two addresses a record needs, and they are not the same.

    Nothing could previously be said ABOUT a run because nothing named one. "B is the null
    twin of A", "show me every run on this cohort", "has this exact analysis been done
    before" are all edges between records, and an unaddressed record has no edges. Validation,
    contrast and applying prior results all need addressed records.

    Identity may be a hash of the spec or fresh per execution. BOTH semantics are wanted —
    dedupe and a stopping rule
    need "same spec, same record"; seed stability and null margins need "each run is its
    own". So both are emitted rather than forcing the choice: `spec_id` is what was asked
    for, `run_id` is this asking of it."""
    import hashlib
    import json
    import uuid

    spec = {
        "recipe": recipe.get("name"),
        "steps": [{"name": s.get("name"), "group": s.get("group"),
                   "params": s.get("params") or {}} for s in (recipe.get("steps") or [])],
        "engine": engine, "seed": seed, "dataset": str(dataset) if dataset else None,
    }
    # Keep legacy identities stable when no source contract was supplied. A named
    # variant or generator override is part of what was asked for, just like step params.
    if dataset_name is not None or data_kwargs:
        spec.update(dataset_name=dataset_name, data_kwargs=dict(data_kwargs or {}))
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return uuid.uuid4().hex[:12], hashlib.sha256(canonical.encode()).hexdigest()[:12]


def as_parent(ref: Any) -> "dict | None":
    """`(run_id, index)` or `{"run_id", "index"}` → the ONE form a record carries, or None.

    A `run_id` names one STATE LINEAGE — one chain of steps applied to one starting state —
    and `parent` is the edge to the lineage this one started from. It is the whole difference
    between the two things "phate, then mioflow, then a different mioflow" can mean: chained,
    that is one lineage of three steps; rewound to phate's embedding, it is two lineages
    sharing a prefix, and only a `parent` can say so. Measured on this checkout, the rewind
    through `Session.branch(0)`: two `run_id`s, two `state/<run_id>/` folders
    (`00-phate_emb.npy 01-mioflow_pseudotime.npy` and `00-mioflow_pseudotime.npy`), and the
    parent's own files still in place with its step-0 embedding loading back element for
    element. Writing the branch into the parent's folder instead would need `NN` to be a
    graph node rather than a position, and would make the parent's `COMPLETE` marker mean
    "finished, and also still growing".

    `index` is the parent step whose OUTPUT this lineage starts from, which is the same
    `(run_id, index)` that already addresses the artifact file (`artifacts.save_array` →
    `NN-step_slot.npy`) and that `_persist` already records in the manifest. No new
    identifier, and deliberately no new *kind* of one.

    Coercion happens here and nowhere else. A record that sometimes held a tuple and
    sometimes a dict would be `catalog.py`'s "two forms of one thing" — the failure this
    codebase keeps repeating — and the tuple would not survive the store anyway: JSON has no
    tuple, so it reads back as a list and the field acquires two accessors, one positional
    and one by key.

    Raises on a half-address: a lineage that claims a parent it cannot name is worse than one
    that claims none, because the edge reads as recorded and resolves to nothing.
    """
    if ref is None:
        return None
    if isinstance(ref, dict):
        run_id, index = ref.get("run_id"), ref.get("index")
    else:
        pair = tuple(ref)
        # unpacked by hand so a 1- or 3-tuple reports the address it was given, rather than
        # raising ValueError("not enough values to unpack") with the value nowhere in it
        run_id, index = pair if len(pair) == 2 else (None, None)
    if run_id is None or index is None:
        raise ValueError(
            f"parent needs both a run_id and an index — the address of the step whose state "
            f"this lineage starts from; got {ref!r}"
        )
    return {"run_id": str(run_id), "index": int(index)}


def _finalize(state: dict, g: dict, *, engine: str, recipe: dict, steps: list,
              plots: list, seed: Any = None, dataset: Any = None,
              dataset_name: str | None = None, data_kwargs: dict | None = None,
              caveats: "list | None" = None, ident: "tuple | None" = None,
              parent: Any = None, discarded: "list | None" = None,
              abandoned: bool = False, not_run=(), compute_error: str | None = None) -> dict:
    """THE g-vector assembler + result shape — identical for every engine.

    `trace` and `status` are derived from the step record rather than accumulated
    alongside it, so the three cannot drift. A repeated step name collapses in `status`
    (one key, last write wins — the pre-existing behaviour) while `steps` keeps both
    entries, which is where a caller should look when the two disagree.

    `discarded` NAMES STEPS THAT RAN AND WERE THEN THROWN AWAY, which is a third thing a step
    record cannot express: the step executed (so its record says `ok`, and rightly) but the
    state it produced was rolled back, so the run does not contain it. `run_tune_loop`'s cancel
    path is the only producer today — it `_reset()`s `state` and `g` to the baseline and
    returns None — and before this argument existed the record was silent about it: a recipe
    abandoned at its second step reported `complete: true` with the geometry of its first.
    It counts against `complete` and against nothing else; see that key below."""
    # Computed at the START of the run now and passed in — an artifact written by step 0
    # needs the run's address at step 0, not once the last step has finished. `identity` is
    # unchanged and still the only place ids are minted.
    not_run = [dict(occurrence) for occurrence in not_run]
    for occurrence in not_run:
        if (not isinstance(occurrence.get("index"), int)
                or isinstance(occurrence["index"], bool) or occurrence["index"] < 0
                or not isinstance(occurrence.get("name"), str) or not occurrence["name"]
                or not isinstance(occurrence.get("reason"), str) or not occurrence["reason"]):
            raise ValueError("not_run entries require a nonnegative index, name and reason")
    caveats = list(caveats or [])
    caveats.extend(f"step {entry['index']} ({entry['name']}): not run — {entry['reason']}"
                   for entry in not_run)
    if abandoned:
        caveats.append("run abandoned: final geometry was not measured")
    if compute_error:
        caveats.append(f"final geometry was not measured: {compute_error}")
    run_id, spec_id = ident or identity(recipe, engine=engine, seed=seed, dataset=dataset,
                                        dataset_name=dataset_name, data_kwargs=data_kwargs)
    trace = [f"{s['group']}:{s['name']}" for s in steps]
    status = {s["name"]: _status_line(s) for s in steps}
    X, emb = state.get("X"), state.get("emb")
    # `setdefault`, and every lineage builder now seeds these at OPEN (`open_gvector`) so this
    # is the fallback rather than the producer. It has to stay a fallback: `X` here is the
    # matrix as the run LEFT it, and a `prep` filter narrows it — reading `n_samples` off it
    # after a filter reports the survivors as the count that entered.
    if getattr(X, "ndim", 0) == 2:
        g.setdefault("n_samples", int(X.shape[0]))
        g.setdefault("n_features", int(X.shape[1]))
    if getattr(emb, "ndim", 0) == 2:
        g["n_embedded"] = int(emb.shape[0])
        g["final_dim"] = int(emb.shape[1])
    # The declared suite, measured here because this is the ONE assembler both engines share.
    # It is deliberately not a recipe step: `configs/metrics/default.yaml` is orthogonal to
    # the recipe by design, and a suite each recipe had to remember to list is a suite that
    # drifts — which is how the cross-recipe g-vector came to share ten keys of provenance and
    # no geometry. Merged last and never over a value the engine already produced.
    from manyruns import catalog

    final = ({} if abandoned or compute_error
             else _suite.measure(emb, X, catalog.load_suite(), already=g))
    g.update(final)
    for key in final:
        if not key.startswith("suite_") and not key.endswith("_note"):
            annotate(g, key, kind="measurement", stage="final embedding")
    fallback_dim = int(X.shape[1]) if getattr(X, "ndim", 0) == 2 else 0
    return {
        "served_by": "local",
        "engine": engine,
        # Legal-but-unusual routes to a fact, in the caller's words. Empty for every bundled
        # recipe. A result reached off the beaten path is exactly the one whose provenance a
        # reader needs, so it rides in the record rather than being printed once and lost.
        "caveats": list(caveats or []),
        "recipe": recipe.get("name"),
        "num_steps": len(trace),
        "trace": trace,
        "g_vector": g,
        "status": status,
        "plots": plots,
        # CAN THIS RUN BE TRUSTED — nothing errored. The experiment harness treats a False
        # here as a failed cell rather than reading a short/empty g-vector as real data.
        #
        # THIS USED TO MEAN "every step ran" (`all(s == "ok" …)`), which conflated two events
        # that are not the same. An ERROR means the run is not trustworthy. A DECLINE means a
        # step correctly determined it had nothing to do — `normalize` standing down on a
        # matrix that is not counts is the decline working, not the run failing. Measured on a
        # 120×12 Gaussian after the cutover gave seven recipes a prep block: 9 of the 11
        # bundled recipes reported `ok=False` with ZERO steps errored, so `experiment.py`'s
        # `ok_rows` filter aggregated a sweep over almost nothing while every step it cared
        # about had run and produced a g-vector. See manyruns#66.
        #
        # UNSUPPORTED counts against it too, and that is the correction the first draft of
        # this needed. `skipped` wears two events: a step that RAN and judged (`normalize` on a
        # matrix that is not counts) and a step NOTHING could run (a typo'd name, a stub, a
        # group this engine has no table for). Treating both as benign would have made
        # `run_inproc(X, {"steps": [{"name": "nosuchalgo", "group": "latent"}]})` report
        # `ok=True` — a recipe that did nothing, reporting success. See `_StepUnsupported`.
        #
        # Read off the RECORDS, not off `status`: `status` is a lossy view (one key per name,
        # last write wins) so a recipe issuing a step twice could hide the failed one.
        "ok": not compute_error and not any(
            s["outcome"] == "error" or s.get("unsupported") for s in steps),
        # DID EVERY DECLARED STEP RUN — the question `ok` used to answer, kept as its own key
        # rather than dropped. A reader deciding "is this comparable to that other run" needs
        # it: two runs of one recipe where one skipped `filter_mito` are not the same analysis,
        # and nothing else in the record says so in one field. `reported` counts as having run
        # (it is an analysis step's readout); `DID_NOT_RUN` is the complement, and is the same
        # tuple `executed()` prunes on, so the two answers cannot drift.
        #
        # A DISCARDED STEP COUNTS AGAINST IT TOO, and it has to be read off `discarded` rather
        # than off the records: a step the tune loop ran and then rolled back has an `ok`
        # record (it did run) and left nothing in the state, so every field derived from the
        # records — `status`, `trace`, `ok` — reads as though it had contributed. `complete` is
        # the field whose question it answers: the declared recipe did NOT get through.
        # Rewriting the step's own outcome instead was the alternative and is worse — it would
        # invent a fifth outcome word, put a falsehood ("did not run") in the record of a step
        # that did, and change what `executed()` prunes.
        "cancelled": bool(abandoned or discarded),
        "not_run": not_run,
        "complete": (not any(s["outcome"] in DID_NOT_RUN for s in steps)
                     and not discarded and not abandoned and not not_run and not compute_error),
        "final_dim": int(g.get("final_dim") or fallback_dim),
        # what actually EXECUTED, positionally, with params and durations. `trace`/`status`
        # above are lossy views of this: `trace` cannot express an outcome and `status`
        # cannot express a repeated step name.
        "steps": steps,
        # The seed the run actually used. `None` still means "not reproducible" and is
        # still never invented — but since #29 both engines have a seed channel, so a real
        # run reports the seed that reached the fit rather than a permanent absence.
        "seed": seed,
        "tool_version": __version__,
        # Identity. Without it a record cannot be referred to, and every capability that is
        # an EDGE between records — a null twin, a re-run of the same spec, "every run on
        # this cohort" — is inexpressible rather than merely unbuilt.
        "run_id": run_id,        # this execution
        "spec_id": spec_id,      # what was asked for; equal across re-runs of one spec
        # The lineage this one started from, or None — see `as_parent`. Deliberately NOT part
        # of `spec_id`: "same spec?" and "same starting state?" are two different questions a
        # comparison reader has to be able to ask separately. Two branches that issue the same
        # step from two different embeddings SHOULD collide on `spec_id` (it is the same
        # analysis) and differ here (it is not the same input) — measured: equal `spec_id`,
        # `parent.run_id` differing.
        "parent": as_parent(parent),
        "dataset": str(dataset) if dataset else None,
        "dataset_name": dataset_name,
        "data_kwargs": dict(data_kwargs or {}),
        "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def _run_analysis_step(name, params, state, g, ctx):
    """An `analysis` step — manyruns's own in-process readout, on EITHER engine.

    Not a manylatents algorithm, which is why `analysis` stays a manyruns-owned sentinel
    rather than collapsing into the engine's group names."""
    if state.get("emb") is None and state.get("X") is None:
        raise _StepSkipped("no data to analyze yet")
    fn = _steps._ANALYSIS_STEPS.get(name)
    if fn is None:
        raise _StepUnsupported(f"no analysis impl for {name!r}")
    fn(state, g, params, ctx["out_dir"], ctx["plots"], ctx["target_dim"])


def _run_prep_step(name, params, state, g, ctx):
    """A `prep` step: resolve by name, call the pure function, hand the result to the loop.

    Returns the result DICT rather than a status string. `apply_step` interprets it — that is
    where a mask meets a selection, and it is deliberately the only such place.

    **It does NOT write the step's `report` into `g`, and that is the fix rather than an
    omission.** This function used to merge `out["report"]` here, one frame before
    `_apply_transition` runs — so a step whose transition was then REFUSED still left its
    numbers in the g-vector. Measured in this checkout on a hand-built state whose working
    matrix had already been reduced (`X` 4 x 2 against a 6-gene selection), running
    `filter_genes(min_cells=2)`: the record came back `outcome='error'` with
    `"'filter_genes' filters genes, but 'X' is 4 x 2 against a selection of 6 genes"`, the
    selection was untouched at `[0, 1, 2, 3, 4, 5]` — and `g` held
    `{'filter_genes.min_cells': 2, 'filter_genes.dropped': 3}`, a narrowing of three genes
    recorded for a narrowing that never happened. `_apply_transition`'s own docstring refuses
    exactly that ("refused BEFORE the selection moves, so the state a step leaves on `error`
    is the state it started from"); the g-vector is part of that state. `apply_step` merges
    the report after the transition applies.

    Engine-independent, and on every engine's table, because prep is scanpy rather than an
    entry in an algorithm catalogue. An engine missing it would report `skipped (no 'prep'
    executor on this engine)` for the standard preamble, which is a hole rather than a
    fidelity difference — the distinction `--engine` is supposed to encode.
    """
    from manyruns.pipeline import frame as _frame

    fn = _prep._PREP_STEPS.get(name)
    if fn is None:
        raise _StepUnsupported(f"no prep step named {name!r}")
    if state.get("frame") is None:
        # A state built by something other than `_new_state`, which seeds the frame for all
        # three builders. Otherwise this would record `KeyError: 'frame'`, which says nothing
        # about what to do next; a declined step with a reason is what the record is for.
        raise _StepSkipped("this state carries no frame, so a prep step has nothing to view")
    view = _frame.view(state["frame"], state["rows"], state["cols"])
    if view["counts"] is None and state.get("X") is None:
        # A named manylatents dataset: the engine loads it and this process never sees the
        # array (`bounds.shape_of` records the same limitation, and `run_manylatents` seeds
        # `X=None` on that route). `transform` would hand `np.log1p` a None and record a
        # TypeError; the honest answer is that there is nothing here to prep.
        raise _StepSkipped("no data in this state to prep — the engine loads a named dataset "
                           "itself, so manyruns never sees the matrix")
    return fn(view, state.get("X"), params)


def _run_probe_step(name, params, state, g, ctx):
    """A `probe` step: ask the trained model a question, record the answer in the g-vector.

    Declines rather than erroring in two cases, and both are ordinary rather than exceptional.
    A state with no `model` is every engine whose `lightning` executor trains nothing — the
    in-process stand-in computes a diffusion pseudotime, the mock invents numbers — and
    `vocab.unmet` cannot express that, because it takes no engine argument by design. A model
    that lacks the method is the upstream half: `decode_to_gene_space` and `growth_rate` do not
    exist on MIOFlow yet, and a probe for one should say so rather than raise an AttributeError
    through three frames.

    Answers are prefixed with the probe's own name, the same convention `_ml_latent` uses for
    scores, so two probes cannot collide in the g-vector.
    """
    fn = _probes._PROBE_STEPS.get(name)
    if fn is None:
        raise _StepUnsupported(f"no probe named {name!r}")
    model = state.get("model")
    if model is None:
        raise _StepSkipped(
            "no trained model in this state — a probe asks a question of a fitted object, and "
            "the lightning step on this engine trains nothing")
    try:
        answer = fn(model, params or {})
    except _probes._NoAnswer as e:
        raise _StepSkipped(str(e)) from None
    for key, value in (answer or {}).items():
        g[f"{name}.{key}"] = value
    return None


# ── manylatents executors ────────────────────────────────────────────────────
def implicit_sample_ids(obj):
    """Use source identities when unique; otherwise disclose the positional fallback."""
    import numpy as np
    import warnings

    names = getattr(obj, "obs_names", None)
    if names is None:
        return None
    names = np.asarray(names, dtype=str)
    if len(set(names.tolist())) != len(names):
        warnings.warn("source observation IDs are duplicated; using positional alignment "
                      "without persisted identities", UserWarning, stacklevel=2)
        return None
    return names.copy()


def display_context(state, color=None, sample_ids=None, label_key=None, declared_shape=None):
    """Copy and validate display inputs once, on the original sample axis."""
    import numpy as np

    state["label_key"] = label_key
    n = len(state["rows"])
    if not n and declared_shape is not None:
        n = declared_shape[0]
    labels = state.get("labels")
    if labels is not None:
        if n and len(labels) != n:
            raise ValueError("labels must match the input rows")
        n = n or len(labels)
    channels = []
    for channel in color or ():
        if not isinstance(channel, dict) or channel.get("kind") not in ("categorical", "continuous"):
            raise ValueError("color must contain named categorical or continuous channels")
        values = channel.get("values")
        if getattr(values, "ndim", 1) != 1 or values is None or len(values) != n:
            raise ValueError("color values must match the input rows")
        values = values.copy() if hasattr(values, "copy") else np.asarray(values).copy()
        if isinstance(values, list):
            values = np.asarray(values)
        if isinstance(values, np.ndarray):
            values.setflags(write=False)
        channels.append({**channel, "values": values})
    if sample_ids is not None:
        sample_ids = np.asarray(sample_ids, dtype=str).copy()
        if sample_ids.ndim != 1 or len(sample_ids) != n:
            raise ValueError("sample_ids must match the input rows")
        if len(set(sample_ids.tolist())) != n:
            raise ValueError("sample_ids must be unique")
        sample_ids.setflags(write=False)
    # A caller may supply verified generator dimensions and metadata while the engine
    # owns the matrix. Those explicit values still establish an original sample axis.
    if not len(state["rows"]) and n and (channels or sample_ids is not None):
        state["rows"] = np.arange(n, dtype=np.int64)
    return {"display_channels": tuple(channels), "sample_ids": sample_ids,
            "label_key": label_key}


def current_display(state, ctx):
    """Select explicit channels once; derive default display from current analysis labels."""
    channels = ctx.get("display_channels") or ()
    rows = state.get("rows", [])
    if channels:
        channels = [{**c, "values": c["values"][rows]} for c in channels]
    elif state.get("labels") is not None:
        categorical = state.get("label_kind") in CATEGORICAL_KINDS
        values = state["labels"] if categorical else _io._labels_to_numeric(state["labels"])
        channels = [{"key": state.get("label_key"), "values": values,
                     "kind": "categorical" if categorical else "continuous"}]
    ids = ctx.get("sample_ids")
    return channels, ids[rows] if ids is not None else None


def _check_metadata_alignment(emb, state, ctx, returned_ids=None):
    """Do not attach source metadata when an executor reports an incompatible row axis.

    The normal manylatents API preserves input order. Equal length alone cannot detect
    an opaque permutation; returned identities, when present, let us reject one explicitly.
    `metadata_aligned` is True for attached metadata, None for an unverified positional
    resume, and False for incompatible rows. Both None and False suppress metadata, but
    positional continuations still need these checks for newly incompatible output.
    """
    import numpy as np

    if ctx.get("metadata_aligned", True) is False:
        return
    n = len(state.get("rows", ()))
    if not n:
        n = len(state["labels"]) if state.get("labels") is not None else 0
    reason = None
    if n and len(emb) != n:
        reason = f"engine coordinates have {len(emb)} rows for {n} input rows"
    elif returned_ids is not None:
        ids = ctx.get("sample_ids")
        expected = ids[state["rows"]] if ids is not None else None
        if expected is None or not np.array_equal(np.asarray(returned_ids, dtype=str), expected):
            reason = "engine reported a different or unverified observation order"
    if reason:
        ctx["metadata_aligned"] = False
        ctx.setdefault("caveats", []).append(
            f"metadata alignment unavailable — {reason}; no proven mapping to source metadata")


def _display_scatter(emb, state, ctx, fname, title, *, trajectories=None):
    if getattr(emb, "ndim", 0) != 2 or emb.shape[1] < 2:
        ctx.setdefault("caveats", []).append(
            f"{title}: no scatter — this result has fewer than two coordinate columns")
        return []
    channels, ids = current_display(state, ctx) if ctx.get("metadata_aligned", True) else ([], None)
    if (any(len(c["values"]) != len(emb) for c in channels)
            or (ids is not None and len(ids) != len(emb))):
        note = (f"{title}: metadata alignment unavailable — engine coordinates have "
                f"{len(emb)} rows without a mapping to the input rows; displaying without metadata")
        ctx.setdefault("caveats", []).append(note)
        channels, ids = [], None
    result = _io.save_display_scatter(emb, channels, ctx["out_dir"], fname, ctx["plots"],
                                      title, obs_names=ids, trajectories=trajectories)
    ctx.setdefault("caveats", []).extend(
        f"skipping colour channel {key!r}: {reason}"
        for key, reason in result.skipped_channels)
    return result


def _ml_latent(name, params, state, g, ctx):
    """A latent step → ``manylatents.api.run(algorithms={'latent': name})``."""
    import numpy as np

    from manylatents.api import run

    kwargs: dict[str, Any] = {"algorithms": {"latent": name}, "seed": int(ctx["seed"])}
    # The step's declared `params` — `{n_components: 3}` in cflows.yaml and embed.yaml. These
    # were dropped in TWO places: this function never referenced its own `params` argument,
    # and manylatents' `_resolve_algorithm` forwarded kwargs only on the bare-string form
    # while manyruns uses the dict form. Both halves had to be fixed; either alone leaves
    # `final_dim` at the engine default of 2 regardless of what the recipe declares.
    kwargs.update(params or {})
    # The declared metric suite. manylatents' registry computes these and returns them in
    # `scores`, which we fold into the g-vector below — this is the whole mechanism by which
    # "a suite of 10-20 metrics" reaches the experiment table. Omitted → engine defaults.
    if ctx.get("metrics"):
        kwargs["metrics"] = list(ctx["metrics"])
    # "first" is derived from the state, not a flag: a flag that only cleared on full
    # success meant a step failing *after* producing an embedding sent the next step back
    # to the raw input instead of chaining.
    source_fit = state.get("emb") is None and ctx.get("array") is not None
    if state.get("emb") is None:
        if ctx.get("array") is not None:
            # torch (inside manylatents) rejects negative strides — e.g. scanpy PCA output.
            # `_dense` before `ascontiguousarray`, and the order is the bug: since the cutover
            # `ctx["array"]` can be a CSR, and `np.ascontiguousarray(csr)` returns a `(1,)` object
            # array WITHOUT RAISING. Measured on pbmc3k, that is what the engine received.
            kwargs["input_data"] = np.ascontiguousarray(_dense(ctx["array"]))
        else:
            kwargs["data"] = ctx.get("data_ref")          # a named manylatents dataset
            # Generation params go through the engine's `data_kwargs` channel, NOT spread into
            # the top-level call: `run(**kwargs)` routes its loose kwargs to the ALGORITHM, so
            # spreading them here dropped every one of them silently — two datasets differing
            # only in generation params produced byte-identical embeddings, which the sweep
            # then read as zero within-group variance (every F came back `inf`).
            if ctx.get("data_kwargs"):
                kwargs["data_kwargs"] = dict(ctx["data_kwargs"])
    else:
        kwargs["input_data"] = np.ascontiguousarray(state["emb"])  # chain
    out = run(**kwargs)
    emb = np.ascontiguousarray(np.asarray(out.get("embeddings")))
    if source_fit:
        # A fresh fit starts on the selected source axis, even after an unverified
        # resume. Check its output anew; chained coordinates retain their uncertainty.
        ctx["metadata_aligned"] = True
    _check_metadata_alignment(emb, state, ctx, out.get("obs_names", out.get("sample_ids")))
    state["emb"] = emb
    for k, v in (out.get("scores") or {}).items():
        g[f"{name}.{k}"] = v
        if not k.endswith("_note"):
            annotate(g, f"{name}.{k}", kind="measurement", stage=f"{name} engine evaluation")
    # Everything else the engine handed back — see `_collect_extras`. Measured on a plain
    # PHATE run: `affinity` and `kernel`, both N x N, both previously dropped on this line.
    _collect_extras(name, out, state, g)
    _display_scatter(emb, state, ctx, f"{name}.png", name)


def _ml_lightning(name, params, state, g, ctx):
    """A lightning step: MIOFlow in-process (chained on the embedding), else the CLI."""
    if name != "mioflow":
        # THE SAME DEFECT AS #55, one branch over: this route shells out to `manylatents.main`
        # and `params` was simply never passed, so every param on a `cflows`/`latent_ode` step
        # was dropped while the step recorded `reported`. Said out loud rather than closed,
        # because closing it means synthesising `+algorithms/lightning.<k>=<v>` overrides, which
        # needs the config-group path per algorithm — a product-side copy of the engine's config
        # layout, which is the thing #55's fix exists to not do.
        #
        # `skipped`, not `error`: the params are not wrong, the ROUTE cannot carry them. That is
        # the distinction the outcome vocabulary draws (runner.py's `_StepSkipped` handling),
        # and it is the same shape as the `--dataset` refusal directly below.
        if params:
            raise _StepSkipped(
                "the manylatents CLI path forwards no params, and this step declares "
                f"{sorted(params)} — they would be silently dropped. Only `mioflow` has an "
                "in-process route that carries them today")
        if not ctx.get("data_ref"):
            raise _StepSkipped("lightning step needs a named --dataset for the CLI")
        return _mioflow._run_lightning_cli(name, ctx["data_ref"], ctx["fast_dev_run"])
    # The SAME rule the real engine's `_step_mioflow` applies — one home, so the two engines
    # cannot drift back apart. They did: this refusal existed only here.
    _steps._require_embedding(state, "mioflow")
    # AN ALLOWLIST, and the difference is not style. This read `None if … == "condition"` — a
    # denylist of one — so any kind added upstream fell through and became a time axis. `group`
    # (a cell type, a cluster, a branch) is as much a fabricated trajectory as treated -> healthy
    # is, and it would have arrived silently. Only `time` is a time axis; everything else falls
    # back to pseudotime, exactly as unlabelled manifold data does.
    time_labels = state.get("labels") if state.get("label_kind") == "time" else None
    # `_run_mioflow_experiment`'s returned "embeddings" is MIOFlow's own `encode()` output —
    # every row integrated from t_min to t_max regardless of that row's own time bin, NOT the
    # same coordinate frame as the embedding that was fed in. The trajectory arrows, by
    # contrast, start at the TRUE input coordinates of the earliest time bin
    # (`MIOFlow._generate_trajectories`). Plotting them over `emb` puts correct arrows on a
    # background that has already moved — `base_emb` is what phate.png/dpt.png/
    # discretize_time.png show, so it's the frame the arrows actually belong on.
    #
    # Read BEFORE the call, because `state["emb"]` is rebound to the flow's output below.
    base_emb = state["emb"]
    emb, scores, extras, model = _mioflow._run_mioflow_experiment(
        state["emb"], ctx["seed"], ctx["fast_dev_run"], params,
        labels=time_labels, device=ctx.get("device", "cpu"),
    )
    if time_labels is not None:
        g["mioflow.n_timepoints"] = len(set(map(str, time_labels)))
    # This used to read `state["emb"] = emb` and nothing else, so the trained flow — the simulatable
    # object — was dropped one frame later while its coordinates were kept. It is kept now, and
    # `probe` steps are what it is kept for.
    state["model"] = model
    _check_metadata_alignment(emb, state, ctx)
    state["emb"] = emb
    for k, v in scores.items():
        g[f"{name}.{k}"] = v
        if not k.endswith("_note"):
            annotate(g, f"{name}.{k}", kind="measurement", stage=f"{name} engine evaluation")
    _collect_extras(name, extras, state, g)
    # `base_emb`, not `emb` — see above. The paths come off the RETAINED MODEL rather than a
    # return value of their own: `state["model"]` is the object the flow left behind, and
    # `trajectories_of` is the defensive read of the one property a figure needs from it.
    _display_scatter(base_emb, state, ctx, f"{name}.png", name,
                     trajectories=_mioflow.trajectories_of(model))
    return None


def run_manylatents(
    recipe: dict,
    data_ref: Any = None,
    array: Any = None,
    out_dir: Path = Path("."),
    seed: int = 42,
    fast_dev_run: bool = True,
    data_kwargs: dict | None = None,
    labels: Any = None,
    device: str = "cpu",
    target_dim: int = 3,
    metrics: Any = None,
    embedding: Any = None,
    label_kind: str | None = None,
    on_step: Any = None,
    parent: Any = None,
    counts: Any = None,
    genes: Any = None,
    layers: "dict | None" = None,
    declared_shape: Any = None,
    color: Any = None,
    sample_ids: Any = None,
    label_key: str | None = None,
    dataset_name: str | None = None,
) -> dict:
    """Run the recipe against the real manylatents engine — one ``api.run`` call per
    latent/lightning step, embeddings chaining forward (phate → mioflow); analysis steps
    run in-process. Thin wrapper: the loop and the g-vector schema are shared (`_run_steps`).

    ``parent`` addresses the lineage a supplied ``embedding`` came from (`as_parent`). It is
    optional and None for every run that starts from data, which is why every existing row in
    `index.jsonl` stays valid: an atomic run is the degenerate lineage, with no parent.

    ``counts``/``genes`` are the gene axis (`_STATE_KEYS`), and unlike `run_inproc` this
    engine CANNOT derive them — stated rather than papered over, because the difference is the
    whole reason a caller has to pass them. `run_inproc` receives the loaded OBJECT as its
    `matrix` (`app._read_inputs`'s `real` branch returns `pipeline.load_array(...)`, an
    AnnData) and reads the axis off it. This function receives ``array``, which on the
    `manylatents` branch is already the output of `pipeline.load_labeled` — a matrix, from
    which the AnnData that carried the genes went out of scope at that function's `return`
    (`loading.py`, the `_is_anndata(obj)` branch). There is nothing here to read them from.

    **MEASURED GAP: nothing populates these yet.** The producer would be `app._load_inputs`,
    which returns `(array, labels, kind)` — a 3-tuple unpacked at `app.py:343, 754, 1345`,
    at `app.py:1237` off `_read_inputs`, and in three tests (`test_label_wiring.py:79`,
    `test_named_dataset_facts.py:91`, `test_real_labels.py:124`). Widening it is app.py's
    change, not this module's, and until it happens a `manylatents` run carries
    `counts=None, genes=None`. That
    matters for exactly one thing and it should be said plainly: `markers` (`pca → leiden →
    rank_genes`) can only run on THIS engine, because `steps._INPROC_STEPS` holds `phate` and
    `mioflow` and neither `pca` nor `leiden` — so the gene axis reaching `run_inproc` alone
    does not unblock it. The state schema and the analysis executors are shared, so the
    remaining work is wiring, not design."""
    declared_shape = _bounds.declared_shape(declared_shape)
    try:
        import numpy  # noqa: F401 - fail here with actionable guidance, not mid-loop

        from manylatents.api import run  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "engine=manylatents needs manylatents, which is a BASE dependency — so this "
            "install is incomplete, not missing an extra. Repair it with `uv sync` in a "
            f"checkout, or `{REINSTALL}` for an installed tool. "
            f"(`[datasets]` carries nothing since it was emptied; adding it fixes nothing.) "
            f"(missing: {e.name})"
        ) from e

    out_dir = Path(out_dir)
    plots: list[str] = []
    # X is seeded from `array` when we have the input in hand; with a named dataset the
    # engine loads it and we never see its shape — so n_samples/n_features stay absent.
    from manyruns import vocab as _vocab

    supplied = {"embedding"} if getattr(embedding, "ndim", 0) == 2 else set()
    # Opened EMPTY and filled after the loop — see the `noncanonical` call below `_run_steps`.
    # It is the same list object `_finalize` is handed, so a bound applied mid-loop still
    # reaches the run's caveats through `ctx`.
    caveats: list = []
    # Validated at OPEN, before any compute: a malformed parent is the caller's bug, and
    # discovering it in `_finalize` would mean failing after the fit has already run.
    parent = as_parent(parent)
    ident = identity(recipe, engine="manylatents", seed=seed, dataset=data_ref,
                     dataset_name=dataset_name, data_kwargs=data_kwargs)
    state = _new_state(X=array if getattr(array, "ndim", 0) == 2 else None, labels=labels,
                       emb=embedding, counts=counts, genes=genes, layers=layers)
    if counts is not None and state["X"] is not None:
        from manyruns.aligned import check_aligned

        check_aligned(state["X"], counts, what="run_manylatents(counts= vs array)")
    state["label_kind"] = label_kind
    # The size of the data that ENTERED, recorded before the loop can narrow it —
    # `open_gvector`.
    g: dict[str, Any] = open_gvector(state)
    ctx = {
        "out_dir": out_dir, "plots": plots, "target_dim": target_dim, "seed": seed,
        "fast_dev_run": fast_dev_run, "device": device, "array": array,
        "data_ref": data_ref, "data_kwargs": data_kwargs, "metrics": metrics,
        "dataset_name": dataset_name,
        "declared_shape": declared_shape,
        "color": _io._labels_to_numeric(labels),  # colour embeddings by timepoint when we can
        "run_id": ident[0],
        # The SAME list `_finalize` is handed below, so a bound applied mid-loop reaches the
        # run's caveats without a second channel. A copy here would be a note nobody reads.
        "caveats": caveats,
    }
    ctx.update(display_context(state, color, sample_ids, label_key, declared_shape))
    with _numeric_quiet():
        steps = _run_steps(recipe, state, g, dispatch=dispatch_for("manylatents"), ctx=ctx,
                           on_step=on_step)
    # The provenance notes, folded over what RAN rather than over what was declared (`executed`
    # says why). PREPENDED, so the order a reader sees — route notes, then the bounds a step
    # was clamped by — is byte-identical to what computing them at open produced.
    caveats[:0] = _vocab.noncanonical(executed(recipe, steps), "unknown", provided=supplied)
    return _finalize(state, g, engine="manylatents", recipe=recipe,
                     steps=steps, plots=plots, seed=seed, dataset=data_ref,
                     dataset_name=dataset_name, data_kwargs=data_kwargs,
                     caveats=caveats, ident=ident, parent=parent)



# ── compute ──────────────────────────────────────────────────────────────────
def _inproc_step(name, params, state, g, ctx):
    """A latent/lightning step computed in-process with public libraries (`vocab.INPROC`).

    Dispatch is by step NAME within the group — the same lookup the manylatents executors
    do against the engine's algorithm catalogue, just served from a local table."""
    fn = _steps._INPROC_STEPS.get(name)
    if fn is None:
        raise _StepUnsupported("no in-process implementation")
    before = state.get("emb")
    before_plots = len(ctx["plots"])
    fn(state, g, params, ctx["out_dir"], ctx["plots"], ctx["target_dim"])
    if state.get("emb") is not before and state.get("emb") is not None:
        # Replace only this executor's new embedding figure. Analysis readouts keep
        # their own time/bin/trajectory encodings.
        new_plots = list(ctx["plots"][before_plots:])
        ctx["plots"][before_plots:] = [p for p in new_plots if Path(p).name != f"{name}.png"]
        for path in dict.fromkeys(new_plots):
            if Path(path).name == f"{name}.png":
                _display_scatter(state["emb"], state, ctx, Path(path).name, name)
        ctx["plots"][before_plots:] = list(dict.fromkeys(ctx["plots"][before_plots:]))


def run_inproc(matrix: Any, recipe: dict, out_dir: Path, target_dim: int = 3,
                 labels: Any = None, label_kind: str | None = None,
                 on_step: Any = None, seed: Any = None, dataset: Any = None,
                 embedding: Any = None, parent: Any = None,
                 counts: Any = None, genes: Any = None,
                 color: Any = None, sample_ids: Any = None,
                 label_key: str | None = None) -> dict:
    """Run the recipe on a 2-D array with real, public-only compute (no private stack).

    Thin wrapper: the step loop and the g-vector schema are shared with the manylatents
    engine (`_run_steps` / `_finalize`), so the two produce comparable g-vectors.

    `labels` is what makes this engine capable of `contrast` at all. Without it
    `separation` and `composition` reach their "no condition labels" branch on *every*
    stackless run and return None with a note — which the results table then reads as a
    MISSING metric, indistinguishable from "this recipe doesn't emit that metric". So the
    default engine could never produce the case/control readouts the product ships a recipe
    for. `label_kind` ("time" | "condition" | "group") travels with them, and each consumer
    allowlists the one kind it wants: a trajectory step reads only "time" (so it cannot
    invent a progression out of a case/control or a cell-type axis) and these two read only
    "condition" (so they cannot report a contrast between cell types).
    `label_key` is caller-supplied provenance for those labels. An AnnData's preferred
    label column cannot identify an independently supplied vector, so an omitted key
    remains unknown.

    `embedding=` had a producer before it had any way to say where the coordinates came from:
    `tests/test_artifacts.py` already runs `phate`, loads `00-phate_emb.npy` back, and feeds
    it to a `mioflow`-only recipe. That second run is a new lineage over state another run
    wrote, and `parent` (`as_parent`) is how it names whose.

    `counts=`/`genes=` are the GENE AXIS, and unlike every other channel here they are
    DERIVED by default rather than merely accepted. `matrix` on this engine is what
    `app._read_inputs` returned, and on the file path that is `pipeline.load_array(...)` —
    the AnnData OBJECT, which `as_matrix` above collapses to a bare array one line later.
    So the object carrying the genes is already in this function's hands and was being
    dropped; `loading.gene_axis` reads them off that same object, which is what makes the
    rows shared by construction rather than by hope (the argument `loading.labels_of`
    makes, one axis over). Passing them explicitly overrides the derivation, for a caller
    that holds the axis and only the matrix — `run_manylatents` is that caller's engine."""
    try:
        X = as_matrix(matrix)
    except ImportError as e:
        raise RuntimeError(
            "the in-process step loop needs a scientific stack that isn't installed. Every "
            "package it reaches for (numpy, phate, scikit-learn, scipy, statsmodels, "
            "matplotlib, and pandas/anndata for .csv/.h5ad) is a BASE dependency, so this "
            "install is incomplete rather than missing an extra: repair it with `uv sync` in "
            f"a checkout, or `{REINSTALL}`. (missing: {e.name})"
        ) from e
    if sample_ids is None:
        sample_ids = implicit_sample_ids(matrix)
    if counts is None and genes is None:
        from manyruns.pipeline.loading import gene_axis

        counts, genes = gene_axis(matrix)
    if counts is not None:
        # The same seam `labels` is guarded at, for the same reason: `X` and `counts` are two
        # matrices over the same cells, and off by one row every gene a readout attributes is
        # attributed to the wrong cell. Unreachable on the derived path (both come from one
        # AnnData); this guards the explicit one, which is the only way they can disagree.
        from manyruns.aligned import check_aligned

        check_aligned(X, counts, what="run_inproc(counts= vs X)")
    if labels is not None:
        # Matrix and labels are fetched separately (see `loading.labels_of`), so this is
        # exactly the seam `aligned` exists to guard: off by one row and cells are
        # attributed to the wrong donor with nothing downstream noticing.
        from manyruns.aligned import check_aligned

        check_aligned(X, labels, what="run_inproc(labels=)")
    plots: list[str] = []
    from manyruns import vocab as _vocab

    supplied = {"embedding"} if getattr(embedding, "ndim", 0) == 2 else set()
    caveats: list = []           # filled after the loop — see the same pair in `run_manylatents`
    parent = as_parent(parent)   # at OPEN — see the same call in `run_manylatents`
    ident = identity(recipe, engine=INPROC, seed=seed, dataset=dataset)
    state = _new_state(X=X, labels=labels, seed=seed, emb=embedding,
                       counts=counts, genes=genes)
    state["label_kind"] = label_kind
    # The size of the data that ENTERED, recorded before the loop can narrow it —
    # `open_gvector`.
    g: dict[str, Any] = open_gvector(state)
    ctx = {"out_dir": Path(out_dir), "plots": plots, "target_dim": target_dim,
           "run_id": ident[0], "caveats": caveats}
    ctx.update(display_context(state, color, sample_ids, label_key))
    with _numeric_quiet():
        steps = _run_steps(recipe, state, g, dispatch=dispatch_for(INPROC), ctx=ctx,
                           on_step=on_step)
    caveats[:0] = _vocab.noncanonical(executed(recipe, steps), "unknown", provided=supplied)
    return _finalize(state, g, engine=INPROC, recipe=recipe, steps=steps, plots=plots,
                     seed=seed, dataset=dataset, caveats=caveats, ident=ident, parent=parent)


# ── mock: the dep-free stand-in, as an executor rather than a second loop ─────
#
# The compute stays in `serving` — these are adapters, not a third implementation. The mock
# transforms a 1-D stand-in `vec` rather than an embedding, which is why `state["emb"]` is
# never assigned here: `_persist` writes only slots with a `.shape`, `_attach_geometry` fires
# only when the embedding object changed, and both correctly do nothing for a mock step.
# Measured on `cflows` through a mock session: two `ok` records, no `state/` folder, no
# geometry deltas, `g_vector` = what the steps emitted plus `_finalize`'s suite absences.
def _mock_transform(name, params, state, g, ctx):
    """A `latent`/`lightning` step on `engine=mock` — `serving._apply_op` over `state["vec"]`."""
    from manyruns.serving import _apply_op

    state["vec"] = _apply_op(name, params or {}, _mock_vec(state, ctx))


def _mock_measure(name, params, state, g, ctx):
    """An `analysis` step on `engine=mock` — `serving._measure`, straight into the g-vector."""
    from manyruns.serving import _measure

    g[name] = _measure(name, _mock_vec(state, ctx))


def _mock_vec(state: dict, ctx: dict) -> list:
    """The mock's stand-in data, seeded from `ctx["mock_seed"]` on first use.

    Seeded lazily rather than in `_new_state` because the seed string is a mock-only concern:
    every other engine reads `state["X"]`/`state["emb"]`, and a `vec` in the shared state
    schema would be a key that means something on exactly one backend."""
    if state.get("vec") is None:
        from manyruns.serving import LocalServer

        state["vec"] = LocalServer._as_vector(ctx.get("mock_seed") or "")
    return state["vec"]


# ── engine → dispatch table: ONE home ────────────────────────────────────────
#
# These three dicts were literals inside `run_manylatents` and `run_inproc`, which was fine
# while a run was the only thing that needed one. A session needs the same table for the same
# engine, and rebuilding it at the session's own call site is how two engines' step vocabularies
# drift apart — the failure this module's header records as already having happened once.
#
# The learner's engine is deliberately absent, and its absence is the refusal: it executed a
# whole recipe in one downstream call and reported no per-step outcome, so there is
# nothing to step. `dispatch_for` raising is what makes that a message rather than a session
# whose every step reports `skipped (no 'latent' executor on this engine)`.
#
# `prep` is on ALL THREE ROWS and is the first group that is, which is worth naming because it
# looks like a violation of the fidelity split and is not. The other groups differ per row
# because they ARE the compute — a `latent` step is manylatents' algorithm catalogue, or the
# in-process stand-in for it, or the mock's invention. Prep is scanpy, which every install has
# (`dependencies`, not an extra), so there is nothing for a fidelity axis to select between. A
# row without it would report `skipped (no 'prep' executor on this engine)` for the standard
# preamble — a hole, not a fidelity difference.
#
# `mock` keeps it deliberately, and it is the one row where that could be argued: the mock
# invents numbers rather than reading data, so a "real" filter under it is a mismatch. It is
# here because a prep step returns a MASK over the caller's own frame — there is nothing for a
# mock to invent, and the alternative is `mock` declining every recipe the migration gives a
# prep block to, which is every bundled recipe but `qc`.
_DISPATCH: dict[str, dict] = {
    # `probe` is on manylatents ALONE, and that is the honest asymmetry: it is the only row
    # whose `lightning` executor trains anything. The in-process stand-in computes a diffusion
    # pseudotime and the mock invents numbers, so offering a probe there would put a step in the
    # menu that can only ever decline — which is precisely what `dispatchable` exists to prevent.
    "manylatents": {"latent": _ml_latent, "lightning": _ml_lightning,
                    "analysis": _run_analysis_step, "prep": _run_prep_step,
                    "probe": _run_probe_step},
    INPROC: {"latent": _inproc_step, "lightning": _inproc_step,
             "analysis": _run_analysis_step, "prep": _run_prep_step},
    "mock": {"latent": _mock_transform, "lightning": _mock_transform,
             "analysis": _mock_measure, "prep": _run_prep_step},
}

#: Executors that resolve a step by NAME from a table manyruns owns, so "can this engine run
#: this step" is answerable without running it. An executor absent from this map resolves
#: against something we cannot enumerate — manylatents' algorithm catalogue, or the mock, which
#: synthesizes a result for any name — and every catalogued step is offered for it.
# DECLARED-BUT-UNIMPLEMENTED STEPS JOIN THEIR GROUP'S TABLE HERE, and here specifically: the
# executors resolve by name off these module dicts (`_run_analysis_step`, `_run_probe_step`), so
# a stub has to be IN them — and `stubs.py` imports `_StepSkipped` from `steps`, so `steps`
# cannot import back. This module already imports all three, so the merge has exactly one home.
#
# `setdefault`, never assignment: a real implementation landing under the same name must win, and
# must win silently rather than by someone remembering to delete a line in `stubs.py`.
for _group, _table in (("analysis", _steps._ANALYSIS_STEPS), ("prep", _prep._PREP_STEPS),
                       ("probe", _probes._PROBE_STEPS)):
    for _name, _fn in _stubs.for_group(_group).items():
        _table.setdefault(_name, _fn)

_NAME_TABLES: dict[Any, dict] = {
    _inproc_step: _steps._INPROC_STEPS,
    _run_analysis_step: _steps._ANALYSIS_STEPS,
    _run_prep_step: _prep._PREP_STEPS,
    _run_probe_step: _probes._PROBE_STEPS,
}


def steppable(engine: str) -> bool:
    """Can this engine be driven one step at a time? (Does it have a dispatch table.)"""
    return engine in _DISPATCH


def dispatch_for(engine: str) -> dict:
    """The `group -> executor` table for one engine. Raises for an engine with no step loop."""
    try:
        return dict(_DISPATCH[engine])
    except KeyError:
        raise ValueError(
            f"engine={engine!r} has no step loop: it runs a whole recipe in one call. "
            # `_DISPATCH` also holds `vocab.INPROC`, which is NOT an engine and must not be
            # named in an error a user reads — offering it would be offering the substrate.
            f"Steppable engines: "
            f"{', '.join(sorted(k for k in _DISPATCH if not k.startswith('_')))}."
        ) from None


def dispatchable(engine: str, name: Any, group: Any) -> bool:
    """Could `engine` actually run this step? — the derived answer, never a hand-kept list.

    Two conditions, both read off tables that already exist: the step's GROUP has an executor
    on this engine, and where that executor resolves by NAME from a manyruns-owned table
    (`_INPROC_STEPS`, `_ANALYSIS_STEPS`), the name is in it. Measured on the bundled catalog:
    `real` and `mock` answer True for all four of phate/mioflow/separation/composition;
    `manylatents` likewise, because its latent/lightning names come from the engine's own
    catalogue, which this process cannot enumerate — offering them and letting the engine
    decline is honest, inventing a product-side copy of that catalogue is not."""
    if not steppable(engine):
        return False
    fn = _DISPATCH[engine].get(group)
    if fn is None:
        return False
    table = _NAME_TABLES.get(fn)
    return table is None or name in table
