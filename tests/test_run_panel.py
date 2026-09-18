"""The run panel — a reader over the step record, and the one surface that cannot be wrong.

Four candidate readouts that tried to judge whether a user's structure was real all failed
against a matched control. What survives is a panel of FACTS ABOUT THE RUN: what executed,
how long it took, why something was skipped, and whether the run is reproducible at all.
None of those are inferences about the data, so none of them can be false.

Before this, `manyruns run` printed a device note and a file path. A run where every step
succeeded and a run where every step failed both wrote a summary and exited 0.
"""
from __future__ import annotations

import pytest

from manyruns import narrate


def _rec(name, outcome, seconds=0.1, detail=None, group="latent"):
    return {"index": 0, "name": name, "group": group, "params": {},
            "outcome": outcome, "detail": detail, "seconds": seconds}


def test_a_run_where_everything_failed_does_not_read_like_a_success():
    """THE point of the panel. The two runs below have identical traces and identical step
    counts — that is exactly why `trace` could never be the surface a user reads."""
    good = {"recipe": "r", "engine": "_inproc", "seed": 42,
            "steps": [_rec("a", "ok"), _rec("b", "ok")]}
    bad = {"recipe": "r", "engine": "_inproc", "seed": 42,
           "steps": [_rec("a", "error", detail="ValueError: kaboom"),
                     _rec("b", "error", detail="ValueError: kaboom")]}

    assert "2 of 2 steps produced a result" in narrate.run_panel(good)
    assert "0 of 2 steps produced a result" in narrate.run_panel(bad)
    assert narrate.run_panel(good) != narrate.run_panel(bad)


def test_an_engine_with_no_step_record_says_so_rather_than_implying_execution():
    """`engine=mock` (the default) and the learner's engine report no per-step outcome —
    the latter synthesised its trace from the recipe's *declared* steps. Presenting that as a
    run would be the silent-success failure the panel exists to prevent."""
    panel = narrate.run_panel(
        {"recipe": "r", "engine": "mock", "trace": ["latent:phate", "analysis:separation"]}
    )
    assert "3 steps" not in panel
    assert "no per-step outcome" in panel
    assert "none of the above is evidence that anything ran" in panel
    assert "produced a result" not in panel      # never claims an outcome it wasn't given


def test_an_absent_seed_is_reported_never_invented():
    """The in-process loop threads no seed (issue #29), so those runs cannot be repeated. The
    panel says so; it does not print a default that would imply reproducibility."""
    unseeded = {"recipe": "r", "engine": "_inproc", "seed": None, "steps": [_rec("a", "ok")]}
    seeded = {"recipe": "r", "engine": "_inproc", "seed": 42, "steps": [_rec("a", "ok")]}

    assert "cannot be reproduced" in narrate.run_panel(unseeded)
    assert "cannot be reproduced" not in narrate.run_panel(seeded)


def test_a_skip_shows_its_reason():
    """A skipped step is not a failure and not a success; the reason is the actionable part
    (`no condition labels` tells a user something about their data, not about manyruns)."""
    panel = narrate.run_panel({
        "recipe": "contrast", "engine": "_inproc", "seed": None,
        "steps": [_rec("separation", "skipped", detail="no condition labels in this data",
                       group="analysis")],
    })
    assert "⊘" in panel and "skipped" in panel
    assert "no condition labels in this data" in panel


def test_a_long_message_cannot_wreck_the_layout():
    """Every step is one line. A long traceback message is elided, not wrapped —
    `summary.md` keeps it in full."""
    panel = narrate.run_panel({
        "recipe": "r", "engine": "_inproc", "seed": 1,
        "steps": [_rec("a", "error", detail="X" * 400)],
    })
    assert len(panel.splitlines()) == 4          # header, the step, blank, the count
    assert "…" in panel


def test_the_panel_triggers_no_compute_and_needs_no_optional_dependency():
    """It is a reader. It takes a finished result dict and returns a string — no engine, no
    numpy, no rich. That is what keeps the instrument seam honest and the panel importable
    on the stackless path."""
    import sys

    before = set(sys.modules)
    out = narrate.run_panel({"recipe": "r", "engine": "_inproc", "seed": 1,
                             "steps": [_rec("a", "ok")]})
    assert isinstance(out, str)
    assert not {"numpy", "rich", "phate"} - before & (set(sys.modules) - before)


def test_an_empty_result_degrades_instead_of_raising():
    assert "no steps recorded" in narrate.run_panel({})


# ── the terminal rendering ───────────────────────────────────────────────────
def _result():
    return {"recipe": "contrast", "engine": "_inproc", "seed": None,
            "g_vector": {"n_samples": 300, "n_features": 30, "separation_silhouette": 0.9},
            "steps": [_rec("phate", "ok"), _rec("separation", "error", detail="boom")]}


def test_a_terminal_gets_panels_and_a_pipe_gets_plain_text():
    """Box drawing is for a terminal. Piped or redirected output must stay plain — the
    same facts, without escape codes a downstream reader would have to strip."""
    pytest.importorskip("rich")
    from rich.panel import Panel

    from manyruns import shell

    class Term:
        is_terminal = True

        def __init__(self):
            self.printed = []

        def print(self, *a):
            self.printed.extend(a)

    class Piped:
        is_terminal = False

        def __init__(self):
            self.printed = []

        def print(self, *a):
            self.printed.extend(a)

    term, piped = Term(), Piped()
    shell.render_result(term, _result())
    shell.render_result(piped, _result())

    assert sum(isinstance(p, Panel) for p in term.printed) == 2   # the run, and the finding
    assert all(isinstance(p, str) for p in piped.printed)
    assert "separation" in " ".join(piped.printed)


def test_the_plain_and_panel_paths_report_the_same_outcomes():
    """Two renderings, one source. If they could disagree there would be two truths about
    what ran, which is the failure the step record exists to remove."""
    pytest.importorskip("rich")
    from manyruns import narrate

    plain = narrate.run_panel(_result())
    assert "1 of 2 steps produced a result" in plain
    assert "cannot be reproduced" in plain          # seed is None in both renderings


# ── the geometry panel and the figure pull ───────────────────────────────────
def _sections(g):
    """{section title: {row label: row}} — the panel's content, addressable by section."""
    from manyruns import narrate

    return {s.title: {r.label: r for r in s.rows} for s in narrate.geometry_sections(g)}


#: The g-vector of a real 400×40 `embed` run on the in-process loop, copied off the run. Two of the
#: twelve declared metrics come back absent, and both absences are STRUCTURAL rather than
#: flaky: `geodesic_distance_correlation` needs a dataset exposing `get_gt_dists`, which
#: `suite._Ambient` deliberately does not provide, and `kernel_sparsity` needs a fitted
#: `LatentModule` to read a kernel matrix off, which the suite never passes.
_MEASURED_EMBED_RUN = {
    # Producer metadata, required to call a value a setting. Legacy punctuation is ambiguous.
    "_provenance": {key: {"kind": "setting", "stage": "phate configuration"} for key in
                    ("phate_dims", "phate.knn", "phate.decay", "phate.gamma", "phate.n_pca",
                     "phate.n_landmark")},
    "phate_dims": 3, "phate.knn": 5, "phate.decay": 40, "phate.gamma": 1,
    "phate.n_pca": 100, "phate.n_landmark": 2000,
    "n_samples": 400, "n_features": 40, "n_embedded": 400, "final_dim": 3,
    "trustworthiness": 0.6480052486187845, "continuity": 0.1888, "knn_preservation": 0.151,
    "lid": 3.380809077684398, "loglog_consistency": 0.9893230427765006,
    "anisotropy": 0.37256091990983936, "participation_ratio": 2.6770254792670816,
    "betti_0": 13.0, "betti_1": 0.0,
    "geodesic_distance_correlation": None,
    "geodesic_distance_correlation_note": "returned nan",
    "outlier_score": 1.0950784577775583,
    "kernel_sparsity": None, "kernel_sparsity_note": "returned nan",
    "suite_measured": 10, "suite_declared": 12,
    "suite_null": "data-null unimplemented — comparable, not a finding",
    "suite_seconds": 1.193,
}


def test_only_real_before_and_after_pairs_are_shown_as_changes():
    """`GVECTOR_CORE` carries a genuine baseline — input rows vs embedded rows, input columns
    vs final dimensions — so those get arrows. Everything else is a value this run produced,
    not a delta. Inventing a baseline to subtract would be a comparison nothing validated."""
    sections = _sections({
        "n_samples": 300, "n_features": 30, "n_embedded": 256, "final_dim": 3,
        "separation_silhouette": 0.9956, "separation_note": "ignored",
        "composition_between": "treated vs healthy",   # not numeric → not a geometry row
    })
    changed, measured = sections["input → output"], sections["final measurements"]

    assert changed["cells"].value == "300 ⇢ 256"     # subsampled: a real change, flagged
    assert changed["dimensions"].value == "30 → 3"
    assert measured["separation_silhouette"].value == "0.9956"
    assert not any("_note" in label for s in sections.values() for label in s)
    assert "composition_between" not in measured     # strings are not measurements
    assert "n_samples" not in measured               # shown as the arrow, never twice


def test_a_setting_a_measurement_and_a_stopwatch_are_not_one_list():
    """The panel used to render 21 undifferentiated rows under one heading that said all of
    them "moved geometrically". `phate.knn 5` is what the run was TOLD to do, `lid 3.394` is
    what came out, and `suite_seconds 1.193` is how long the measuring took — three different
    kinds of claim, and only the first section is a before/after at all."""
    sections = _sections(_MEASURED_EMBED_RUN)

    assert sections["settings"].keys() == {
        "phate_dims", "phate.knn", "phate.decay", "phate.gamma", "phate.n_pca",
        "phate.n_landmark"}
    assert sections["bookkeeping"].keys() == {
        "suite_measured", "suite_declared", "suite_seconds"}
    assert {"lid", "betti_0", "trustworthiness"} <= sections["final measurements"].keys()
    # the three must not leak into each other — that is the whole defect
    for stray in ("phate.knn", "suite_seconds"):
        assert stray not in sections["final measurements"]


def test_a_metric_that_could_not_be_measured_is_named_with_its_reason():
    """The point of `suite.measure`'s declared-not-conditional design, and what the panel
    threw away by skipping `None`: the run above showed `10` of `12` measured with no way to
    learn which two were missing or why."""
    measured = _sections(_MEASURED_EMBED_RUN)["final measurements"]

    for name in ("geodesic_distance_correlation", "kernel_sparsity"):
        assert name in measured, f"{name} vanished — an absence is a fact, not a gap"
        assert measured[name].measured is False
        assert measured[name].value == "returned nan"

    absent = [label for label, row in measured.items() if not row.measured]
    assert len(absent) == _MEASURED_EMBED_RUN["suite_declared"] - \
        _MEASURED_EMBED_RUN["suite_measured"]

    plain = narrate.run_panel({"recipe": "embed", "engine": "_inproc", "seed": 42,
                               "steps": [_rec("phate", "ok")],
                               "g_vector": _MEASURED_EMBED_RUN})
    assert "not measured — returned nan" in plain
    assert "geodesic_distance_correlation" in plain and "kernel_sparsity" in plain


def test_an_absence_never_looks_like_a_value():
    """A reason is prose sitting in a column of numbers. It is labelled, clipped, and it never
    reaches the panel as a bare string that could be read as a result."""
    long_reason = "raised ValueError: " + "x" * 200
    row = _sections({"lid": None, "lid_note": long_reason})["final measurements"]["lid"]

    assert row.measured is False
    assert len(row.value) <= 44 and row.value.endswith("…")

    # `None` with nothing to say is still shown — silence would put it back in the gap
    quiet = _sections({"lid": None})["final measurements"]["lid"]
    assert quiet.measured is False and quiet.value == "no reason recorded"


def test_the_numbers_carry_the_sentence_that_says_they_are_not_a_finding():
    """Every suite metric is derived from the embedding, so `vocab.NULL_KIND` says each needs
    a `data` null and none is implemented. `suite_null` states that once; before this it was a
    string in the g-vector that the numeric-only panel dropped on the floor, leaving 21 numbers
    on screen with nothing saying what they are not."""
    sections = {s.title: s for s in narrate.geometry_sections(_MEASURED_EMBED_RUN)}

    assert sections["final measurements"].note == _MEASURED_EMBED_RUN["suite_null"]
    assert "suite_null" not in {r.label for s in sections.values() for r in s.rows}
    assert _MEASURED_EMBED_RUN["suite_null"] in narrate.run_panel(
        {"recipe": "embed", "engine": "_inproc", "seed": 42, "steps": [_rec("phate", "ok")],
         "g_vector": _MEASURED_EMBED_RUN})

    # no suite, no blanket caveat: `separation_silhouette` has a labels null and claiming
    # otherwise would be as wrong as claiming nothing.
    mock = {s.title: s for s in narrate.geometry_sections({"separation_silhouette": 0.4})}
    assert mock["final measurements"].note == ""


# ── per-step geometry deltas ─────────────────────────────────────────────────
def test_a_first_embedding_is_an_appearance_not_a_delta_from_zero():
    """`rec["geometry"]` is `{metric: [from, to]}` and `from` is None when there was no prior
    embedding. Treating the absent side as 0 would report a 3.394-unit move that nothing
    measured — the same fabricated baseline `geometry_sections` refuses for the g-vector."""
    rows = {r.label: r.value for r in narrate.geometry_delta_rows(
        {"geometry": {"lid": [None, 3.394], "betti_0": [None, 14]}})}

    assert rows == {"lid": "— → 3.394", "betti_0": "— → 14"}
    assert "0 →" not in " ".join(rows.values())

    # a second embedding step HAS both numbers, and that pair is a real measured change
    moved = narrate.geometry_delta_rows({"geometry": {"lid": [3.394, 2.71]}})
    assert moved[0].value == "3.394 → 2.71"


def test_a_step_that_moved_no_geometry_says_nothing_and_a_broken_payload_cannot_crash():
    """The key is absent on steps that changed no embedding, so no rows is the correct
    rendering of "this step moved nothing" — not of "nobody looked"."""
    assert narrate.geometry_delta_rows(_rec("separation", "ok")) == []
    assert narrate.geometry_delta_rows({"geometry": None}) == []
    assert narrate.geometry_delta_rows({"geometry": {"lid": 3.4, "betti_0": [1],
                                                     "x": [None, None]}}) == []


def test_per_step_geometry_lands_in_the_steps_panel_the_live_panel_and_the_plain_text():
    """A scientist watching a run sees geometry move as each step lands — in all three
    renderings, because `run_panel` is the fallback for the other two and a fallback that
    showed fewer facts would be a second account of the same run."""
    pytest.importorskip("rich")
    from rich.console import Console

    from manyruns import shell

    steps = [
        {**_rec("phate", "ok", seconds=1.63), "index": 0, "produced": "ndarray(400, 3)",
         "geometry": {"lid": [None, 3.394], "betti_0": [None, 14]}},
        {**_rec("mioflow", "ok", seconds=2.1), "index": 1, "produced": "ndarray(400, 3)",
         "geometry": {"lid": [3.394, 2.71], "betti_0": [14, 9]}},
    ]
    declared = [{"name": "phate", "group": "latent"}, {"name": "mioflow", "group": "lightning"}]

    def draw(renderable):
        out = Console(force_terminal=False, width=100)
        with out.capture() as cap:
            out.print(renderable)
        return cap.get()

    body = draw(shell._steps_body({"steps": steps, "seed": 42}, narrate))
    live = draw(shell._progress_panel(declared, steps, "t"))
    plain = narrate.run_panel({"recipe": "cflows", "engine": "_inproc", "seed": 42, "steps": steps})

    for text in (body, plain):
        assert "— → 3.394" in text and "3.394 → 2.71" in text   # both steps, in order
        assert "14 → 9" in text
    # the live panel deliberately shows only the newest step: the suite declares 12 metrics
    # and rich.Live has to repaint the whole thing inside one screen
    assert "3.394 → 2.71" in live and "— → 3.394" not in live


def test_the_plain_and_panel_renderings_show_the_same_geometry():
    """Two renderings, one source. A fact visible in the terminal and missing from the pipe
    would be exactly the drift the step record was built to remove."""
    pytest.importorskip("rich")
    from rich.console import Console

    from manyruns import shell

    results = {"recipe": "embed", "engine": "_inproc", "seed": 42,
               "steps": [_rec("phate", "ok")], "g_vector": _MEASURED_EMBED_RUN}
    out = Console(force_terminal=False, width=100)
    with out.capture() as cap:
        out.print(shell._geometry_panel(_MEASURED_EMBED_RUN))
    panel, plain = cap.get(), narrate.run_panel(results)

    for section in narrate.geometry_sections(_MEASURED_EMBED_RUN):
        assert section.title in panel and section.title in plain
        for row in section.rows:
            assert row.label in panel, f"{row.label} missing from the panel"
            assert row.label in plain, f"{row.label} missing from the plain fallback"
    assert "not measured" in panel and "not measured" in plain


def test_a_g_vector_with_nothing_renderable_draws_no_empty_panel():
    """The mock writes a g-vector only for `analysis` steps, so an empty one is a real case."""
    pytest.importorskip("rich")
    from manyruns import shell

    assert narrate.geometry_sections({}) == []
    assert shell._geometry_panel({}) is None
    assert shell._geometry_panel({"composition_between": "a vs b"}) is None


def test_the_figure_panel_only_claims_files_that_exist(tmp_path):
    """A path in `plots` that was never written must not be advertised."""
    from manyruns import shell

    real = tmp_path / "phate.png"
    real.write_bytes(b"\x89PNG\r\n")

    class Term:
        is_terminal = True

        def __init__(self):
            self.printed = []

        def print(self, *a):
            self.printed.extend(a)

    con = Term()
    shell._render_plots(con, {"plots": [str(real), str(tmp_path / "missing.png")]})
    body = " ".join(str(getattr(p, "renderable", p)) for p in con.printed)
    assert "phate.png" in body
    assert "missing.png" not in body

    con2 = Term()
    shell._render_plots(con2, {"plots": []})
    assert con2.printed == []                     # nothing to show → no empty panel


def test_inline_images_are_attempted_only_where_the_terminal_speaks_the_protocol(tmp_path, monkeypatch):
    """Detection covers the terminals people actually use, and declines where it would corrupt.

    The first version keyed on `TERM_PROGRAM == "iTerm.app"` alone, so kitty, Ghostty, WezTerm
    and the VS Code integrated terminal — which implements the iTerm2 protocol — all fell
    through to printing a file path."""
    from manyruns import shell

    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNG\r\n")
    for var in ("TERM_PROGRAM", "TERM", "KITTY_WINDOW_ID", "LC_TERMINAL", "TMUX", "STY",
                "MANYRUNS_INLINE_IMAGES"):
        monkeypatch.delenv(var, raising=False)

    assert shell._image_protocol() is None            # a terminal that claims nothing gets nothing

    for program in ("iTerm.app", "WezTerm", "vscode"):
        monkeypatch.setenv("TERM_PROGRAM", program)
        assert shell._image_protocol() == "iterm", program
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    assert shell._image_protocol() == "kitty"
    monkeypatch.setenv("TERM", "xterm-kitty")
    assert shell._image_protocol() == "kitty"

    # a multiplexer swallows the escape unless passthrough is on, so decline rather than
    # dump 100k characters of base64 into the scrollback
    monkeypatch.setenv("TMUX", "/tmp/tmux-501/default,123,0")
    assert shell._image_protocol() is None
    monkeypatch.delenv("TMUX")

    # the override wins both ways — detection is env sniffing and env vars lie
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", "off")
    assert shell._image_protocol() is None
    monkeypatch.delenv("TERM_PROGRAM")
    monkeypatch.delenv("TERM")
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", "iterm")
    assert shell._image_protocol() == "iterm"

    assert shell._image_escape(png, "iterm", 60).startswith("\033]1337;File=inline=1")
    kitty = shell._image_escape(png, "kitty", 60)
    assert kitty.startswith("\033_Gf=100,a=T,t=d,c=60,m=0;") and kitty.endswith("\033\\")


def test_a_large_figure_is_chunked_and_never_wrapped_by_rich(tmp_path, monkeypatch):
    """The payload goes straight to the file descriptor, and kitty's 4096-byte cap is honoured.

    Routing it through `console.print` would let rich count ~100k base64 characters as
    printable width and wrap a newline into the middle of the image."""
    from manyruns import shell

    png = tmp_path / "big.png"
    png.write_bytes(b"\x89PNG\r\n" + b"\x00" * 20_000)          # ~27k base64 chars → 7 chunks

    chunks = shell._image_escape(png, "kitty", 60).split("\033_G")[1:]
    assert len(chunks) > 1
    assert all(len(c.split(";", 1)[1].rstrip("\033\\")) <= 4096 for c in chunks)
    assert all("m=1" in c for c in chunks[:-1]) and "m=0" in chunks[-1]
    assert "f=100" not in "".join(chunks[1:])       # control keys ride the first chunk only

    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", "kitty")

    class _Tty:
        is_terminal = True

        def __init__(self):
            self.file = self
            self.buf = ""

        def write(self, s):
            self.buf += s

        def flush(self):
            pass

    tty = _Tty()
    assert shell._emit_inline_image(tty, png) is True
    assert "\n" not in tty.buf[:-1]                 # exactly one newline, at the very end

    # not a terminal (a pipe, a test, `| less`) → never emit binary
    class _Pipe(_Tty):
        is_terminal = False

    assert shell._emit_inline_image(_Pipe(), png) is False


def test_the_size_keys_have_one_declaration_not_two():
    """The panel hides the size descriptors from the measured section — they are already the
    two arrows above it — and it used to do that by matching against its own frozenset while
    `runner.GVECTOR_CORE` declared the same four as the keys `_finalize` emits. Identical as
    sets, different container, nothing keeping them equal: add a fifth size key at the
    producer and the panel would list it as a measurement of the embedding. It now reads the
    producer's declaration, and this fails if that stops being true."""
    from manyruns.pipeline.runner import GVECTOR_SIZE

    assert narrate._size_keys() == frozenset(GVECTOR_SIZE)

    sections = _sections({**{k: 4 for k in GVECTOR_SIZE}, "lid": 3.394})
    measured = sections["final measurements"]
    assert set(measured) == {"lid"}
    for key in GVECTOR_SIZE:
        assert key not in measured, f"{key} is a size descriptor, not a measurement"
    assert sections["input → output"]["cells"].value == "4 → 4"


def test_a_core_key_that_is_not_a_size_key_still_reaches_the_panel():
    """Suppression is keyed on SIZE, not on CORE, and the two must be free to differ.

    The test above pins `_size_keys()` to the producer's declaration; on its own that is
    satisfied just as well by reading `GVECTOR_CORE`, because it derives both sides of its
    assertion from whichever tuple `_size_keys` happens to read. This one does not: it adds a
    geometry key to `GVECTOR_CORE` ALONE and asserts the panel still reports it.

    Measured with `_size_keys` reading `GVECTOR_CORE`: extending that tuple with an
    always-computed geometry key put `margin: 0.42`
    in the g-vector and left it out of every panel row, with the whole suite green. A number
    the record carries and the panel silently drops is the failure this file exists to catch,
    so it is pinned here rather than left for a future geometry key to expose."""
    import manyruns.pipeline.runner as runner

    original = runner.GVECTOR_CORE
    try:
        runner.GVECTOR_CORE = (*original, "margin")
        sections = _sections({**{k: 4 for k in runner.GVECTOR_SIZE}, "margin": 0.42})
        measured = sections["final measurements"]
        assert "margin" in measured, "a core key that is not a size key vanished from the panel"
    finally:
        runner.GVECTOR_CORE = original
