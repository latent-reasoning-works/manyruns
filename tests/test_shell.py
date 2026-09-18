"""The interactive shell (manyruns/shell.py) + intent layer (manyruns/intent.py).

Driven with an injected prompter (scripted answers) and a capturing console, so the whole
read → offer → run → narrate loop runs off a real terminal, stackless, on the mock engine.
The two load-bearing harness invariants are asserted directly: (1) importing the shell/intent
never pulls an SDK or the TUI packages into the base chain, and (2) intent refuses an analysis
the data can't honestly support before it ever routes to a recipe.
"""
from __future__ import annotations

import pytest


# ── injectable I/O ───────────────────────────────────────────────────────────
class ScriptedPrompter:
    def __init__(self, selects, texts=()):
        self._selects = list(selects)
        self._texts = list(texts)

    def select(self, message, choices, default=None):
        assert self._selects, f"unexpected select: {message}"
        return self._selects.pop(0)

    def text(self, message, default=""):
        return self._texts.pop(0) if self._texts else ""


class CaptureConsole:
    is_terminal = False

    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args):
        self.lines.append(" ".join(str(a) for a in args))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture(autouse=True)
def _isolated_drop_folder(tmp_path, monkeypatch):
    """Point the drop folder at a tmp dir so tests never create/scan ./data in the repo.

    `chdir` for the same reason, one directory over: a completed run now appends to the run
    store (`shell._record`), whose default `outputs/index.jsonl` is CWD-relative — so without
    this the end-to-end tests below would write a history file into the working copy. The
    behaviour is asserted in `tests/test_lineage.py`, against a store under `tmp_path`."""
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("MANYRUNS_DATASET_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    def no_network(*args, **kwargs):
        pytest.fail("shell tests must inject the download transport")

    monkeypatch.setattr("urllib.request.urlopen", no_network)


def _shell_args():
    from manyruns import app

    return app._build_parser().parse_args(["shell", "--engine", "mock"])


# ── the loop, end to end on the mock ─────────────────────────────────────────
def test_explore_a_preset_dataset_end_to_end_on_mock():
    """explore → pick swissroll → run cflows → narrate → quit, all on the $0 mock."""
    pytest.importorskip("numpy")
    from manyruns import shell

    con = CaptureConsole()
    pr = ScriptedPrompter(selects=["explore", ("dataset", "swissroll"), "cflows", "quit"])
    rc = shell.run_shell(_shell_args(), prompter=pr, console=con)
    out = con.text

    assert rc == 0
    assert "swissroll" in out                      # it read + named the data
    assert "phate" in out and "mioflow" in out     # the step trace ran
    assert "path" in out.lower()                   # narrate spoke a plain-English finding


def test_plain_prompter_drives_the_shell_without_a_tty(monkeypatch):
    """No terminal / no questionary: the numbered input() fallback must run the same loop.

    Under pytest stdin isn't a TTY, so run_shell picks PlainPrompter on its own — we only
    script the keystrokes: explore(1) → swissroll(3) → "run a specific recipe"(2) →
    cflows(by name) → quit(5).

    `cflows` is now ONE LEVEL DOWN for this data, and that is the behaviour under test rather
    than an inconvenience. swissroll is `shape: manifold`, which `cflows` does not declare in
    its `suits:`, so the top menu offers only the question the data is suited to and puts the
    rest behind a single agnostic entry. Previously every legal recipe was a top-level row —
    `cflows` appearing as the bare label "Run the cflows recipe" beside a question written in
    English — so the menu grew by a row per recipe and mixed two kinds of thing.

    The last recipe answer goes in BY NAME, exercising `PlainPrompter`'s substring match: a
    positional script would silently pass while selecting the wrong entry the next time the
    menu is reordered, which is exactly how this test broke."""
    pytest.importorskip("numpy")
    from manyruns import shell

    answers = iter(["1", "3", "2", "cflows", "5"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(answers))
    con = CaptureConsole()
    rc = shell.run_shell(_shell_args(), console=con)   # prompter defaulted → PlainPrompter
    assert rc == 0
    assert "swissroll" in con.text and "mioflow" in con.text


def test_explore_a_dropped_file_shows_data_loaded(tmp_path, monkeypatch):
    """A file dropped into the data folder is listed, loaded (with a ✓ confirmation), and run."""
    pytest.importorskip("numpy")
    from manyruns import shell

    drop = tmp_path / "data"
    drop.mkdir()
    f = drop / "cohort.csv"
    f.write_text("a,b\n1,2\n")
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(drop))

    con = CaptureConsole()
    pr = ScriptedPrompter(selects=["explore", ("path", f), "embed", "quit"])
    rc = shell.run_shell(_shell_args(), prompter=pr, console=con)
    out = con.text
    assert rc == 0
    assert "cohort.csv" in out
    assert "data loaded" in out          # the explicit validation step


def test_scan_drop_folder_lists_data_and_folders_only(tmp_path):
    from manyruns import shell

    (tmp_path / "cohort.h5ad").write_bytes(b"")
    (tmp_path / "notes.md").write_text("not data")
    (tmp_path / "treated_vs_healthy").mkdir()
    labels = [label for label, _ in shell._scan_drop_folder(tmp_path)]
    assert any(lbl == "cohort.h5ad" for lbl in labels)
    assert any("treated_vs_healthy/" in lbl for lbl in labels)
    assert not any("notes.md" in lbl for lbl in labels)   # .md is not a data suffix


def test_data_dir_defaults_and_respects_env(monkeypatch, tmp_path):
    """Both branches, pinned explicitly — this test used to assert `Path("data")`
    unconditionally and passed only because the repo it ran in happened to have a `./data`.
    CI has none (it is gitignored), so the assertion encoded the developer's directory
    rather than the contract."""
    from pathlib import Path

    from manyruns import shell

    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path / "drop"))
    assert shell.data_dir() == tmp_path / "drop"          # the env var always wins

    monkeypatch.delenv("MANYRUNS_DATA_DIR")
    monkeypatch.chdir(tmp_path)
    # no ./data here → the stable home folder, so "where do I put my file" has one answer
    # that does not change every time you cd
    assert shell.data_dir() == Path.home() / ".manyruns" / "data"

    (tmp_path / "data").mkdir()
    assert shell.data_dir() == Path("data")               # a checkout keeps its own


def test_back_out_of_source_returns_to_menu_then_quits():
    from manyruns import shell

    con = CaptureConsole()
    pr = ScriptedPrompter(selects=["explore", ("back", None), "quit"])
    assert shell.run_shell(_shell_args(), prompter=pr, console=con) == 0


def test_recipes_and_datasets_verbs_list_the_registries():
    from manyruns import shell

    con = CaptureConsole()
    pr = ScriptedPrompter(selects=["recipes", "datasets", "quit"])
    shell.run_shell(_shell_args(), prompter=pr, console=con)
    out = con.text
    assert "cflows" in out and "swissroll" in out


# ── intent: rule-based refuse/route, and the LLM tier is genuinely optional ───
def test_route_recipe_refuses_a_trajectory_with_no_time_axis():
    from manyruns import intent
    from manyruns.narrate import Observation

    obs = Observation(shape="case-control", conditions=["treated", "healthy"])
    recipe, refusal = intent.route_recipe(
        "show me the trajectory of these cells over time", obs, ["contrast", "embed"]
    )
    assert recipe is None
    assert refusal and ("clock" in refusal.lower() or "progression" in refusal.lower())


def test_route_recipe_maps_compare_to_contrast():
    from manyruns import intent
    from manyruns.narrate import Observation

    obs = Observation(shape="case-control", conditions=["treated", "healthy"])
    recipe, refusal = intent.route_recipe(
        "compare the two groups", obs, ["contrast", "embed"]
    )
    assert recipe == "contrast" and refusal is None


def test_llm_tier_is_off_without_a_key(monkeypatch):
    from manyruns import intent

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert intent.llm_available() is False


def test_route_verb_is_rule_based_by_default():
    from manyruns import intent

    assert intent.route_verb("please validate the catalog") == "check"
    assert intent.route_verb("what recipes are there") == "recipes"


# ── the two harness invariants ───────────────────────────────────────────────
def test_bare_manyruns_opens_the_shell():
    from manyruns import app

    assert app._normalize_argv([]) == ["shell"]
    assert app._normalize_argv(["swissroll"]) == ["run", "swissroll"]  # back-compat kept


def test_base_import_chain_pulls_no_sdk_or_tui():
    """Importing the shell + intent must not drag rich / questionary / anthropic into the base
    process — that is what keeps the shipped app (and a compiled binary) SDK-free."""
    import importlib
    import sys

    # Popping only the TOP-LEVEL key orphaned every `rich.*` submodule: a later `import rich`
    # built a fresh package object, while `import rich.repr` short-circuited on the stale
    # `sys.modules["rich.repr"]` and never re-attached the attribute. Any Textual test running
    # after this one then died on "module 'rich' has no attribute 'repr'" — a failure that
    # depended on test ORDER, which is why it looked intermittent.
    roots = ("rich", "questionary", "anthropic")
    saved = {k: v for k, v in sys.modules.items()
             if k in roots or any(k.startswith(r + ".") for r in roots)}
    for k in saved:
        sys.modules.pop(k, None)
    try:
        importlib.import_module("manyruns.shell")
        importlib.import_module("manyruns.intent")
        importlib.import_module("manyruns.verbs")
        for m in roots:
            assert m not in sys.modules, f"{m} leaked into the base import chain"
    finally:
        # Restore the whole tree, parents included, so the next test sees rich intact.
        for k, v in saved.items():
            sys.modules.setdefault(k, v)


# ── the front door's half of the step seam ───────────────────────────────────
#
# `Session` owns "a step is a dict, and its params reach the fit"; `shell` owns the driver that
# hands it one. Both halves were pinned separately and their JOINT behaviour was not: I removed
# each of the two lines below in turn and ran the whole suite (`pytest tests/ -q
# --ignore=tests/harness`) — 516 passed, 1 skipped, unchanged, both times. These two tests are
# what turned that into a failure.
def test_the_stepped_front_door_hands_the_session_the_step_not_its_name():
    """The recipe's declared params must reach the session, on the shell path too.

    `session.step(s["name"])` re-resolves the step through the CATALOG, which silently
    substitutes the catalog's declared params for the loaded recipe's. On the bundled three
    that is invisible — `phate` is `{n_components: 3}` in cflows, contrast and embed alike — so
    the recipe below declares **4**, which no bundled recipe does. That is the whole reason the
    mutation survived the suite: every fixture agreed with the catalog it was being replaced by.

    The return value is asserted too, and for a separate reason: `_run_stepped` must end with
    `close()`, not `results()`. `finish_carry` is the only writer of the `COMPLETE` marker, so
    a driver that finalises without closing leaves every front-door lineage looking interrupted
    on disk — while returning a byte-identical record, which is why nothing downstream notices.
    """
    from manyruns import shell

    handed: list = []

    class _Session:
        steps: list = []

        def step(self, action):
            handed.append(action)
            return {"ok": True, "name": "phate", "detail": "ok"}

        def results(self):
            return {"ended_with": "results"}

        def close(self):
            return {"ended_with": "close"}

    recipe = {"name": "odd", "steps": [
        {"name": "phate", "group": "latent", "params": {"n_components": 4}}]}
    out = shell._run_stepped(CaptureConsole(), _Session(), recipe)

    assert handed == [{"name": "phate", "group": "latent", "params": {"n_components": 4}}]
    assert out == {"ended_with": "close"}


def test_the_front_door_steps_every_engine_that_has_a_step_loop(monkeypatch):
    """`shell._run` carried the same stale `engine in ("mock", "manylatents")` predicate
    `app.cmd_init` did, and only `cmd_init`'s deletion was pinned (`test_session.py::
    test_init_opens_the_prompt_on_a_public_install`). `real` is what `_default_engine` returns
    on a public install, so under the stale predicate the bare `manyruns` front door — the
    default entry point — never reached the stepped path at all on the shipped configuration.

    The learner's engine was the one that must still take the blocking branch: it executes a whole
    recipe in one downstream call and reports no per-step outcome, so there is nothing to step.
    """
    from manyruns import app, shell

    class _Session:      # `_run` stamps `.source` on it, so it cannot be a bare sentinel
        pass

    monkeypatch.setattr(app, "_build_session", lambda proj, args: _Session())
    monkeypatch.setattr(shell, "_run_stepped", lambda con, s, r: {"stepped": True})
    monkeypatch.setattr(app, "explore_once", lambda **kw: {"stepped": False})

    def _route(engine):
        obs = type("O", (), {"source": "swissroll"})()
        return shell._run(_shell_args(), CaptureConsole(), None, "swissroll", "scrna",
                          "cflows", obs, engine=engine)["stepped"]

    assert _route("_inproc") is True               # the public default — was unreachable
    assert _route("mock") is True
    assert _route("manylatents") is True
    # The negative arm used to be the learner's engine — the one that ran a whole recipe in a
    # single downstream call, so there was nothing to step. It is not an engine any more
    # (§3.5), and EVERY engine now steps. What is left to check is that a name which is not an
    # engine does not silently acquire a step loop.
    assert _route("not_an_engine") is False


@pytest.fixture
def downloadable_sample(tmp_path, monkeypatch):
    import hashlib
    import io

    import yaml

    from manyruns import catalog

    payload = b"a,b\n1,2\n3,4\n"
    cfg = {"name": "sample", "shape": "clusters", "modality": "scrna", "handle": {
        "kind": "path", "ref": "elsewhere/sample.csv", "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(), "url": "https://example.org/sample.csv",
    }}
    root = tmp_path / "catalog"
    root.mkdir()
    (root / "sample.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("MANYRUNS_DATASET_DIR", str(root))
    requests = []

    def opener(url, *, timeout):
        requests.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr("urllib.request.urlopen", opener)
    assert root.is_dir()
    return catalog.load_dataset("sample"), requests


class RecordingPrompter(ScriptedPrompter):
    def __init__(self, selects):
        super().__init__(selects)
        self.offered = []

    def select(self, message, choices, default=None):
        self.offered.append((message, choices, default))
        return super().select(message, choices, default)


def test_fetch_on_selection_returns_a_newly_resolved_absolute_drop_path(downloadable_sample):
    from manyruns import shell

    ds, requests = downloadable_sample
    con = CaptureConsole()
    pr = RecordingPrompter([("dataset", "sample"), "fetch"])
    picked = shell._pick_source(pr, con)
    assert picked[0] == shell.data_dir().resolve() / "sample.csv"
    assert picked[1] is None
    assert picked[3].source == "sample"
    assert picked[0].read_bytes() == b"a,b\n1,2\n3,4\n"
    assert requests == [ds["handle"]["url"]]
    assert "download on selection" in str(pr.offered[0])
    assert pr.offered[1][2] == "back"
    assert [label for label, value in pr.offered[1][1]] == ["Fetch", "Back"]
    for expected in ("sample.csv", ds["handle"]["url"],
                     str(shell.data_dir().resolve() / "sample.csv"), str(ds["handle"]["bytes"])):
        assert expected in con.text


def test_fetch_back_stays_in_the_picker_with_manual_instructions(downloadable_sample):
    from manyruns import shell

    ds, requests = downloadable_sample
    con = CaptureConsole()
    pr = RecordingPrompter([("dataset", "sample"), "back", ("back", None)])
    assert shell._pick_source(pr, con) is None
    assert requests == []
    assert len(pr.offered) == 3
    assert ds["handle"]["url"] in con.text
    assert str(shell.data_dir().resolve() / "sample.csv") in con.text
    assert "local" in con.text


def test_missing_file_instructions_are_plain_text_and_include_manual_pins(downloadable_sample):
    from manyruns import shell

    ds, requests = downloadable_sample
    text = "\n".join(shell.where_the_file_should_be("[sample]", ds))
    assert "[sample]" in text and "[bold]" not in text
    assert ds["handle"]["url"] in text
    assert ds["handle"]["sha256"] in text
    assert str(ds["handle"]["bytes"]) in text
    assert str(shell.data_dir().resolve() / "sample.csv") in text
    assert requests == []


def test_shared_missing_file_instructions_use_surface_neutral_recovery(downloadable_sample):
    from manyruns import shell

    ds, requests = downloadable_sample
    text = "\n".join(shell.where_the_file_should_be("sample", ds))
    for surface_specific in ("rescan", "file or folder path", "ctrl+r", "find data"):
        assert surface_specific not in text.lower()
    assert "choose the sample again" in text.lower()
    assert "sample.csv" in text
    assert ds["handle"]["url"] in text
    assert str(shell.data_dir().resolve() / "sample.csv") in text
    assert requests == []


def test_rich_shell_refusal_preserves_its_rescan_and_local_path_hints(downloadable_sample):
    """Preservation: the rich-shell renderer still explains its own recovery choices."""
    from manyruns import shell

    _, requests = downloadable_sample
    con = CaptureConsole()
    assert not shell._say_where_the_file_should_be(con, "sample")
    assert "pick “rescan”" in con.text
    assert "choose “a file or folder path…”" in con.text
    assert requests == []


@pytest.mark.parametrize(("source", "dataset", "file_source", "expected"), [
    ("arch_soft", "archetypal", False, "arch_soft"),
    ("arch_soft", "swissroll", False, None),
    ("not_a_catalog_row", "swissroll", False, None),
    ("arch_soft", None, True, None),
])
def test_run_preserves_only_verified_named_generator_identity(
    monkeypatch, tmp_path, source, dataset, file_source, expected,
):
    from types import SimpleNamespace

    from manyruns import app, shell

    monkeypatch.delenv("MANYRUNS_DATASET_DIR", raising=False)
    captured = []

    def build(project, args):
        captured.append(project)
        return SimpleNamespace()

    monkeypatch.setattr(app, "_build_session", build)
    monkeypatch.setattr(shell, "_run_stepped", lambda *args: {})
    shell._run(_shell_args(), CaptureConsole(), tmp_path if file_source else None,
               dataset, "synthetic", "embed", SimpleNamespace(source=source), engine="mock")
    assert captured[0]["dataset_name"] == expected


@pytest.mark.parametrize("failure", ["offline", "unicode"])
def test_failed_fetch_stays_in_picker_then_retries_with_the_same_instructions(
    downloadable_sample, monkeypatch, failure,
):
    import io
    from urllib.error import URLError

    from manyruns import shell

    ds, _ = downloadable_sample
    calls = []
    error = (URLError("offline today") if failure == "offline"
             else UnicodeEncodeError("ascii", "é", 0, 1, "URL cannot be encoded"))

    def opener(url, *, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise error
        return io.BytesIO(b"a,b\n1,2\n3,4\n")

    monkeypatch.setattr("urllib.request.urlopen", opener)
    pr = RecordingPrompter([("dataset", "sample"), "fetch", ("dataset", "sample"), "fetch"])
    con = CaptureConsole()
    result = shell._pick_source(pr, con)
    assert result[0] == shell.data_dir().resolve() / "sample.csv"
    assert str(error) in con.text
    assert len(calls) == 2 and len(pr.offered) == 4
    for part in ("sample.csv", ds["handle"]["url"], str(result[0])):
        assert con.text.count(part) >= 2


def test_existing_noncanonical_sample_skips_fetch_confirmation(downloadable_sample):
    from manyruns import shell

    _, requests = downloadable_sample
    target = shell.ensure_drop_folder() / "sample.csv"
    target.write_bytes(b"user,column\n5,6\n")
    pr = RecordingPrompter([("dataset", "sample")])
    assert shell._pick_source(pr, CaptureConsole())[0] == target
    assert len(pr.offered) == 1 and requests == []
    assert target.read_bytes() == b"user,column\n5,6\n"
    assert "download on selection" not in str(pr.offered[0])


def test_listing_never_fetches_or_creates_the_drop_folder(downloadable_sample):
    from manyruns import shell

    _, requests = downloadable_sample
    con = CaptureConsole()
    shell._list_datasets(con)
    assert requests == []
    assert not shell.data_dir().exists()
    assert "sample" in con.text and "not on this machine" in con.text


def test_pbmc3k_missing_menu_label_names_download_size_without_fetching():
    from manyruns import catalog, shell

    assert catalog.dataset_dir().is_dir()
    pr = RecordingPrompter([("back", None)])
    shell._pick_source(pr, CaptureConsole())
    label = next(label for label, value in pr.offered[0][1] if value == ("dataset", "pbmc3k"))
    assert "download on selection · 5.6 MB" in label


def test_fetch_progress_output_is_bounded_for_many_small_chunks(downloadable_sample, monkeypatch):
    import hashlib
    import io

    import yaml

    from manyruns import catalog, shell

    ds, _ = downloadable_sample
    payload = b"a,b\n" + b"1,2\n" * 100
    ds["handle"].update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (catalog.dataset_dir() / "sample.yaml").write_text(yaml.safe_dump(ds))

    class SmallChunks(io.BytesIO):
        def read(self, size=-1):
            return super().read(1)

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: SmallChunks(payload))
    con = CaptureConsole()
    shell._pick_source(ScriptedPrompter([("dataset", "sample"), "fetch"]), con)
    progress = [line for line in con.lines if "downloading sample.csv:" in line]
    assert 2 <= len(progress) <= 11
    assert "(0%)" in progress[0] and "(100%)" in progress[-1]


def test_refusal_renderer_escapes_catalog_text(downloadable_sample, monkeypatch):
    import io

    from rich.console import Console

    from manyruns import shell

    monkeypatch.setattr(shell, "where_the_file_should_be", lambda *a: ["[bold]literal[/bold]"])
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)
    assert not shell._say_where_the_file_should_be(console, "sample")
    assert "[bold]literal[/bold]" in output.getvalue()


def test_shared_instructions_are_empty_for_present_and_generated_data(downloadable_sample):
    from manyruns import shell

    ds, requests = downloadable_sample
    (shell.ensure_drop_folder() / "sample.csv").write_bytes(b"local")
    assert shell.where_the_file_should_be("sample", ds) == []
    ds["handle"] = {"kind": "path", "ref": "synthetic:time-course"}
    assert shell.where_the_file_should_be("sample", ds) == []
    ds["handle"] = {"kind": "manylatents", "ref": "swissroll"}
    assert shell.where_the_file_should_be("sample", ds) == []
    assert requests == []


def test_sample_without_download_url_refuses_with_citation_and_placement(
    downloadable_sample,
):
    import yaml

    from manyruns import catalog, shell

    ds, requests = downloadable_sample
    del ds["handle"]["url"]
    ds["source"] = {"kind": "tutorial", "url": "https://example.org/citation"}
    (catalog.dataset_dir() / "sample.yaml").write_text(yaml.safe_dump(ds))
    con = CaptureConsole()
    pr = RecordingPrompter([("dataset", "sample"), ("back", None)])
    assert shell._pick_source(pr, con) is None
    assert len(pr.offered) == 2 and requests == []
    for part in ("sample.csv", ds["source"]["url"],
                 str(shell.data_dir().resolve() / "sample.csv")):
        assert part in con.text
