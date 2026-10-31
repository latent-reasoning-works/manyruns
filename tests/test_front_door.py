"""The front door — bare `manyruns`, and the flags it falls back on.

Four things made this hard to drive, all of them hit in real use:

  * `--data cohort.h5ad` failed with `ambiguous option: could match --dataset, --data-set`,
    which reads like the flag does not exist;
  * `--project` was effectively required, because without it `input()` raised a bare
    EOFError in any non-TTY context (a script, CI, a pipe);
  * the shell never asked how to run, so the bare front door silently used whatever
    `_default_engine()` returned — on a stackless install the mock, which invents numbers
    and reads no data, with nothing on screen saying so;
  * an engine whose dependencies are missing was indistinguishable from one that does not
    exist.
"""
from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def _isolated_drop_folder(tmp_path, monkeypatch):
    """Point the drop folder at a tmp dir. Without this these tests would resolve the real
    `~/.manyruns/data` and read — or create — a developer's actual data."""
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path / "drop"))


def _parse(argv):
    from manyruns import app

    return app._build_parser().parse_args(argv)


# ── the flags ────────────────────────────────────────────────────────────────
def test_data_is_accepted_both_positionally_and_as_a_flag():
    """`--data` has to be declared explicitly: as a mere prefix it is ambiguous against
    `--dataset` and `--data-set`, and argparse rejects it before the command runs."""
    assert _parse(["run", "cohort.h5ad"]).data_folder == "cohort.h5ad"
    assert _parse(["run", "--data", "cohort.h5ad"]).data_arg == "cohort.h5ad"


def test_ask_takes_the_default_instead_of_dying_without_a_tty(monkeypatch):
    """Under pytest stdin is not a TTY — the same condition as a script or a CI job."""
    from manyruns import app

    def explode(*_a, **_k):
        raise AssertionError("prompted with no terminal to prompt")

    monkeypatch.setattr("builtins.input", explode)
    assert app._ask("project name", "cohort") == "cohort"


def test_a_project_name_is_derived_from_the_data_source():
    """Nobody should have to invent a name to look at a file."""
    from manyruns import app

    assert app._project_name_for("cohort.h5ad") == "cohort"
    assert app._project_name_for("swissroll") == "swissroll"
    assert app._project_name_for("/tmp/treated_vs_healthy/") == "treated_vs_healthy"
    assert app._project_name_for("My Cohort v2.h5ad") == "my-cohort-v2"   # slugged, not raw


def test_setup_project_runs_unattended(tmp_path, monkeypatch):
    """End to end: a path and nothing else. This used to raise EOFError before it reached
    any work, which is what made the CLI feel unusable from a script."""
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: pytest.fail("prompted"))
    (tmp_path / "cohort.csv").write_text("a,b\n1,2\n")

    proj = app._setup_project(_parse(["run", "cohort.csv", "--engine", "mock"]), None)
    assert proj["project"] == "cohort"


def test_an_empty_drop_folder_says_where_to_put_the_data(tmp_path, monkeypatch):
    """`manyruns run` with no arguments should be a reasonable thing to type. With nothing
    dropped yet the answer is an absolute path to drag files into — not an error about a
    missing argument, and not a traceback."""
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    drop = tmp_path / "drop"
    with pytest.raises(SystemExit) as e:
        app._setup_project(_parse(["run", "--engine", "mock"]), None)

    msg = str(e.value)
    assert str(drop.resolve()) in msg          # the actual folder, absolute
    assert ".h5ad" in msg                      # what it accepts
    assert drop.is_dir()                       # and it was created for them


def test_a_single_dropped_file_is_used_without_being_named(tmp_path, monkeypatch, capsys):
    """One file in the folder → just use it. This is the friction being removed: having to
    pass an explicit path on every run."""
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    drop = tmp_path / "drop"
    drop.mkdir(parents=True)
    (drop / "cohort.csv").write_text("a,b\n1,2\n")

    proj = app._setup_project(_parse(["run", "--engine", "mock"]), None)
    assert proj["project"] == "cohort"
    assert "using cohort.csv" in capsys.readouterr().out


def test_several_dropped_files_are_listed_rather_than_guessed_between(tmp_path, monkeypatch):
    """Ambiguity is reported with the choices, never resolved by picking one silently."""
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    drop = tmp_path / "drop"
    drop.mkdir(parents=True)
    (drop / "a.csv").write_text("x\n1\n")
    (drop / "b.csv").write_text("x\n1\n")

    with pytest.raises(SystemExit) as e:
        app._setup_project(_parse(["run", "--engine", "mock"]), None)
    msg = str(e.value)
    assert "a.csv" in msg and "b.csv" in msg


# ── the engine vocabulary, and the shell's picker ────────────────────────────
def test_the_engine_list_has_one_home():
    """The CLI's `--engine` choices and the shell's picker both read `app.ENGINES`, so the
    two surfaces cannot drift — the failure `vocab.py` exists to prevent, in miniature."""
    from manyruns import app

    action = next(a for a in _parse(["run"])._get_kwargs() if a[0] == "engine")  # noqa: SLF001
    assert action is not None
    # TWO, not three: the learner's engine was removed from this list — manyruns holds no track to the
    # learner (the environment-contract spec, §3.5), so there is no backend to offer.
    assert set(app.ENGINE_NAMES) == {"mock", "manylatents"}
    # `_inproc` is a step loop, not a backend: it must never reach this list, or `--engine`
    # and the picker start offering the suite's test substrate to a researcher.
    from manyruns.vocab import INPROC
    assert INPROC not in app.ENGINE_NAMES
    from manyruns.serving import LocalServer
    assert INPROC not in LocalServer.SERVES
    assert app.engine_available("mock") is True          # no import target — always runnable


def test_the_picker_is_skipped_when_engine_is_given():
    """The flag path is unchanged: `--engine` wins and nothing is asked."""
    from manyruns import shell

    class Never:
        def select(self, *a, **k):
            raise AssertionError("prompted despite an explicit --engine")

    assert shell._pick_engine(Never(), None, _parse(["shell", "--engine", "mock"])) == "mock"


class _Console:
    """The narrowest thing `_pick_engine` needs. Its no-backend branch prints, so a test that
    passes `None` here asserts nothing about the picker — it raises on the first print."""

    def __init__(self):
        self.printed: list = []

    def print(self, *a):
        self.printed.append(" ".join(str(x) for x in a))


class _Spy:
    def __init__(self):
        self.seen: dict = {}

    def select(self, message, choices, default=None):
        self.seen["labels"] = [label for label, _ in choices]
        self.seen["default"] = default
        return "mock"


def test_the_picker_never_offers_a_development_backend(monkeypatch):
    """`mock` is a test fixture, not a fidelity a scientist chooses between.

    It invents a g-vector and never opens the file, and the panel it produces is
    indistinguishable at a glance from a measured one — so it is omitted from the menu
    entirely, while backends that are merely *uninstalled* are still shown and marked (a user
    who wants a real run needs to know it is installable, not that it is absent).

    `engine_available` is FORCED rather than read from the machine, and that is the fix for a
    real defect in the test: `_pick_engine` short-circuits when nothing is available — it prints
    "no compute backend installed" and returns before `prompter.select` is ever called — so on
    an install without the private stack this reached `spy.seen["labels"]` with nothing in it,
    and with `console=None` it died on `NoneType.print` first. The menu was only ever built on a
    developer's machine that happened to have manylatents. Forcing one available backend is what
    makes the assertion about the MENU instead of about the install.
    """
    from manyruns import app, shell

    monkeypatch.setattr(app, "engine_available", lambda name: name == "manylatents")
    spy = _Spy()
    shell._pick_engine(spy, _Console(), _parse(["shell"]))
    labels = " ".join(spy.seen["labels"])

    assert "invents numbers" not in labels
    assert "mock" not in labels
    offered = len(shell_engine_names()) - len(app.DEV_ENGINES)
    assert len(spy.seen["labels"]) == offered + 1                       # + "back"


def test_the_flag_still_reaches_the_development_backend():
    """Omitted from the menu, not removed from the product: CLAUDE.md requires the mock so the
    flow and the suite run with no private stack, and every test that passes `--engine mock`
    depends on it. `--engine` wins over the picker, so that path is untouched."""
    from manyruns import shell

    assert shell._pick_engine(_Spy(), None, _parse(["shell", "--engine", "mock"])) == "mock"


def test_the_mock_is_never_preferred_when_something_real_is_installed(monkeypatch):
    """The property that matters, asserted against a controlled environment rather than
    against whatever happens to be installed. The first version of this test asserted
    `default != "mock"` unconditionally and failed in CI — correctly, because there the mock
    is the ONLY runnable backend and preferring it is the honest answer.

    `manylatents` rather than `real` since the compute was fixed in: `real` joined `mock` in
    `DEV_ENGINES`, because it implements two latent steps and several bundled recipes report
    "no in-process implementation" on it. It was the answer to "what if the engine is not installed",
    and there is no such install now."""
    from manyruns import app, shell

    monkeypatch.setattr(app, "engine_available", lambda n: n in ("manylatents", "mock"))
    spy = _Spy()
    shell._pick_engine(spy, None, _parse(["shell"]))
    assert spy.seen["default"] == "manylatents"


def test_nothing_installed_says_what_to_install_rather_than_inventing_numbers(monkeypatch):
    """The base install — no extras — used to answer a real cohort with fabricated geometry.

    `_default_engine` returned `mock` as its last resort, so a user who installed manyruns and
    pointed it at their data got a full panel of invented numbers with nothing on screen
    saying so. There is now no fallback: the picker asks nothing (a menu whose every option
    reads "(not installed)" is a question with no right answer) and says what to install."""
    from manyruns import app, shell

    monkeypatch.setattr(app, "engine_available", lambda n: n == "mock")
    assert app._default_engine() is None

    printed = []

    class _Console:
        def print(self, *a):
            printed.extend(str(x) for x in a)

    spy = _Spy()
    assert shell._pick_engine(spy, _Console(), _parse(["shell"])) is None
    assert "labels" not in spy.seen                       # never asked
    assert "no compute backend installed" in " ".join(printed)
    # It says REINSTALL, not "install the extra". The compute is a hard dependency now, so a
    # missing backend is a broken install rather than a menu item nobody picked — and the
    # advice a user is given has to match which of those two it is.
    assert "reinstall" in " ".join(printed).lower()


def shell_engine_names():
    from manyruns import app

    return app.ENGINE_NAMES


def test_choosing_an_uninstalled_engine_says_so_rather_than_silently_backing_out(monkeypatch):
    """The sentinel matters: returning None for 'not installed' is indistinguishable from
    'the user backed out', so the message would never be reached.

    One backend is forced available so the picker gets PAST its no-backend short-circuit and the
    `__missing__` branch is reachable at all. Without it the assertion read the wrong message —
    "no compute backend installed" — which is a different sentence about a different situation,
    and the branch this test exists to cover never ran.
    """
    from manyruns import app, shell

    monkeypatch.setattr(app, "engine_available", lambda name: name == "manylatents")
    console = _Console()

    class PicksMissing:
        def select(self, *a, **k):
            return "__missing__mock"

    assert shell._pick_engine(PicksMissing(), console, _parse(["shell"])) is None
    assert "mock isn't installed here" in " ".join(console.printed)


def test_mock_is_never_a_silent_default():
    """On an install without the private stack — the shipped configuration — the default used
    to be the mock, so pointing manyruns at a real cohort produced invented numbers with
    nothing saying so.

    THE FALLBACK IS GONE, not redirected. `real` used to catch this case because manylatents was
    optional; it is a hard dependency now, so every install has it, and falling back to an engine
    that cannot run several bundled recipes would trade a loud absence for a quiet incapacity.
    A dev engine is never the answer and neither is a crippled one — `None` is, and the caller
    says what is wrong."""
    from manyruns import app

    assert app._default_engine() != "mock"

    monkey = app.engine_available
    try:
        # the shipped shape: manylatents present, private stack absent
        app.engine_available = lambda name: name in ("manylatents", "_inproc", "mock")
        assert app._default_engine() == "manylatents"

        # and a broken install — no manylatents — refuses rather than limping on `real`
        app.engine_available = lambda name: name in ("_inproc", "mock")
        assert app._default_engine() is None
    finally:
        app.engine_available = monkey


# ── `run` with nothing said, at a terminal, is a request for the console ──────
def test_bare_run_at_a_terminal_opens_the_console(monkeypatch):
    """`run` is the verb people reach for. Routing it straight to a one-shot meant the
    console — where you choose the data, the question and how to run it — was reachable
    only by OMITTING the verb, which nobody guesses."""
    from manyruns import app

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    opened = []
    monkeypatch.setattr("manyruns.shell.run_shell", lambda a, **k: opened.append(a) or 0)

    assert app.cmd_run(_parse(["run"]), None) == 0
    assert opened, "bare `run` at a terminal did not open the console"


@pytest.mark.parametrize("argv", [
    ["run", "cohort.h5ad"],
    ["run", "--dataset", "swissroll"],
    ["run", "--engine", "manylatents"],
    ["run", "--recipe", "embed"],
    ["run", "--project", "p"],
])
def test_anything_that_says_what_to_do_keeps_the_one_shot(argv, monkeypatch):
    from manyruns import app

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert app._wants_the_console(_parse(argv)) is False


def test_a_pipe_or_script_never_gets_the_console(monkeypatch):
    """Automation must not be handed an interactive loop. Under pytest stdin is already
    not a TTY — the same condition as a pipe, a cron job, or CI."""
    from manyruns import app

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert app._wants_the_console(_parse(["run"])) is False


def test_a_mock_result_says_so_in_its_own_record():
    """The panel a mock run produces is indistinguishable at a glance from a measured one, so
    the record carries what it is — in `caveats`, which rides into `index.jsonl` (as of the
    store gaining that column; this docstring asserted it before it was true) and into the
    piped rendering. Reaching this path takes an explicit `--engine mock`; the caveat is what
    stops the result outliving that intent in a summary or a screenshot read a month later.

    Pinned on the RECIPE path specifically: `_run_mock` early-returns `_run_recipe` when a
    recipe is supplied, which is the branch the product actually takes, and the first version
    of this marked only the adaptive branch — so a real product call came back unmarked."""
    from manyruns import narrate
    from manyruns.serving import LocalServer

    recipe = {"name": "r", "steps": [{"name": "phate", "group": "latent", "params": {}}]}
    res = LocalServer(engine="mock").predict({"recipe": recipe})

    assert res["engine"] == "mock"
    assert res["caveats"] == ["engine=mock — every number here is invented; no data was read"]
    assert "every number here is invented" in narrate.run_panel(res)


def test_the_class_still_has_the_methods_the_mock_path_needs():
    """A guard with a real history: lifting a helper to module level between two methods left
    `_run_mock`, `_run_recipe`, `_as_vector` and `_next_step` nested inside it at the same
    indent, so they silently left the class. It still imported — the file was valid Python and
    merely wrong — and only an attribute check caught it."""
    from manyruns.serving import LocalServer

    for method in ("_run_mock", "_run_recipe", "_as_vector", "_next_step", "predict"):
        assert hasattr(LocalServer, method), method


# ── the recipe menu is agnostic to the catalog ───────────────────────────────
def _obs(shape):
    from manyruns.narrate import Observation

    return Observation(source="x", n_obs=100, n_vars=10, shape=shape, modality="scrna")


def test_the_menu_never_hides_a_recipe_the_data_can_legally_run():
    """There were two answers to "what can this data run" and they disagreed.

    `narrate.offer` was three hardcoded shape branches naming cflows/contrast/embed as
    literals; `vocab.unmet` computes legality from the catalog. Measured across the six shapes,
    the menu offered fewer recipes than the calculus allowed on FIVE of them — `cflows` was
    hidden everywhere except `time-course`."""
    from manyruns import narrate
    from manyruns.catalog import discover_recipes, load_recipe
    from manyruns.vocab import unmet

    # A `dev.` fixture is legal and is still not offered, which is the one deliberate exception
    # to "never hide a runnable recipe" — it is a surface for driving the app against unbuilt
    # steps, and a scientist learns nothing about their data from being offered it. Excluded
    # from `legal` here by the same prefix `offer` filters on, so this test keeps asserting the
    # real invariant: no ANALYSIS the data can run is ever hidden.
    for shape in ("time-course", "case-control", "single", "manifold", "clusters", "unknown"):
        legal = [n for n in discover_recipes()
                 if not unmet(load_recipe(n), shape)
                 and not str(load_recipe(n).get("id") or "").startswith("dev.")]
        offered = {o["recipe"] for o in narrate.offer(_obs(shape), available=legal)}
        assert offered == set(legal), f"{shape}: hid {set(legal) - offered}"


def test_a_recipe_carries_its_own_question_so_the_menu_needs_no_table():
    """Adding a recipe to `configs/recipe/` must add its own menu entry. The label comes from
    the recipe's `question:`, so `narrate` holds no list of recipe names at all."""
    import inspect

    from manyruns import narrate
    from manyruns.catalog import discover_recipes, load_recipe

    # The docstring names the recipes it USED to hardcode, as the record of why this changed.
    # Checking it would fail on the explanation rather than on the behaviour, so only the code
    # is examined — which is what "no recipe name drives this function" actually means.
    src = inspect.getsource(narrate.offer)
    body = src.replace(narrate.offer.__doc__ or "", "")

    for name in discover_recipes():
        assert name not in body, f"{name} is hardcoded in offer()"
        assert load_recipe(name).get("question"), f"{name} declares no question"


def test_suitability_orders_the_menu_and_never_prunes_it():
    """`suits:` marks the ★ recommendation only. Legality is structural, suitability is advice
    — the same split as `unmet` vs `noncanonical`. Collapsing them is how a menu comes to hide
    a runnable analysis."""
    from manyruns import narrate

    offered = narrate.offer(_obs("time-course"), available=["cflows", "embed"])

    assert offered[0]["recipe"] == "cflows" and offered[0]["recommended"]
    assert offered[1]["recipe"] == "embed" and not offered[1]["recommended"]


def test_the_top_menu_offers_an_agnostic_entry_not_a_bare_recipe_name():
    """The complaint that started this: a scientific question and the raw label "Run the
    cflows recipe" sat side by side in one list, and the list grew by a row per recipe."""
    from manyruns import shell

    spy = _Spy()
    shell._pick_recipe(spy, None, _obs("manifold"), ["cflows", "embed"], allow_llm=False)
    labels = spy.seen["labels"]

    assert not [x for x in labels if x.startswith("Run the ")], labels
    assert any("Run a specific recipe" in x for x in labels), labels
    # one question + the agnostic entry + free text + back — NOT one row per recipe
    assert len(labels) == 4


# ── the drop folder follows you, or the file is gone from the front door ─────
def test_every_drop_folder_is_read_not_just_the_first_one(tmp_path, monkeypatch):
    """`data/pbmc3k_raw.h5ad` (5.6 MB) vanished from the roster the moment the app ran from a
    git worktree — `/data/` is gitignored, so the worktree had none, and `data_dir()` prefers
    `./data` WHEN IT EXISTS and otherwise falls back to `~/.manyruns/data`. The fallback won,
    the roster read an empty folder, and a real dataset was simply not on the front door.

    That is the failure `data_dir`'s own docstring says the home fallback prevents — "a folder
    resolved from the current directory silently becomes a *different* folder every time you
    `cd`" — reintroduced by the `./data`-wins branch in front of it. The two are additive now.
    """
    from manyruns import shell

    home, here = tmp_path / "home" / ".manyruns" / "data", tmp_path / "here" / "data"
    home.mkdir(parents=True)
    here.mkdir(parents=True)
    (home / "cohort.h5ad").write_bytes(b"x")
    (here / "checkout.h5ad").write_bytes(b"x")
    monkeypatch.delenv("MANYRUNS_DATA_DIR", raising=False)
    monkeypatch.setattr(shell.Path, "home", staticmethod(lambda: tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "here")

    assert {p.name for p in shell.drop_folder_entries()} == {"cohort.h5ad", "checkout.h5ad"}


def test_an_explicit_data_dir_means_that_folder_alone(tmp_path, monkeypatch):
    """`$MANYRUNS_DATA_DIR` is how a test points at a tmp dir and how a user pins a cohort
    directory. Adding the home folder on top would let a suite run pick up whatever is sitting
    in the developer's `~/.manyruns/data` — isolation lost, and lost intermittently."""
    from manyruns import shell

    home, pinned = tmp_path / "home" / ".manyruns" / "data", tmp_path / "pinned"
    home.mkdir(parents=True)
    pinned.mkdir()
    (home / "stray.h5ad").write_bytes(b"x")
    (pinned / "wanted.h5ad").write_bytes(b"x")
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(pinned))
    monkeypatch.setattr(shell.Path, "home", staticmethod(lambda: tmp_path / "home"))

    assert [p.name for p in shell.drop_folder_entries()] == ["wanted.h5ad"]


def test_a_name_in_two_folders_is_one_row(tmp_path, monkeypatch):
    """Two identical rows that open different files is worse than either.

    `data_dir()` wins, because it is where a FETCH lands — if the folder you download into did
    not win the clash, you could fetch `cohort.h5ad` and be shown a different file of that
    name. Here `./data` exists, so that is `data_dir()`, so the checkout's copy shadows home."""
    from manyruns import shell

    home, here = tmp_path / "home" / ".manyruns" / "data", tmp_path / "here" / "data"
    home.mkdir(parents=True)
    here.mkdir(parents=True)
    (home / "same.h5ad").write_bytes(b"home")
    (here / "same.h5ad").write_bytes(b"checkout")
    monkeypatch.delenv("MANYRUNS_DATA_DIR", raising=False)
    monkeypatch.setattr(shell.Path, "home", staticmethod(lambda: tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "here")

    found = shell.drop_folder_entries()
    assert len(found) == 1 and found[0].read_bytes() == b"checkout"
