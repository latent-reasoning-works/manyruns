"""The command interface: the registry, its `--help` / REPL rendering, and that it stays
consistent with the session's real actions. Dep-free (argparse + the mock REPL)."""
from __future__ import annotations

from manyruns import app, catalog, commands
from manyruns.session import UNOFFERED_GROUPS, Session, available_actions, resolve_step


# ── the registry ─────────────────────────────────────────────────────────────
def test_resolve_canonical_and_aliases_case_space_insensitive():
    assert commands.resolve("phate").name == "phate"
    assert commands.resolve("MIIOFlow").name == "mioflow"          # alias, casing
    assert commands.resolve("MIIOFlow").name == "mioflow"  # alias, spacing
    assert commands.resolve("Granger Causality") is None    # deleted with the step
    assert commands.resolve("?").name == "help"
    assert commands.resolve("EXIT").name == "quit"
    assert commands.resolve("q").name == "quit"
    assert commands.resolve("nope") is None


def test_render_help_lists_every_command_and_alias():
    h = commands.render_help()
    for c in commands.COMMANDS:
        assert c.name in h
        for alias in c.aliases:
            assert alias in h
    assert "steps" in h and "meta" in h


def _help_sections(text: str) -> dict[str, list[str]]:
    """`{heading: [canonical name, …]}`, parsed back out of `render_help`'s two-space headings
    and four-space rows. Reads the RENDERED page, because the page is what a person reads and
    the defect this file's `…_not_offered…` tests cover was invisible in the registry."""
    sections: dict[str, list[str]] = {}
    heading = None
    for line in text.splitlines():
        if line.startswith("    ") and heading is not None:
            sections[heading].append(line.split()[0])
        elif line.startswith("  ") and line.strip():
            heading = line.strip()
            sections[heading] = []
    return sections


def _step_headings(text: str) -> tuple[str, str]:
    """`(offered heading, not-offered heading)` — exactly one of each, or the test fails here
    rather than in a confusing assertion further down."""
    steps = [h for h in _help_sections(text) if h.startswith("steps")]
    unoffered = [h for h in steps if "not offered" in h]
    offered = [h for h in steps if "not offered" not in h]
    assert len(offered) == 1 and len(unoffered) == 1, f"step headings: {steps}"
    return offered[0], unoffered[0]


def test_help_separates_the_steps_the_menu_offers_from_the_ones_it_does_not():
    """The step-vocabulary drift the catalog derivation exists to prevent, in its second form.

    Deriving both halves from the catalog closed the "two hand-kept lists" version of it, then
    `session.UNOFFERED_GROUPS` opened a new one: `--help` and the REPL's `help` render the whole
    catalog, while the session banner's `steps: …` and the `unknown action …; try …` line render
    only the offered moves. Measured on the bundled catalog before this test existed: 23 rows
    under one undifferentiated `steps` heading, 16 in the banner, and nothing on either page
    accounting for `normalize`, `transform`, `filter_cells`, `filter_genes`, `filter_mito`,
    `growth_rate`, `sample_trajectories`.

    The resolution is NOT to drop them from `help` — `session.UNOFFERED_GROUPS`' docstring
    settles that question ("absent from the menu and still typeable"), and the banner points at
    `help` for "the full alias list". It is that the page has to SAY which is which, so the two
    surfaces disagree only where they mean to. Asserted from both sides, so a step cannot hide
    in the unoffered section by quietly falling out of the menu."""
    groups = {name: step.get("group") for name, step in catalog.known_steps().items()}
    offered_heading, unoffered_heading = _step_headings(commands.render_help())
    sections = _help_sections(commands.render_help())

    assert sections[offered_heading] == available_actions()
    assert sections[unoffered_heading] == [n for n, g in groups.items() if g in UNOFFERED_GROUPS]
    assert sections[unoffered_heading], "no bundled step is unoffered — this proves nothing"
    # every step still appears SOMEWHERE: the split is a labelling, never a filter
    assert set(sections[offered_heading]) | set(sections[unoffered_heading]) == set(groups)
    # …and the heading's "typeable" is a claim about behaviour, so it is checked as one
    for name in sections[unoffered_heading]:
        assert "typeable" in unoffered_heading
        assert resolve_step(name) is not None, f"help calls {name!r} typeable; it resolves to None"


def test_the_offered_flag_is_the_session_s_rule_not_a_second_copy_of_it():
    """`Command.offered` reads `session.offered`. Verified by mutation: hardcoding it to `True`
    in `step_commands()` fails this at the `prep`/`probe` rows."""
    for cmd in commands._kind("step"):
        group = catalog.known_steps()[cmd.name].get("group")
        assert cmd.offered is (group not in UNOFFERED_GROUPS), cmd.name
    assert all(c.offered for c in commands._kind("meta")), "meta has no menu to be absent from"


def test_an_install_with_no_unoffered_step_prints_no_heading_for_them(tmp_path, monkeypatch):
    """A heading over zero rows advertises a distinction this install does not have."""
    (tmp_path / "solo.yaml").write_text(
        "name: solo\nsteps:\n  - {name: umap, group: latent, params: {}}\n")
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(tmp_path))

    sections = _help_sections(commands.render_help())
    assert [h for h in sections if h.startswith("steps")] == ["steps  (from the recipe catalog)"]
    assert sections["steps  (from the recipe catalog)"] == ["umap"]


def test_cli_help_accounts_for_the_steps_the_banner_will_not_repeat():
    """The distinction has to survive argparse, not just `render_help` — the epilog is rendered
    by `RawDescriptionHelpFormatter`, which is the only formatter that does not reflow it."""
    help_text = app._build_parser().format_help()
    offered_heading, unoffered_heading = _step_headings(help_text)

    assert help_text.index(offered_heading) < help_text.index(unoffered_heading)
    for name in ("normalize", "filter_mito", "growth_rate"):
        assert help_text.index(name) > help_text.index(unoffered_heading), (
            f"{name!r} is in --help above the heading that explains why the banner omits it")


# ── surfaces: --help and the REPL both render the registry ───────────────────
def test_cli_help_shows_the_alias_list():
    help_text = app._build_parser().format_help()
    for token in ("phate", "mioflow", "miioflow", "separation", "run", "quit", "exit"):
        assert token in help_text, f"{token!r} missing from --help"


def test_repl_help_command_renders_the_registry():
    inputs = iter(["help", "quit"])
    out: list = []
    session = Session(
        project="x", engine="mock", modality="scrna", recipe=app.load_recipe("cflows"),
    )
    app.interactive_session(session, read=lambda _="": next(inputs), write=out.append)
    joined = "\n".join(out)
    assert "miioflow" in joined and "separation" in joined
    assert "PHATE manifold embedding" in joined


# ── the step half is DERIVED from the catalog, not kept in step with it ──────
#
# These two used to import `session._ACTIONS` and assert that the registry covered it — a test
# over two hand-kept lists, which fails AFTER someone forgets rather than making forgetting
# impossible. Both lists are gone; the property is now that adding a recipe adds its steps.
def test_the_registry_s_step_half_is_the_recipe_catalog(tmp_path, monkeypatch):
    """A recipe dropped into `$MANYRUNS_RECIPE_DIR` reaches `--help` and the REPL's `help`
    with no source edit. Verified by mutation: restoring `step_commands()` to the four literal
    `Command(...)` rows it used to be fails this at `["phate", …] != ["umap"]` — which is the
    whole point, since that literal is what a new recipe could not reach."""
    # The registry and the menu PARTED AT THE CUTOVER, so this asserts the relation between
    # them rather than equality. `session.UNOFFERED_GROUPS = ("prep", "probe")` is the whole
    # gap: the registry stays the whole catalog, because `--help` and the REPL's `help`
    # describe every step that EXISTS, while `available_actions()` is the interactive menu and
    # drops seven of them — `normalize`/`transform`/`filter_cells`/`filter_genes`/`filter_mito`
    # (prep: the loader preamble, now declared) and `growth_rate`/`sample_trajectories`
    # (probe). Measured on the bundled catalog 2026-08-17: 23 steps in the registry, 16
    # offered. What has not moved, and is what this test is for, is that BOTH are derived from
    # the recipe catalog rather than hand-kept beside it. That the two are DISTINGUISHED on the
    # page they share is `test_help_separates_the_steps_the_menu_offers_from_the_ones_it_does_not`.
    groups = {name: step.get("group") for name, step in catalog.known_steps().items()}
    registry_steps = {c.name for c in commands._kind("step")}
    unoffered = {n for n, g in groups.items() if g in UNOFFERED_GROUPS}
    assert registry_steps == set(groups)
    assert unoffered, "no bundled step is in an unoffered group — the next line proves nothing"
    assert registry_steps - unoffered == set(available_actions())

    (tmp_path / "solo.yaml").write_text(
        "name: solo\nsteps:\n  - {name: umap, group: latent, params: {}}\n")
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(tmp_path))

    assert [c.name for c in commands._kind("step")] == ["umap"]
    assert commands.resolve("umap").kind == "step"
    # a step with no hand-written blurb is still OFFERED, described by its group — the summary
    # table must never become a filter on what a person can reach
    assert commands.resolve("umap").summary == "latent step"
    # the meta half is this REPL's own verbs and does not move with the catalog
    assert {c.name for c in commands._kind("meta")} == {"run", "summary", "trace", "help", "quit"}


def test_registry_and_session_resolve_an_alias_to_the_same_step():
    """Two surfaces, one answer: what the help list calls a command and what the session
    actually dispatches must be the same step, aliases included."""
    for cmd in commands._kind("step"):
        for token in (cmd.name, *cmd.aliases):
            assert commands.resolve(token).name == cmd.name
            assert resolve_step(token)["name"] == cmd.name
