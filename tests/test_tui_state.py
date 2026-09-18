"""The app-facing state seam (`manyruns.tui.state`) — component 1 of the TUI rewrite.

Every test here runs WITHOUT a terminal, which is the property the "no Textual import" rule
buys. Nothing below imports `textual`, and one test asserts that importing the seam does not
pull it in either.

Several of these guard `narrate` and `vocab` rather than `state`. They live here because this
component added those functions; `test_every_fact_the_calculus_names_has_a_phrase_and_no_others`
and its two siblings would sit more naturally in `tests/test_narrate.py`, and moving them there
is a tidy-up, not a change.
"""
from __future__ import annotations

import json
import sys

import pytest

from manyruns import narrate, vocab
from manyruns.narrate import Observation
from manyruns.tui import state


def _obs(shape: str = "single", **kw) -> Observation:
    return Observation(shape=shape, modality=kw.pop("modality", "scrna"), **kw)


def test_entry_identity_is_captured_before_source_changes(tmp_path, monkeypatch):
    from dataclasses import replace
    from pathlib import Path

    target = tmp_path / 'target.csv'
    target.write_text('a,b\n1,2\n')
    path = tmp_path / 'source.csv'
    path.symlink_to(target)
    entry = state.DataEntry('source.csv', 'file', _obs(), path=path)
    expected = ('file', str(target.resolve()))
    path.unlink()
    path.symlink_to(path)

    def forbidden(*args, **kwargs):
        pytest.fail('reading a captured row identity touched the filesystem')

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'resolve', forbidden)
        patch.setattr(Path, 'stat', forbidden)
        assert entry.identity == expected
        assert replace(entry, missing=True).identity == expected


def test_local_entry_identity_uses_observed_target(tmp_path, monkeypatch):
    target = tmp_path / 'target.csv'
    target.write_text('a,b\n1,2\n')
    path = tmp_path / 'source.csv'
    path.symlink_to(target)
    expected = ('file', str(target.resolve()))
    data_entry = state.DataEntry

    def invalidated_entry(*args, **kwargs):
        # The source can change after the final signature check but before row assembly.
        path.unlink()
        path.symlink_to(path)
        return data_entry(*args, **kwargs)

    monkeypatch.setattr(state, 'DataEntry', invalidated_entry)
    entry = state.local_entry(path)
    assert entry is not None
    assert entry.identity == expected


def test_old_inspection_cache_does_not_make_unreadable_bytes_selectable(tmp_path):
    from manyruns import inspected
    from manyruns.tui import dropwatch

    path = tmp_path / 'empty.h5ad'
    path.touch()
    row = inspected.as_row('scrna', _obs())
    row.update(observed_signature=list(dropwatch.fingerprint(path)), observed_source=str(path))
    inspected.put(path, row)
    with pytest.raises(ValueError, match='Could not read.*empty.h5ad'):
        state.local_entry(path)


def test_unknown_h5ad_encoding_is_reported_as_a_source_read_error(tmp_path):
    import anndata
    import h5py
    import numpy as np

    path = tmp_path / 'unknown-encoding.h5ad'
    anndata.AnnData(np.ones((2, 2))).write_h5ad(path)
    with h5py.File(path, 'r+') as handle:
        handle['X'].attrs['encoding-type'] = 'unknown-matrix-encoding'
    with pytest.raises(ValueError, match='Could not read.*IORegistryError'):
        state.local_entry(path)


def test_folder_inspection_never_executes_a_contained_script(tmp_path):
    folder = tmp_path / 'source'
    folder.mkdir()
    assert folder.is_dir()
    marker = tmp_path / 'executed'
    (folder / 'a.py').write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
        'import numpy as np\nX = np.ones((2, 2))\n')
    (folder / 'z.csv').write_text('a,b\n1,2\n')
    assert state.local_entry(folder) is not None
    assert not marker.exists()


@pytest.mark.parametrize('suffix', ['.csv', '.tsv', '.txt', '.mtx', '.npy', '.h5'])
def test_source_inspection_uses_headers_and_bounded_table_previews(tmp_path, monkeypatch, suffix):
    import h5py
    import numpy as np
    from manyruns.pipeline import loading
    from manyruns.tui import samplefetch

    path = tmp_path / ('source' + suffix)
    if suffix in {'.csv', '.tsv', '.txt'}:
        sep = ',' if suffix == '.csv' else '\t'
        path.write_text(sep.join(['a', 'b']) + '\n' + (sep.join(['1', '2']) + '\n') * 10000)
        import pandas as pd
        read_csv = pd.read_csv
        previews = []
        def bounded_csv(source, *args, **kwargs):
            assert hasattr(source, 'getvalue')
            previews.append(source.getvalue())
            assert len(source.getvalue().splitlines()) <= 6
            return read_csv(source, *args, **kwargs)
        monkeypatch.setattr(pd, 'read_csv', bounded_csv)
    elif suffix == '.mtx':
        # A correct header with an invalid payload: deeper corruption waits for a run.
        path.write_text('%%MatrixMarket matrix coordinate real general\n2000 20000 1\ninvalid\n')
        import scipy.io
        monkeypatch.setattr(scipy.io, 'mmread', lambda *a, **k: pytest.fail('read mtx payload'))
    elif suffix == '.npy':
        np.save(path, np.ones((4, 3)))
        load = np.load
        def mapped(*args, **kwargs):
            assert kwargs.get('mmap_mode') == 'r' and kwargs.get('allow_pickle') is False
            return load(*args, **kwargs)
        monkeypatch.setattr(np, 'load', mapped)
    else:
        with h5py.File(path, 'w') as handle:
            matrix = handle.create_group('matrix')
            matrix['shape'] = [3, 4]
            for key in ('data', 'indices', 'indptr', 'barcodes'):
                matrix.create_dataset(key, shape=(1,), dtype='i4')
        getitem = h5py.Dataset.__getitem__
        def shape_only(node, *args, **kwargs):
            assert node.name == '/matrix/shape'
            return getitem(node, *args, **kwargs)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', shape_only)
    monkeypatch.setattr(loading, 'load_array', lambda *a, **k: pytest.fail('execution loader called'))
    entry = state.local_entry(path)
    assert entry is not None and not entry.pending and not entry.refusal
    assert samplefetch._resolve(entry).path == path
    if suffix in {'.csv', '.tsv', '.txt'}:
        assert previews and entry.obs.n_obs is None  # no invented count from a preview
    else:
        assert entry.size == ('2,000 × 20,000' if suffix == '.mtx' else '4 × 3')
        assert entry.obs.provides() == (frozenset({'genes'}) if suffix == '.h5' else frozenset())


@pytest.mark.parametrize('layout', ['tenx', 'samples'])
def test_matrix_folders_probe_headers_without_loading_or_concatenating(tmp_path, monkeypatch, layout):
    from manyruns.pipeline import loading

    folder = tmp_path / 'cohort'
    folder.mkdir()
    assert folder.is_dir()
    samples = [folder] if layout == 'tenx' else [folder / 'T0', folder / 'T1']
    for sample in samples:
        sample.mkdir(exist_ok=True)
        assert sample.is_dir()
        (sample / 'matrix.mtx').write_text(
            '%%MatrixMarket matrix coordinate real general\n20000 2000 1\ninvalid\n')
        (sample / 'features.tsv').write_text('gene1\n')
        (sample / 'barcodes.tsv').write_text('cell1\n')
    monkeypatch.setattr(loading, 'load_array', lambda *a, **k: pytest.fail('execution loader called'))
    entry = state.local_entry(folder)
    assert entry is not None and not entry.refusal
    assert entry.obs.n_obs == (2000 if layout == 'tenx' else 4000)
    if layout == 'samples':
        assert entry.shape == 'time-course' and entry.obs.n_timepoints == 2
        assert entry.obs.n_vars is None  # the width after an outer join is not in the headers
    (samples[-1] / 'barcodes.tsv').unlink()
    with pytest.raises(ValueError, match='missing.*barcodes'):
        state.local_entry(folder)


def test_header_probe_preserves_vector_npy_sources(tmp_path, monkeypatch):
    """The execution path treats a vector as one feature; the roster must still accept it."""
    import numpy as np
    from manyruns.pipeline import loading

    path = tmp_path / 'vector.npy'
    np.save(path, np.arange(4))
    monkeypatch.setattr(loading, 'load_array', lambda *a, **k: pytest.fail('execution loader called'))
    entry = state.local_entry(path)
    assert entry is not None and entry.size == '4 × 1'
    assert not entry.obs.provides()


@pytest.mark.parametrize('layout', ['wide', 'multiline'])
def test_table_preview_preserves_complete_records(tmp_path, layout):
    """Preservation: a preview must not mistake its own cutoff for a corrupt source."""
    path = tmp_path / 'table.csv'
    if layout == 'wide':
        path.write_text(','.join(f'feature{i}' for i in range(20000)) + '\n'
                        + ','.join('1' for _ in range(20000)) + '\n')
    else:
        path.write_text('value,note\n1,"' + '\n'.join(['line'] * 8) + '"\n2,short\n')
    entry = state.local_entry(path)
    assert entry is not None and not entry.refusal


@pytest.mark.parametrize('column,values', [
    ('timepoint', [0, 1, 2]), ('condition', ['a', 'b', 'a']), ('cell_type', ['a', 'b', 'a']),
])
@pytest.mark.parametrize('encoding', ['modern', 'legacy', 'legacy-categorical'])
@pytest.mark.parametrize('cached_metadata', [False, True])
def test_header_inspection_preserves_h5ad_narration_and_cached_qc(
        tmp_path, monkeypatch, column, values, encoding, cached_metadata):
    from dataclasses import replace

    import anndata
    import h5py
    import numpy as np
    import pandas as pd
    from manyruns import inspected
    from manyruns.pipeline import loading
    from manyruns.tui import dropwatch

    path = tmp_path / 'cells.h5ad'
    data = anndata.AnnData(np.ones((3, 2)))
    data.obs[column] = values
    if encoding == 'modern':
        data.layers['counts'] = data.X.copy()
        data.obsm['embedding'] = data.X.copy()
        data.write_h5ad(path)
    else:
        with h5py.File(path, 'w') as handle:
            matrix = handle.create_group('X')
            matrix.attrs['h5sparse_format'] = 'csr'
            matrix.attrs['h5sparse_shape'] = (3, 2)
            matrix['data'] = [1.]
            matrix['indices'] = [0]
            matrix['indptr'] = [0, 1, 1, 1]
            if encoding == 'legacy-categorical':
                category = pd.Categorical(values)
                encoded = category.codes
                levels = np.asarray(category.categories)
                if levels.dtype.kind == 'O':
                    levels = levels.astype('S')
                handle[f'uns/{column}_categories'] = levels
            else:
                encoded = np.asarray(values)
                if encoded.dtype.kind == 'U':
                    encoded = encoded.astype('S')
            handle['obs'] = np.array(list(zip([b'c1', b'c2', b'c3'], encoded)),
                                     dtype=[('index', 'S2'), (column, encoded.dtype)])
            handle['var'] = np.array([(b'g1',), (b'g2',)], dtype=[('index', 'S2')])
    with h5py.File(path, 'r+') as handle:
        handle['uns/unrelated_matrix'] = np.ones((3, 3))
    expected = narrate.read_data(path, 'scrna', source=path.name)
    assert expected.columns == [[column, 'categorical', len(set(values))]]
    measured = {'qc_n_cells': 3, 'qc_median_genes_per_cell': 2}
    cached = {'qc': measured}
    if cached_metadata:
        # The merged header reader wrote a complete cache schema with columns=None.
        cached.update(inspected.as_row('scrna', replace(expected, columns=None)))
        cached.update(observed_signature=list(dropwatch.fingerprint(path)),
                      observed_source=str(path), source_readable=True)
    inspected.put(path, cached)

    getitem = h5py.Dataset.__getitem__

    def no_matrix_payload(node, *args, **kwargs):
        assert node.name != '/X'
        assert not node.name.startswith(('/X/', '/layers/', '/obsm/', '/uns/unrelated_matrix'))
        return getitem(node, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, '__getitem__', no_matrix_payload)
    monkeypatch.setattr(loading, 'load_array', lambda *a, **k: pytest.fail('execution loader called'))
    monkeypatch.setattr(anndata, 'read_h5ad', lambda *a, **k: pytest.fail('read whole AnnData'))
    entry = state.local_entry(path)
    assert entry.obs == expected
    assert entry.qc == measured
    monkeypatch.setattr(state, '_h5ad_observation', lambda *a, **k: pytest.fail('ignored cache'))
    cached_entry = state.local_entry(path)
    assert cached_entry.obs == expected
    assert cached_entry.qc == measured


@pytest.mark.parametrize('error', [TypeError, AttributeError, KeyError, IndexError, EOFError])
@pytest.mark.parametrize('source_kind', ['local', 'catalog'])
def test_source_read_errors_stay_inside_the_row_boundary(tmp_path, monkeypatch, error, source_kind):
    import yaml
    from manyruns import shell

    path = tmp_path / 'source.csv'
    path.write_text('a,b\n1,2\n')

    def broken(*args, **kwargs):
        raise error('malformed source metadata')

    if source_kind == 'catalog':
        configs = tmp_path / 'catalog'
        configs.mkdir()
        assert configs.is_dir()
        (configs / 'source.yaml').write_text(yaml.safe_dump({
            'name': 'source', 'shape': 'single', 'modality': 'bulk',
            'handle': {'kind': 'path', 'ref': str(path)},
        }))
        monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
        monkeypatch.setattr(shell, '_resolve_dataset', broken)
        def resolve():
            return state.catalog_entry('source')
    else:
        monkeypatch.setattr(shell, '_resolve_path_source', broken)
        def resolve():
            return state.local_entry(path)
    with pytest.raises(ValueError, match=f'Could not read.*{error.__name__}.*malformed source metadata'):
        resolve()


# ── the seam's defining property ─────────────────────────────────────────────
def test_the_state_seam_pulls_in_no_terminal_machinery():
    """§4 of the spec: `state.py` has no Textual import, and that is what keeps the app
    testable without a terminal AND keeps `manyruns` startable where textual is absent.

    Asserted on the SOURCE as well as on `sys.modules`, because a module already imported by
    another test would make the `sys.modules` half pass vacuously."""
    import inspect
    import subprocess

    src = inspect.getsource(state)
    assert "import textual" not in src and "from textual" not in src

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import manyruns.tui.state; "
         "print('textual' in sys.modules, 'rich' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False False", out.stdout


# ── the phrasing tables are views of the vocabulary, not second copies ───────
def test_every_fact_the_calculus_names_has_a_phrase_and_no_others():
    """The drift guard on `narrate.MISSING_FACT`.

    `vocab`'s own docstring says a vocabulary declared twice drifts silently, and this repo has
    paid for it (`kind` vs `group`, the forked `.obs` time words). The phrase table is allowed
    to exist because its keys are DERIVED-checked against `vocab.facts()` — add a fact to
    `STEP_NEEDS` with no phrase and this fails, write a phrase for a fact that does not exist
    and this fails too. Both directions, or it is not a guard."""
    assert set(narrate.MISSING_FACT) == vocab.facts()

    # and the consequence, not just the shape: every phrase must actually PHRASE the fact, not
    # restate it. `needs <fact>` is the fallback `blocked` uses for a word nobody wrote a
    # sentence for, so a table entry equal to the fallback would be a phrase that is not one.
    for fact, phrase in narrate.MISSING_FACT.items():
        assert phrase.startswith("needs "), f"{fact!r}: a hollow row's reason starts 'needs'"
        assert phrase != f"needs {fact}", f"{fact!r} restates the machine word"


def test_every_fact_a_narrowing_can_clear_has_its_own_phrase_and_no_others():
    """The same drift guard for the SECOND reason a fact can be missing, and the key set is
    derived too: exactly the facts a step produces (`vocab.step_facts`), which is exactly what
    `have &= keep` can take away. A frame fact survives every narrowing, so writing a
    cleared-phrase for `time` or `conditions` would be a sentence no code path can reach."""
    assert set(narrate.CLEARED_FACT) == vocab.step_facts()
    assert vocab.step_facts() & vocab.frame_facts() == frozenset()

    for fact, phrase in narrate.CLEARED_FACT.items():
        assert phrase.startswith("needs "), f"{fact!r}: a hollow row's reason starts 'needs'"
        assert phrase != narrate.MISSING_FACT[fact], (
            f"{fact!r}: the two reasons are the same sentence, so the distinction is invisible"
        )


def test_a_refusal_caused_by_a_filter_does_not_claim_that_nothing_embeds():
    """THE DEFECT THIS PAIR OF TABLES EXISTS FOR. `vocab.unmet` returns `embedding` for two
    opposite recipes now: one that never lays the cells out, and one that lays them out and
    then filters the cells they were fitted over. `MISSING_FACT`'s sentence is a claim about
    the STEP LIST — "nothing here lays the cells out" — and it is false about the second, whose
    first step is phate. Every surface that asks "why can't I run this" shows that one
    sentence (`tui/state.py:314` → `LedgerRow.blocked`, `narrate.offer` via `blocked`), and the
    fix it implies is already done, so the user adds a second embedder and is refused again.

    Not hypothetical after Task 8: it gives seven bundled recipes a prep block."""
    filtered = {"name": "tidytraj", "steps": [
        {"name": "phate", "group": "latent", "params": {}},
        # `filter_cells`, not `filter_genes`: this test is about the reason a narrowing
        # gives, and `filter_genes` declares `STEP_NEEDS = ("genes",)`, so on a synthetic
        # manifold it would be refused for TWO facts and the assertion would stop isolating
        # the one under test.
        {"name": "filter_cells", "group": "prep", "params": {"min_genes": 200}},
        {"name": "mioflow", "group": "lightning", "params": {}}]}
    never_embeds = {"name": "backwards", "steps": [
        {"name": "mioflow", "group": "lightning", "params": {}}]}

    assert vocab.unmet(filtered, "manifold") == vocab.unmet(never_embeds, "manifold")
    assert narrate.blocked(never_embeds, "manifold") == [narrate.MISSING_FACT["embedding"]]
    assert narrate.blocked(filtered, "manifold") == [narrate.CLEARED_FACT["embedding"]]
    # and the reason names the CAUSE, so the fix it implies is the one that works
    assert "filter" in narrate.CLEARED_FACT["embedding"]


def test_every_shape_and_topology_has_a_contents_phrase_and_no_others():
    """Same guard for the roster's "what's in it" column, over the other two closed lists."""
    assert set(narrate._SHAPE_PHRASE) == set(vocab.SHAPES)
    assert set(narrate._TOPOLOGY_PHRASE) == set(vocab.TOPOLOGIES)


def test_vocab_facts_is_derived_from_the_tables_and_not_a_fifth_literal():
    """`vocab.facts()` must be computed from the four tables. If it were a literal it would be
    the same closed list stated a fifth time — so adding a need must move it."""
    assert vocab.facts() == {"time", "conditions", "embedding", "clusters", "genes",
                             "model", "splicing", "velocity"}
    tables = (vocab.SHAPE_PROVIDES, vocab.GROUP_PROVIDES, vocab.STEP_PRODUCES, vocab.STEP_NEEDS)
    assert vocab.facts() == {f for t in tables for v in t.values() for f in v}


def test_a_fact_no_step_needs_can_never_reach_a_ledger_row():
    """MEASURED, and it contradicts the mockup in §3.2 — say so rather than pretend.

    §3.2 draws a hollow `trace a path — needs timepoints` row. `vocab.unmet` cannot produce it:
    `time` is provided by `time-course` data and required by NOTHING, because `mioflow` treats a
    real clock as a preference and falls back to a diffusion pseudotime (`STEP_NEEDS`'s own
    comment). So `cflows` is legal on all six shapes and that row cannot appear until some step
    declares a `time` need. This test fails the day one does — which is the day the mockup
    becomes reachable."""
    needed = {f for v in vocab.STEP_NEEDS.values() for f in v}
    assert "time" not in needed
    for shape in vocab.SHAPES:
        rows = {r.recipe: r for r in state.ledger(_obs(shape))}
        assert rows["cflows"].can_run, f"cflows unexpectedly blocked on {shape}"
        assert not any("timepoints" in b for r in rows.values() for b in r.blocked)


def test_blocked_says_exactly_what_unmet_refuses_and_agrees_on_legality():
    """One call, one answer. A caller must never hold a legality verdict and a reason computed
    separately — that is how the two come to disagree. The invariant is iff, over every bundled
    recipe × every shape."""
    from manyruns.catalog import discover_recipes, load_recipe

    checked = 0
    for name in discover_recipes():
        recipe = load_recipe(name)
        for shape in vocab.SHAPES:
            unmet = vocab.unmet(recipe, shape)
            why = narrate.blocked(recipe, shape)
            assert bool(why) == bool(unmet), f"{name}/{shape}: {why} vs {sorted(unmet)}"
            assert len(why) == len(unmet)
            checked += 1
    assert checked == len(discover_recipes()) * len(vocab.SHAPES)


def test_an_unphrased_fact_degrades_to_the_machine_word_instead_of_vanishing():
    """The fallback must never DROP a fact: a silent drop turns an illegal recipe into a
    runnable-looking one with no reason attached, which is worse than a machine word on screen.
    Exercised by naming a need nothing phrases."""
    recipe = {"steps": [{"name": "__probe__", "group": "analysis", "params": {}}]}
    original = dict(vocab.STEP_NEEDS)
    vocab.STEP_NEEDS["__probe__"] = ("lineage_barcodes",)
    try:
        why = narrate.blocked(recipe, "single")
        assert why == ["needs lineage_barcodes"]
        assert bool(why) == bool(vocab.unmet(recipe, "single"))
    finally:
        vocab.STEP_NEEDS.clear()
        vocab.STEP_NEEDS.update(original)


# ── the question splits into the ledger's two columns, from the catalog ──────
def test_every_bundled_recipe_splits_into_an_ask_and_a_gloss():
    """§3.2 draws two columns. They are DERIVED from each recipe's own `question:` — measured
    2026-08-16, all 11 bundled recipes write it as `<ask> — <what it will do>`. A hand-kept
    table of short verbs beside the questions would drift the moment someone reworded one and
    not its twin, so this pins that the derivation still covers the catalog."""
    from manyruns.catalog import discover_recipes, load_recipe

    names = discover_recipes()
    assert len(names) == 11, "catalog changed; re-read the measurement in this test"
    for name in names:
        ask, gloss = narrate.split_question(load_recipe(name)["question"])
        assert ask and gloss, f"{name}: {ask!r} / {gloss!r}"
        assert "—" not in ask and "—" not in gloss
        assert len(ask) < len(gloss), f"{name}: the ask should be the short half"


def test_a_question_with_no_dash_is_all_ask_and_no_invented_gloss():
    assert narrate.split_question("Just do the thing") == ("Just do the thing", "")
    assert narrate.split_question("") == ("", "")
    assert narrate.split_question(None) == ("", "")


# ── the ledger ───────────────────────────────────────────────────────────────
def test_the_ledger_shows_the_refused_recipe_instead_of_hiding_it():
    """§0's third defect, and the one commitment this product actually keeps: today a recipe
    that cannot run is simply ABSENT from the menu, so the scientist never learns that a disease
    column would unlock `contrast`. The ledger must carry it as a hollow row with its reason."""
    from manyruns.catalog import discover_recipes

    from manyruns.catalog import load_recipe

    rows = {r.recipe: r for r in state.ledger(_obs("single"))}
    # EVERY recipe except the `dev.` fixtures, which `narrate.offer` filters on the prefix they
    # declare. The commitment this test guards is that an analysis a scientist cannot run is
    # still SHOWN, with its reason; a recipe whose own header says its steps "DO NOT WORK YET"
    # is not an analysis, and offering it taught the reader nothing about their data.
    assert set(rows) == {n for n in discover_recipes()
                         if not str(load_recipe(n).get("id") or "").startswith("dev.")}

    contrast = rows["contrast"]
    assert contrast.can_run is False
    assert contrast.blocked == ("needs a healthy/disease column",)
    assert contrast.detail == "needs a healthy/disease column"   # the reason IS the 2nd column
    # Measured on the bundled 11, and the hollow rows are hollow for THREE different reasons:
    # `contrast` wants a fact this data lacks (`vocab.unmet` → `conditions`), `qc`, `markers` and
    # `preprocess` want a different one (`genes`), and `archetypes` cannot run against any data at
    # all until manylatents' `aa` stops rejecting the `method` kwarg (`narrate.unavailable`). The
    # ledger joins them; none is folded into the others.
    #
    # 8 -> 6 at the cutover (#54): `qc` and `rank_genes` now declare `("genes",)` in
    # `vocab.STEP_NEEDS`, and `genes` is a DATASET fact that `SHAPE_PROVIDES` never supplies —
    # `_obs("single")` is a bare Observation with no `n_vars` and no `provided`, so both refuse.
    # That is the point of the fact: a Gaussian point cloud has no gene names to rank. It STAYS 6
    # with `preprocess`: the eleventh recipe arrives hollow on this observation for the same
    # reason, so the catalog grew by one and the runnable count did not move.
    # 6 -> 5 with the `dev.` filter: `sandbox` was runnable here and no longer reaches the pane.
    assert sum(1 for r in rows.values() if r.can_run) == 5
    assert {name for name, r in rows.items() if not r.can_run} == {
        "contrast", "archetypes", "qc", "markers", "preprocess"}
    # The reason is written in the register `contrast` set — what the reader lacks, never what
    # the code did. It used to read "blocked upstream — manylatents' `aa` rejects the `method`
    # kwarg (archetypes-package skew)", which is a dependency's API drift on a scientist's screen.
    assert rows["archetypes"].detail == "not available yet"
    # the three gene-axis rows say WHICH fact, and it is not the one `contrast` is missing
    assert rows["qc"].detail == narrate.MISSING_FACT["genes"] == rows["markers"].detail
    assert rows["preprocess"].detail == narrate.MISSING_FACT["genes"]


def test_the_same_recipe_fills_in_once_the_data_carries_the_fact():
    """The hollow row is a statement about the DATA, not about the recipe."""
    single = {r.recipe: r for r in state.ledger(_obs("single"))}
    labelled = {r.recipe: r for r in state.ledger(_obs("case-control", conditions=["a", "b"]))}
    assert single["contrast"].can_run is False
    assert labelled["contrast"].can_run is True
    assert labelled["contrast"].blocked == ()
    assert labelled["contrast"].detail == single["contrast"].gloss   # back to the gloss


def test_a_caller_who_already_holds_the_fact_is_not_refused():
    """`provided` is passed straight through to `unmet`, whose docstring is explicit that the
    test is structural — does the object exist — never canonical. A precomputed condition
    labelling unlocks `contrast` on data whose shape does not declare one."""
    rows = {r.recipe: r for r in state.ledger(_obs("single"), provided={"conditions"})}
    assert rows["contrast"].can_run is True


def test_runnable_rows_come_first_and_the_head_is_the_recommendation():
    """Hollow rows are not selectable, so a list that interleaves them makes the keyboard skip.
    Inside each group `narrate.offer`'s order survives — this must not become a second ranking.
    """
    rows = state.ledger(_obs("case-control", conditions=["a", "b"]))
    assert [r.can_run for r in rows] == sorted((r.can_run for r in rows), reverse=True)
    assert rows[0].recommended and rows[0].recipe == "contrast"

    mixed = state.ledger(_obs("single"))
    assert [r.can_run for r in mixed] == sorted((r.can_run for r in mixed), reverse=True)
    runnable = [r.recipe for r in mixed if r.can_run]
    offered = [o["recipe"] for o in narrate.offer(_obs("single"))
               if o["recipe"] in set(runnable)]
    assert runnable == offered, "the ledger re-ranked what offer already ordered"


def test_the_ledger_is_catalog_derived_and_names_no_recipe():
    """`narrate.offer` is guarded this way in `test_front_door`; the ledger inherits the rule.
    A recipe added as a YAML file must reach the ledger with no code change here."""
    import inspect

    from manyruns.catalog import discover_recipes

    src = inspect.getsource(state.ledger) + inspect.getsource(state.LedgerRow)
    body = src.replace(state.ledger.__doc__ or "", "").replace(state.LedgerRow.__doc__ or "", "")
    for name in discover_recipes():
        assert name not in body, f"{name} is hardcoded in the ledger"


# ── the roster ───────────────────────────────────────────────────────────────
def test_the_roster_carries_the_drop_folder_and_the_bundled_set(tmp_path, monkeypatch):
    """The drop folder is the primary surface, so it comes first — the same order
    `shell._pick_source` uses, so moving between the two pickers is not a re-learn."""
    from manyruns.catalog import discover_datasets

    (tmp_path / "mine.csv").write_text("a,b\n1,2\n")
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path))

    rows = state.roster()
    assert rows[0].name == "mine.csv" and rows[0].kind == "file"
    assert {r.name for r in rows if r.kind == "bundled"} == set(discover_datasets())
    assert all(r.kind == "bundled" for r in rows[1:]), "bundled rows must not interleave"


def test_a_generated_dataset_says_synthetic_and_never_a_byte_count():
    """`synthetic` is not a missing value. A bundled handle names a GENERATOR — there is no
    file and no matrix until the engine makes one — and both handle kinds can do it:
    `synthetic_timecourse` is `{kind: path, ref: "synthetic:time-course"}`, which is why the
    test is "no file on disk" rather than "kind is manylatents"."""
    by_name = {r.name: r for r in state.roster()}
    assert by_name["swissroll"].size == "synthetic"          # kind: manylatents
    assert by_name["synthetic_timecourse"].size == "synthetic"  # kind: path, generated ref
    assert by_name["tree_wide"].contents == "a branching tree"


def test_topology_beats_shape_in_the_what_is_in_it_column():
    """`tree_wide` is `shape: manifold, topology: [multi-branching]`, and its own YAML says the
    shape vocabulary "CANNOT express this dataset". Reading `shape` first prints "a continuum"
    over a branching tree."""
    assert narrate.contents("manifold", ["multi-branching"]) == "a branching tree"
    assert narrate.contents("manifold") == "a continuum"
    # multi-label is joined, not collapsed: TOPOLOGIES is explicitly a union over annotators
    assert narrate.contents("manifold", ["cycle", "surface"]) == "a closed loop, a curved sheet"
    assert narrate.contents("nonsense-shape") == narrate.contents("unknown")


def test_the_capability_count_discriminates_once_a_row_reads_its_own_facts():
    """**The revisit this test's earlier form asked for.** It banned a "you can run N/8" column
    on §3.1's grounds and said what would end the ban: "this fails when datasets gain labels — at
    which point the column becomes real and the ban should be revisited."

    The cutover (#54) is that moment, one fact earlier than labels: `qc` and `rank_genes` declare
    `("genes",)` in `vocab.STEP_NEEDS`, and `configs/dataset/pbmc3k.yaml` is the first bundled
    dataset to declare `provides: [genes]`. So the two readings of "how many can I run" no longer
    agree, and both are pinned here rather than one being called the answer:

      * BY SHAPE ALONE — what `state.ledger(obs)` answers when the caller hands it nothing — is
        still one constant on every roster row. It moved 8 -> 6, because `qc` and `markers` are
        now refused on anything with no gene axis, and it moved by the same 2 everywhere, so it
        still discriminates nothing.
      * BY THE ROW'S OWN FACTS — `obs.provides()`, which is what `shell.py:554` passes — reads 3
        higher wherever there IS a gene axis. Measured 2026-08-16: `pbmc3k` (declared) and
        `pbmc3k_raw.h5ad` (the drop folder) read 9, the 13 synthetics read 6. It was 2 and 8
        before `preprocess` landed; the gap WIDENS with every recipe that reads gene names,
        which is the sense in which the column is now a real statistic.

    The ban therefore now rests on the first reading only, and the column has become the real
    statistic §3.1 said it was not. `DataEntry` still carries no `capability` field — that is a
    source change and not this test's to make, so that assertion stands unchanged.

    Which reading the TUI is on is not academic: `tui/app.py:151` builds `LedgerScreen(entry)`
    without `entry.obs.provides()`, so the screen is on the first one and refuses `qc` and
    `markers` on real scRNA with "a point cloud has no genes to read". Reported, not patched
    here.
    """
    rows = state.roster()
    by_shape = {r.name: sum(1 for row in state.ledger(r.obs) if row.can_run) for r in rows}
    # 6 -> 5: `sandbox` no longer reaches the menu (it declares `id: dev.sandbox.stubs`), so
    # every roster row lost the same one. The POINT of this assertion is the constancy, not the
    # number — a reading that is the same on all 14 datasets discriminates nothing.
    assert set(by_shape.values()) == {5}, f"the shape-only reading is not constant: {by_shape}"

    with_facts = {r.name: sum(1 for row in state.ledger(r.obs, r.obs.provides()) if row.can_run)
                  for r in rows}
    # Stated as "which rows moved" rather than as a set of counts, so this holds in a checkout
    # where `data/pbmc3k_raw.h5ad` is absent — the roster then carries no gene-bearing row and
    # both readings collapse back onto one constant.
    assert {n for n in by_shape if with_facts[n] > by_shape[n]} == {
        r.name for r in rows if r.obs.provides()}
    assert all(with_facts[r.name] == by_shape[r.name] + 3 for r in rows if r.obs.provides()), \
        "a gene axis unlocks exactly three recipes: `qc`, `markers` and `preprocess`"
    assert not hasattr(state.DataEntry, "capability")


def test_the_declared_shape_survives_wherever_there_is_nothing_to_infer_from(
        isolated_roster, tmp_path, monkeypatch):
    """Preservation: real anchors partition generated/missing refs from a temporary file."""
    import yaml
    from manyruns import catalog, shell

    assert isolated_roster.is_dir()
    configs = tmp_path / "catalog"
    configs.mkdir()
    declarations = [catalog.load_dataset(n) for n in catalog.discover_datasets()]
    assert declarations
    for ds in declarations:
        (configs / f"{ds['name']}.yaml").write_text(yaml.safe_dump(ds))
    import anndata
    import numpy as np
    anndata.AnnData(np.ones((3, 2))).write_h5ad(isolated_roster / "declared.h5ad")
    (configs / "local.yaml").write_text(yaml.safe_dump({
        "name": "local", "modality": "scrna", "shape": "single",
        "handle": {"kind": "path", "ref": "elsewhere/declared.h5ad"},
    }))
    monkeypatch.setenv("MANYRUNS_DATASET_DIR", str(configs))
    generated, on_disk = {}, {}
    assert configs.is_dir()
    for row in state.roster():
        if row.kind != "bundled":
            continue
        ds = catalog.load_dataset(row.name)
        ref = str((ds.get("handle") or {}).get("ref") or "")
        target = on_disk if shell.dataset_ref_path(ref) is not None else generated
        target[row.name] = (ds["shape"], row)
    assert generated and on_disk
    assert all(declared == row.shape for declared, row in generated.values())
    assert "local" in on_disk
    assert all(row.obs.n_obs and row.obs.n_vars for _, row in on_disk.values())


def test_a_bundled_dataset_that_is_a_real_file_is_still_read():
    """The fallback is for a ref that is NOT on disk. A dataset pointing at a real file must
    still be inspected — otherwise a declaration could quietly outrank the data it names, which
    is the opposite failure and the harder one to notice."""
    from manyruns import shell

    seen: list = []
    real = narrate.read_data

    class _Spy:
        def __call__(self, *a, **k):
            seen.append(a[0])
            return real(*a, **k)

    import unittest.mock

    with unittest.mock.patch.object(narrate, "read_data", _Spy()):
        for name in ("pbmc3k", "swissroll", "synthetic_timecourse"):
            try:
                shell._resolve_dataset(name)
            except Exception:  # noqa: BLE001 - not every name is in every checkout
                pass

    assert not any("synthetic:" in str(p) for p in seen), "a generator ref was opened as a file"


def test_a_selected_row_hands_the_existing_run_path_its_own_tuple():
    """`as_source()` is `(data_folder, dataset, modality, obs)` — exactly what
    `shell._pick_source` returns and `shell._run` takes. A second calling convention here would
    be a second way to start a run."""
    from manyruns import shell

    by_name = {r.name: r for r in state.roster()}
    got = by_name["swissroll"].as_source()
    assert got[:3] == (None, "swissroll", "synthetic")
    assert isinstance(got[3], Observation)

    # the same tuple `_pick_source` would have produced for that pick, element for element —
    # so a screen can hand it to `_run` unchanged rather than growing a second convention.
    want = shell._resolve_dataset("swissroll")
    assert got[:3] == want[:3] and got[3].shape == want[3].shape


# ── the run feed ─────────────────────────────────────────────────────────────
_RECIPE = {"steps": [{"name": "phate", "group": "latent"},
                     {"name": "mioflow", "group": "lightning"}]}


def _running(index, name):
    return {"index": index, "name": name, "group": "latent", "params": {},
            "outcome": None, "detail": None, "seconds": 0.0,
            "state": "running", "started": 1000.0}


def _settled(index, name, outcome="ok", **kw):
    rec = {"index": index, "name": name, "group": "latent", "params": {},
           "outcome": outcome, "detail": None, "seconds": 1.5}
    rec.update(kw)
    return rec


def test_a_declared_step_is_on_screen_as_queued_before_anything_runs():
    """The overlay is what makes a two-minute embedding watchable — `shell.live_dashboard`'s
    docstring: the dashboard exists BEFORE the run does."""
    feed = state.RunFeed(_RECIPE)
    rows = feed.rows()
    assert [r.name for r in rows] == ["phate", "mioflow"]
    assert [r.state for r in rows] == ["queued", "queued"]
    assert all(r.seconds is None and r.started is None for r in rows)
    assert (feed.done, feed.total) == (0, 2)


def test_the_feed_shows_running_then_the_outcome_as_the_records_land():
    feed = state.RunFeed(_RECIPE)
    live = _running(0, "phate")
    feed.on_step(live, [live])
    first = feed.rows()[0]
    assert first.state == "running" and first.running is True
    assert first.started == 1000.0
    assert first.seconds is None, "a duration must not exist before the step has one"
    assert feed.done == 0

    done = _settled(0, "phate", geometry={"lid": [None, 3.4]})
    feed.on_step(done, [done])
    first = feed.rows()[0]
    assert first.state == "ok" and first.running is False
    assert first.seconds == 1.5 and first.started is None
    assert [(d.label, d.value) for d in first.deltas] == [("lid", "— → 3.4")]
    assert (feed.done, feed.total) == (1, 2)


def test_the_feed_copies_the_lineage_instead_of_aliasing_it():
    """`apply_step` mutates the live record in place and hands back the SAME list every time.
    A feed that stored the list would render a settled step it was never told about — and a
    screen that repaints on notification only would show state it did not receive."""
    feed = state.RunFeed(_RECIPE)
    lineage = [_running(0, "phate")]
    feed.on_step(lineage[0], lineage)
    snapshot = feed.rows()[0].state

    lineage.append(_settled(1, "mioflow"))          # the runner moves on, silently
    assert snapshot == "running"
    assert [r.state for r in feed.rows()] == ["running", "queued"]


def test_a_step_that_ran_is_never_missing_from_the_run_view():
    """The one failure a run view must not have. `shell._progress_panel` drops a record whose
    index is past the declared list, which is right for a fixed recipe and wrong here:
    `Session.step` issues ad-hoc actions with no declaration, and `Session.run_recipe`
    deliberately does not de-duplicate, so a re-run lands at its own index."""
    feed = state.RunFeed(_RECIPE)
    records = [_settled(0, "phate"), _settled(1, "mioflow"), _settled(2, "phate")]
    feed.on_step(records[-1], records)

    rows = feed.rows()
    assert [r.name for r in rows] == ["phate", "mioflow", "phate"]
    assert [r.index for r in rows] == [0, 1, 2]
    assert feed.total == 3

    adhoc = state.RunFeed()          # a session with no recipe at all
    adhoc.on_step(records[0], [records[0]])
    assert [r.name for r in adhoc.rows()] == ["phate"]


def test_the_feed_records_an_observer_error_instead_of_losing_it():
    """`runner._report` swallows anything an observer raises, so a screen bug would vanish. It
    is kept here instead — and the run is still not taken down."""
    def boom(_feed):
        raise RuntimeError("bad width")

    feed = state.RunFeed(_RECIPE, listener=boom)
    rec = _settled(0, "phate")
    feed.on_step(rec, [rec])            # must not raise
    assert feed.observer_errors == ["RuntimeError: bad width"]
    assert feed.rows()[0].state == "ok"


def test_the_run_view_names_an_outcome_the_same_way_the_plain_panel_does():
    """Two surfaces, one table. If they could disagree there would be two truths about what
    ran — the failure `test_watch.test_every_renderer_covers_the_whole_outcome_vocabulary`
    exists to catch between the plain and rich panels, extended to the third surface."""
    from manyruns.watch import OUTCOMES

    for outcome in OUTCOMES:
        rec = _settled(0, "s", outcome=outcome)
        feed = state.RunFeed({"steps": [{"name": "s", "group": "latent"}]})
        feed.on_step(rec, [rec])
        view = feed.rows()[0]
        glyph, word = narrate._OUTCOME_MARKS[outcome]
        assert (view.glyph, view.word) == (glyph, word)
        assert f"{glyph} s" in narrate.run_panel(
            {"recipe": "r", "engine": "_inproc", "seed": 1, "steps": [rec]})


def test_the_live_states_are_not_smuggled_into_the_outcome_vocabulary():
    """`running` is a live state, not an outcome — `runner.apply_step` says so, and `_settle`
    strips it off every finished record. Merging the two tables would make `queued` a way a
    step can END."""
    from manyruns.watch import OUTCOMES

    assert set(narrate._OUTCOME_MARKS) == set(OUTCOMES)
    assert set(narrate._LIVE_MARKS).isdisjoint(OUTCOMES)
    assert narrate.step_mark(None) == narrate._LIVE_MARKS["queued"]
    assert narrate.step_mark({"state": "running"}) == narrate._LIVE_MARKS["running"]
    assert narrate.step_mark({"outcome": "nonsense"}) == ("?", "nonsense")


def test_the_detail_reaches_the_screen_unclipped():
    """`run_panel` clips at 58 characters because plain text has no layout engine. A Textual
    screen does, and §2 says making layout a constraint rather than a magic number is the whole
    reason for it — so a pre-truncated string it cannot widen would be the wrong hand-off."""
    long = "ValueError: " + "x" * 300
    rec = _settled(0, "phate", outcome="error", detail=long)
    feed = state.RunFeed(_RECIPE)
    feed.on_step(rec, [rec])
    assert feed.rows()[0].detail == long
    assert "…" in narrate.run_panel({"recipe": "r", "engine": "_inproc", "steps": [rec]})


# ── the whole structure is assertable without a terminal ─────────────────────
def test_every_view_serialises_to_plain_json():
    """The point of `to_dict`: a test (or a future `--json` surface) can assert on the whole
    structure with no terminal and no rich."""
    rec = _settled(0, "phate", geometry={"lid": [1.0, 3.4]}, plots=["a.png"])
    feed = state.RunFeed(_RECIPE)
    feed.on_step(rec, [rec])

    entry = state.roster()[0]
    payload = {"roster": entry.to_dict(),
               "ledger": [r.to_dict() for r in state.ledger(entry.obs)],
               "run": feed.to_dict()}
    assert json.loads(json.dumps(payload)) == payload
    assert payload["run"]["rows"][0]["deltas"] == [
        {"label": "lid", "value": "1 → 3.4", "measured": True}]


def test_a_misspelled_field_fails_loudly_rather_than_rendering_blank():
    """Why these are dataclasses and not dicts. Nothing here round-trips — they are built once
    and read once by a renderer — so a type costs nothing and buys an AttributeError in a test
    where a dict would have given a column a silent `None`."""
    row = state.ledger(_obs("single"))[0]
    with pytest.raises(AttributeError):
        row.recipie          # noqa: B018
    with pytest.raises(Exception):
        row.recipe = "x"     # frozen: a view must not be edited in place


def test_the_two_surfaces_cannot_disagree_about_a_blocked_recipe():
    """FOUND BY DRIVING BOTH, and it was a real inconsistency rather than a hypothetical.

    `narrate.unavailable` had exactly one caller — `state.ledger` — so the app drew
    `archetypes` as a hollow row reading "blocked upstream", while the rich shell, which reads
    `narrate.offer`, listed the same recipe as its TOP recommendation for `swissroll`. One
    recipe, two answers, on the same data.

    The suppression lives in `offer` now, so every surface that reads it gets one answer and
    there is no second call for a new surface to forget. Both halves are asserted: the blocked
    recipe is never RECOMMENDED, and it is still RETURNED — the ledger's hollow row is
    informative precisely because it is on screen with its reason, and dropping it from the
    menu would take a stated blockage back to an invisible one.
    """
    from manyruns.catalog import load_recipe

    obs = _obs("manifold", modality="synthetic")
    offered = {o["recipe"]: o for o in narrate.offer(obs)}

    blocked_upstream = [n for n in offered if narrate.unavailable(load_recipe(n))]
    assert blocked_upstream, "no recipe declares `blocked:`; this test proves nothing"
    for name in blocked_upstream:
        assert name in offered, f"{name} vanished from the menu instead of being un-starred"
        assert not offered[name]["recommended"], f"{name} is blocked upstream and recommended"

    # and the two surfaces now agree, recipe for recipe
    rows = {r.recipe: r for r in state.ledger(obs)}
    for name, row in rows.items():
        if not row.can_run:
            assert not offered[name]["recommended"], f"{name}: ledger hollow, menu recommends"


def test_a_recipe_a_NARROWING_blocks_is_hollow_on_both_surfaces_too(tmp_path, monkeypatch):
    """The same invariant, against the second way a recipe can now be blocked — and the test
    above cannot see it. It runs over the 8 BUNDLED recipes, none of which declares a `prep`
    step, so it fires on `unavailable` and never on `unmet`.

    `offer` computed `recommended` from `suits` and `unavailable` only, which was safe while a
    recipe on its own declared suit could only be blocked by a MISSING DATA FACT — something
    `suits` rules out by construction. A narrowing blocks for a reason internal to step ORDER,
    which is shape-independent and about which `suits` says nothing, so the recipe below sorted
    FIRST and starred in the rich shell's menu while the ledger drew it as a hollow row. One
    recipe, two answers on the same data: the failure `vocab.py`'s header describes, and the
    one commit 240f222 fixed for the other half of `offer` one commit earlier."""
    (tmp_path / "tidytraj.yaml").write_text(
        "name: tidytraj\n"
        "question: Trace the path — where are these tidy cells heading?\n"
        "suits: [manifold]\n"
        "steps:\n"
        "  - {name: phate, group: latent, params: {n_components: 3}}\n"
        # `filter_cells` for the reason the sibling test above records: `filter_genes` needs
        # a gene axis, which a synthetic manifold has not got.
        "  - {name: filter_cells, group: prep, params: {min_genes: 200}}\n"
        "  - {name: mioflow, group: lightning, params: {}}\n"
    )
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(tmp_path))
    obs = _obs("manifold", modality="synthetic")

    offered = {o["recipe"]: o for o in narrate.offer(obs)}
    row = {r.recipe: r for r in state.ledger(obs)}["tidytraj"]

    assert row.can_run is False and row.blocked == (narrate.CLEARED_FACT["embedding"],)
    assert "tidytraj" in offered, "still RETURNED — a hollow row with its reason is the point"
    assert offered["tidytraj"]["recommended"] is False


def test_a_declared_shape_wins_even_when_the_file_can_be_read():
    """The half of the declared-vs-sniffed defect that survived the first fix.

    `shell._resolve_dataset` honoured the YAML's `shape:` when the file was MISSING and ran
    `narrate.read_data` when it was present — so THE MORE INFORMATION THE PRODUCT HAD, THE MORE
    LIKELY IT WAS TO DISCARD THE DECLARATION. Measured on `data/pbmc3k_raw.h5ad`, which declares
    `shape: clusters`: the roster reported `single`, and `cluster` — the recipe that dataset is
    for — was demoted out of the suited set on the one real dataset this repo ships.

    MERGE, NOT CHOOSE, is the property asserted here: the declaration governs `shape`, and the
    read still fills in everything it alone knows (`n_vars`). Taking either wholesale throws the
    other away, and both are needed on the same screen.
    """
    import pytest

    pytest.importorskip("anndata")
    from pathlib import Path

    from manyruns import catalog, shell

    if not Path("data/pbmc3k_raw.h5ad").is_file():
        pytest.skip("pbmc3k is a checkout-relative ref; see manyruns#12 for fetch-and-verify")

    declared = catalog.load_dataset("pbmc3k")["shape"]
    _folder, _ds, _modality, obs = shell._resolve_dataset("pbmc3k")

    assert obs.shape == declared == "clusters", "the sniff overrode the declaration"
    assert obs.n_vars == 32738, "the read no longer fills in what only it knows"
    assert "genes" in obs.provides()

    rows = state.ledger(obs, obs.provides())
    suited = {r.recipe for r in rows if r.recommended and r.can_run}
    assert "cluster" in suited, f"cluster is not suited to clusters-shaped data: {sorted(suited)}"


@pytest.fixture
def isolated_roster(tmp_path, monkeypatch):
    from manyruns import inspected
    monkeypatch.chdir(tmp_path)
    drop = tmp_path / 'drop'
    drop.mkdir()
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(drop))
    monkeypatch.delenv('MANYRUNS_DATASET_DIR', raising=False)
    monkeypatch.setattr(inspected, 'HOME', tmp_path / 'cache')
    return drop


def test_absent_sample_is_downloadable_and_keeps_declared_facts(isolated_roster):
    assert isolated_roster.is_dir()
    row = next(r for r in state.roster() if r.name == 'pbmc3k')
    assert row.missing and row.downloadable and not row.pending
    assert row.size == 'download · 5.6 MB'
    assert row.shape == 'clusters' and 'genes' in row.obs.provides()
    assert row.obs.n_obs is None and row.obs.n_vars is None
    assert str(isolated_roster / 'pbmc3k_raw.h5ad') in '\n'.join(row.refusal)
    assert json.loads(json.dumps(row.to_dict())) == row.to_dict()
    assert isinstance(row.refusal, tuple)


def test_held_alias_and_drop_never_inspect_or_read_cache(isolated_roster, monkeypatch):
    from manyruns import inspected, shell
    path = isolated_roster / 'pbmc3k_raw.h5ad'
    path.write_bytes(b'partial')
    assert isolated_roster.is_dir()

    def forbidden(*args, **kwargs):
        pytest.fail('a held source was inspected')

    monkeypatch.setattr(inspected, 'get', forbidden)
    monkeypatch.setattr(narrate, 'read_data', forbidden)
    real = shell._resolve_dataset
    monkeypatch.setattr(shell, '_resolve_dataset',
                        lambda name: forbidden() if name == 'pbmc3k' else real(name))
    rows = state.roster(skip={path.parent / '..' / path.parent.name / path.name})
    assert not any(r.kind == 'file' for r in rows)
    row = next(r for r in rows if r.name == 'pbmc3k')
    assert row.pending and row.size == 'copying…'
    assert row.obs.n_vars is None and 'genes' in row.obs.provides()


@pytest.mark.parametrize('ref', ['generator.py', '.hidden.csv', '.hidden/generator.py'])
def test_catalog_only_sources_settle_without_becoming_dropped_rows(
        isolated_roster, monkeypatch, tmp_path, ref):
    import yaml
    from manyruns import shell
    from manyruns.pipeline.loading import load_array
    from manyruns.tui import dropwatch, samplefetch

    assert isolated_roster.is_dir()
    path = isolated_roster / ref
    path.parent.mkdir(exist_ok=True)
    path.write_text('import numpy as np\nX = np.array([[1, 2], [3, 4]])\n'
                    if path.suffix == '.py' else 'a,b\n1,2\n3,4\n')
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    (configs / 'declared.yaml').write_text(yaml.safe_dump({
        'name': 'declared', 'shape': 'manifold', 'modality': 'synthetic',
        'handle': {'kind': 'path', 'ref': str(path)},
    }))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    reads = []
    original = shell._resolve_dataset

    def resolve(name):
        reads.append(name)
        return original(name)

    monkeypatch.setattr(shell, '_resolve_dataset', resolve)
    watch = dropwatch.DropWatch()

    def rows():
        poll = watch.observe(dropwatch.snapshot())
        result = state.roster(snapshot=poll.snapshot, skip=poll.held)
        assert len(result) == 1 and result[0].kind == 'bundled'
        return result[0]

    assert rows().pending and not reads
    settled = rows()
    assert not settled.pending and not settled.refusal
    assert samplefetch._resolve(settled).path == path
    assert load_array(settled.as_source()[0]).shape == (2, 2)
    reads.clear()
    path.write_text(path.read_text() + '\n')
    assert rows().pending and not reads
    assert not rows().pending and reads == ['declared']


@pytest.mark.parametrize('appears_during', ['startup', 'rescan', 'path_resolution'])
def test_external_catalog_appearance_after_snapshot_waits_for_two_polls(
        isolated_roster, tmp_path, monkeypatch, appears_during):
    import yaml
    from manyruns import shell
    from manyruns.tui import dropwatch

    assert isolated_roster.is_dir()
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    source = tmp_path / 'external.csv'
    (configs / 'external.yaml').write_text(yaml.safe_dump({
        'name': 'external', 'shape': 'single', 'modality': 'bulk',
        'handle': {'kind': 'path', 'ref': str(source)},
    }))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    watch = dropwatch.DropWatch()
    if appears_during == 'rescan':
        watch.observe(dropwatch.snapshot())
    poll = watch.observe(dropwatch.snapshot())
    assert not poll.snapshot.entries
    if appears_during == 'path_resolution':
        resolve_path = shell.dataset_ref_path

        def appear(ref):
            found = resolve_path(ref)
            source.write_text('a,b\n1,2\n')
            monkeypatch.setattr(shell, 'dataset_ref_path', resolve_path)
            return found

        monkeypatch.setattr(shell, 'dataset_ref_path', appear)
    else:
        source.write_text('a,b\n1,2\n')
    reads = []
    resolve = shell._resolve_dataset

    def read(name):
        reads.append(name)
        return resolve(name)

    monkeypatch.setattr(shell, '_resolve_dataset', read)

    def row(poll):
        return state.roster(snapshot=poll.snapshot, skip=poll.held,
                            on_changed=watch.invalidate)[0]

    appeared = row(poll)
    assert not reads, 'a source absent from the snapshot was inspected'
    assert (appeared.pending or appeared.missing) and appeared.observation is None
    assert row(watch.observe(dropwatch.snapshot())).pending and not reads
    settled = row(watch.observe(dropwatch.snapshot()))
    assert not settled.pending and settled.observation is not None
    assert reads == ['external']


@pytest.mark.parametrize('source_kind', [
    'drop_file', 'drop_directory', 'catalog_alias', 'external_catalog',
    'fetched_sample', 'symlink', 'symlink_target', 'nested_file_symlink',
    'nested_directory_symlink',
])
def test_all_source_kinds_share_inspection_and_selection_settling(
        isolated_roster, tmp_path, monkeypatch, source_kind):
    import hashlib
    import io
    import yaml
    from manyruns import datasetfetch, shell
    from manyruns.pipeline.loading import load_array
    from manyruns.tui import dropwatch, samplefetch

    assert isolated_roster.is_dir()
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    source = isolated_roster / 'cells.csv'
    if source_kind in {'drop_directory', 'catalog_alias', 'nested_file_symlink',
                       'nested_directory_symlink'}:
        source = isolated_roster / 'cohort' / 'cells.csv'
        source.parent.mkdir()
    elif source_kind == 'external_catalog':
        source = tmp_path / 'external.csv'
    target = source
    if source_kind in {'symlink', 'symlink_target', 'nested_file_symlink'}:
        target = tmp_path / 'storage'
        source.symlink_to(target)
    elif source_kind == 'nested_directory_symlink':
        storage = tmp_path / 'storage'
        storage.mkdir()
        (source.parent / 'linked').symlink_to(storage, target_is_directory=True)
        source = source.parent / 'linked' / 'cells.h5ad'
        target = storage / 'cells.h5ad'
    payload = b'a,b\n1,2\n'
    if source_kind == 'nested_directory_symlink':
        import anndata
        import numpy as np

        anndata.AnnData(np.ones((1, 2))).write_h5ad(target)
        payload = target.read_bytes()
    handle = {'kind': 'path', 'ref': str(source)}
    if source_kind == 'nested_directory_symlink':
        handle['ref'] = str(source.parent.parent)
    declaration = {'name': 'declared', 'shape': 'single', 'modality': 'bulk',
                   'handle': handle}
    bundled = source_kind in {'catalog_alias', 'external_catalog', 'fetched_sample',
                             'symlink_target', 'nested_directory_symlink'}
    if source_kind == 'fetched_sample':
        handle.update(ref='data/cells.csv', bytes=len(payload),
                      sha256=hashlib.sha256(payload).hexdigest(),
                      url='https://example.invalid/cells.csv')
    if bundled:
        (configs / 'declared.yaml').write_text(yaml.safe_dump(declaration))
    if source_kind == 'fetched_sample':
        datasetfetch.fetch_dataset(declaration, opener=lambda *a, **k: io.BytesIO(payload))
    else:
        target.write_bytes(payload)
    reads = []
    for attr in ('_resolve_path_source', '_resolve_dataset'):
        original = getattr(shell, attr)

        def read(*args, _original=original, **kwargs):
            reads.append(args[0])
            return _original(*args, **kwargs)

        monkeypatch.setattr(shell, attr, read)
    watch = dropwatch.DropWatch()

    def rows():
        poll = watch.observe(dropwatch.snapshot())
        return state.roster(snapshot=poll.snapshot, skip=poll.held,
                            on_changed=watch.invalidate)

    assert all(r.pending for r in rows()) and not reads
    settled = next(r for r in rows() if (r.kind == 'bundled') == bundled)
    assert not settled.pending and settled.observation is not None
    prepared = samplefetch._resolve(settled)
    assert not prepared.pending and not prepared.refusal
    assert load_array(prepared.path).shape == (1, 2)
    reads.clear()
    if source_kind == 'nested_directory_symlink':
        anndata.AnnData(np.ones((2, 2))).write_h5ad(target)
    else:
        target.write_bytes(payload + b'3,4\n')
    assert samplefetch._resolve(settled).pending and not reads
    assert all(r.pending for r in rows()) and not reads
    refreshed = next(r for r in rows() if (r.kind == 'bundled') == bundled)
    assert not refreshed.pending and refreshed.observation != settled.observation
    prepared = samplefetch._resolve(refreshed)
    assert not prepared.pending and not prepared.refusal
    assert load_array(prepared.path).shape == (2, 2)


@pytest.mark.parametrize('from_snapshot', [False, True])
def test_source_symlinks_preserve_names_suffixes_and_execution_paths(
        isolated_roster, tmp_path, monkeypatch, from_snapshot):
    import yaml
    from manyruns import inspected
    from manyruns.pipeline.loading import load_array
    from manyruns.tui import dropwatch, samplefetch

    assert isolated_roster.is_dir()
    blob = tmp_path / 'storage-blob'
    blob.write_text('a,b\n1,2\n3,4\n')
    source = isolated_roster / 'experiment.csv'
    source.symlink_to(blob)
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    (configs / 'experiment.yaml').write_text(yaml.safe_dump({
        'name': 'experiment', 'shape': 'single', 'modality': 'bulk',
        'handle': {'kind': 'path', 'ref': str(source)},
    }))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    # A cache row inspected through a different suffix cannot describe this reference.
    # Seed a claimed inspection directly: new rows reject this bare blob. Mark it
    # readable so the reference mismatch, rather than cache migration, is exercised.
    cached = inspected.as_row('unknown', _obs(modality='unknown'))
    cached.update(observed_signature=list(dropwatch.fingerprint(blob)), observed_source=str(blob),
                  source_readable=True)
    inspected.put(blob, cached)
    if from_snapshot:
        watch = dropwatch.DropWatch()
        first = watch.observe(dropwatch.snapshot())
        assert not any(r.kind == 'file' for r in
                       state.roster(snapshot=first.snapshot, skip=first.held))
        poll = watch.observe(dropwatch.snapshot())
        rows = state.roster(snapshot=poll.snapshot, skip=poll.held)
    else:
        rows = [state.local_entry(source), state.catalog_entry('experiment')]
    assert len(rows) == 2
    for row in rows:
        assert row.path == source and row.path.is_symlink()
        assert row.name == ('experiment.csv' if row.kind == 'file' else 'experiment')
        assert row.modality == 'bulk'
        prepared = samplefetch._resolve(row)
        assert not prepared.pending and not prepared.refusal
        assert prepared.path == source
        assert load_array(prepared.as_source()[0]).shape == (2, 2)


def test_catalog_symlink_retargeted_during_inspection_is_held(
        isolated_roster, tmp_path, monkeypatch):
    import yaml
    from manyruns import shell
    from manyruns.tui import dropwatch

    assert isolated_roster.is_dir()
    source = isolated_roster / 'experiment.csv'
    old = tmp_path / 'old-blob'
    new = tmp_path / 'new-blob'
    old.write_text('a,b\n1,2\n')
    new.write_text('a,b\n3,4\n5,6\n')
    source.symlink_to(old)
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    (configs / 'experiment.yaml').write_text(yaml.safe_dump({
        'name': 'experiment', 'shape': 'single', 'modality': 'bulk',
        'handle': {'kind': 'path', 'ref': str(source)},
    }))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    original = shell._resolve_dataset

    def retarget(name):
        source.unlink()
        source.symlink_to(new)
        return original(name)

    monkeypatch.setattr(shell, '_resolve_dataset', retarget)
    changed = []
    row = state.catalog_entry('experiment', snapshot=dropwatch.snapshot(),
                              on_changed=changed.append)
    assert row.pending and row.obs.n_obs is None
    assert row.observation is None and changed == [old]


def test_inspection_race_is_held_and_not_cached(isolated_roster, monkeypatch):
    from manyruns import inspected, shell
    from manyruns.tui import dropwatch
    path = isolated_roster / 'changing.csv'
    path.write_text('a,b\n1,2\n')
    assert isolated_roster.is_dir()
    snap = dropwatch.snapshot()
    changed = []

    def read(*args):
        path.write_text('a,b\n1,2\n3,4\n')
        return path, None, 'tabular', _obs('single')

    monkeypatch.setattr(shell, '_resolve_path_source', read)
    monkeypatch.setattr(inspected, 'put', lambda *a: pytest.fail('cached moving bytes'))
    rows = state.roster(snapshot=snap, on_changed=changed.append)
    assert path in changed
    assert not any(r.path == path for r in rows)


def test_cache_publication_race_reloads_metadata_after_resettling(isolated_roster, monkeypatch):
    import anndata
    import numpy as np
    from manyruns import inspected
    from manyruns.tui import dropwatch
    assert isolated_roster.is_dir()
    path = isolated_roster / 'changing.h5ad'
    anndata.AnnData(np.ones((2, 3))).write_h5ad(path)
    watch = dropwatch.DropWatch()
    watch.observe(dropwatch.snapshot())
    stable = watch.observe(dropwatch.snapshot())
    reads = []
    resolve, put = state._resolve_path_source, inspected.put

    def read(p, console):
        result = resolve(p, console)
        reads.append(result[3].n_obs)
        return result

    def publish(p, row):
        # put obtains its own key after the caller's pre-publication check.
        anndata.AnnData(np.ones((4, 3))).write_h5ad(path)
        put(p, row)

    monkeypatch.setattr(state, '_resolve_path_source', read)
    monkeypatch.setattr(inspected, 'put', publish)
    assert state.local_entry(path, expected=stable.snapshot.entries[path],
                             on_changed=watch.invalidate) is None
    monkeypatch.setattr(inspected, 'put', put)
    assert path in watch.observe(dropwatch.snapshot()).held
    settled = watch.observe(dropwatch.snapshot())
    assert path in settled.ready
    fresh = state.local_entry(path, expected=settled.snapshot.entries[path])
    assert fresh.obs.n_obs == 4
    assert reads == [2, 4]
    # A valid published observation remains usable without another metadata read.
    assert state.local_entry(path).obs.n_obs == 4
    assert reads == [2, 4]


def test_held_directory_alias_never_reads_descendant_bytes(isolated_roster, tmp_path, monkeypatch):
    import yaml
    from manyruns import inspected
    assert isolated_roster.is_dir()
    folder = isolated_roster / 'cohort'
    folder.mkdir()
    matrix = folder / 'matrix.csv'
    matrix.write_text('a,b\n1,2\n')
    configs = tmp_path / 'catalog'
    configs.mkdir()
    (configs / 'cohort.yaml').write_text(yaml.safe_dump({
        'name': 'cohort', 'shape': 'single', 'modality': 'tabular',
        'handle': {'kind': 'path', 'ref': str(matrix)},
    }))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    monkeypatch.setattr(inspected, 'get', lambda *a: pytest.fail('held directory cache read'))
    monkeypatch.setattr(narrate, 'read_data', lambda *a, **k: pytest.fail('held alias read'))
    rows = state.roster(skip={folder})
    assert len(rows) == 1 and rows[0].pending


@pytest.mark.parametrize('change_at', ['before_roster', 'local_read', 'before_alias', 'alias_read'])
def test_directory_alias_uses_the_observed_parent_snapshot(
        isolated_roster, tmp_path, monkeypatch, change_at):
    import yaml
    from manyruns import shell
    from manyruns.tui import dropwatch
    assert isolated_roster.is_dir()
    folder = isolated_roster / 'cohort'
    folder.mkdir()
    matrix = folder / 'matrix.csv'
    matrix.write_text('a,b\n1,2\n')
    configs = tmp_path / 'catalog'
    configs.mkdir()
    (configs / 'cohort.yaml').write_text(yaml.safe_dump({
        'name': 'cohort', 'shape': 'single', 'modality': 'tabular',
        'handle': {'kind': 'path', 'ref': str(matrix)},
    }))
    assert configs.is_dir()
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    snap = dropwatch.snapshot()
    changed, reads = [], []

    def grow():
        matrix.write_text('a,b\n1,2\n3,4\n')

    def local_read(p, console):
        grow()
        return p, None, 'tabular', _obs('single')

    def alias_read(name):
        reads.append(name)
        if change_at == 'alias_read':
            grow()
        return matrix, None, 'tabular', _obs('single')

    monkeypatch.setattr(shell, '_resolve_path_source', local_read)
    monkeypatch.setattr(shell, '_resolve_dataset', alias_read)
    if change_at.startswith('before_'):
        grow()
    if change_at in {'before_alias', 'alias_read'}:
        row = state.catalog_entry('cohort', snapshot=snap, on_changed=changed.append)
    else:
        rows = state.roster(snapshot=snap, on_changed=changed.append)
        assert not any(r.kind == 'folder' for r in rows)
        row = next(r for r in rows if r.name == 'cohort')
    assert row.pending and row.obs.n_obs is None
    assert changed == [folder]
    assert reads == (['cohort'] if change_at == 'alias_read' else [])


def test_directory_growth_does_not_reuse_the_parent_timestamp_cache(
        isolated_roster):
    from manyruns import inspected
    from manyruns.tui import dropwatch
    assert isolated_roster.is_dir()
    folder = isolated_roster / 'cohort'
    folder.mkdir()
    control, treated = folder / 'control', folder / 'treated'
    control.mkdir()
    treated.mkdir()
    matrix = '%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n2 2 1\n'
    (control / 'matrix.mtx').write_text(matrix)
    assert folder.is_dir() and control.is_dir() and treated.is_dir()
    for sample in (control, treated):
        # Complete legacy 10x inputs: a matrix-only placeholder is now unreadable.
        (sample / 'genes.tsv').write_text('g1\tG1\ng2\tG2\n')
        (sample / 'barcodes.tsv').write_text('c1\nc2\n')
    signature = dropwatch.fingerprint(folder)
    before = state.local_entry(folder)
    assert before.obs.shape == 'single' and before.obs.conditions is None
    # Preservation: live-row reuse also checks descendant signatures for folders.
    assert state.local_entry(folder, previous=before) is before
    assert not next(r for r in state.ledger(before.obs, before.obs.provides())
                    if r.recipe == 'contrast').can_run
    inspected.put(folder, inspected.as_row('scrna', before.obs))
    old_key = inspected.key(folder)

    # Only a descendant changes; the parent cache key still names the old observation.
    (treated / 'matrix.mtx').write_text(matrix)
    assert inspected.key(folder) == old_key
    assert dropwatch.fingerprint(folder) != signature
    after = state.local_entry(folder, previous=before)
    assert after is not before
    assert after.obs.shape == 'case-control'
    assert after.obs.conditions == ['control', 'treated']
    assert next(r for r in state.ledger(after.obs, after.obs.provides())
                if r.recipe == 'contrast').can_run
