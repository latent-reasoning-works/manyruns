"""Every recipe in the catalog is valid, legal somewhere, and actually runs.

Three recipes was not a menu — the shell hardcoded "run the cflows recipe" because there was
nothing to choose between. Eleven is only an improvement if each one EXECUTES; a catalog of YAML
that parses and cannot run is a longer dropdown, not a product.
"""
import pytest

np = pytest.importorskip("numpy")

from manyruns import catalog, vocab  # noqa: E402

#: `archetypes` is blocked UPSTREAM, not here: manylatents' `aa` raises
#: `TypeError: AA.__init__() got an unexpected keyword argument 'method'` — a version skew
#: against the `archetypes` package, one line, and theirs. Named rather than skipped silently,
#: so that when it is fixed this list shrinks and the test tells us.
UPSTREAM_BLOCKED = {"archetypes"}

#: Legal on no BUNDLED dataset, each for a stated reason. A DATA gap, not a dead recipe:
#: `contrast` needs `conditions`, which only a `case-control` shape provides, and the starter
#: set is thirteen synthetic point clouds plus `pbmc3k` — no cohort among them. That gap is
#: already on the record at authoring time, as a WARN and not a FAIL, from `manyruns check`'s
#: coverage pass (pinned in `tests/test_check.py`). Named here rather than tolerated silently,
#: so that declaring a case-control dataset shrinks this set and the test tells us.
NO_BUNDLED_DATA = {"contrast"}


def test_every_recipe_validates():
    """`check_recipe` covers the provenance fields too — a malformed `source:` or a `via:` on a
    step that does not exist is refused here rather than at run time."""
    for name in catalog.discover_recipes():
        assert catalog.check_recipe(catalog.load_recipe(name), name) == [], name


def test_every_recipe_declares_its_provenance():
    """A recipe that cannot say where it came from is a workflow someone made up. `id` is the
    stable canonical name; `source` is the pointer out to literature no code resolves."""
    for name in catalog.discover_recipes():
        cfg = catalog.load_recipe(name)
        assert cfg.get("id"), f"{name} has no id"
        src = cfg.get("source")
        assert src, f"{name} has no source"
        # `kind: none` is a legitimate answer, not a gap: `contrast`'s two steps are
        # manyruns's own readouts with no published method behind them. What is refused is
        # SILENCE — a recipe that neither cites a source nor says it has none.
        if src.get("kind") == "none":
            assert src.get("note"), f"{name} claims no source but does not say why"
            continue
        assert src.get("cite"), f"{name} source has no citation"
        # a DOI or a URL — never neither, and never an invented DOI
        assert src.get("doi") or src.get("url"), f"{name} source has neither doi nor url"


def test_every_recipe_is_legal_on_some_bundled_dataset():
    """A recipe legal on nothing is a dead entry in the menu. `contrast` was exactly that — it
    needs `conditions` and no bundled dataset declared `case-control`.

    **RE-FRAMED AT THE CUTOVER, and the re-framing is the finding.** This used to sweep SHAPES
    alone, which stopped being the right question when `qc` and `rank_genes` declared `genes`:
    that fact comes from the BYTES, so `SHAPE_PROVIDES` can never supply it and both recipes were
    "legal on no shape at all" while being perfectly runnable on a real `.h5ad`. Legality is a
    function of (shape x dataset) and always was — `unmet` has taken a `handle` since the gene
    axis landed; only this test had not caught up.

    So it sweeps the DECLARATIONS, which is also the stronger question: not "does some
    hypothetical shape admit this recipe" but "can anyone actually run it on something we ship".
    Measured: before `configs/dataset/pbmc3k.yaml` existed the answer for `qc` and `markers` was
    NO — all thirteen bundled datasets were synthetic point clouds with no gene axis.

    But the two questions are not the same, and dropping the shape sweep lost one of them: the
    dataset sweep alone reports `contrast` as a failure, when what it actually measures is that
    we ship no cohort. "Dead" and "unexercised" are different defects with different fixes —
    delete the recipe vs. declare a dataset — so this asks BOTH. Unexercised is allowed, once,
    by name, in `NO_BUNDLED_DATA`, and only for a recipe that some declarable shape still
    satisfies. Measured on the bundled catalog: `contrast` is legal on `case-control` and on
    nothing else, which is why one `.yaml` closes the gap and no code change would.
    """
    datasets = catalog.load_datasets()
    for name in catalog.discover_recipes():
        rec = catalog.load_recipe(name)
        legal = [d["name"] for d in datasets
                 if not vocab.unmet(rec, d.get("shape", "unknown"), handle=d.get("handle"))]
        if name in NO_BUNDLED_DATA:
            assert not legal, (
                f"{name} now runs on {legal} — the data gap closed, so drop it from "
                "NO_BUNDLED_DATA and let this test hold it to the same bar as the rest")
            assert [s for s in vocab.SHAPE_PROVIDES if not vocab.unmet(rec, s)], (
                f"{name} is legal on no bundled dataset AND on no declarable shape — that is "
                f"dead rather than unexercised; it needs {sorted(vocab.unmet(rec, 'unknown'))}")
            continue
        assert legal, (
            f"{name} is legal on none of the {len(datasets)} bundled datasets; it needs "
            f"{sorted(vocab.unmet(rec, 'unknown'))} and nothing declares them")


def test_the_gene_recipes_are_legal_on_real_data_and_refused_on_a_swissroll():
    """The refusal has to be REAL and SATISFIABLE, which are two separate failures.

    A need nothing can satisfy is a dead recipe (the test above). A need everything satisfies is
    decoration. `genes` is neither: measured, `qc` and `markers` come back clean on `pbmc3k` and
    unmet on every synthetic declaration — which is correct, because a swissroll has no genes and
    both steps used to run on one and emit a note instead of declining."""
    by_name = {d["name"]: d for d in catalog.load_datasets()}
    pbmc, roll = by_name["pbmc3k"], by_name["swissroll"]

    for name in ("qc", "markers"):
        rec = catalog.load_recipe(name)
        assert vocab.unmet(rec, pbmc["shape"], handle=pbmc["handle"]) == frozenset()
        assert "genes" in vocab.unmet(rec, roll["shape"], handle=roll["handle"])


def _point_cloud():
    """Three Gaussian blobs, 180 x 30 — the fixture every recipe here has always run on.

    Structure enough to be worth measuring and NOT counts: it is signed, so `normalize` and
    `transform` decline on it, which is the behaviour the block at the end of the sweep pins.
    """
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(c, 0.6, (60, 30)) for c in (0, 5, 10)])


def _scrna_counts():
    """Counts WITH A GENE AXIS, 180 x 400 — what a recipe declaring `genes` has to run on.

    A second fixture rather than a wider first one, because the point cloud is what makes the
    preamble stand down and that is a property the sweep pins for the other nine recipes.

    THE THRESHOLDS ARE WHY IT IS 400 GENES AND NOT 30. `preprocess` declares the Scanpy pbmc3k
    tutorial's cuts against a 32,738-gene matrix; measured, `min_genes=200` over the 30-column
    point cloud drops all 180 cells, so the recipe "runs" having annihilated its own data. A
    fixture has to be wide enough for that cut to be a cut.

    EVERY FILTER BITES, or a recipe passes this sweep having removed nothing and the ordering
    that the recipe's comment calls load-bearing is never exercised. Measured 2026-08-16 through
    `preprocess`: `filter_cells` drops 6 of 180 cells, `filter_genes` 10 of 400 genes,
    `filter_mito` 6 of the remaining 174 — each from the defect planted for it below.
    """
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, (180, 400)).astype(np.float32)
    for k in range(3):  # three cell groups, so `leiden` and `rank_genes` have something to find
        counts[k * 60:(k + 1) * 60, k * 40:(k + 1) * 40] += rng.poisson(20.0, (60, 40))
    genes = np.array([f"G{i:04d}" for i in range(400)], dtype="<U8")
    genes[300:313] = [f"MT-{i}" for i in range(13)]   # 13, the number pbmc3k carries
    counts[:6, 120:] = 0.0                            # 6 low-complexity cells, under min_genes
    counts[:, 390:] = 0.0                             # 10 genes seen in one cell, under min_cells
    counts[20, 390:] = 3.0
    counts[6:12, 300:313] += 200.0                    # 6 cells over max_pct mitochondrial
    return counts, genes


@pytest.mark.parametrize("name", sorted(catalog.discover_recipes()))
def test_the_recipe_runs(name, tmp_path):
    """THE acceptance criterion. Not that the YAML parses — that the steps execute and the run
    reports ok, on data with enough structure to be worth measuring.

    WHICH DATA IS DERIVED FROM THE RECIPE'S OWN DECLARATION, not from a list of names here. A
    recipe whose `vocab.unmet` still asks for `genes` on a `clusters` shape is asking for a fact
    the point cloud does not have, and running it there is running an ILLEGAL cell — precisely
    what the calculus exists to refuse at plan time. It used to be tolerable because `qc` and
    `rank_genes` degrade on a nameless matrix and write a note; `filter_genes` and `filter_mito`
    REFUSE (`prep._genes_or_refuse`), so `preprocess` cannot degrade and the tolerance ran out.
    Measured 2026-08-16: three recipes take this branch — `preprocess`, `qc`, `markers` — and the
    other eight are unaffected, `contrast` included (it asks for `conditions`, not `genes`).
    """
    pytest.importorskip("manylatents")
    # AND THE OMICS EXTRA, because every bundled recipe but `qc` now declares a prep block and
    # the prep executors delegate to `manylatents.singlecell.preprocessing`. CI installs
    # `manylatents` and not `manylatents[omics]`, so without this the sweep reports nine
    # recipes erroring on an INSTALL problem and reads as nine broken recipes.
    #
    # That this guard became necessary is itself a product fact and is filed as manyruns#68:
    # CLAUDE.md states a bare `uv sync` runs every bundled recipe, and since the cutover that
    # is only true with the extra. The guard makes the SUITE honest; it does not make the
    # promise true.
    pytest.importorskip("manylatents.singlecell.preprocessing",
                        reason="every bundled recipe now declares prep steps ([omics] extra)")
    from manyruns.pipeline import runner

    recipe = catalog.load_recipe(name)
    if "genes" in vocab.unmet(recipe, "clusters"):
        X, genes = _scrna_counts()
        counts = X
    else:
        X, genes, counts = _point_cloud(), None, None

    res = runner.run_manylatents(recipe, tmp_path, array=X, counts=counts, genes=genes,
                                 seed=0, fast_dev_run=True)

    if name in UPSTREAM_BLOCKED:
        failed = [s for s in res["steps"] if s["outcome"] == "error"]
        assert failed, f"{name} is listed as upstream-blocked but every step passed — unlist it"
        return

    # A recipe DECLARING A STUB cannot report ok, by construction: a stub declines
    # (`pipeline/stubs.py`), which is `outcome="skipped"`. Derived from the registry rather than
    # a second hand-kept list, so implementing a stub moves its recipe back into the criterion
    # below with no edit here. What is asserted instead is the property that matters — every
    # stub SKIPPED with a reason, and nothing ERRORED, which is what makes such a recipe usable
    # for driving the app rather than a broken one.
    from manyruns.pipeline import stubs

    declared_stubs = [s["name"] for s in recipe["steps"] if s["name"] in stubs.names()]
    if declared_stubs:
        by_name = {s["name"]: s for s in res["steps"]}
        for stub in declared_stubs:
            assert by_name[stub]["outcome"] == "skipped", (stub, by_name[stub])
            assert by_name[stub]["detail"], f"{stub} declined without saying why"
        assert not [s for s in res["steps"] if s["outcome"] == "error"], (
            f"{name} declares stubs, which SKIP — an error is a real failure: "
            f"{[(s['name'], s['detail']) for s in res['steps'] if s['outcome'] == 'error']}")
        return
    # THE PREAMBLE STANDING DOWN IS A PASS, and only the preamble. Every bundled recipe declares
    # `normalize → transform` since the cutover, and the point-cloud fixture is never counts and
    # is signed. Both steps say so and stand down: a library size is meaningless without counts,
    # and `log1p` of a negative is NaN. That is what the loader used to do silently behind
    # `looks_like_counts`; the change is that it is on the record.
    #
    # A PASS, not a requirement — which is the half `_scrna_counts` now exercises. On the counts
    # fixture the same two steps RUN (measured: `normalize.target_sum` and `transform.method`
    # reach the g-vector), so this block reads "nothing skipped" there. Both branches through one
    # assertion, because the property is about which steps MAY skip, not about how many did.
    #
    # Narrow on purpose. The names are pinned, so a THIRD step learning to skip still fails this,
    # and each must still say why — a skip with an empty `detail` is the silence being reinvented.
    PREAMBLE = {"normalize", "transform"}
    stood_down = [s for s in res["steps"] if s["outcome"] == "skipped"]
    assert all(s["name"] in PREAMBLE and s["detail"] for s in stood_down), [
        (s["name"], s["detail"]) for s in stood_down]
    assert res["ok"] is True or stood_down, [
        (s["name"], s["outcome"], s["detail"]) for s in res["steps"]]
    assert not [s for s in res["steps"] if s["outcome"] == "error"], [
        (s["name"], s["detail"]) for s in res["steps"] if s["outcome"] == "error"]
    assert len(res["g_vector"]) > 20, "a run that emits nothing measurable is not a recipe"


def test_adding_recipes_grows_the_dropdown_not_the_top_menu():
    """The property that makes a catalog possible at all. The old menu appended one row per
    recipe, so a scientific question and a raw recipe name sat side by side and the list grew
    without bound. Recipes now carry their own `question:`, and everything but the recommended
    one sits behind a single agnostic entry."""
    from manyruns import narrate
    from manyruns.narrate import Observation

    obs = Observation(shape="clusters", modality="scrna", conditions=None, n_timepoints=0,
                      n_obs=500, n_vars=2000, source="x", signals=())
    legal = catalog.discover_recipes()
    offered = narrate.offer(obs, available=legal)

    # Measured 2026-08-16 at 11 recipes: 11 offered, 2 (`cluster`, `embed`) `recommended` for a
    # clusters shape, because `recommended` is a predicate over `suits:` and is true of more than
    # one. The menu shows the HEAD. The comment this replaces said "five at 8 recipes" and was
    # already wrong before `preprocess` landed — which cannot have moved it either way, since
    # `suits: []` makes a recipe unrecommendable by construction.
    recommended = [o for o in offered if o.get("recommended")]
    assert len(recommended) > 1, "this test is vacuous unless several recipes suit the shape"

    suited = (recommended or offered)[:1]
    assert len(suited) == 1, "the top menu must not grow with the catalog"
    assert len(offered) >= 6, "the dropdown should carry the whole legal set"
