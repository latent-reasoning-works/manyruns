"""`manyruns check` — the author's self-check, including the coverage pass.

Coverage runs the precondition calculus over (recipe × current datasets) so an author sees,
at check time, a recipe that no dataset can exercise — the exact gap a sweep would otherwise
prune silently. It is a WARN, not a FAIL: the recipe is valid, the data to run it is missing.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from manyruns import app


def _check(dataset_dir=None):
    return app.cmd_check(SimpleNamespace(dataset_dir=dataset_dir), argparse.ArgumentParser())


def test_coverage_warns_when_a_recipe_has_no_legal_dataset(tmp_path, capsys):
    # only a manifold dataset present → contrast (needs conditions) is unexercisable
    (tmp_path / "swiss.yaml").write_text(
        "name: swiss\nhandle: {kind: manylatents, ref: swissroll}\nshape: manifold\n")
    rc = _check(str(tmp_path))
    out = capsys.readouterr().out

    assert "coverage" in out
    assert "WARN  contrast" in out and "needs a shape providing ['conditions']" in out
    # `qc` and `markers` JOINED contrast at the cutover: both declare `("genes",)` in
    # `vocab.STEP_NEEDS` now, and `genes` is a fact of the BYTES — a `manylatents:swissroll`
    # handle declares none, so a swissroll-only directory exercises neither. Their WARN is
    # worded differently on purpose: "go find a shape providing ['genes']" is advice nobody
    # can act on, because no member of `vocab.SHAPES` provides it.
    #
    # `preprocess` is the third, and it is the one that makes the count worth keeping: the
    # scRNA preamble is the recipe a swissroll-only directory MOST looks like it should be able
    # to run, and it cannot — `filter_genes` and `filter_mito` read gene names.
    assert "WARN  qc" in out and "WARN  markers" in out and "WARN  preprocess" in out
    assert "comes from the data itself and not from any shape" in out
    assert "4 unexercisable" in out   # was 1 — contrast, plus the three that read gene names
    assert rc == 0, "an unexercisable recipe is a WARN, not a FAIL — exit stays 0"


def test_coverage_clears_when_a_case_control_dataset_exists(tmp_path, capsys):
    (tmp_path / "cohort.yaml").write_text(
        "name: cohort\nhandle: {kind: path, ref: /data/x.h5ad}\nshape: case-control\n")
    _check(str(tmp_path))
    out = capsys.readouterr().out

    assert "ok    contrast" in out and "runs on case-control" in out
    assert "unexercisable" not in out


def test_an_invalid_file_is_a_FAIL_and_flips_the_exit_code(tmp_path, capsys):
    (tmp_path / "bad.yaml").write_text(
        "name: bad\nhandle: {kind: s3, ref: b}\nshape: nonsense\n")
    rc = _check(str(tmp_path))
    out = capsys.readouterr().out

    assert "FAIL  bad" in out and "INVALID" in out
    assert rc == 1, "a real validation failure must flip the exit code, unlike a coverage WARN"


def test_the_bundled_set_is_valid_but_flags_contrast(capsys):
    rc = _check()  # the shipped recipes + datasets
    out = capsys.readouterr().out

    assert "all valid" in out           # nothing is broken
    assert "WARN  contrast" in out      # …but the starter data can't exercise contrast
    assert rc == 0


# ── `preprocess`, the scRNA preamble as a recipe ─────────────────────────────
def test_preprocess_is_a_recipe_and_validates_like_any_other():
    """`preprocess` is the backlog's "normal scRNA preamble in one line". As a STEP it would be
    an expansion mechanism — one declared name silently becoming five — which is a second
    abstraction over the recipe. As a recipe the standard order is readable on the page, the
    defaults are forkable, and `check_recipe` validates it like the other ten.

    THE ORDER IS THE ASSERTION, not just the membership. Filters first because a narrowing
    invalidates every derived array (`runner.DERIVED_SLOTS`), and `normalize` before `transform`
    because `_step_normalize` reads the counts while `_step_transform` reads the working matrix
    — reversed, the log1p is silently discarded by a library-size rescale of log-ratios.
    """
    from manyruns import catalog

    recipe = catalog.load_recipe("preprocess")

    assert catalog.check_recipe(recipe, "preprocess") == []
    assert [s["name"] for s in recipe["steps"]] == [
        "filter_cells", "filter_genes", "filter_mito", "normalize", "transform"]
    assert all(s["group"] == "prep" for s in recipe["steps"])
    # `suits: []` and no `claims:` — the same abstention `qc` records, checked rather than
    # described: a preamble answers nobody's question and asserts no structure.
    assert recipe["suits"] == [] and "claims" not in recipe


def test_the_preamble_names_the_implementation_that_actually_runs():
    """`via: manylatents`, and this is not cosmetic. The compute landed in manylatents-omics#61;
    `pipeline/prep.py` imports no scanpy at all. `via:` is what `vocab.standins` compares against
    the engine name to raise a "not the canonical implementation" caveat, so `via: scanpy` here
    would aim a real caveat at a library the run never loads."""
    from manyruns import catalog

    recipe = catalog.load_recipe("preprocess")
    assert {s.get("via") for s in recipe["steps"]} == {"manylatents"}


def test_no_bundled_prep_step_declares_a_parameter_its_executor_cannot_read():
    """THE SILENT-IGNORE GUARD, and the reason `preprocess`'s `normalize` carries no `method:`.

    A declared parameter that nothing reads is worse than a missing one: it appears on the page,
    is copied into a fork, is quoted in a review, and changes nothing. `prep._step_normalize`
    takes `target_sum` and `force`; an author following the Scanpy tutorial reaches for
    `method: library`, and the run would report `ok` having ignored it.

    Checked by SOURCE rather than by a hand-kept table of readable keys, because a table would
    be the second home for a fact `prep.py` already states — a name that never appears in its
    executor's body cannot be read by it. Measured 2026-08-16: 19 bundled prep steps across 8
    recipes declare 20 params, and every one of the 20 is named by its executor.
    """
    import inspect

    from manyruns import catalog
    from manyruns.pipeline import prep

    checked = 0
    for name in catalog.discover_recipes():
        for step in catalog.load_recipe(name)["steps"]:
            if step.get("group") != "prep":
                continue
            body = inspect.getsource(prep._PREP_STEPS[step["name"]])
            for key in (step.get("params") or {}):
                assert f'"{key}"' in body, (
                    f"{name}: {step['name']} declares {key!r} and its executor never names it — "
                    "a param nothing reads is a silent lie on the page")
                checked += 1
    assert checked == 20, "the bundled prep params moved; re-read the measurement above"


def test_the_preamble_asks_the_short_half_like_every_other_row():
    """The ledger draws `<ask> — <gloss>` in two weights and folds the ask column, so a question
    written the other way round overflows the narrow half. Measured with `split_question` itself
    rather than by eye: 19 characters of ask against 65 of gloss."""
    from manyruns import narrate
    from manyruns.catalog import load_recipe

    ask, gloss = narrate.split_question(load_recipe("preprocess")["question"])
    assert ask == "Clean this up first"
    assert gloss.startswith("filter, normalize and transform")
    assert len(ask) < len(gloss)


def _downloadable_dataset():
    return {"name": "sample", "shape": "clusters", "handle": {
        "kind": "path", "ref": "data/sample.h5ad", "sha256": "a" * 64,
        "bytes": 12, "url": "https://example.org/sample.h5ad",
    }}


@pytest.mark.parametrize(("field", "value"), [
    (None, []), (None, "tutorial"), (None, 42), (None, True), (None, None),
    ("handle", "path"), ("handle", []), ("handle", None),
    ("handle.ref", ["sample.h5ad"]), ("handle.ref", 42), ("handle.ref", None),
    ("handle.ref", ""), ("handle.ref", " \t"),
    ("handle.provides", "genes"), ("handle.provides", {}), ("handle.provides", None),
    ("shape", 42), ("shape", []), ("shape", {}),
    ("topology", {}), ("topology", None),
    ("params", []), ("params", None),
    ("source", "10x tutorial"), ("source", ["tutorial"]), ("source", None),
])
@pytest.mark.parametrize("boundary", ["check", "load"])
def test_malformed_dataset_structure_is_reported(field, value, boundary, tmp_path):
    import yaml

    from manyruns import catalog

    ds = {"name": "sample", "shape": "clusters",
          "handle": {"kind": "path", "ref": "sample.h5ad"}}
    if field is None:
        ds = value
    elif field.startswith("handle."):
        ds["handle"][field.split(".")[1]] = value
    else:
        ds[field] = value

    if boundary == "check":
        problems = catalog.check_dataset(ds, "sample")
        assert isinstance(problems, list) and problems
        assert all(isinstance(problem, str) for problem in problems)
    else:
        (tmp_path / "sample.yaml").write_text(yaml.safe_dump(ds))
        assert tmp_path.is_dir()
        with pytest.raises(ValueError, match="invalid dataset 'sample'"):
            catalog.load_dataset("sample", config_dir=tmp_path)


def test_dataset_validation_reports_all_independent_structure_errors():
    from manyruns import catalog

    problems = catalog.check_dataset({
        "handle": {"kind": "path", "ref": ["sample.h5ad"], "provides": "genes"},
        "shape": 42, "topology": {}, "params": [], "source": "tutorial",
    })
    for field in ("ref", "provides", "shape", "topology", "params", "source"):
        assert any(field in problem for problem in problems), (field, problems)


def test_check_reports_every_malformed_dataset_file(tmp_path, monkeypatch, capsys):
    import yaml

    from manyruns import catalog

    monkeypatch.delenv("MANYRUNS_RECIPE_DIR", raising=False)
    declarations = {
        "bad_list": ["dataset"],
        "bad_scalar": 42,
        "bad_source": {"name": "bad_source", "shape": "clusters",
                       "handle": {"kind": "path", "ref": "sample.h5ad"},
                       "source": "tutorial"},
        "good": {"name": "good", "shape": "clusters",
                 "handle": {"kind": "path", "ref": "sample.h5ad"}},
    }
    for name, ds in declarations.items():
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(ds))
    assert tmp_path.is_dir()
    assert catalog.recipe_dir().is_dir()

    assert _check(str(tmp_path)) == 1
    out = capsys.readouterr().out
    for name in ("bad_list", "bad_scalar", "bad_source"):
        assert f"FAIL  {name}" in out
    assert "ok    good" in out
    assert "3 INVALID" in out
    assert "AttributeError" not in out and "TypeError" not in out


def test_all_bundled_datasets_preserve_valid_structure(monkeypatch):
    """Preservation: stricter structure checks must accept every shipped declaration."""
    from manyruns import catalog

    monkeypatch.delenv("MANYRUNS_DATASET_DIR", raising=False)
    assert catalog.dataset_dir().is_dir()
    names = catalog.discover_datasets()
    assert names
    for name in names:
        ds = catalog.load_dataset(name)
        assert catalog.check_dataset(ds, name) == [], name


@pytest.mark.parametrize("source", [
    {}, {"cite": "A local sample"}, {"url": "https://example.org/citation"},
    {"kind": "none"}, {"kind": "tutorial", "url": "https://example.org/citation"},
])
def test_dataset_source_mapping_preserves_optional_kind(source):
    """Preservation: dataset citations do not have to declare a recipe's source kind."""
    from manyruns import catalog

    ds = _downloadable_dataset()
    ds["source"] = source
    assert catalog.check_dataset(ds) == []


def test_pbmc3k_declares_executable_download_pins(monkeypatch):
    from manyruns import catalog

    monkeypatch.delenv("MANYRUNS_DATASET_DIR", raising=False)
    assert catalog.dataset_dir().is_dir()
    handle = catalog.load_dataset("pbmc3k")["handle"]
    assert handle["sha256"] == "89a96f1beaa2dd83a687666d3f19a4513ac27a2a2d12581fcd77afed7ea653a1"
    assert handle["bytes"] == 5855727
    assert handle["url"] == "https://exampledata.scverse.org/scanpy/pbmc3k_raw.h5ad"


@pytest.mark.parametrize(("key", "value"), [
    ("sha256", None), ("sha256", "A" * 64), ("sha256", "g" * 64),
    ("sha256", "a" * 63), ("sha256", 123),
    ("bytes", None), ("bytes", True), ("bytes", False), ("bytes", 0),
    ("bytes", -1), ("bytes", 1.2), ("bytes", "12"),
    ("url", None), ("url", 42), ("url", "http://example.org/file"),
    ("url", "https://" + "/file"), ("url", "https://" + "user:secret@example.org/file"),
    ("url", "https://example.org:bad/file"), ("url", "https://[bad/file"),
    ("url", "https://bad host/file"), ("url", "https://example.org/\nfile"),
    ("url", "https://example.org/données.h5ad"),
    ("url", "https://exa\u200bmple.org/file"),
    ("url", "https://example..org/file"), ("url", "https://.example.org/file"),
    ("url", "https://" + "a" * 64 + ".org/file"),
    ("kind", "manylatents"), ("ref", "/tmp/sample.h5ad"),
    ("ref", "synthetic:time-course"), ("ref", "../sample.h5ad"),
    ("ref", "~/sample.h5ad"), ("ref", "data/"), ("ref", 42),
])
def test_download_declarations_reject_malformed_fields(key, value):
    from manyruns.catalog import check_dataset

    ds = _downloadable_dataset()
    ds["handle"][key] = value
    assert check_dataset(ds), (key, value)


@pytest.mark.parametrize("url", [
    "https://example.org/donn%C3%A9es.h5ad", "https://xn--bcher-kva.org/file",
    "https://127.0.0.1:8443/file", "https://[::1]:8443/file", "https://example.org./file",
])
def test_download_url_accepts_encoded_paths_and_usable_hosts(url):
    from manyruns import catalog, datasetfetch

    ds = _downloadable_dataset()
    ds["handle"]["url"] = url
    assert catalog.check_dataset(ds) == []
    assert datasetfetch.can_fetch(ds)


@pytest.mark.parametrize("missing", ["sha256", "bytes"])
def test_download_url_requires_both_pins(missing):
    from manyruns.catalog import check_dataset

    ds = _downloadable_dataset()
    del ds["handle"][missing]
    assert check_dataset(ds)


def test_manual_data_pins_and_legacy_handles_remain_valid():
    """Preservation: declarations need no download URL or new fields."""
    from manyruns.catalog import check_dataset

    ds = _downloadable_dataset()
    assert check_dataset(ds) == []
    del ds["handle"]["url"]
    ds["handle"]["ref"] = "/manual/sample.h5ad"
    assert check_dataset(ds) == []
    del ds["handle"]["sha256"]
    del ds["handle"]["bytes"]
    assert check_dataset(ds) == []


@pytest.mark.parametrize("dims", [
    None, [], 12, {}, {"n_samples": 100}, {"n_features": 3},
    {"n_samples": True, "n_features": 3}, {"n_samples": 0, "n_features": 3},
    {"n_samples": -1, "n_features": 3}, {"n_samples": "100", "n_features": 3},
    {"n_samples": 100, "n_features": False}, {"n_samples": 100, "n_features": 0},
    {"n_samples": 100, "n_features": -3}, {"n_samples": 100, "n_features": 3.5},
])
def test_generator_dims_require_two_positive_integer_counts(dims):
    from manyruns.catalog import check_dataset

    ds = {"shape": "manifold", "handle": {"kind": "manylatents", "ref": "swissroll"},
          "dims": dims}
    assert any("dims" in error for error in check_dataset(ds))


def test_dims_are_only_meaningful_for_a_manylatents_handle():
    from manyruns.catalog import check_dataset

    ds = _downloadable_dataset()
    ds["dims"] = {"n_samples": 100, "n_features": 3}
    assert any("dims" in error for error in check_dataset(ds))
    ds["handle"] = {"kind": "manylatents", "ref": "swissroll"}
    assert check_dataset(ds) == []


def test_catalog_validation_and_loading_never_fetch_or_verify_data_bytes(tmp_path, monkeypatch):
    """Preservation: validating a declaration must remain cheap, local and metadata-only."""
    import yaml

    from manyruns import catalog

    ds = _downloadable_dataset()
    (tmp_path / "sample.yaml").write_text(yaml.safe_dump(ds))

    def no_data_io(*args, **kwargs):
        pytest.fail("catalog validation attempted data I/O or verification")

    monkeypatch.setattr("urllib.request.urlopen", no_data_io)
    monkeypatch.setattr("hashlib.sha256", no_data_io)
    monkeypatch.setattr("manyruns.shell.dataset_ref_path", no_data_io)
    assert tmp_path.is_dir()
    assert catalog.load_dataset("sample", config_dir=tmp_path) == ds
    assert catalog.check_dataset(ds) == []
