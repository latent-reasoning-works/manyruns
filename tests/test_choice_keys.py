"""`run --choice` names the time column and the condition column (#26).

A choice document may carry `time_key` and `condition_key`. `check_choice` refuses a key that
is not a column of the dataset, and a key of the wrong type, before anything is written. The
loader uses a given key before name detection, and the run row records the key that the run
used. Plain `run` has the matching `--condition-key` flag.

The runs here use the manylatents loading path with `modes.run` forwarded to the mock engine:
the loader reads the real file, and no geometry is computed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from manyruns import action_space, actions
from manyruns.action_space import Refused, check_choice, offer_sha256

CHOOSER = "prior:test.0123456789ab"
DATASET = "twoarms"

#: Two candidate condition columns. `condition` is found by name; `arm` is not. `day` is a
#: time name, so with no key given, name detection takes `day` as the time axis.
CONDITION = ["a", "b"] * 20
ARM = ["x"] * 10 + ["y"] * 30


def _write_h5ad(path: Path) -> Path:
    import anndata
    import pandas as pd

    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"condition": CONDITION, "arm": ARM, "day": [0, 1] * 20},
                       index=[f"c{i}" for i in range(40)])
    anndata.AnnData(X=rng.poisson(3.0, size=(40, 6)).astype("float32"), obs=obs).write_h5ad(path)
    return path


@pytest.fixture
def store(tmp_path, monkeypatch) -> Path:
    """An empty store, a catalog of one AnnData dataset, and a private home."""
    data = _write_h5ad(tmp_path / "twoarms.h5ad")
    configs = tmp_path / "datasets"
    configs.mkdir()
    (configs / f"{DATASET}.yaml").write_text(
        f"name: {DATASET}\n"
        f"handle: {{kind: path, ref: {json.dumps(str(data))}}}\n"
        "modality: scrna\nshape: case-control\ntopology: [clusters]\ndesign: case-control\n")
    monkeypatch.setenv("MANYRUNS_DATASET_DIR", str(configs))
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "store"
    root.mkdir()
    monkeypatch.chdir(root)
    return root


def _choice(store: Path, **keys) -> dict:
    assert store.is_dir()
    doc = json.loads(json.dumps(actions.enumerate_actions(
        "ledger", dataset=DATASET, gate=False, seed=42, store=store)))
    legal = [o["recipe"] for o in doc["offered"] if o["can_run"]]
    assert "embed" in legal, legal
    return {"schema_version": 1, "surface": "ledger", "dataset": DATASET,
            "offer_sha256": offer_sha256(doc), "chosen": {"recipe": "embed"},
            "chooser": CHOOSER, "bound": "logits", **keys}


def _forward_to_mock(monkeypatch) -> list[dict]:
    """`modes.run` sees the loaded request, then the mock engine runs it."""
    from manyruns import modes

    real = modes.run
    seen: list[dict] = []

    def forward(mode, req, *args, **kwargs):
        seen.append(req)
        return real(mode, {**req, "engine": "mock"}, *args, **kwargs)

    monkeypatch.setattr(modes, "run", forward)
    return seen


def _run_choice(store: Path, choice: dict, name: str, engine: str = "manylatents") -> int:
    from manyruns import app

    path = store.parent / f"{name}.json"
    path.write_text(json.dumps(choice))
    return app.main(["run", "--choice", str(path), "--engine", engine, "--project", name])


def _runs(store: Path) -> list[dict]:
    from manyruns import store as records

    return list(records.read(store / "outputs"))


# ── the acceptance test ──────────────────────────────────────────────────────────


def test_two_condition_keys_each_recorded_and_used(store, monkeypatch):
    seen = _forward_to_mock(monkeypatch)
    assert _run_choice(store, _choice(store, condition_key="condition"), "p1") == 0
    assert _run_choice(store, _choice(store, condition_key="arm"), "p2") == 0
    first, second = _runs(store)
    assert first["condition_key"] == "condition" and first["time_key"] is None
    assert second["condition_key"] == "arm" and second["time_key"] is None
    assert [r["label_kind"] for r in seen] == ["condition", "condition"]
    assert list(seen[0]["labels"]) == CONDITION
    assert list(seen[1]["labels"]) == ARM


def test_a_given_time_key_is_recorded_and_used(store, monkeypatch):
    seen = _forward_to_mock(monkeypatch)
    assert _run_choice(store, _choice(store, time_key="day"), "p1") == 0
    (run,) = _runs(store)
    assert run["time_key"] == "day" and run["condition_key"] is None
    assert seen[0]["label_kind"] == "time" and list(seen[0]["labels"]) == [0, 1] * 20


def test_with_no_key_name_detection_is_unchanged(store, monkeypatch):
    seen = _forward_to_mock(monkeypatch)
    assert _run_choice(store, _choice(store), "p1") == 0
    (run,) = _runs(store)
    assert run["time_key"] == "day" and run["condition_key"] is None
    assert list(seen[0]["labels"]) == [0, 1] * 20


def test_a_mock_run_reads_no_column(store):
    assert _run_choice(store, _choice(store, condition_key="arm"), "p1", engine="mock") == 0
    (run,) = _runs(store)
    assert run["condition_key"] is None and run["time_key"] is None


# ── the refusals, each before the first write ─────────────────────────────────────


def _refused(choice: dict, store: Path, *texts: str) -> None:
    with pytest.raises(Refused) as caught:
        check_choice(choice, store=store, seed=42)
    assert caught.value.exit_code == 2, caught.value.message
    for text in texts:
        assert text in caught.value.message, (text, caught.value.message)


@pytest.mark.parametrize("key", ["time_key", "condition_key"])
def test_a_key_that_is_not_a_column_is_refused(store, key, capsys):
    choice = _choice(store, **{key: "nosuch"})
    _refused(choice, store, key, "nosuch", "arm")
    with pytest.raises(SystemExit) as stopped:
        _run_choice(store, choice, "p1")
    assert stopped.value.code == 2
    assert "nosuch" in capsys.readouterr().err
    assert not (store / "outputs").exists()


@pytest.mark.parametrize("value", [3, "", ["arm"], None, True])
@pytest.mark.parametrize("key", ["time_key", "condition_key"])
def test_a_key_of_the_wrong_type_is_refused(store, key, value):
    _refused(_choice(store, **{key: value}), store, key)


def test_one_column_for_both_keys_is_refused(store):
    _refused(_choice(store, time_key="arm", condition_key="arm"), store, "arm")


def test_a_key_on_a_dataset_without_columns_is_refused(store, monkeypatch):
    monkeypatch.delenv("MANYRUNS_DATASET_DIR")
    doc = json.loads(json.dumps(actions.enumerate_actions(
        "ledger", dataset="swissroll", gate=True, seed=42, store=store)))
    choice = {"schema_version": 1, "surface": "ledger", "dataset": "swissroll",
              "offer_sha256": offer_sha256(doc), "chosen": {"recipe": "embed"},
              "chooser": CHOOSER, "bound": "logits", "condition_key": "label"}
    _refused(choice, store, "swissroll", "condition_key")


def test_review_keeps_the_keys(store):
    choice = _choice(store, time_key="day", condition_key="arm")
    reviewed = action_space.review(choice, ask=lambda c: "accept", human_id="a3")
    assert reviewed["time_key"] == "day" and reviewed["condition_key"] == "arm"
    check_choice(reviewed, store=store, seed=42)


# ── the loader and plain `run` ────────────────────────────────────────────────────


def test_label_key_of_uses_a_given_condition_key_before_detection(tmp_path):
    import anndata

    from manyruns.pipeline import loading

    obj = anndata.read_h5ad(_write_h5ad(tmp_path / "x.h5ad"))
    assert loading.label_key_of(obj) == ("day", "time")
    assert loading.label_key_of(obj, condition_key="arm") == ("arm", "condition")
    assert loading.label_key_of(obj, time_key="day", condition_key="arm") == ("day", "time")
    labels, kind = loading.labels_of(obj, condition_key="arm")
    assert kind == "condition" and list(labels) == ARM


def test_plain_run_takes_condition_key(store, monkeypatch):
    from manyruns import app

    seen = _forward_to_mock(monkeypatch)
    data = store.parent / "twoarms.h5ad"
    assert data.is_file()
    assert app.main(["run", str(data), "--recipe", "embed", "--engine", "manylatents",
                     "--condition-key", "arm", "--project", "plain"]) == 0
    (run,) = _runs(store)
    assert run["condition_key"] == "arm" and list(seen[0]["labels"]) == ARM
