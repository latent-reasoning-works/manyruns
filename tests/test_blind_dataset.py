"""`MANYRUNS_BLIND_DATASET`: true dataset names are hidden where a run or a prompt leaves the process."""
from __future__ import annotations

import json

import pytest

from manyruns import blind, narrate, store
from manyruns.narrate import Observation
from manyruns.pipeline import runner
from manyruns.tui.state import DataEntry

_BLINDED = "ds-5568e6d0fb02"

#: The spec input of the store scenarios.
_RESULTS = {"run_id": "r1", "dataset_name": "pbmc3k", "dataset": "dataset:pbmc3k",
            "data_kwargs": {"path": "/data/pbmc3k.h5ad"}}
_EXTRA = {"source": "pbmc3k"}

#: What the UNMODIFIED `store.append` wrote for `_RESULTS` and `_EXTRA`, captured before
#: blinding was added. Off must stay byte-identical to it. The model seam later added the core
#: field `non_commercial_steps` after `caveats`, which every row now carries.
_GOLDEN = (
    '{"schema_version":1,"run_id":"r1","spec_id":null,"parent":null,'
    '"dataset":"dataset:pbmc3k","dataset_name":"pbmc3k",'
    '"data_kwargs":{"path":"/data/pbmc3k.h5ad"},"at":null,"recipe":null,"engine":null,'
    '"seed":null,"tool_version":null,"ok":null,"complete":null,"caveats":[],'
    '"non_commercial_steps":[],"steps":[],"g_vector":{},"source":"pbmc3k"}\n'
)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("MANYRUNS_BLIND_DATASET", "1")


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv("MANYRUNS_BLIND_DATASET", raising=False)


def _entry():
    return DataEntry(name="pbmc3k", kind="bundled",
                     obs=Observation(shape="single", source="pbmc3k"), dataset="pbmc3k")


# ── the switch and the name ──────────────────────────────────────────────────
@pytest.mark.parametrize("value, expected", [
    (None, False), ("", False), ("0", False), ("false", False), (" OFF ", False), ("no", False),
    ("1", True), ("yes", True),
])
def test_enabled_table(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("MANYRUNS_BLIND_DATASET", raising=False)
    else:
        monkeypatch.setenv("MANYRUNS_BLIND_DATASET", value)
    assert blind.enabled() is expected


def test_blind_name_known_value():
    assert blind.blind_name("pbmc3k") == _BLINDED


def test_blind_name_idempotent():
    assert blind.blind_name(_BLINDED) == _BLINDED


# ── the side file ────────────────────────────────────────────────────────────
def test_side_file_round_trip(tmp_path):
    assert blind.side_path(tmp_path) == tmp_path / "blind_names.jsonl"
    assert blind.remember("pbmc3k", out_dir=tmp_path) == _BLINDED
    assert blind.remember("pbmc3k", out_dir=tmp_path) == _BLINDED
    lines = blind.side_path(tmp_path).read_text().splitlines()
    assert len(lines) == 1
    assert list(json.loads(lines[0])) == ["schema_version", "blinded", "dataset_name"]
    assert json.loads(lines[0]) == {"schema_version": 1, "blinded": _BLINDED,
                                    "dataset_name": "pbmc3k"}
    assert blind.unblind(_BLINDED, out_dir=tmp_path) == "pbmc3k"
    assert blind.unblind("ds-000000000000", out_dir=tmp_path) == "ds-000000000000"


def test_blinded_name_is_not_remembered(tmp_path):
    assert blind.remember(_BLINDED, out_dir=tmp_path) == _BLINDED
    assert not blind.side_path(tmp_path).exists()


def test_unblind_rows_copies(tmp_path):
    blind.remember("pbmc3k", out_dir=tmp_path)
    rows = [{"dataset_name": _BLINDED}, {"dataset": _BLINDED, "annotator": "a1"},
            {"dataset_name": None, "dataset": 3}]
    out = blind.unblind_rows(rows, out_dir=tmp_path)
    assert out == [{"dataset_name": "pbmc3k"}, {"dataset": "pbmc3k", "annotator": "a1"},
                   {"dataset_name": None, "dataset": 3}]
    assert rows[0]["dataset_name"] == _BLINDED
    assert rows[1]["dataset"] == _BLINDED


# ── the run row ──────────────────────────────────────────────────────────────
def test_store_named_fields_hold_no_true_name(on, tmp_path):
    store.append(dict(_RESULTS), out_dir=tmp_path, extra=dict(_EXTRA))
    line = store.index_path(tmp_path).read_text()
    assert "pbmc3k" not in line
    row = json.loads(line)
    assert row["dataset_name"] == _BLINDED
    assert row["dataset"] == f"dataset:{_BLINDED}"
    assert row["source"] == _BLINDED
    assert row["data_kwargs"] == {"path": f"/data/{_BLINDED}.h5ad"}
    side = [json.loads(s) for s in blind.side_path(tmp_path).read_text().splitlines()]
    assert len(side) == 1
    assert side[0]["dataset_name"] == "pbmc3k"


def test_store_does_not_rewrite_steps(on, tmp_path):
    """Step records hold artifact paths that `artifacts.complete()` reads back."""
    steps = [{"name": "pca", "artifact": "outputs/pbmc3k/pca.npy"}]
    store.append({**_RESULTS, "steps": steps}, out_dir=tmp_path, extra=dict(_EXTRA))
    (row,) = list(store.read(tmp_path))
    assert row["steps"] == steps
    assert row["dataset_name"] == _BLINDED


def test_store_off_is_byte_identical(off, tmp_path):
    store.append(dict(_RESULTS), out_dir=tmp_path, extra=dict(_EXTRA))
    assert store.index_path(tmp_path).read_text() == _GOLDEN
    assert not blind.side_path(tmp_path).exists()


def test_identity_ignores_blinding(monkeypatch):
    recipe = {"name": "embed", "steps": [{"name": "pca", "group": "latent"}]}
    kwargs = dict(engine="mock", seed=0, dataset="pbmc3k", dataset_name="pbmc3k")
    monkeypatch.setenv("MANYRUNS_BLIND_DATASET", "1")
    blinded = runner.identity(recipe, **kwargs)
    monkeypatch.delenv("MANYRUNS_BLIND_DATASET")
    clear = runner.identity(recipe, **kwargs)
    assert blinded[1] == clear[1]


# ── the model-facing payloads ────────────────────────────────────────────────
def test_prompt_holds_no_true_name(on):
    text = narrate.record_for_prompt(entry={"name": "pbmc3k"}, said=["kept pbmc3k attempt 1"])
    assert f"dataset: {_BLINDED}" in text
    assert "pbmc3k" not in text


def test_prompt_off_is_unchanged(off):
    text = narrate.record_for_prompt(entry={"name": "pbmc3k"}, said=["kept pbmc3k attempt 1"])
    assert text.startswith("dataset: pbmc3k\n")
    assert "kept pbmc3k attempt 1" in text


def test_entry_payload_holds_no_true_name(on):
    result = _entry().to_dict()
    assert "pbmc3k" not in json.dumps(result)
    assert result["name"] == _BLINDED
    assert result["dataset"] == _BLINDED
    assert result["obs"]["source"] == _BLINDED


def test_entry_payload_blinds_the_path(on, tmp_path):
    entry = DataEntry(name="pbmc3k", kind="file", path=tmp_path / "pbmc3k.h5ad",
                      obs=Observation(shape="single", source=str(tmp_path / "pbmc3k.h5ad")))
    result = entry.to_dict()
    assert "pbmc3k" not in result["path"]
    assert "pbmc3k" not in result["obs"]["source"]
    assert result["path"].endswith(f"{_BLINDED}.h5ad")


def test_entry_fields_unchanged_in_memory(on):
    entry = _entry()
    entry.to_dict()
    assert entry.obs.source == "pbmc3k"
    assert entry.name == "pbmc3k"
    assert entry.dataset == "pbmc3k"


def test_entry_payload_off_is_unchanged(off):
    result = _entry().to_dict()
    assert (result["name"], result["dataset"], result["obs"]["source"]) == ("pbmc3k",) * 3


# ── scoring after a blinded run ──────────────────────────────────────────────
def test_scoring_after_blinding(tmp_path):
    """A verdict on a blinded run and a label on the true name meet after `unblind_rows`."""
    from manyruns import score, vocab

    blind.remember("pbmc3k", out_dir=tmp_path)
    probs = {t: 0.0 for t in vocab.TOPOLOGIES}
    probs["clusters"] = 0.9
    verdict_rows = [{"run_id": "r1", "dataset_name": _BLINDED, "dataset": None,
                     "extra": {"verdict": {"topology_probs": probs, "chooser": "human:a3",
                                           "bound": "none", "legal_id": "a37fab6bef12",
                                           "model": None}}}]
    truth_rows = [{"dataset": "pbmc3k", "annotator": "human:a1", "topology": ["clusters"]},
                  {"dataset": "pbmc3k", "annotator": "human:a2", "topology": ["clusters"]}]
    assert score.class_auprc(verdict_rows, truth_rows)["unmatched"] == 1
    out = score.class_auprc(blind.unblind_rows(verdict_rows, out_dir=tmp_path), truth_rows)
    assert out["unmatched"] == 0
    assert out["n"] == 1
    assert out["per_class"]["clusters"] == 1.0


def test_run_screen_prompt_holds_no_true_name(on):
    # The screen hands `record_for_prompt` the already-blinded `to_dict()`, so the true name in
    # what the loop said must be blinded by the screen, which still holds the entry itself.
    from manyruns.tui.run import RunScreen

    recipe = {"name": "embed", "steps": [{"name": "phate", "group": "latent", "params": {}}]}
    screen = RunScreen(recipe, entry=_entry(), concerns=("pbmc3k has few cells",))
    screen.said.append("loaded pbmc3k")
    context = screen.ask_context()
    assert "pbmc3k" not in context
    assert _BLINDED in context


def test_run_screen_prompt_off_keeps_the_name(off):
    from manyruns.tui.run import RunScreen

    recipe = {"name": "embed", "steps": [{"name": "phate", "group": "latent", "params": {}}]}
    screen = RunScreen(recipe, entry=_entry())
    screen.said.append("loaded pbmc3k")
    assert "loaded pbmc3k" in screen.ask_context()
