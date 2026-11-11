import hashlib
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from manyruns import artifacts
from manyruns.pipeline import steps, tables


@pytest.fixture
def context(tmp_path):
    return {"out_dir": tmp_path / "out", "run_id": "r1", "step_index": 3}


def _publish(context, state, g, **kwargs):
    return tables.publish(
        "step",
        {"t": pd.DataFrame({"value": [1.0, 2.0], "gene": ["g0", "g1"]})},
        required={"t": ["gene", "value"]},
        state=state,
        g=g,
        ctx=context,
        **kwargs,
    )


def test_publish_saves_records_and_hashes(context):
    state, g = {}, {"existing": 7}
    saved = _publish(context, state, g)
    path = context["out_dir"] / "state" / "r1" / "03-step_t.tsv"
    assert saved == state["tables"] == {"t": str(path)}
    assert path.read_text() == "value\tgene\n1.0\tg0\n2.0\tg1\n"
    assert g == {
        "existing": 7,
        "step.t.rows": 2,
        "step.t.columns": ["gene", "value"],
        "step.t.sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_publish_refuses_missing_columns_and_changes_nothing(context):
    state, g = {"tables": {"previous": "previous.tsv"}}, {"existing": 7}
    before = deepcopy((state, g))
    with pytest.raises(steps._StepSkipped, match="missing columns.*value"):
        tables.publish(
            "step",
            {"valid": pd.DataFrame({"gene": ["g0"]}), "t": pd.DataFrame({"gene": ["g1"]})},
            required={"valid": ["gene"], "t": ["gene", "value"]},
            state=state,
            g=g,
            ctx=context,
        )
    assert (state, g) == before
    assert not context["out_dir"].exists()


def test_publish_refuses_a_persisted_slot_name(context):
    state, g = {}, {}
    with pytest.raises(steps._StepSkipped, match="has an artifact name conflict; rename the table"):
        tables.publish(
            "step", {"emb": pd.DataFrame({"value": [1]})},
            required={"emb": ["value"]}, state=state, g=g, ctx=context,
        )
    assert state == g == {}
    assert not context["out_dir"].exists()


def test_publish_refuses_a_reserved_name(context):
    state, g = {}, {}
    with pytest.raises(steps._StepSkipped, match="artifact name conflict"):
        _publish(context, state, g, reserved=iter(["t"]))
    assert state == g == {}
    assert not context["out_dir"].exists()


def test_publish_refuses_an_existing_extra_name(context):
    state, g = {"extras": {"t": [1, 2]}}, {}
    before = deepcopy(state)
    with pytest.raises(steps._StepSkipped, match="artifact name conflict"):
        _publish(context, state, g)
    assert state == before
    assert g == {}
    assert not context["out_dir"].exists()


@pytest.mark.parametrize("missing", ["out_dir", "step_index"])
def test_publish_refuses_without_out_dir_or_step_index(context, missing):
    out_dir = context["out_dir"]
    del context[missing]
    state, g = {}, {}
    with pytest.raises(steps._StepSkipped, match="supply out_dir and step_index"):
        _publish(context, state, g)
    assert state == g == {}
    assert not out_dir.exists()


def test_publish_replaces_the_tables_dict(context):
    before = {"previous": "previous.tsv"}
    state = {"tables": before}
    saved = _publish(context, state, {})
    assert before == {"previous": "previous.tsv"}
    assert state["tables"] is not before
    assert state["tables"] == before | saved


@pytest.mark.parametrize("failure", ["save", "hash"])
def test_publish_failure_on_second_table_changes_no_state_or_measurements(
    context, monkeypatch, failure
):
    state, g = {"tables": {"previous": "previous.tsv"}}, {"existing": 7}
    before = deepcopy((state, g))
    original_save = artifacts.save_table
    original_read = Path.read_bytes

    def save(out_dir, run_id, index, step, name, df):
        if name == "second":
            raise OSError("simulated write failure")
        return original_save(out_dir, run_id, index, step, name, df)

    def read(path):
        if path.name.endswith("_second.tsv"):
            raise OSError("simulated read failure")
        return original_read(path)

    if failure == "save":
        monkeypatch.setattr(artifacts, "save_table", save)
    else:
        monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(steps._StepSkipped, match="cannot save table" if failure == "save" else "hash"):
        tables.publish(
            "step",
            {name: pd.DataFrame({"value": [1]}) for name in ("first", "second")},
            required={name: ["value"] for name in ("first", "second")},
            state=state, g=g, ctx=context,
        )
    assert (state, g) == before


def test_publish_no_tables_needs_no_context_and_changes_nothing():
    before = {"previous": "previous.tsv"}
    state, g = {"tables": before}, {"existing": 7}
    assert tables.publish("step", {}, required={}, state=state, g=g, ctx={}) == {}
    assert state["tables"] is before
    assert g == {"existing": 7}
