"""`manyruns claim` and `manyruns.verdict`: a structure-class verdict appended to the run store."""
from __future__ import annotations

import datetime as _dt
import json

import pytest

from manyruns import app, commands, decisions, score, store, verdict, vocab

_SIX = {t: 0.0 for t in vocab.TOPOLOGIES}


def _probs(**given):
    named = {k.replace("_", "-"): v for k, v in given.items()}
    return {t: float(named.get(t, 0.0)) for t in vocab.TOPOLOGIES}


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    """One run row `r1` in `outputs/index.jsonl`, with the working directory at `tmp_path`."""
    store.append({"run_id": "r1", "spec_id": "s1", "dataset_name": "pbmc3k", "seed": 7},
                 out_dir=tmp_path / "outputs")
    monkeypatch.chdir(tmp_path)
    return tmp_path / "outputs"


def _lines(outputs):
    return store.index_path(outputs).read_text().splitlines()


def _bytes(outputs):
    return store.index_path(outputs).read_bytes()


def _exit2(argv, capsys):
    with pytest.raises(SystemExit) as info:
        app.main(argv)
    assert info.value.code == 2
    return capsys.readouterr().err


# ── parse_topology ───────────────────────────────────────────────────────────
def test_parse_unnamed_classes_are_zero():
    out = verdict.parse_topology("clusters=0.7,cycle=0.2")
    assert out == {"clusters": 0.7, "single-trajectory": 0.0, "multi-branching": 0.0,
                   "archetypal": 0.0, "cycle": 0.2, "surface": 0.0}
    assert list(out) == list(vocab.TOPOLOGIES)


@pytest.mark.parametrize("text, token", [
    ("cluster=0.7", "cluster"),
    ("clusters=high", "high"),
    ("clusters=1.4", "1.4"),
    ("clusters=nan", "nan"),
    ("clusters=0.2,clusters=0.3", "clusters"),
    ("", "no NAME=P pair"),
])
def test_parse_refuses_each_bad_token(text, token):
    with pytest.raises(ValueError) as info:
        verdict.parse_topology(text)
    assert token in str(info.value)


def test_parse_does_not_normalize():
    out = verdict.parse_topology(",".join(f"{t}=1" for t in vocab.TOPOLOGIES))
    assert sum(out.values()) == 6.0


# ── legal_id ─────────────────────────────────────────────────────────────────
def test_legal_id_known_value():
    assert vocab.VERSION == "1"
    assert verdict.legal_id(vocab.TOPOLOGIES) == "a37fab6bef12"


def test_legal_id_order_free():
    assert verdict.legal_id(["cycle", "clusters"]) == verdict.legal_id(["clusters", "cycle"])
    assert verdict.legal_id(["cycle", "clusters"]) == "4b3f0cfed9e1"


def test_legal_id_set_sensitive():
    assert verdict.legal_id(vocab.TOPOLOGIES[:5]) == "761bf121ce02"


def test_legal_id_reads_vocab_version(monkeypatch):
    monkeypatch.setattr(vocab, "VERSION", "2")
    assert verdict.legal_id(vocab.TOPOLOGIES) != "a37fab6bef12"


def test_legal_id_matches_the_decision_row_formula():
    """One formula for every `legal_id`: a verdict's is a decision row's over the same set."""
    offered = [{"label": t} for t in vocab.TOPOLOGIES]
    assert verdict.legal_id(vocab.TOPOLOGIES) == decisions._legal_id(offered)


# ── write_verdict ────────────────────────────────────────────────────────────
def test_unknown_run_raises_lookup_error(outputs):
    before = _bytes(outputs)
    with pytest.raises(LookupError) as info:
        verdict.write_verdict("nope", _probs(clusters=0.9), chooser="human:a3",
                              out_dir=outputs)
    assert "nope" in str(info.value)
    assert str(store.index_path(outputs)) in str(info.value)
    assert _bytes(outputs) == before


def test_payload_has_five_keys(outputs, tmp_path):
    out = verdict.write_verdict("r1", _probs(archetypal=0.6), chooser="llm:local-model-7b",
                                bound="schema", out_dir=outputs)
    assert set(out) == {"topology_probs", "chooser", "bound", "legal_id", "model"}
    assert out["model"] == "local-model-7b"
    assert out["bound"] == "schema"
    assert out["legal_id"] == "a37fab6bef12"
    row = list(store.read(outputs))[-1]
    assert row["extra"]["verdict"] == out

    plain_dir = tmp_path / "plain"
    store.append({"run_id": "r1"}, out_dir=plain_dir)
    (plain,) = list(store.read(plain_dir))
    assert len(plain) == 17
    assert set(row) - set(plain) == {"extra"}
    assert set(plain) <= set(row)


def test_copied_fields(tmp_path):
    results = {"run_id": "r9", "spec_id": "s9", "parent": {"run_id": "r8", "index": 1},
               "dataset": "dataset:pbmc3k", "dataset_name": "pbmc3k",
               "data_kwargs": {"n_top_genes": 500}, "recipe": "embed", "engine": "mock",
               "seed": 3, "tool_version": "0.0.0", "at": "2026-01-01T00:00:00+00:00",
               "ok": True, "complete": True, "caveats": ["mock"],
               "steps": [{"name": "pca", "group": "latent"}], "g_vector": {"x": 1}}
    store.append(results, out_dir=tmp_path)
    verdict.write_verdict("r9", _probs(clusters=0.5), chooser="human:a3", out_dir=tmp_path)
    run, claim = list(store.read(tmp_path))
    for key in ("run_id", "spec_id", "parent", "dataset", "dataset_name", "data_kwargs",
                "recipe", "engine", "seed", "tool_version"):
        assert claim[key] == run[key], key
    assert (claim["ok"], claim["complete"]) == (None, None)
    assert (claim["caveats"], claim["steps"], claim["g_vector"]) == ([], [], {})
    assert claim["at"] != run["at"]
    stamp = _dt.datetime.fromisoformat(claim["at"])
    assert stamp.tzinfo is not None and stamp.microsecond == 0


def test_verdict_row_is_never_a_target(outputs):
    store.append({"run_id": "r1", "spec_id": "s1", "dataset_name": "pbmc3k", "seed": 99},
                 out_dir=outputs, extra={"extra": {"verdict": {"topology_probs": _SIX}}})
    verdict.write_verdict("r1", _probs(clusters=0.4), chooser="human:a3", out_dir=outputs)
    rows = list(store.read(outputs))
    assert len(rows) == 3
    assert rows[2]["seed"] == 7
    assert score.verdict_of(rows[2]) is not None


@pytest.mark.parametrize("chooser, ok", [
    ("human:a3", True),
    ("human:scshapebench_a4", True),
    ("llm:local-model-7b", True),
    ("llm:org/local-model@rev1", True),
    ("learner:policy@0123abc", True),
    ("prior:fixed_geometry_rule.0123456789ab", True),
    ("human", False),
    ("human:", False),
    ("human:A3", False),
    ("human:Ada Lovelace", False),
    ("llm:", False),
    ("robot:x", False),
    ("prior:has space", False),
])
def test_chooser_forms(outputs, chooser, ok):
    assert (verdict.CHOOSER_PATTERN.fullmatch(chooser) is not None) is ok
    before = _bytes(outputs)
    if ok:
        out = verdict.write_verdict("r1", _probs(clusters=0.5), chooser=chooser,
                                    bound="logits", out_dir=outputs)
        assert out["chooser"] == chooser
        assert out["model"] == (chooser[len("llm:"):] if chooser.startswith("llm:") else None)
    else:
        with pytest.raises(ValueError):
            verdict.write_verdict("r1", _probs(clusters=0.5), chooser=chooser,
                                  bound="logits", out_dir=outputs)
        assert _bytes(outputs) == before


def test_chooser_check_comes_before_the_index_is_read(tmp_path):
    """A bad chooser is a caller bug; it is refused even when the index does not exist."""
    with pytest.raises(ValueError):
        verdict.write_verdict("r1", _probs(), chooser="human", bound="none", out_dir=tmp_path)


@pytest.mark.parametrize("chooser, bound, expect", [
    ("human:a3", None, "none"),
    ("llm:local-model-7b", None, ValueError("--bound")),
    ("learner:policy@0123abc", None, ValueError("--bound")),
    ("prior:suits", None, ValueError("--bound")),
    ("human:a3", "prompt", "prompt"),
    ("llm:local-model-7b", "logits", "logits"),
    ("human:a3", "enum", ValueError("enum")),
    ("llm:local-model-7b", "enum", ValueError("enum")),
])
def test_bound_rule(outputs, chooser, bound, expect):
    assert verdict.CHOOSER_PATTERN.fullmatch(chooser)
    if isinstance(expect, Exception):
        before = _bytes(outputs)
        with pytest.raises(ValueError) as info:
            verdict.write_verdict("r1", _probs(clusters=0.5), chooser=chooser, bound=bound,
                                  out_dir=outputs)
        assert str(expect) in str(info.value)
        assert _bytes(outputs) == before
    else:
        out = verdict.write_verdict("r1", _probs(clusters=0.5), chooser=chooser, bound=bound,
                                    out_dir=outputs)
        assert out["bound"] == expect
        assert out["bound"] in decisions.BOUNDS


# ── the `claim` verb ─────────────────────────────────────────────────────────
def test_claim_routes_without_run():
    argv = ["claim", "abc123", "--topology", "clusters=0.7"]
    assert "claim" in app._SUBCOMMANDS
    assert app._normalize_argv(argv) == argv


def test_help_lists_claim(capsys):
    with pytest.raises(SystemExit):
        app.main(["--help"])
    assert "claim" in capsys.readouterr().out


def test_claim_is_not_a_prompt_command():
    assert commands.resolve("claim") is None


def test_claim_known_run_exits_0(outputs, capsys):
    rc = app.main(["claim", "r1", "--topology", "clusters=0.9", "--chooser", "human:a3"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert len(_lines(outputs)) == 2
    row = list(store.read(outputs))[1]
    assert row["run_id"] == "r1"
    assert row["dataset_name"] == "pbmc3k"
    assert row["ok"] is None and row["complete"] is None
    assert row["steps"] == [] and row["g_vector"] == {}
    assert row["extra"]["verdict"]["topology_probs"]["clusters"] == 0.9
    assert row["extra"]["verdict"]["bound"] == "none"
    assert row["extra"]["verdict"]["model"] is None
    assert json.loads(captured.out) == row["extra"]["verdict"]


def test_claim_llm_with_bound_records_the_model(outputs, capsys):
    rc = app.main(["claim", "r1", "--topology", "archetypal=0.6", "--chooser",
                   "llm:local-model-7b", "--bound", "schema"])
    assert rc == 0
    assert capsys.readouterr().err == ""
    assert len(_lines(outputs)) == 2
    v = list(store.read(outputs))[1]["extra"]["verdict"]
    assert (v["model"], v["bound"], v["legal_id"]) == ("local-model-7b", "schema", "a37fab6bef12")


def test_claim_unknown_run_exits_1(outputs, capsys):
    before = _bytes(outputs)
    rc = app.main(["claim", "nope", "--topology", "clusters=0.9", "--chooser", "human:a3"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "nope" in err
    assert len(err.strip().splitlines()) == 1
    assert _bytes(outputs) == before
    assert len(_lines(outputs)) == 1


def test_claim_out_of_range_exits_2(outputs, capsys):
    err = _exit2(["claim", "r1", "--topology", "clusters=1.4", "--chooser", "human:a3"], capsys)
    assert "1.4" in err
    assert len(_lines(outputs)) == 1


def test_claim_real_name_exits_2(outputs, capsys):
    before = _bytes(outputs)
    err = _exit2(["claim", "r1", "--topology", "clusters=0.5", "--chooser", "human:Ada Lovelace"],
                 capsys)
    assert "chooser" in err
    assert _bytes(outputs) == before
    assert len(_lines(outputs)) == 1


def test_claim_llm_without_bound_exits_2(outputs, capsys):
    before = _bytes(outputs)
    err = _exit2(["claim", "r1", "--topology", "clusters=0.5", "--chooser", "llm:local-model-7b"],
                 capsys)
    assert "--bound" in err
    assert _bytes(outputs) == before
    assert len(_lines(outputs)) == 1


def test_claim_without_chooser_exits_2(outputs, capsys):
    before = _bytes(outputs)
    err = _exit2(["claim", "r1", "--topology", "clusters=0.5"], capsys)
    assert "--chooser" in err
    assert _bytes(outputs) == before
    assert len(_lines(outputs)) == 1


def test_bare_human_verdict_refused(outputs, capsys):
    assert verdict.CHOOSER_PATTERN.fullmatch("human") is None
    with pytest.raises(ValueError):
        verdict.write_verdict("r1", _probs(clusters=0.5), chooser="human", bound="none",
                              out_dir=outputs)
    err = _exit2(["claim", "r1", "--topology", "clusters=0.5", "--chooser", "human"], capsys)
    assert "chooser" in err
    assert len(_lines(outputs)) == 1


def test_upper_case_pseudonym_is_refused_as_in_a_label_row(outputs):
    with pytest.raises(ValueError):
        verdict.write_verdict("r1", _probs(clusters=0.5), chooser="human:A3", out_dir=outputs)
