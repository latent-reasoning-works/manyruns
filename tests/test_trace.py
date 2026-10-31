"""The trace sink — that it writes what it was given, and that it cannot break the app.

WHAT IS TESTED IS THE WIRE, THE SILENCE AND THE EMITTERS. The wire: one `emit` is one line, the
second one does not eat the first, and every way of getting the configuration wrong ends in
`NullTracer` rather than a traceback. The emitters: a session's `step` and `run_complete` events,
a written decision's `decision` event, and the module docstring's table of every stamped kind.

THE SILENCE IS THE HALF THAT ROTS. A sink whose failures are dropped is indistinguishable from a
sink that works until somebody looks, so the drops are pinned here one by one — an event with no
run to name, a run id that is really a path, a directory that cannot be made — each asserted to
write nothing AND to raise nothing. Without these the file's promise ("a sink may only ever
change a duration") is a sentence in a docstring.

`tmp_path` throughout: this suite must never write to a real home, and a tracer pointed at one by
a stray `$MANYRUNS_TRACE` in the developer's shell would do exactly that. Every test sets the
variable itself, so an exported one cannot reach the code under test.
"""
from __future__ import annotations

import json

import pytest

from manyruns import trace


def _lines(path):
    """Every event in one trace file, parsed, oldest first."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── what the setting selects ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [
    None,                     # unset — the shipped configuration
    "",                       # exported and emptied, which is how a person turns it off
    "   ",
    "jsonl",                  # the word without the scheme
    "/var/traces",            # a bare directory: a sink is named by a scheme, not a path
    "sqlite:///runs.db",      # sub-project 2's value, arriving early on a shared profile
    "jsonl://",               # the scheme with nowhere to put anything
    "jsnol://~/traces",       # the typo the module docstring is about
    "1",                      # `MANYRUNS_TRACEBACK` reached for and stopped short of
])
def test_anything_not_understood_is_no_sink_at_all(monkeypatch, value):
    """THE CENTRAL PROMISE. None of these is an error; all of them are off."""
    monkeypatch.delenv("MANYRUNS_TRACE", raising=False)
    monkeypatch.delenv("GEOMANCER_TRACE", raising=False)
    if value is not None:
        monkeypatch.setenv("MANYRUNS_TRACE", value)

    sink = trace.current()

    assert isinstance(sink, trace.NullTracer)
    assert isinstance(sink, trace.Tracer)      # the protocol is what callers depend on


def test_a_jsonl_url_selects_a_writer_at_that_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("MANYRUNS_TRACE", f"jsonl://{tmp_path}")

    sink = trace.current()

    assert isinstance(sink, trace.JsonlTracer)
    assert sink.directory == tmp_path


def test_a_tilde_is_expanded_because_that_is_how_a_person_writes_it(monkeypatch, tmp_path):
    """`jsonl://~/traces` is the spelling that goes in a shell profile. Unexpanded it makes a
    directory literally named `~` in whatever the working directory happened to be."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MANYRUNS_TRACE", "jsonl://~/traces")

    assert trace.current().directory == tmp_path / "traces"


def test_the_setting_is_read_through_the_shim_and_not_around_it(monkeypatch):
    """The other half of `test_no_module_reads_the_environment_around_the_shim`: that guard
    catches an `os.environ` read by its shape, and would miss a read that went through some
    third path. This one names the call — `env.get("TRACE")` — and fails if the module stops
    making it, which is also what keeps the pre-rename prefix working here for free."""
    asked = []

    def _get(name, default=None):
        asked.append(name)
        return None

    monkeypatch.setattr("manyruns.env.get", _get)

    assert isinstance(trace.current(), trace.NullTracer)
    assert asked == ["TRACE"]


# ── what lands on disk ───────────────────────────────────────────────────────────────────────

def test_one_emit_is_one_line_in_the_file_named_for_the_run(tmp_path):
    sink = trace.JsonlTracer(tmp_path)

    sink.emit({"kind": "ask", "run_id": "2d730cc5c591", "question": "is this filtered?"})

    written = tmp_path / "2d730cc5c591.jsonl"
    assert _lines(written) == [
        {"kind": "ask", "run_id": "2d730cc5c591", "question": "is this filtered?"}]


def test_a_second_event_appends_and_never_truncates_the_first(tmp_path):
    """APPEND-ONLY IS THE WHOLE FORMAT. A read-modify-write would lose one of these under the
    process pool the sweep runs, and lose it silently."""
    sink = trace.JsonlTracer(tmp_path)

    sink.emit({"kind": "ask", "run_id": "abc", "question": "first"})
    sink.emit({"kind": "ask", "run_id": "abc", "question": "second"})

    got = _lines(tmp_path / "abc.jsonl")
    assert [e["question"] for e in got] == ["first", "second"]


def test_two_runs_are_two_files(tmp_path):
    """One file per run, because the first reader is a person with `cat` and the question is
    always about one run."""
    sink = trace.JsonlTracer(tmp_path)

    sink.emit({"kind": "ask", "run_id": "aaa"})
    sink.emit({"kind": "ask", "run_id": "bbb"})

    assert sorted(p.name for p in tmp_path.iterdir()) == ["aaa.jsonl", "bbb.jsonl"]


def test_constructing_a_tracer_touches_nothing(tmp_path):
    """A session that traces nothing leaves nothing behind — an empty directory that appears
    because a variable is exported reads as a bug in the sink."""
    where = tmp_path / "traces"

    trace.JsonlTracer(where)

    assert not where.exists()


def test_the_directory_is_made_on_the_first_event(tmp_path):
    sink = trace.JsonlTracer(tmp_path / "deep" / "traces")

    sink.emit({"kind": "ask", "run_id": "abc"})

    assert _lines(tmp_path / "deep" / "traces" / "abc.jsonl") == [{"kind": "ask", "run_id": "abc"}]


def test_a_value_json_cannot_hold_is_stringified_rather_than_dropped(tmp_path):
    """`default=str`: an event is scalars the emitter chose, so the only thing this meets is the
    occasional `Path`, and a stringified value is worth more than a lost event."""
    sink = trace.JsonlTracer(tmp_path)

    sink.emit({"kind": "ask", "run_id": "abc", "where": tmp_path})

    assert _lines(tmp_path / "abc.jsonl")[0]["where"] == str(tmp_path)


# ── the silence: every failure drops the event and none of them raises ───────────────────────

def test_an_event_with_no_run_still_lands_somewhere_greppable(tmp_path):
    """Dropping it would make the sink lossy in exactly the case worth investigating."""
    sink = trace.JsonlTracer(tmp_path)

    sink.emit({"kind": "ask"})

    assert _lines(tmp_path / f"{trace.UNKNOWN_RUN}.jsonl") == [{"kind": "ask"}]


def test_a_run_id_that_is_really_a_path_cannot_escape_the_directory(tmp_path):
    """A `run_id` is minted as twelve hex characters, but this class writes whatever the EVENT
    carries — and a sink that can be pointed at a file outside its own directory by a field is
    doing more than changing a duration."""
    sink = trace.JsonlTracer(tmp_path / "traces")

    sink.emit({"kind": "ask", "run_id": "../../escaped"})

    assert not (tmp_path.parent / "escaped.jsonl").exists()
    assert [p.name for p in (tmp_path / "traces").iterdir()] == ["escaped.jsonl"]


def test_a_directory_that_cannot_be_made_drops_the_event_and_raises_nothing(tmp_path):
    """The sink is pointed inside a FILE, so `mkdir` fails with `NotADirectoryError`. The app
    that was mid-run when this happened must not learn about it."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    sink = trace.JsonlTracer(blocker / "traces")

    sink.emit({"kind": "ask", "run_id": "abc"})     # must not raise

    assert blocker.read_text() == ""


def test_the_null_sink_writes_nothing_anywhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    trace.NullTracer().emit({"kind": "ask", "run_id": "abc"})

    assert list(tmp_path.iterdir()) == []


# ── the envelope ─────────────────────────────────────────────────────────────────────────────

def test_stamp_puts_the_four_fields_on_every_event():
    got = trace.stamp({"kind": "ask", "answer": "no"}, run_id="2d730cc5c591")

    assert got["kind"] == "ask"
    assert got["run_id"] == "2d730cc5c591"
    assert got["agent_id"] == trace.HUMAN == "human"
    assert got["at"].startswith("20") and got["at"].endswith("+00:00")
    assert got["answer"] == "no"                    # and the payload survives


def test_the_agent_is_the_person_unless_a_caller_says_otherwise():
    """`agent_id` is on every event from the first one, used or not — a corpus cannot go back
    and label the rows written before anyone thought to ask who caused them."""
    assert trace.stamp({"kind": "ask"}, run_id="abc")["agent_id"] == "human"
    assert trace.stamp({"kind": "ask"}, run_id="abc", agent_id="sweep")["agent_id"] == "sweep"


def test_an_event_with_no_kind_is_named_rather_than_refused():
    """A reader dispatching on `kind` would otherwise meet a `KeyError` from a line that is
    already on disk and cannot be fixed."""
    assert trace.stamp({"answer": "no"}, run_id="abc")["kind"] == "event"


def test_the_stamp_wins_over_a_payload_that_carries_the_same_names():
    """One clock, one identity. Two callers stamping their own times is how a record becomes
    unorderable, and an event that renames its own run is a join that silently misses."""
    got = trace.stamp({"kind": "ask", "at": "1999-01-01T00:00:00+00:00", "run_id": "other"},
                      run_id="abc")

    assert got["run_id"] == "abc"
    assert not got["at"].startswith("1999")


def test_stamp_does_not_mutate_what_it_was_given():
    """An emitter may keep and reuse the payload it built; stamping it must not be a write."""
    payload = {"kind": "ask", "answer": "no"}

    trace.stamp(payload, run_id="abc")

    assert payload == {"kind": "ask", "answer": "no"}


def test_a_stamped_event_round_trips_through_the_sink(tmp_path):
    """The two halves together, in the shape §2's `ask` event has."""
    sink = trace.JsonlTracer(tmp_path)

    sink.emit(trace.stamp({"kind": "ask", "question": "should I filter first?",
                           "context_sha256": "0" * 64, "context_tokens": 147,
                           "model": "qwen3:8b", "backend": "ollama", "latency_s": 1.5,
                           "answer": "no"}, run_id="2d730cc5c591"))

    got = _lines(tmp_path / "2d730cc5c591.jsonl")[0]
    assert set(got) == {"kind", "at", "run_id", "agent_id", "question", "context_sha256",
                        "context_tokens", "model", "backend", "latency_s", "answer"}
    assert got["latency_s"] == 1.5 and got["context_tokens"] == 147


def test_short_writes_finish_the_line_before_the_next_event(monkeypatch, tmp_path):
    """Seven bytes per write still produce two complete JSON lines, including UTF-8 payloads."""
    write = trace.os.write
    monkeypatch.setattr(trace.os, "write", lambda fd, data: write(fd, data[:7]))
    sink = trace.JsonlTracer(tmp_path)
    events = [{"run_id": "abc", "answer": text} for text in ("细胞", "next")]
    for event in events:
        sink.emit(event)
    assert _lines(tmp_path / "abc.jsonl") == events
    assert (tmp_path / "abc.jsonl").read_bytes().endswith(b"\n")


@pytest.mark.parametrize("failure", ["raise", "zero"])
def test_a_torn_write_rolls_back_without_losing_the_previous_event(monkeypatch, tmp_path, failure):
    """After seven bytes, ENOSPC or no progress must leave only the earlier complete event."""
    import errno

    sink = trace.JsonlTracer(tmp_path)
    first, last = {"run_id": "abc", "n": 1}, {"run_id": "abc", "n": 3}
    sink.emit(first)
    write = trace.os.write
    calls = 0

    def torn(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return write(fd, data[:7])
        if failure == "zero":
            return 0
        raise OSError(errno.ENOSPC, "no space")

    with monkeypatch.context() as patch:
        patch.setattr(trace.os, "write", torn)
        sink.emit({"run_id": "abc", "n": 2})
    assert _lines(tmp_path / "abc.jsonl") == [first]
    sink.emit(last)
    assert _lines(tmp_path / "abc.jsonl") == [first, last]


def test_an_existing_torn_tail_cannot_swallow_the_next_event(tmp_path):
    """A killed writer can leave bytes rollback never saw; the next emit must not extend them."""
    path = tmp_path / "abc.jsonl"
    path.write_bytes(b'{"run_id":')
    trace.JsonlTracer(tmp_path).emit({"run_id": "abc", "n": 2})
    assert path.read_bytes() == b'{"run_id":'


def test_a_symlink_named_for_a_run_is_not_followed(tmp_path):
    """The sanitized name `abc.jsonl` can still be a link; the outside file stays byte-identical."""
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"keep":true}\n')
    directory = tmp_path / "traces"
    directory.mkdir()
    (directory / "abc.jsonl").symlink_to(outside)
    trace.JsonlTracer(directory).emit({"run_id": "abc"})
    assert outside.read_text() == '{"keep":true}\n'
    assert (directory / "abc.jsonl").is_symlink()


# ── the emit sites: who drives a session, and what it emits ──────────────────────────────────

STEP = {"name": "phate", "group": "latent", "params": {}}


def _traced(monkeypatch, tmp_path):
    traces = tmp_path / "traces"
    monkeypatch.setenv("MANYRUNS_TRACE", f"jsonl://{traces}")
    return traces


def _mock_session(tmp_path, **kw):
    from manyruns.session import Session

    return Session("p", engine="mock", out_dir=tmp_path / "out", **kw)


def _kind(path, kind):
    return [e for e in _lines(path) if e["kind"] == kind] if path.is_file() else []


def test_the_default_driver_is_the_person(tmp_path):
    assert _mock_session(tmp_path).agent_id == "human" == trace.HUMAN


def test_a_branch_keeps_the_agent(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("phate")
    from manyruns.session import Session

    s = Session(project="p", engine="_inproc", out_dir=tmp_path, modality="scrna",
                array=np.random.default_rng(0).normal(size=(150, 12)), seed=0,
                agent_id="llm:qwen3:8b")
    s.step({"name": "phate", "group": "latent", "params": {"n_components": 3}})
    assert s.branch(0).agent_id == "llm:qwen3:8b"


def test_a_two_step_session_writes_two_step_events(monkeypatch, tmp_path):
    traces = _traced(monkeypatch, tmp_path)
    s = _mock_session(tmp_path)
    s.step(dict(STEP))
    s.step(dict(STEP))

    events = _kind(traces / f"{s.run_id}.jsonl", "step")
    assert [e["index"] for e in events] == [0, 1]
    for event, rec in zip(events, s.steps):
        assert event["run_id"] == s.run_id
        assert event["agent_id"] == "human"
        assert (event["name"], event["group"], event["outcome"]) == (
            rec["name"], rec["group"], rec["outcome"])


def test_a_step_event_names_the_session_driver(monkeypatch, tmp_path):
    traces = _traced(monkeypatch, tmp_path)
    s = _mock_session(tmp_path, agent_id="llm:x")
    s.step(dict(STEP))
    (event,) = _kind(traces / f"{s.run_id}.jsonl", "step")
    assert event["agent_id"] == "llm:x"


def test_the_null_sink_is_the_default(monkeypatch, tmp_path):
    monkeypatch.delenv("MANYRUNS_TRACE", raising=False)
    s = _mock_session(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    rec = s.apply(dict(STEP))
    assert sorted(tmp_path.rglob("*")) == before
    assert (rec["index"], rec["name"], rec["group"]) == (0, "phate", "latent")
    assert rec["outcome"] in ("ok", "skipped", "error")


class _Compute:
    """An offloaded-compute stand-in: `call` runs in process on COPIES of its arguments, as a
    worker process receives pickled ones, or raises what it is told to."""

    def __init__(self, raises=None):
        self.raises = raises

    def call(self, fn, *args, **kwargs):
        import copy

        if self.raises is not None:
            raise self.raises
        return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))

    def close(self):
        pass


@pytest.mark.parametrize("raises, outcome", [(None, None), (RuntimeError("worker died"), "error")])
def test_the_offloaded_paths_emit_one_step_event(monkeypatch, tmp_path, raises, outcome):
    traces = _traced(monkeypatch, tmp_path)
    s = _mock_session(tmp_path)
    s.compute = _Compute(raises)
    rec = s.apply(dict(STEP))
    (event,) = _kind(traces / f"{s.run_id}.jsonl", "step")
    assert event["index"] == 0 and event["outcome"] == rec["outcome"]
    if outcome:
        assert rec["outcome"] == outcome


def test_a_cancelled_step_emits_nothing(monkeypatch, tmp_path):
    from manyruns.pipeline import runner

    traces = _traced(monkeypatch, tmp_path)
    s = _mock_session(tmp_path)
    s.compute = _Compute(runner.ComputeCancelled())
    with pytest.raises(runner.ComputeCancelled):
        s.apply(dict(STEP))
    assert _kind(traces / f"{s.run_id}.jsonl", "step") == []


def test_one_completion_event_per_close(monkeypatch, tmp_path):
    traces = _traced(monkeypatch, tmp_path)
    s = _mock_session(tmp_path)
    s.step(dict(STEP))
    res = s.close()

    events = _lines(traces / f"{s.run_id}.jsonl")
    assert [e["kind"] for e in events].count("run_complete") == 1
    last = events[-1]
    assert last["kind"] == "run_complete"
    assert last["ok"] == res["ok"] and last["complete"] == res["complete"]
    assert last["spec_id"] == res["spec_id"]
    assert last["n_steps"] == 1
    assert last["run_id"] == s.run_id and last["agent_id"] == "human"


def test_a_close_that_raises_emits_nothing(monkeypatch, tmp_path):
    traces = _traced(monkeypatch, tmp_path)
    s = _mock_session(tmp_path)

    def boom(**kw):
        raise ValueError("no record")

    monkeypatch.setattr(s, "results", boom)
    with pytest.raises(ValueError):
        s.close()
    assert _kind(traces / f"{s.run_id}.jsonl", "run_complete") == []


DECISION_KEYS = {"schema_version", "decision_id", "run_id", "at", "surface", "chooser", "bound",
                 "legal_id", "dataset", "shape", "topology", "offered", "chosen"}


def test_a_tune_decision_is_attributed(monkeypatch, tmp_path):
    import types

    from manyruns import tune

    traces = _traced(monkeypatch, tmp_path)
    session = types.SimpleNamespace(run_id="abc123def456", agent_id="llm:claude-opus-4-8",
                                    out_dir=tmp_path / "out", dataset=None, shape="unknown")
    attempts = [{"name": "phate", "params": {"knn": 5}, "label": "a"},
                {"name": "phate", "params": {"knn": 9}, "label": "b"}]
    tune._record_decision(session, attempts, "b")

    from manyruns import decisions

    rows = [r for d in tmp_path.rglob("decisions.jsonl") for r in decisions.read(d.parent)]
    assert len(rows) == 1 and rows[0]["chooser"] == "llm:claude-opus-4-8"
    (event,) = _kind(traces / "abc123def456.jsonl", "decision")
    assert event["agent_id"] == "llm:claude-opus-4-8"
    assert event["surface"] == "tune"
    assert event["n_offered"] == 2
    assert event["decision_id"] == rows[0]["decision_id"]
    assert event["chosen"] == "b"


def test_an_empty_offer_emits_nothing(monkeypatch, tmp_path):
    from manyruns import decisions

    traces = _traced(monkeypatch, tmp_path)
    assert decisions.append(offered=[], chosen=None, run_id="abc123def456",
                            out_dir=tmp_path) is None
    assert not traces.exists() or not any(
        e["kind"] == "decision" for f in traces.glob("*.jsonl") for e in _lines(f))


def test_the_decision_row_schema_is_unchanged(monkeypatch, tmp_path):
    from manyruns import decisions

    traces = _traced(monkeypatch, tmp_path)
    decision_id = decisions.append(offered=[{"recipe": "embed"}], chosen="embed",
                                   chooser="llm:x", out_dir=tmp_path)
    (row,) = list(decisions.read(tmp_path))
    assert row["chooser"] == "llm:x" and "agent_id" not in row
    assert set(row) == DECISION_KEYS
    (event,) = _kind(traces / f"{trace.UNKNOWN_RUN}.jsonl", "decision")
    assert event["agent_id"] == "llm:x"
    assert event["decision_id"] == decision_id
    assert event["run_id"] is None


def test_no_second_keyword_for_the_chooser(tmp_path):
    from manyruns import decisions

    with pytest.raises(TypeError):
        decisions.append(offered=[{"recipe": "embed"}], chosen="embed", agent_id="llm:x",
                         out_dir=tmp_path)
    assert not (tmp_path / "decisions.jsonl").exists()


# ── the docstring lists every stamped kind ───────────────────────────────────────────────────

#: Kinds built at run time (`"proposer_" + kind` in `agents.propose`), which a scan for string
#: literals cannot see.
DYNAMIC_KINDS = {"proposer_assistant", "proposer_tool_call", "proposer_tool_result"}


def stamped_kinds(source: str) -> set[str]:
    """Every literal `"kind"` value in a dict literal passed first to `trace.stamp`,
    `_trace.stamp` or `stamp`."""
    import ast

    kinds: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        fn = node.func
        name = (fn.attr if isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name) and fn.value.id in ("trace", "_trace")
                else fn.id if isinstance(fn, ast.Name) else None)
        if name != "stamp" or not isinstance(node.args[0], ast.Dict):
            continue
        for key, value in zip(node.args[0].keys, node.args[0].values):
            if (isinstance(key, ast.Constant) and key.value == "kind"
                    and isinstance(value, ast.Constant) and isinstance(value.value, str)):
                kinds.add(value.value)
    return kinds


def _missing(kinds) -> list[str]:
    return sorted(k for k in kinds if k not in trace.__doc__)


def test_a_new_kind_without_documentation_is_found():
    kinds = stamped_kinds('trace.stamp({"kind": "tune_attempt"}, run_id=None)')
    assert kinds == {"tune_attempt"}
    assert _missing(kinds) == ["tune_attempt"]


def test_the_helper_reads_all_three_spellings_and_ignores_other_calls():
    source = ('_trace.stamp({"kind": "a"}, run_id=None)\nstamp({"kind": "b"}, run_id=None)\n'
              'other.stamp({"kind": "c"})\nemit({"kind": "d"})\n'
              'trace.stamp({"kind": "p_" + k}, run_id=None)')
    assert stamped_kinds(source) == {"a", "b"}


def test_every_stamped_kind_in_the_package_is_documented():
    from pathlib import Path

    package = Path(trace.__file__).parent
    assert package.is_dir()
    kinds = set(DYNAMIC_KINDS)
    for path in package.rglob("*.py"):
        kinds |= stamped_kinds(path.read_text())
    assert {"ask", "step", "run_complete", "decision", "proposer_call", "message"} <= kinds
    assert _missing(kinds) == []


def test_the_stale_sentence_is_gone():
    assert "Nothing imports this module yet" not in trace.__doc__
    assert "any emit site at all" not in trace.__doc__
