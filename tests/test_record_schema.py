"""Versioned append-only records remain compatible with rows already on disk."""
import json

import pytest

from manyruns import decisions, store


@pytest.mark.parametrize('ledger', [store, decisions], ids=['runs', 'decisions'])
def test_new_rows_write_version_one_first(tmp_path, ledger):
    if ledger is store:
        ledger.append({'run_id': 'run-one'}, out_dir=tmp_path)
    else:
        ledger.append(offered=[{'recipe': 'embed'}], chosen='embed', out_dir=tmp_path)
    line = ledger.index_path(tmp_path).read_text()
    assert line.startswith('{"schema_version":1,')
    assert next(ledger.read(tmp_path))['schema_version'] == 1


@pytest.mark.parametrize('ledger', [store, decisions], ids=['runs', 'decisions'])
@pytest.mark.parametrize('version', [None, 1, 2])
def test_legacy_and_future_rows_preserve_unknown_fields_on_json_round_trip(tmp_path, ledger, version):
    fixture = {'run_id': 'older-run', 'unknown_top_level': {'items': [1, 'kept']},
               'extra': {'downstream': {'annotation': 'kept'}},
               'offered': [{'recipe': 'embed', 'unknown_offer_key': True}]}
    if version is not None:
        fixture['schema_version'] = version
    path = ledger.index_path(tmp_path)
    path.write_text(json.dumps(fixture) + '\n')
    original = path.read_bytes()

    row = next(ledger.read(tmp_path))

    assert row == {**fixture, 'schema_version': 1 if version is None else version}
    assert path.read_bytes() == original, 'reading must not migrate an append-only ledger'
    # Neither writer is a stored-row update API; consumers reserialize the dictionaries.
    copy = tmp_path / 'roundtrip'
    copy.mkdir()
    ledger.index_path(copy).write_text(json.dumps(row) + '\n')
    assert next(ledger.read(copy)) == row


def test_existing_extra_merge_keeps_core_fields_and_downstream_namespace(tmp_path):
    extra = {'source': 'data/input.csv', 'modality': 'bulk', 'decision_id': 'choice-one',
             'future_key': 'preserved', 'extra': {'downstream': {'note': 'annotation'}},
             'schema_version': 99}
    store.append({'run_id': 'run-one'}, out_dir=tmp_path, extra=extra)
    row = next(store.read(tmp_path))
    for key in ('source', 'modality', 'decision_id', 'future_key', 'extra'):
        assert row[key] == extra[key]
    assert row['schema_version'] == 1
    assert next(iter(json.loads(store.index_path(tmp_path).read_text()))) == 'schema_version'
