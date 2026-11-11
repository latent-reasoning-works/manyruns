"""Four representative rows from the dataset_features/1 helper contract."""

DEFINITION_VERSION = "dataset_features/1"
FIELDS = ("key", "value", "type", "status", "reason", "source", "definition_version")
CALLS = []


def stub_rows(data, roles, *, layer=None, overrides=None) -> list[dict]:
    CALLS.append((data, roles, layer, overrides))
    return [
        dict(key="has_mito_genes", value=True, type="bool", status="ok", reason=None,
             source="data", definition_version=DEFINITION_VERSION),
        dict(key="n_cells", value=3, type="int", status="ok", reason=None,
             source="data", definition_version=DEFINITION_VERSION),
        dict(key="tested_matrix", value="X" if layer is None else f"layers:{layer}",
             type="str", status="ok", reason=None, source="data", definition_version=DEFINITION_VERSION),
        dict(key="sample_column", value=None, type="str", status="not_supplied",
             reason="Role not supplied: sample", source="role", definition_version=DEFINITION_VERSION),
    ]
