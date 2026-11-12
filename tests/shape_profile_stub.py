"""Fixed helper fixtures; no statistical computation."""
GRID = (5, 10, 20, 40, 80, 160)


def stub_rows(points, *, selection=None, layout=None, extended=False):
    if extended:
        raise ValueError("not_implemented")
    rows = []
    for kind in ("dataset", "selection") if selection is not None else ("dataset",):
        core = []
        keys = [("log_rk_r15", k, a) for k in GRID for a in ("median", "iqr")]
        keys += [("window_lo", None, "value"), ("window_hi", None, "value"), ("n_cells", None, "value")]
        for name, k, aggregation in keys:
            core.append(dict(entity_kind=kind, stat_name=name, scale_kind="k" if k else "none",
                             scale_value=k, aggregation=aggregation, value=0.12345678901234568,
                             n_cells_used=600, status="ok", reason="", knn_backend="scipy_exact",
                             dtype="float64", metric_version="manylatents==test:shape_profile"))
        rows.extend(core)
        if kind == "selection":
            for name, aggregation in (("conductance", "share"), ("selection_fidelity", "value")):
                for k in GRID:
                    rows.append({**core[0], "stat_name":name, "scale_value":k, "aggregation":aggregation})
            rows.extend({**r, "stat_name":"relative_"+r["stat_name"]} for r in core[:14])
    return rows
