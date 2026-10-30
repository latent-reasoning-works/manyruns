"""Cut `tests/fixtures/annotations.jsonl` with the real writer.

    uv run python tests/fixtures/cut_annotations.py tests/fixtures/annotations.jsonl

Every row goes through `annotations.append` into a temporary directory, and the result is
copied to the one path argument. The fixture is therefore whatever the writer writes, so a change
to the row shape shows up as a diff here rather than as a hand-edited file that drifted.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from manyruns import annotations


def cut(out_dir: Path) -> None:
    def add(kind: str, **kwargs) -> None:
        if annotations.append(kind, out_dir=out_dir, **kwargs) is None:
            raise SystemExit(f"the {kind} row was not written")

    # a label on one run
    add("label", run_id="abc123def456", chooser="human:a3",
        payload={"dataset": "pbmc3k", "topology": ["clusters", "multi-branching"], "raw": []})
    # an imported label: no run, joined on the dataset
    add("label", run_id=None, chooser="human:scshapebench_a4",
        payload={"dataset": "scd_0047", "topology": ["clusters"], "raw": ["blob", "clusters"],
                 "source": {"name": "scshapebench", "revision": "0123abc",
                            "sha256": "0" * 64}})
    # one rank rating per run, for two runs
    compared = ["r1", "r2"]
    for run_id, rank in zip(compared, (2, 1)):
        add("rating", run_id=run_id, chooser="human:a3",
            payload={"mode": "rank", "rank": rank, "compared": compared,
                     "novelty": None, "impact": None})
    # a blind rating by a model
    add("rating", run_id="r1", chooser="llm:local-model-7b",
        payload={"mode": "blind", "blind": True, "accuracy": 4, "completeness": 3})
    # a correction on a ledger run, and one on a run with no decision
    add("correction", run_id="r1", chooser="human:a3", decision_id="d1", step=0,
        should="umap", payload={"did": "phate"})
    add("correction", run_id="r2", chooser="human:a3", step=1, should="pca",
        payload={"did": "umap"})


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        cut(Path(tmp))
        shutil.copy(annotations.index_path(tmp), argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
