"""Documentation citations in shipped sources and tests must resolve in the checkout."""
from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.source_checkout


def test_documentation_citations_exist():
    root = Path.cwd()
    assert root.is_dir()
    sources = [root / "README.md", root / "CLAUDE.md", root / "CONTRIBUTING.md"]
    for name in ("manyruns", "tests"):
        scanned = root / name
        assert scanned.is_dir(), f"missing citation scan root: {scanned}"
        sources.extend(path for path in scanned.rglob("*") if path.is_file()
                       and "__pycache__" not in path.parts)
    missing = []
    for source in sources:
        # Byte scanning also covers comments in bundled YAML and stylesheets.
        for cited in re.findall(rb"docs/[A-Za-z0-9_./-]+\.md", source.read_bytes()):
            target = cited.decode("ascii")
            if not (root / target).is_file():
                missing.append(f"{source.relative_to(root)}: {target}")
    assert not missing, "missing documentation citations:\n" + "\n".join(sorted(set(missing)))


def test_docs_keep_only_contract_and_sidecars():
    docs = Path(__file__).resolve().parents[1] / "docs"
    assert docs.is_dir()
    assert {path.name for path in docs.iterdir()} == {"environment-contract.md", "sidecars"}
