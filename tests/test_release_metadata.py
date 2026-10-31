"""Release metadata: one version in four sources, a dated changelog section, the licence.

The helpers are pure: each takes text (or a dict, or a list) and returns a list of
problems, empty when the input is correct. The tests apply them to the checkout. The
version is read from `pyproject.toml`, never hard-coded, so these tests keep passing at
the next release. The only text pinned for 0.1.1 is the content of its changelog section,
which stays in the file after later releases.

The sdist omits `CITATION.cff` and `uv.lock`, so the whole module needs a source checkout.
"""

from __future__ import annotations

import ast
import datetime
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.source_checkout

ROOT = Path(__file__).resolve().parents[1]

REPO_URL = "https://github.com/latent-reasoning-works/manyruns"

#: Each serial-queue item of 0.1.1, by a name its changelog line must contain.
#: `bound` keeps its backticks: a bare "bound" would match "the manylatents lower bound".
REQUIRED_CHANGELOG_NAMES_0_1_1 = (
    "partition",
    "run --ask",
    "chooser",
    "`bound`",
    "legal_id",
    "manyruns actions",
    "annotations.jsonl",
    "manyruns label",
    "manyruns import-labels",
    "manyruns rate",
    "manyruns correct",
    "manyruns claim",
    "class_auprc",
    "branches",
    "rung_support",
    "MANYRUNS_PROPOSER_MODEL",
    "run_id",
    "model seam",
    "tools verify",
    "MANYRUNS_COMMERCIAL",
    "manyruns.action_space",
    "run --choice",
)

#: The aa-fix release of manylatents, the lower bound that 0.1.1 states.
MANYLATENTS_BOUND = "manylatents>=0.1.9"

COPYRIGHT_LINE = "Copyright (c) 2026 Latent Reasoning Works and the manyruns contributors"
NOTICE_LICENCE_LINE = "Licensed under the Apache License, Version 2.0 (see LICENSE)."


# --------------------------------------------------------------------------- helpers


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def version_mismatches(versions: dict[str, str]) -> list[str]:
    """The sources whose version differs from the one in `pyproject.toml`."""
    reference = versions["pyproject.toml"]
    return [source for source, value in versions.items() if value != reference]


def _section(text: str, heading_prefix: str) -> list[str] | None:
    """Lines after the first heading that starts with `heading_prefix`, up to the next `## [`."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(heading_prefix):
            body = []
            for following in lines[index + 1 :]:
                if following.startswith("## ["):
                    break
                body.append(following)
            return body
    return None


def changelog_layout_problems(text: str, version: str) -> list[str]:
    """Faults in the dated section of `version` and in the link lines."""
    problems = []
    lines = text.splitlines()
    try:
        start = lines.index("## [Unreleased]")
    except ValueError:
        return ["no `## [Unreleased]` heading"]
    following = next((line for line in lines[start + 1 :] if line.startswith("## [")), None)
    match = re.fullmatch(rf"## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})", following or "")
    if match is None:
        problems.append(
            f"the first `## [` heading after `## [Unreleased]` is {following!r}, "
            f"not `## [{version}] - YYYY-MM-DD`"
        )
    else:
        try:
            datetime.date.fromisoformat(match.group(1))
        except ValueError:
            problems.append(f"`## [{version}]` carries an invalid date {match.group(1)!r}")
    for link in (
        f"[Unreleased]: {REPO_URL}/compare/v{version}...HEAD",
        f"[{version}]: {REPO_URL}/releases/tag/v{version}",
    ):
        if link not in lines:
            problems.append(f"missing link line {link!r}")
    return problems


def unreleased_bullets(text: str) -> list[str]:
    """Bullet lines under `## [Unreleased]`, as the empty-Unreleased awk command finds them."""
    body = _section(text, "## [Unreleased]") or []
    return [line for line in body if line.startswith("- ")]


def missing_changelog_names(text: str) -> list[str]:
    """Required names absent from the `## [0.1.1]` section (text elsewhere does not count)."""
    body = _section(text, "## [0.1.1]")
    section = "\n".join(body) if body is not None else ""
    return [name for name in REQUIRED_CHANGELOG_NAMES_0_1_1 if name not in section]


def _nonempty_stripped(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def license_problems(
    license_text: str, notice_text: str, pyproject_text: str, citation_text: str
) -> list[str]:
    """Faults against Apache-2.0 with the NOTICE file, in all four places that state it."""
    problems = []
    head = _nonempty_stripped(license_text)[:2]
    if head != ["Apache License", "Version 2.0, January 2004"]:
        problems.append(f"LICENSE does not open with the Apache License 2.0 title: {head!r}")
    notice_lines = [line.strip() for line in notice_text.splitlines()]
    if NOTICE_LICENCE_LINE not in notice_lines:
        problems.append(f"NOTICE lacks the line {NOTICE_LICENCE_LINE!r}")
    if COPYRIGHT_LINE not in notice_lines:
        problems.append(f"NOTICE lacks the copyright line {COPYRIGHT_LINE!r}")
    project = tomllib.loads(pyproject_text).get("project", {})
    if project.get("license") != "Apache-2.0":
        problems.append(f"pyproject.toml license is {project.get('license')!r}, not 'Apache-2.0'")
    if project.get("license-files") != ["LICENSE", "NOTICE"]:
        problems.append(
            f"pyproject.toml license-files is {project.get('license-files')!r}, "
            "not ['LICENSE', 'NOTICE']"
        )
    if not re.search(r"^license:\s*Apache-2\.0\s*$", citation_text, re.MULTILINE):
        problems.append("CITATION.cff lacks `license: Apache-2.0`")
    return problems


def bound_problems(dependencies: list[str]) -> list[str]:
    """Faults in the manylatents lower bound among `[project] dependencies`."""
    entries = [dep for dep in dependencies if re.match(r"manylatents\s*(\[|[<>=!~;]|$)", dep)]
    if not entries:
        return [f"no manylatents dependency; expected {MANYLATENTS_BOUND}"]
    return [
        f"{entry} is not the aa-fix bound {MANYLATENTS_BOUND} (0.1.9)"
        for entry in entries
        if re.sub(r"\s+", "", entry) != MANYLATENTS_BOUND
    ]


# --------------------------------------------------------------------------- readers


def _read(name: str) -> str:
    assert ROOT.is_dir(), f"repository root {ROOT} is not a directory"
    return (ROOT / name).read_text(encoding="utf-8")


def _pyproject_version() -> str:
    return tomllib.loads(_read("pyproject.toml"))["project"]["version"]


def _version_sources() -> dict[str, str]:
    init = ast.parse(_read("manyruns/__init__.py"))
    init_version = next(
        node.value.value
        for node in init.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    )
    citation = re.search(r"^version:\s*['\"]?([^'\"\s]+)", _read("CITATION.cff"), re.MULTILINE)
    lock = tomllib.loads(_read("uv.lock"))
    lock_version = next(p["version"] for p in lock["package"] if p["name"] == "manyruns")
    return {
        "pyproject.toml": _pyproject_version(),
        "manyruns/__init__.py": init_version,
        "CITATION.cff": citation.group(1) if citation else "<missing>",
        "uv.lock": lock_version,
    }


# --------------------------------------------------------------------------- fixtures

_GOOD_SECTION = """\
### Added

- `manyruns actions` and `run --ask`; decision rows carry `chooser`, `bound` and `legal_id`.
- `outputs/annotations.jsonl` with `manyruns label`, `manyruns import-labels`,
  `manyruns rate` and `manyruns correct`; `manyruns claim` and `class_auprc`.
- The `branches` recipe and `vocab.rung_support`; `MANYRUNS_PROPOSER_MODEL` and `run_id`.
- The model seam and `manyruns tools verify`; `MANYRUNS_COMMERCIAL`.
- `manyruns.action_space` and `manyruns run --choice`.

### Changed

- The `manylatents` lower bound is `manylatents>=0.1.9`.

### Fixed

- A partition step is no longer scored by the geometry suite.
"""

_LINKS = f"""\
[Unreleased]: {REPO_URL}/compare/v0.1.1...HEAD
[0.1.1]: {REPO_URL}/releases/tag/v0.1.1
[0.1.0]: {REPO_URL}/releases/tag/v0.1.0
"""


def _changelog(unreleased: str = "", section: str = _GOOD_SECTION, links: str = _LINKS) -> str:
    return (
        f"# Changelog\n\n## [Unreleased]\n\n{unreleased}\n## [0.1.1] - 2026-10-31\n\n"
        f"{section}\n## [0.1.0] - 2026-09-17\n\n- First release.\n\n{links}"
    )


_UV_LINE = "- Installation and runtime repair guidance consistently uses uv tool installs.\n"

_LICENSE = "\n                                 Apache License\n                           Version 2.0, January 2004\n"
_NOTICE = f"manyruns\n\n{NOTICE_LICENCE_LINE}\n\nMIT License\n\n{COPYRIGHT_LINE}\n"
_PYPROJECT = '[project]\nname = "manyruns"\nlicense = "Apache-2.0"\nlicense-files = ["LICENSE", "NOTICE"]\n'
_CITATION = "cff-version: 1.2.0\nversion: 0.1.1\nlicense: Apache-2.0\n"


def test_helpers_accept_good_fixtures():
    """Positive control, so each rejection below is caused by its one fault."""
    good = _changelog()
    assert changelog_layout_problems(good, "0.1.1") == []
    assert unreleased_bullets(good) == []
    assert missing_changelog_names(good) == []
    assert license_problems(_LICENSE, _NOTICE, _PYPROJECT, _CITATION) == []
    assert bound_problems(["numpy>=1.26", "manylatents>=0.1.9", "manylatents-omics[singlecell]>=0.1"]) == []
    assert version_mismatches(dict.fromkeys(("pyproject.toml", "manyruns/__init__.py"), "0.1.1")) == []


@pytest.mark.parametrize(
    ("problems", "names"),
    [
        pytest.param(
            lambda: version_mismatches(
                {
                    "pyproject.toml": "0.1.1",
                    "manyruns/__init__.py": "0.1.0",
                    "CITATION.cff": "0.1.1",
                    "uv.lock": "0.1.1",
                }
            ),
            "manyruns/__init__.py",
            id="missed-version-source",
        ),
        pytest.param(
            lambda: unreleased_bullets(_changelog(unreleased=_UV_LINE)),
            "uv tool installs",
            id="bullet-left-in-unreleased",
        ),
        pytest.param(
            lambda: missing_changelog_names(
                _changelog(section=_GOOD_SECTION.replace("and `class_auprc`", ""))
            ),
            "class_auprc",
            id="missing-class-auprc",
        ),
        pytest.param(
            lambda: missing_changelog_names(
                _changelog(
                    section=_GOOD_SECTION.replace("`chooser`, `bound` and", "`chooser` and")
                    + "- The manylatents lower bound moves.\n"
                )
            ),
            "`bound`",
            id="lower-bound-without-backticked-bound",
        ),
        pytest.param(
            lambda: missing_changelog_names(
                _changelog(
                    unreleased="- `manyruns actions` prints the action space.\n",
                    section=_GOOD_SECTION.replace("`manyruns actions` and ", ""),
                )
            ),
            "manyruns actions",
            id="name-only-outside-the-section",
        ),
        pytest.param(
            lambda: license_problems(
                _LICENSE,
                _NOTICE.replace(COPYRIGHT_LINE, "Copyright (c) 2026 Latent Reasoning Works"),
                _PYPROJECT,
                _CITATION,
            ),
            "copyright",
            id="changed-copyright-holder",
        ),
        pytest.param(
            lambda: license_problems(
                _LICENSE,
                _NOTICE,
                '[project]\nlicense = "MIT"\nlicense-files = ["LICENSE"]\n',
                _CITATION,
            ),
            "license-files",
            id="mit-pyproject",
        ),
        pytest.param(
            lambda: changelog_layout_problems(
                _changelog().replace("## [0.1.1] - 2026-10-31", "## [0.1.1] - 2026-02-30"), "0.1.1"
            ),
            "invalid date",
            id="invalid-section-date",
        ),
        pytest.param(
            lambda: changelog_layout_problems(_changelog(links=f"[0.1.0]: {REPO_URL}/releases/tag/v0.1.0\n"), "0.1.1"),
            "compare/v0.1.1...HEAD",
            id="missing-link-lines",
        ),
        pytest.param(
            lambda: bound_problems(["manylatents>=0.1.7"]),
            "manylatents>=0.1.7",
            id="old-manylatents-bound",
        ),
    ],
)
def test_helpers_reject_fixture_texts(problems, names):
    found = problems()
    assert found, "the helper accepted a faulty fixture"
    assert any(names in problem for problem in found), found


def test_helper_results_match_the_spec_scenarios():
    assert version_mismatches(
        {
            "pyproject.toml": "0.1.1",
            "manyruns/__init__.py": "0.1.0",
            "CITATION.cff": "0.1.1",
            "uv.lock": "0.1.1",
        }
    ) == ["manyruns/__init__.py"]
    assert len(unreleased_bullets(_changelog(unreleased=_UV_LINE))) == 1
    mit = license_problems(
        _LICENSE, _NOTICE, '[project]\nlicense = "MIT"\nlicense-files = ["LICENSE"]\n', _CITATION
    )
    assert any("license is" in p for p in mit) and any("license-files" in p for p in mit)
    (old,) = bound_problems(["manylatents>=0.1.7"])
    assert "manylatents>=0.1.7" in old and "0.1.9" in old
    later = _changelog(unreleased="- New line.\n")
    assert changelog_layout_problems(later, "0.1.1") == []


# --------------------------------------------------------------------------- the checkout


def test_version_sources_agree():
    versions = _version_sources()
    mismatched = version_mismatches(versions)
    assert mismatched == [], f"version sources disagree with pyproject.toml: {mismatched} ({versions})"


def test_changelog_has_section_for_version():
    assert changelog_layout_problems(_read("CHANGELOG.md"), _pyproject_version()) == []


def test_unreleased_is_empty_at_release_tag():
    version = _pyproject_version()
    described = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if described.returncode != 0 or described.stdout.strip() != f"v{version}":
        pytest.skip(f"HEAD is not the release tag v{version}")
    assert unreleased_bullets(_read("CHANGELOG.md")) == []


def test_changelog_0_1_1_names_queue_items():
    if _version_tuple(_pyproject_version()) < (0, 1, 1):
        pytest.skip("the 0.1.1 section is written by the 0.1.1 release")
    missing = missing_changelog_names(_read("CHANGELOG.md"))
    assert missing == [], f"the ## [0.1.1] section lacks: {missing}"


def test_license_is_apache_with_notice():
    problems = license_problems(
        _read("LICENSE"), _read("NOTICE"), _read("pyproject.toml"), _read("CITATION.cff")
    )
    assert problems == []


def test_manylatents_bound_is_aa_fix():
    dependencies = tomllib.loads(_read("pyproject.toml"))["project"]["dependencies"]
    assert bound_problems(dependencies) == []
    if _version_tuple(_pyproject_version()) < (0, 1, 1):
        return
    section = "\n".join(_section(_read("CHANGELOG.md"), "## [0.1.1]") or [])
    assert MANYLATENTS_BOUND in section, f"the ## [0.1.1] section does not name {MANYLATENTS_BOUND}"
