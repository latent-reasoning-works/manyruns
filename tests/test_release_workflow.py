"""The public release must verify artifacts before granting publishing authority."""
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.source_checkout
ROOT = Path(__file__).resolve().parents[1]


def workflow():
    assert ROOT.is_dir()
    return yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())


def test_publishing_is_opt_in_and_oidc_only():
    release = workflow()
    assert release["permissions"] == {}
    assert release["on"]["push"]["tags"] == ["v*"]
    assert release["on"]["workflow_dispatch"]["inputs"]["publish"]["default"] is False
    publish = release["jobs"]["publish"]
    assert set(publish["needs"]) == {"build", "verify"}
    assert publish["environment"] == "release"
    assert publish["permissions"] == {"id-token": "write"}
    assert "github.event_name == 'push'" in publish["if"]
    assert "startsWith(github.ref, 'refs/tags/v')" in publish["if"]
    assert "github.event_name == 'workflow_dispatch' && inputs.publish" in publish["if"]
    publisher, = [step for step in publish["steps"]
                  if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@")]
    assert not publisher.get("with"), "use the default PyPI OIDC exchange without credentials"
    assert all("env" not in step and "run" not in step for step in publish["steps"])


def test_github_attachments_use_separate_authority_after_publication():
    jobs = workflow()["jobs"]
    attachment = jobs["github-release"]
    assert set(attachment["needs"]) == {"build", "publish"}
    assert attachment["permissions"] == {"contents": "write"}
    for name, job in jobs.items():
        if name != "publish":
            assert "id-token" not in job.get("permissions", {})
    # All downstream jobs consume the exact artifacts uploaded by the build job.
    upload, = [step["with"] for step in jobs["build"]["steps"]
               if step.get("uses", "").startswith("actions/upload-artifact@")]
    assert {"dist/*.whl", "dist/*.tar.gz"} <= set(upload["path"].splitlines())
    for name in ("verify", "publish", "github-release"):
        download, = [step["with"] for step in jobs[name]["steps"]
                     if step.get("uses", "").startswith("actions/download-artifact@")]
        assert download["name"] == upload["name"]


def test_verification_covers_supported_python_and_platforms():
    verify = workflow()["jobs"]["verify"]
    assert verify["needs"] == "build"
    matrix = verify["strategy"]["matrix"]
    assert matrix["python-version"] == ["3.11", "3.12"]
    assert set(matrix["os"]) == {"ubuntu-latest", "macos-latest"}
