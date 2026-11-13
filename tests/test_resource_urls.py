"""Check bundled resource declarations offline, and optionally check their URLs."""

from pathlib import PurePosixPath
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from manyruns import toolchain


@pytest.fixture
def bundled_resources(monkeypatch):
    monkeypatch.delenv('MANYRUNS_TOOL_DIR', raising=False)
    root = toolchain.tool_dir()
    assert root.is_dir(), f'bundled tool directory missing: {root}'
    resources = []
    for path in sorted(root.glob('*.yaml')):
        cfg = toolchain.load_tool(path.stem)
        for name, block in cfg.get('resource', {}).items():
            resources.append((path.stem, name, block))
    return resources


def test_bundled_resource_declarations(bundled_resources):
    for tool, name, block in bundled_resources:
        context = f'{tool}: resource {name}'
        for field in ('url', 'release', 'prepare'):
            assert block.get(field), f'{context}: missing {field}'
        if name == 'gene_ids':
            filename = PurePosixPath(urlsplit(block['url']).path).name
            assert str(block['release']) in filename, (
                f'{context}: release {block["release"]} is absent from URL filename {filename}'
            )


@pytest.mark.network
def test_bundled_resource_urls_answer(bundled_resources):
    for tool, name, block in bundled_resources:
        url = block['url']
        context = f'{tool}: resource {name}: {url}'
        request = Request(url, method='GET' if url.endswith('/') else 'HEAD')
        try:
            with urlopen(request, timeout=30) as response:
                assert response.status < 400, f'{context}: HTTP {response.status}'
        except OSError as exc:
            pytest.fail(f'{context}: {exc}')
