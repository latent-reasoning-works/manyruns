"""Block a commit whose staged files or message hold a forbidden name.

Turn it on once per clone with `git config core.hooksPath .githooks`. It runs the
public-safety gate's own matcher, `forbidden_count` from `tests/public_safety.py`, with
that gate's default digest set, so it matches exactly what the gate matches. It prints
locations only, never the matched text. Standard library only.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _git(*args: str) -> bytes:
    return subprocess.run(['git', *args], check=True, capture_output=True).stdout


def _matcher():
    root = Path(_git('rev-parse', '--show-toplevel').decode().strip())
    sys.path.insert(0, str(root / 'tests'))
    from public_safety import forbidden_count
    return forbidden_count


def _line_hits(count, text: str):
    return [number for number, line in enumerate(text.splitlines(), start=1) if count(line)]


def staged_locations(count) -> list[str]:
    raw = _git('diff', '--cached', '--name-only', '-z', '--diff-filter=ACMR')
    paths = [name.decode('utf-8', errors='replace') for name in raw.split(b'\0') if name]
    locations = []
    for index, path in enumerate(paths, start=1):
        content = _git('show', f':{path}').decode('utf-8', errors='replace')
        lines = _line_hits(count, content)
        if count(path):
            label = f'staged path {index} of {len(paths)}'
            locations.append(label)
            locations.extend(f'{label}, line {number}' for number in lines)
        else:
            locations.extend(f'{path}:{number}' for number in lines)
    return locations


def message_locations(count, message_file: str) -> list[str]:
    text = Path(message_file).read_text(encoding='utf-8', errors='replace')
    return [f'commit message line {number}' for number in _line_hits(count, text)]


def main(argv: list[str]) -> int:
    if argv[:1] == ['pre-commit'] and len(argv) == 1:
        locations = staged_locations(_matcher())
    elif argv[:1] == ['commit-msg'] and len(argv) == 2:
        locations = message_locations(_matcher(), argv[1])
    else:
        print('usage: forbidden_names.py pre-commit | commit-msg <message-file>',
              file=sys.stderr)
        return 2
    if not locations:
        return 0
    print(f'forbidden-name hook: commit blocked, {len(locations)} matches', file=sys.stderr)
    for location in locations:
        print(location, file=sys.stderr)
    print('the hook never prints the name; fix the lines above', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
