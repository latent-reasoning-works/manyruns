#!/bin/sh
# Install from the public repository on macOS/Linux; requires Git, no GitHub login.
# Run from a checkout or a copy of this file.
set -eu

installer_script=''
cleanup() {
    if [ -n "$installer_script" ]; then
        rm -f "$installer_script"
    fi
}
trap cleanup 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if ! command -v uv >/dev/null 2>&1; then
    installer_script=$(mktemp "${TMPDIR:-/tmp}/manyruns-install.XXXXXX")
    if curl -LsSf https://astral.sh/uv/install.sh -o "$installer_script"; then
        :
    else
        status=$?
        printf '%s\n' 'uv bootstrap download failed; see the error above.' >&2
        exit "$status"
    fi
    if /bin/sh "$installer_script"; then
        :
    else
        status=$?
        printf '%s\n' 'uv bootstrap failed; see the error above.' >&2
        exit "$status"
    fi
    rm -f "$installer_script"
    installer_script=''
    PATH="$HOME/.local/bin:$PATH"
    export PATH
    if ! command -v uv >/dev/null 2>&1; then
        printf '%s\n' 'uv is not on PATH. Activate the PATH printed by its installer and rerun this script.' >&2
        exit 1
    fi
    printf '%s\n' 'uv bootstrapped. Open a new terminal or activate its printed PATH for later commands.'
fi

if uv tool install --python 3.12 --force git+https://github.com/latent-reasoning-works/manyruns; then
    :
else
    status=$?
    printf '%s\n' 'Installation failed; see the error above.' \
        'Check Git, your network connection, and the resolver error above.' \
        'Public source: https://github.com/latent-reasoning-works/manyruns' >&2
    exit "$status"
fi

if ! command -v co-science >/dev/null 2>&1; then
    tool_bin=$(uv tool dir --bin 2>/dev/null) || tool_bin="$HOME/.local/bin"
    tool_bin=${tool_bin:-"$HOME/.local/bin"}
    printf 'Installation succeeded. Tool bin directory: %s\n' "$tool_bin"
    printf '%s\n' 'Run uv tool update-shell and open a new terminal, then run co-science.'
    exit 0
fi

co-science --version
