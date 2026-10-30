"""Frozen-binary entry point (distribution Path "B").

PyInstaller needs a real script to analyze; the wheel's console-script
(`manyruns = manyruns.app:main`) isn't one. This is that thin shim — nothing
lives here but the call into the same `main`, so the binary and the
`uv tool install` console-script are the identical program.

The wheel and the frozen binary both install one command, `manyruns`.
"""
import sys

from manyruns.app import main

if __name__ == "__main__":
    sys.exit(main())
