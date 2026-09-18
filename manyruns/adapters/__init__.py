"""Adapters — code that runs INSIDE a tool's interpreter, never inside manyruns's.

Every module here is executed by `<tool-env>/bin/python`, an environment chosen to satisfy the
TOOL (pyrovelocity's is numpy 1.26 against this product's 2.2.6 — mutually exclusive, measured).
So the rules are inverted from the rest of the package:

  * an adapter MAY import its tool at module scope — that is the whole job, and
    `tests/test_base_install.py` excludes this directory for exactly that reason;
  * an adapter MUST NOT import manyruns, ever. It could not succeed if it tried, and a test
    pins the prohibition so nobody discovers it the slow way.

The contract in both directions is a FILE, never a call: a request JSON in, an `.h5ad` and a
result JSON out. That is what lets two irreconcilable dependency closures cooperate at all.
"""
