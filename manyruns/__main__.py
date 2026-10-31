"""`python -m manyruns` is the `manyruns` command. Guarded, so importing the module runs nothing."""
from manyruns.app import main

if __name__ == "__main__":
    raise SystemExit(main())
