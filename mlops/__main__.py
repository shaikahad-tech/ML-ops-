"""__main__ shim so ``python -m mlops`` dispatches to the CLI."""
from run import main

if __name__ == "__main__":
    raise SystemExit(main())
