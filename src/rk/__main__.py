"""`python -m rk <command>`: same as the `rk` console script."""

from rk.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
