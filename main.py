"""Entry point: python main.py [--voice|--speak] [-m "message"]"""

from cronus.interfaces.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
