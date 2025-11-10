"""Entry point for ``python -m crewctl``."""

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
