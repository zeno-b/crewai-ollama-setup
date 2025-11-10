import typer

from .agents import agents_app
from .models import models_app
from .utils import DeploymentError

app = typer.Typer(help="CrewAI deployment control utility.")
app.add_typer(agents_app, name="agents")
app.add_typer(models_app, name="models")


@app.callback()
def main_callback() -> None:
    """Entry point for the CLI."""
    return


def main() -> None:
    try:
        app()
    except DeploymentError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    main()
