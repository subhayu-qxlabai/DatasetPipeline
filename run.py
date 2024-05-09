#!/usr/bin/env python
import typer

from scripts import pipeline


app = typer.Typer(no_args_is_help=True)
app.add_typer(pipeline.app, name="pipeline")


if __name__ == "__main__":
    app()
