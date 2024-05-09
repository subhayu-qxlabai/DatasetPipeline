from pathlib import Path

import typer
import rich

from app import Pipeline, Job, JobConfig
from app.sample_job import config as sample_job_config

app = typer.Typer(name="pipeline", no_args_is_help=True)


def load_pipeline_from_path(path: str):
    path: Path = Path(path)
    if path.is_dir():
        pipeline = Pipeline.from_dir(path)
    elif path.is_file():
        try:
            pipeline = Pipeline.from_file(path)
        except Exception:
            pipeline = Pipeline(jobs=[])
        try:
            configs = [JobConfig.from_file(path)]
        except Exception:
            configs = []
        
        pipeline.jobs = pipeline.jobs + configs
    else:
        raise ValueError("Invalid path. Must be a directory or a file.")
    return pipeline

@app.command(name="list", help="List all jobs in the pipeline.")
def list_jobs(
    path: str = typer.Argument(..., help="Path to load config(s) from. Can be a directory or a file."),
):
    pipeline = load_pipeline_from_path(path)
    rich.print(f"Total jobs: {pipeline.total_jobs}")
    for job in pipeline.get_jobs():
        rich.print(job.config, end="\n\n")
    

@app.command(name="run", help="Run a pipeline from config(s).")
def run_pipeline(
    path: str = typer.Argument(..., help="Path to load config(s) from. Can be a directory or a file."),
):
    pipeline = load_pipeline_from_path(path)
    pipeline.run()

@app.command(help="Dump a sample job config to a file. Prints to stdout if no file is specified.")
def sample(
    file: str = typer.Argument(None, help="Path to dump the config to. Can have a .json or .yaml extension."),
):
    job = Job(config=sample_job_config)
    if file is None:
        rich.print("File path not specified. Printing to stdout.", end="\n\n")
        rich.print(job, end="\n\n")
    else:
        if Path(file).exists():
            if not typer.confirm(f"File {file} already exists. Overwrite?", abort=True):
                return
        job.to_file(file)
