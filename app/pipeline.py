"""
The `app.pipeline` module defines the `Pipeline` class, which represents a collection of `Job` objects.

The `Pipeline` class has the following methods:
- `get_jobs()`: Returns an iterator of `Job` objects based on the `JobConfig` objects in the `jobs` list.
- `total_jobs`: Returns the number of `JobConfig` objects in the `jobs` list.
- `run()`: Runs all the jobs in the pipeline and returns a list of the results.
- `from_dir()`: Creates a `Pipeline` object from a directory containing JSON, YAML, or YML files. It recursively searches the directory for files with these extensions and creates `JobConfig` objects from each file. It then creates a `Pipeline` object with the `JobConfig` objects.

Example usage:

```python
from app.pipeline import Pipeline

# Create a Pipeline object from a directory
pipeline = Pipeline.from_dir('/path/to/directory')

# Get the jobs in the pipeline
jobs = pipeline.get_jobs()

# Run the pipeline
results = pipeline.run()
```
"""

from pathlib import Path
from itertools import chain

from .job import Job, JobConfig, BaseModel


class Pipeline(BaseModel):
    jobs: list[JobConfig] = []
    
    def get_jobs(self):
        return (Job(config=config) for config in self.jobs)
    
    @property
    def total_jobs(self):
        return len(self.jobs)
    
    def run(self):
        return [job.run() for job in self.get_jobs()]
    
    @classmethod
    def from_dir(cls, directory: Path | str):
        def get_jobs_from_job_or_pipeline(p: Path):
            try:
                jobs = [JobConfig.from_file(p, fuzzy=False)]
            except Exception:
                try:
                    jobs = cls.from_file(p, fuzzy=False).jobs
                except Exception:
                    jobs = []
            return jobs
        
        jobs = list(set(chain(*[
            get_jobs_from_job_or_pipeline(p) 
            for p in Path(directory).glob("*") 
            if p.suffix in [".json", ".yaml", ".yml"]
        ])))
        return cls(jobs=jobs)
    