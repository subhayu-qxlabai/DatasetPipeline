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
        return [pipeline.run() for pipeline in self.get_jobs()]
    
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
    