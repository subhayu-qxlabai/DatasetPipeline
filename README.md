# DatasetPipeline

A data processing and analysis pipeline designed to handle various jobs related to data transformation, quality assessment, deduplication, and formatting. The pipeline can be configured and executed using YAML configuration files.

## Directory Structure

The directory structure of the project is organized as follows:

- `app/`: Contains the core application code.
  - `translators/`: Modules related to data translation.
  - `analyzer/`: Modules for analyzing data quality and output.
  - `dedup/`: Modules for deduplication logic.
  - `format/`: Modules for data formatting.
  - `helpers/`: Utility modules and helper functions.
  - `loader/`: Modules for loading data from various sources.
  - `models/`: Data models used in the application.
  - `saver/`: Modules for saving processed data.
  - `constants.py`: Contains constant values used across the application.
  - `job.py`: Defines the job structure.
  - `pipeline.py`: Manages the pipeline execution logic.
  - `sample_job.py`: Contains a sample job configuration.

- `jobs/`: Directory for storing job configuration files in YAML format.
- `processed/`: Directory for storing processed data outputs.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `Pipfile` and `Pipfile.lock`: Specify Python dependencies.
- `README.md`: Project documentation.
- `run`: Entry point script for executing the application.
- `run.py`: Main script to run the Typer CLI application.
- `scripts/`: Additional scripts for the project.
  - `pipeline.py`: Script to manage the pipeline commands.

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**
   Make sure you have `pipenv` installed. If not, install it via pip:
   ```bash
   pip install pipenv
   ```
   Then, install the project dependencies:
   ```bash
   pipenv install
   ```

## Usage

The project utilizes Typer for the command-line interface. The main entry point for the CLI is `run.py`. You can perform various operations like listing jobs, running the pipeline, and generating sample job configurations.

### Listing Jobs

To list all jobs in the pipeline, use the following command:

```bash
./run pipeline list <path-to-config-or-directory>
```

### Running the Pipeline

To run the pipeline based on the specified configuration, use:

```bash
./run pipeline run <path-to-config-or-directory>
```

### Generating a Sample Job Configuration

To generate a sample job configuration and print it to stdout or save it to a file, use:

```bash
./run pipeline sample <path-to-output-file>
```

If no file path is specified, the configuration will be printed to stdout.

## Configuration Files

Job configurations are specified in YAML format and stored in the `jobs/` directory. Each configuration file defines the parameters for a specific job or set of jobs to be executed by the pipeline.

### Example Job Configuration

Below is an example job configuration for the pipeline. This configuration file is designed to load data from Hugging Face, format the data, deduplicate it, analyze its quality, and then save the processed data locally.

```yaml
# jobs/aeroboros-conv.yml

load:
    huggingface:
        - path: jondurbin/airoboros-3.2
          split: train
          save: false
          take_rows: 1000

format:
    to_text:
        system:
            template: 'SYSTEM: {system}'
            key: system
        user:
            template: 'USER: {user}'
            key: user
        assistant:
            template: 'ASSISTANT: {assistant}'
            key: assistant
        message_role_field: role
        message_content_field: content
        separator: '\n\n'
    output:
        return_only_messages: true

deduplicate:
    semantic:
        column: messages
        threshold: 0.2

analyze:
    quality:
        column_name: messages
        categories:
            - code
            - math
            - job
            - essay
            - translation
            - literature
            - history
            - science

save:
    local:
        directory: processed
        filetype: parquet
```

## Running the Job

To run this job, use the following command:

```bash
./run pipeline run jobs/aeroboros-conv.yml
```

This command will execute the pipeline according to the configuration specified in the `aeroboros-conv.yml` file, processing the data and saving the results as described.
