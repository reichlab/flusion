# flusion

Influenza forecasting using data fusion

## Repository organization

This repository has the following directories:

- `code/`: all code for flusion and its component models, exploratory data analyses, and so on. See the readme in that folder for further information.
- `data-raw/`: raw data measuring influenza activity, pulled from various sources. See the readme in that folder for further information.
- `retrospective-hub/`: a folder with the same structure as the FluSight forecast hub, used for storing model outputs generated in retrospective analyses.
- `submissions-hub/`: a folder with the same structure as the FluSight forecast hub, used for storing model outputs geneated in real time for purposes of weekly model submissions. This includes model outputs from flusion as well as its component models.

## Environment setup

### Docker-free

```
conda env create -f environment.yml
```

If you have an existing `flusion` environment that you want to update to match changes in `environment.yml`, you can do so with the following command:

```
conda env update --file environment.yml --prune
```

### Using Docker

Build the flusion image with the following command:

```
docker build -t flusion .
```

Now, you can run commands such as the following. Note that these commands mount the `retrospective-hub` and `submissions-hub` directories as volumes in the container, and when commands run within the container, the `flusion` conda environment is activated by default.
```
docker run -dit \
    -w /flusion/code/gbq \
    -v ./retrospective-hub:/flusion/retrospective-hub \
    -v ./submissions-hub:/flusion/submissions-hub \
    flusion python /flusion/code/gbq/gbq.py --short_run --output_root ../../retrospective-hub/model-output

docker run -dit \
    -w /flusion/code/gbq \
    -v ./retrospective-hub:/flusion/retrospective-hub \
    -v ./submissions-hub:/flusion/submissions-hub \
    flusion pytest
```

## Running unit tests

Unit tests for `gbq` functionality can be run as follows, working within the `code/gbq` subdirectory:

```
conda activate flusion
pytest
```
