# flusion

Influenza forecasting using data fusion.

## Repository organization

This repository has the following directories:

- `code/`: all code for flusion and its component models, exploratory data analyses, and so on. See the readme in that folder for further information.
- `data-raw/`: raw data measuring influenza activity, pulled from various sources. See the readme in that folder for further information.
- `retrospective-hub/`: a folder with the same structure as the FluSight forecast hub, used for storing model outputs generated in retrospective analyses.
- `submissions-hub/`: a folder with the same structure as the FluSight forecast hub, used for storing model outputs geneated in real time for purposes of weekly model submissions. This includes model outputs from flusion as well as its component models.

## Environment setup

### Without Docker

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

On a Mac with an M-series processor, you may need to use the following command to build the image:

```
docker build --platform linux/amd64 -t flusion .
```

Now you can run commands such as the following:
```
docker run -it \
    -w /flusion/code/gbq \
    -v ./retrospective-hub:/flusion/retrospective-hub \
    -v ./submissions-hub:/flusion/submissions-hub \
    flusion python /flusion/code/gbq/gbq.py --short_run --output_root ../../retrospective-hub/model-output
```

Here are a few notes about this:

- Here, we set the working directory to be `/flusion/code/gbq`, which is suitable for running variations on the GBQ model.
- We mount the `retrospective-hub` and `submissions-hub` directories as volumes in the container; any outputs saved to those locations will be persisted outside of the container.
- The Dockerfile is set up so that when commands run within the container, the `flusion` conda environment is activated by default.

As another example, the following command will put you into a `bash` shell running in the container:
```
docker run -it \
    -w /flusion/code/gbq \
    -v ./retrospective-hub:/flusion/retrospective-hub \
    -v ./submissions-hub:/flusion/submissions-hub \
    flusion bash
```

## Running unit tests

**Note:** Lightgbm model runs don't consistently yield the same results on different machines, so you should expect one failure for an integration test that compares model predictions to stored results for a small model run (unless you get lucky). This is true regardless of whether or not you're using Docker.

### Without Docker

Unit tests for `gbq` functionality can be run as follows, working within the `code/gbq` subdirectory:

```
conda activate flusion
pytest
```

#### Using Docker

Unit tests for `gbq` functionality can be run as follows, working within the root of this repository:

```
docker run -it \
    -w /flusion/code/gbq \
    -v ./retrospective-hub:/flusion/retrospective-hub \
    -v ./submissions-hub:/flusion/submissions-hub \
    flusion pytest
```
