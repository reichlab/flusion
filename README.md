# flusion
Influenza forecasting using data fusion

## Environment setup

```
conda env create -f environment.yml
```

If you have an existing `flusion` environment that you want to update to match changes in `environment.yml`, you can do so with the following command:

```
conda env update --file environment.yml --prune
```

## Running unit tests

Unit tests for `gbq` functionality can be run as follows, working within the `code/gbq` subdirectory:

```
conda activate flusion
pytest
```
