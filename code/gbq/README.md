# gbq

## File organization

This folder contains code related to the GBQ models.  It contains the following files and directories:
- Current code for model running:
    - `gbq.py`: the main entry point for running GBQ models
    - `run.py`: internal functions for running GBQ models
    - `preprocess.py`: internal functions for running GBQ models
    - `utils.py`: internal functions for running GBQ models
    - `tests/`: has a single integration test, used to ensure code changes don't break functionality.
    - `configs/`: defines configuration settings for the `gbq_qr` and `gbq_qr_no_level` models.
- Legacy notebook files. These were used for model development and for generating real-time submissions up through reference date 2024-04-13. They are not currently used; eventually, they may be deleted once all necessary code is removed from them.
    - gbq_qr.ipynb: obsolete except for plotting code and historical interest
    - gbq_qr_no_level.ipynb: obsolete except for plotting code and historical interest
    - gbq_bootstrap.ipynb: methods not yet implemented in gbq python modules. Anecdotally, this method seemed to generate conservative (wide) prediction intervals and so porting over implementation is a low priority for now. However, this is the method used for covid forecasting by our group.
    - gbq_qr_no_source.ipynb: methods not yet implemented in gbq python modules. Anecdotally, this method seemed to have minimal impact on predictions relative to "plain vanilla" `gbq_qr`.

## Running the integration test

The integration test for GBQ functionality can be run as follows:

```
conda activate flusion
pytest
```

## Generating weekly forecast submission files

Weekly submission files for the `gbq_qr` and `gbq_qr_no_level` models can be generated as follows. Model output files in csv format will be written to `flusion/submissions-hub/model-output`.

```
python gbq.py --model_name gbq_qr
python gbq.py --model_name gbq_qr_no_level
```
