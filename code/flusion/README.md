# flusion

Scripts in this folder were used to create flusion ensembles in real time in the 2023/34 season, while scripts in the `retrospective-experiments` subfolder contain script for creating retrospective ensemble predictions to support manuscript writing. All of these scripts take as input component model submission files located in the `submissions-hub` and `retrospective-hub` folders in the root of the `flusion` repository.

These scripts are as follows:

- `flusion.R` is the script we ran to create real-time ensembles for submission on a weekly basis.
- `flusion-manual-2023-11-22.R` was used to produce ensemble forecasts for the reference date 2023-11-25. That week, reporting in Alaska appeared to be artificially low, and for that location only we used an ad hoc method that was a linear pool of two ensembles: one produced using all available data, and one produced using all data up through the previous week (i.e., omitting the last observation).
- `validate_and_plot.R` was used to create plots of the predictions each week for manual inspection before submission.
- `retrospective-experiments/flusion-retrospective.R` creates a retrospective flusion ensemble for a given reference date and combination of component models
- `retrospective-experiments/run-flusion-retrospective.R` runs the `retrospective-experiments/flusion-retrospective.R` script repeatedly to create retrospective flusion ensembles based on different subsets of the component models.
