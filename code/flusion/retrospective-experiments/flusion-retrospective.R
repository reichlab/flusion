# Create a retrospective flusion ensemble for a given reference date and
# combination of component models

library(dplyr)
library(lubridate)
library(readr)

library(hubEnsembles)
library(hubData)

library(here)
setwd(here::here())

# Parse arguments.  We're expecting a call like
# Rscript --vanilla flusion-retrospective.R 2023-12-30 UMass-gbq_qr UMass-gbq_qr_no_level
args <- commandArgs(trailingOnly = TRUE)
ref_date <- as.Date(args[1])
models <- args[-1]

#' load forecasts
#'
#' @param hub_path string specifying path to hub directory structure
#' @param models character vector of model names to load
#'
#' @return data frame with model outputs
load_forecasts <- function(hub_path, models) {
  hub_con <- connect_hub(hub_path)

  forecasts <- hub_con |>
    dplyr::filter(
      reference_date == ref_date,
      model_id %in% models
    ) |>
    dplyr::collect() |>
    as_model_out_tbl()

  # note: component model horizons are off by 1 due to different accounting in
  # model fitting and hub formats: is horizon relative to last observation or
  # the reference date?  We adjust here to standardize on hub format
  forecasts <- forecasts %>%
    dplyr::filter(horizon < 3) %>%
    dplyr::mutate(horizon = horizon + 1)

  return(forecasts)
}

# load forecasts from retrospective-hub if available
# retrospective predictions exist for dates where a model was not fit in real
# time or there was a bug affecting its real-time predictions
forecasts <- load_forecasts("retrospective-hub", models)

# load forecasts from submissions-hub if they weren't available in
# retrospective-hub
submissions_models <- models[!models %in% forecasts$model_id]
if (length(submissions_models) > 0) {
  submissions_forecasts <- load_forecasts("submissions-hub", submissions_models)

  forecasts <- dplyr::bind_rows(
    forecasts,
    submissions_forecasts
  )
}

# generate mean ensemble
ensemble_outputs <- hubEnsembles::simple_ensemble(
    forecasts,
    agg_fun = "mean",
    task_id_cols = c("reference_date", "location", "horizon", "target",
                     "target_end_date"),
  ) |>
  dplyr::select(-model_id)

# save outputs
if (length(models) == 3) {
  # all 3 component models; ensemble is just flusion
  ensemble_name <- "UMass-flusion"
} else {
  # just 2 component models; put them in the ensemble model name
  components_short <- gsub("UMass-", "", models)
  ensemble_name <- paste0(
    "UMass-flusion__",
    paste0(components_short, collapse = "__")
  )
}

output_dir <- paste0("retrospective-hub", "/model-output/", ensemble_name, "/")
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

readr::write_csv(
  ensemble_outputs,
  paste0(output_dir, ref_date, "-", ensemble_name, ".csv")
)
