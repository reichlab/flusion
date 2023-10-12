library(tidyverse)
library(stringr)

library(lubridate)
library(readr)
library(hubEnsembles)
library(hubUtils)
library(here)
setwd(here::here())

current_ref_date <- lubridate::ceiling_date(Sys.Date(), "week") - days(1)

hub_path <- "submissions-hub"
hub_con <- connect_hub(hub_path)
forecasts <- hub_con |>
  dplyr::filter(
    reference_date == current_ref_date,
    model_id %in% c("UMass-gbq_bootstrap", "UMass-gbq_qr", "UMass-sarix")
  ) |>
  dplyr::collect() |>
  as_model_out_tbl()

forecasts <- forecasts %>%
  dplyr::filter(horizon < 3) %>%
  dplyr::mutate(horizon = horizon + 1)

# generate median ensemble
ensemble_outputs <- hubEnsembles::simple_ensemble(
    forecasts,
    agg_fun="median",
    task_id_cols=c("reference_date", "location", "horizon", "target", "target_end_date")
  ) |>
  dplyr::select(-model_id)


if (!dir.exists(paste0(hub_path, "/model-output/UMass-flusion/"))) {
  dir.create(paste0(hub_path, "/model-output/UMass-flusion/"))
}
readr::write_csv(
  ensemble_outputs,
  paste0(hub_path, "/model-output/UMass-flusion/", current_ref_date, "-UMass-flusion.csv"))





# Stage 2: different data used
forecasts <- hub_con |>
  dplyr::filter(
    reference_date == current_ref_date,
    model_id %in% c("UMass-flusion_first_draft", "UMass-flusion_second_draft")
  ) |>
  dplyr::collect() |>
  as_model_out_tbl()

forecasts <- forecasts %>%
  dplyr::filter(horizon >= 0)

ensemble_outputs <- hubEnsembles::linear_pool(
    forecasts |> dplyr::mutate(output_type_id = as.numeric(output_type_id)),
    task_id_cols=c("reference_date", "location", "horizon", "target", "target_end_date")
  ) |>
  dplyr::select(-model_id)

if (!dir.exists(paste0(hub_path, "/model-output/UMass-flusion/"))) {
  dir.create(paste0(hub_path, "/model-output/UMass-flusion/"))
}
readr::write_csv(
  ensemble_outputs,
  paste0(hub_path, "/model-output/UMass-flusion/", current_ref_date, "-UMass-flusion.csv"))

