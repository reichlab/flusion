library(tidyverse)
library(stringr)

library(lubridate)
library(readr)
library(hubEnsembles)
library(hubUtils)
library(here)
setwd(here::here())

current_ref_date <- lubridate::ceiling_date(Sys.Date(), "week") - days(1)
last_ref_date <- current_ref_date - 7

hub_path <- "submissions-hub"
hub_con <- connect_hub(hub_path)
forecasts <- hub_con |>
  dplyr::filter(
    reference_date == current_ref_date,
    model_id %in% c("UMass-gbq_qr", "UMass-sarix")
    # model_id %in% c("UMass-gbq_bootstrap", "UMass-gbq_qr", "UMass-sarix")
  ) |>
  dplyr::collect() |>
  as_model_out_tbl()

forecasts <- forecasts %>%
  dplyr::filter(horizon < 3) %>%
  dplyr::mutate(horizon = horizon + 1)

# generate median ensemble
ensemble_outputs <- hubEnsembles::simple_ensemble(
    forecasts,
    # agg_fun="median",
    agg_fun="mean",
    task_id_cols=c("reference_date", "location", "horizon", "target", "target_end_date"),
  ) |>
  dplyr::select(-model_id)


# repeat for last week
last_forecasts <- hub_con |>
  dplyr::filter(
    reference_date == last_ref_date,
    model_id %in% c("UMass-gbq_qr", "UMass-sarix")
  ) |>
  dplyr::collect() |>
  as_model_out_tbl()

last_forecasts <- last_forecasts |>
  dplyr::mutate(
    reference_date = as.Date(current_ref_date),
    horizon = as.integer((target_end_date - reference_date) / 7)) |>
  dplyr::filter(horizon >= 0, location == "02")

# generate median ensemble
last_ensemble_outputs <- hubEnsembles::simple_ensemble(
    last_forecasts,
    # agg_fun="median",
    agg_fun="mean",
    task_id_cols=c("reference_date", "location", "horizon", "target", "target_end_date"),
  ) |>
  dplyr::select(-model_id)


# linear pool to combine last and current
combined_outputs <- dplyr::bind_rows(
  ensemble_outputs |> dplyr::mutate(model_id = "current"),
  last_ensemble_outputs |> dplyr::mutate(model_id = "last")
)

final_outputs <- hubEnsembles::linear_pool(
    combined_outputs |> dplyr::mutate(output_type_id = as.numeric(output_type_id)),
    task_id_cols=c("reference_date", "location", "horizon", "target", "target_end_date")
  ) |>
  dplyr::select(-model_id)


if (!dir.exists(paste0(hub_path, "/model-output/UMass-flusion/"))) {
  dir.create(paste0(hub_path, "/model-output/UMass-flusion/"))
}
readr::write_csv(
  final_outputs,
  paste0(hub_path, "/model-output/UMass-flusion/", current_ref_date, "-UMass-flusion.csv"))





# # Stage 2: different data used
# forecasts <- hub_con |>
#   dplyr::filter(
#     reference_date == current_ref_date,
#     model_id %in% c("UMass-flusion_first_draft", "UMass-flusion_second_draft")
#   ) |>
#   dplyr::collect() |>
#   as_model_out_tbl()

# forecasts <- forecasts %>%
#   dplyr::filter(horizon >= 0)

# ensemble_outputs <- hubEnsembles::linear_pool(
#     forecasts |> dplyr::mutate(output_type_id = as.numeric(output_type_id)),
#     task_id_cols=c("reference_date", "location", "horizon", "target", "target_end_date")
#   ) |>
#   dplyr::select(-model_id)

# if (!dir.exists(paste0(hub_path, "/model-output/UMass-flusion/"))) {
#   dir.create(paste0(hub_path, "/model-output/UMass-flusion/"))
# }
# readr::write_csv(
#   ensemble_outputs,
#   paste0(hub_path, "/model-output/UMass-flusion/", current_ref_date, "-UMass-flusion.csv"))


