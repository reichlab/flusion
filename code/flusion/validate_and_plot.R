library(hubValidations)
library(hubVis)
library(hubUtils)
library(lubridate)

current_ref_date <- lubridate::ceiling_date(Sys.Date(), "week") - days(1)

locations <- read.csv("submissions-hub/auxiliary-data/locations.csv")

# hubValidations::validate_submission(
#     hub_path="submissions-hub",
#     file_path=paste0("UMass-flusion/", current_ref_date, "-UMass-flusion.csv"))

# hubValidations::validate_submission(
#     hub_path="submissions-hub",
#     file_path="UMass-gbq_bootstrap/2023-10-14-UMass-gbq_bootstrap.csv")

# hubValidations::validate_submission(
#     hub_path="submissions-hub",
#     file_path="UMass-gbq_qr/2023-10-14-UMass-gbq_qr.csv")

# hubValidations::validate_submission(
#     hub_path="submissions-hub",
#     file_path="UMass-sarix_4rt/2023-10-14-UMass-sarix_4rt.csv")

# hubValidations::validate_submission(
#     hub_path="submissions-hub",
#     file_path="UMass-sarix_sqrt/2023-10-14-UMass-sarix_sqrt.csv")

# hubValidations::validate_submission(
#     hub_path=".",
#     file_path="FluSight-ensemble/2023-10-07-FluSight-ensemble.csv")


forecast <- dplyr::bind_rows(
  # read.csv("submissions-hub/model-output/UMass-flusion_first_draft/2023-10-14-UMass-flusion_first_draft.csv") |>
  #   dplyr::mutate(model_id = "UMass-flusion_first_draft"),
  # read.csv("submissions-hub/model-output/UMass-flusion_second_draft/2023-10-14-UMass-flusion_second_draft.csv") |>
  #   dplyr::mutate(model_id = "UMass-flusion_second_draft"),
  read.csv(paste0("submissions-hub/model-output/UMass-flusion/", current_ref_date, "-UMass-flusion.csv")) |>
    dplyr::mutate(model_id = "UMass-flusion"),
  # read.csv(paste0("submissions-hub/model-output/UMass-flusion_frog/", current_ref_date, "-UMass-flusion_frog.csv")) |>
  #  dplyr::mutate(model_id = "UMass-flusion_frog"),
  # read.csv(paste0("submissions-hub/model-output/UMass-flusion_hamster/", current_ref_date, "-UMass-flusion_hamster.csv")) |>
  #  dplyr::mutate(model_id = "UMass-flusion_hamster"),
  # read.csv(paste0("submissions-hub/model-output/UMass-gbq_bootstrap/", current_ref_date, "-UMass-gbq_bootstrap.csv")) |>
  #   dplyr::mutate(model_id = "UMass-gbq_bootstrap"),
  read.csv(paste0("submissions-hub/model-output/UMass-gbq_qr/", current_ref_date, "-UMass-gbq_qr.csv")) |>
    dplyr::mutate(model_id = "UMass-gbq_qr"),
  read.csv(paste0("submissions-hub/model-output/UMass-gbq_qr_no_level/", current_ref_date, "-UMass-gbq_qr_no_level.csv")) |>
    dplyr::mutate(model_id = "UMass-gbq_qr_no_level"),
  read.csv(paste0("submissions-hub/model-output/UMass-sarix/", current_ref_date, "-UMass-sarix.csv")) |>
    dplyr::mutate(model_id = "UMass-sarix"),
  # read.csv("submissions-hub/model-output/UMass-sarix_4rt/2023-10-14-UMass-sarix_4rt.csv") |>
  #   dplyr::mutate(model_id = "UMass-sarix_4rt")
) |>
  dplyr::left_join(locations)
head(forecast)

target_data <- readr::read_csv("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/target-data/target-hospital-admissions.csv")
head(target_data)


for (timespan in c("last_season", "rolling_12wk")) {
  if (timespan == "last_season") {
    data_start = "2022-10-01"
  } else {
    data_start = max(target_data$date) - 12 * 7
  }

  for (incl_models in c("components", "flusion_compare")) {
  # for (incl_models in c("all")) {
  # for (incl_models in c("flusion_compare")) {
  # for (incl_models in c("all", "qr", "sarix", "flusion")) {
  # for (incl_models in c("all", "bootstrap", "qr", "sarix")) {
  # for (incl_models in c("sarix_sqrt", "sarix_4rt", "sarix")) {
  # for (incl_models in c("flusion")) {
  # for (incl_models in c("flusion_compare", "flusion")) {
    if (incl_models == "all") {
      incl_models_vec = c("UMass-gbq_qr", "UMass-sarix", "UMass-flusion")
    } else if (incl_models == "sarix") {
      incl_models_vec = "UMass-sarix"
    } else if (incl_models == "bootstrap") {
      incl_models_vec = "UMass-gbq_bootstrap"
    } else if (incl_models == "flusion") {
      incl_models_vec = "UMass-flusion"
    } else if (incl_models == "flusion_compare") {
      incl_models_vec = c("UMass-flusion", "UMass-flusion_frog", "UMass-flusion_hamster")
    } else if (incl_models == "qr") {
      incl_models_vec = "UMass-gbq_qr"
    } else if (incl_models == "qr_compare") {
      incl_models_vec = c("UMass-gbq_qr", "UMass-gbq_qr_no_level")
    } else if (incl_models == "components") {
      incl_models_vec = c("UMass-gbq_qr", "UMass-gbq_qr_no_level", "UMass-sarix")
    } else if (incl_models == "sarix_sqrt") {
      incl_models_vec = "UMass-sarix_sqrt"
    } else if (incl_models == "sarix_4rt") {
      incl_models_vec = "UMass-sarix_4rt"
    }

    p <- plot_step_ahead_model_output(
      forecast |> dplyr::filter(model_id %in% incl_models_vec),
      target_data |> dplyr::filter(date >= data_start),
      x_col_name = "target_end_date",
      x_truth_col_name = "date",
      intervals = 0.95,
      facet = "location_name",
      facet_scales = "free_y",
      facet_nrow = 15,
      use_median_as_point = TRUE,
      interactive = FALSE,
      show_plot = FALSE
    )

    save_dir <- paste0("plots/", current_ref_date)
    if (!dir.exists(save_dir)) {
      dir.create(save_dir, recursive = TRUE)
    }

    save_path <- file.path(save_dir, paste0(current_ref_date, "_", incl_models, "_", timespan, ".pdf"))

    pdf(save_path, height = 24, width = 14)
    print(p)
    dev.off()
  }
}
