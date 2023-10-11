library(hubValidations)
library(hubVis)
library(hubUtils)

hubValidations::validate_submission(
    hub_path="submissions-hub",
    file_path="UMass-gbq_bootstrap/2023-10-14-UMass-gbq_bootstrap.csv")

hubValidations::validate_submission(
    hub_path="submissions-hub",
    file_path="UMass-gbq_qr/2023-10-14-UMass-gbq_qr.csv")

hubValidations::validate_submission(
    hub_path=".",
    file_path="FluSight-ensemble/2023-10-07-FluSight-ensemble.csv")


forecast <- dplyr::bind_rows(
  read.csv("submissions-hub/model-output/UMass-gbq_bootstrap/2023-10-14-UMass-gbq_bootstrap.csv") |>
    dplyr::mutate(model_id = "UMass-gbq_bootstrap"),
  read.csv("submissions-hub/model-output/UMass-gbq_qr/2023-10-14-UMass-gbq_qr.csv") |>
    dplyr::mutate(model_id = "UMass-gbq_qr")
)
forecast <- as_model_out_tbl(forecast)
head(forecast)

target_data <- readr::read_csv("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/target-data/target-hospital-admissions.csv")
head(target_data)


for (timespan in c("last_season", "rolling_12wk")) {
  if (timespan == "last_season") {
    data_start = "2022-10-01"
  } else {
    data_start = max(target_data$date) - 12 * 7
  }
  
  for (incl_models in c("both", "bootstrap", "qr")) {
    if (incl_models == "both") {
      incl_models_vec = c("UMass-gbq_bootstrap", "UMass-gbq_qr")
    } else if (incl_models == "bootstrap") {
      incl_models_vec = "UMass-gbq_bootstrap"
    } else {
      incl_models_vec = "UMass-gbq_qr"
    }
    
    p <- plot_step_ahead_model_output(
      forecast |> dplyr::filter(model_id %in% incl_models_vec),
      target_data |> dplyr::filter(date >= data_start),
      x_col_name = "target_end_date",
      x_truth_col_name = "date",
      facet = "location",
      facet_scales = "free_y",
      facet_nrow = 15,
      use_median_as_point = TRUE,
      interactive = FALSE,
      show_plot = FALSE
    )
    
    save_dir <- paste0("plots/2023-10-14")
    if (!dir.exists(save_dir)) {
      dir.create(save_dir, recursive = TRUE)
    }
    
    save_path <- file.path(save_dir, paste0("2023-10-14", "_", incl_models, "_", timespan, ".pdf"))
    
    pdf(save_path, height = 24, width = 14)
    print(p)
    dev.off()
  }
}
