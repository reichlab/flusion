# Plot predictions in tests/test_gbq_qr/2024-03-30-UMass-gbq_qr.csv
# Intended as a visual check that these are plausible predictions
# With working directory as code/gbq, can run with
# Rscript tests/test_gbq_qr/plot_prediction.R

library(hubVis)
library(dplyr)

forecast <- read.csv("tests/test_gbq_qr/2024-03-30-UMass-gbq_qr.csv") |>
    dplyr::mutate(model_id = "UMass-gbq_qr")

target_data <- read.csv("../../data-raw/influenza-hhs/hhs-2024-03-27.csv") |>
    dplyr::mutate(observation = inc)

plot_step_ahead_model_output(
    forecast,
    target_data |> dplyr::filter(date >= "2023-10-01"),
    x_col_name = "target_end_date",
    x_target_col_name = "date",
    intervals = 0.95,
    facet = "location",
    facet_scales = "free_y",
    facet_nrow = 15,
    use_median_as_point = TRUE,
    interactive = FALSE,
    show_plot = TRUE
  )
