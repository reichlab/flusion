library(hubUtils)
library(hubData)
library(scoringutils)

library(lubridate)
library(dplyr)
library(ggplot2)
library(plotly)

library(here)
setwd(here::here())

source("code/eval/scoring_helpers.R")

current_ref_date <- lubridate::ceiling_date(Sys.Date(), "week") - lubridate::days(1)

hub_path <- "../FluSight-forecast-hub"

hub_con <- connect_hub(hub_path)
forecasts <- hub_con |>
  dplyr::filter(
    output_type == "quantile"
  ) |>
  dplyr::collect() |>
  as_model_out_tbl() |>
  dplyr::filter(horizon >= 0,
                reference_date >= "2023-10-14",
                location != "US",
                location != "78")

target_data <- readr::read_csv("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/target-data/target-hospital-admissions.csv")
head(target_data)

by <- list("model",
          #  c("model", "horizon"),
           c("model", "horizon", "reference_date"))

by <- list(#"model",
          #  c("model", "horizon"),
           c("model", "horizon", "reference_date"))


scores <- compute_scores(forecasts = forecasts,
                         target_data = target_data,
                         by = by,
                         submission_threshold = 0.75)

scores[[1]][, 1:8]

scores_w_horizon <- scores[[2]]
p_by_horizon <- ggplot(data = scores_w_horizon |>
                # dplyr::filter(model %in% models_to_keep) |>
                dplyr::mutate(is_umass = grepl("UMass", model, fixed = TRUE))) +
  geom_line(mapping = aes(x = reference_date, y = wis_scaled_relative_skill,
                          color = model, size = factor(is_umass))) +
  geom_point(mapping = aes(x = reference_date, y = wis_scaled_relative_skill,
                           color = model, size = factor(is_umass))) +
  scale_size_manual(values = c(0.25, 1)) +
  facet_wrap(~ horizon) +
  theme_bw()

ggplotly(p_by_horizon)




# p_by_horizon <- ggplot(data = scores_w_horizon |>
#                 dplyr::filter(model %in% models_to_keep) |>
#                 dplyr::mutate(is_umass = grepl("UMass", model))) +
#   geom_line(mapping = aes(x = reference_date, y = ae_median, color = model, size = factor(is_umass))) +
#   geom_point(mapping = aes(x = reference_date, y = ae_median, color = model, size = factor(is_umass))) +
#   scale_size_manual(values = c(0.25, 1)) +
#   facet_wrap(~ horizon) +
#   theme_bw()

ggplotly(p_by_horizon)



coverage_by_horizon <- ggplot(data = scores_w_horizon |>
                # dplyr::filter(model %in% models_to_keep) |>
                dplyr::mutate(is_umass = grepl("UMass", model))) +
  geom_line(mapping = aes(x = reference_date, y = interval_coverage_50, color = model, size = factor(is_umass))) +
  geom_point(mapping = aes(x = reference_date, y = interval_coverage_50, color = model, size = factor(is_umass))) +
  geom_hline(yintercept=0.5, linetype=2) +
  scale_size_manual(values = c(0.25, 1)) +
  facet_wrap(~ horizon) +
  theme_bw()

ggplotly(coverage_by_horizon)



coverage_by_horizon <- ggplot(data = scores_w_horizon |>
                # dplyr::filter(model %in% models_to_keep) |>
                dplyr::mutate(is_umass = grepl("UMass", model))) +
  geom_line(mapping = aes(x = reference_date, y = interval_coverage_95, color = model, size = factor(is_umass))) +
  geom_point(mapping = aes(x = reference_date, y = interval_coverage_95, color = model, size = factor(is_umass))) +
  geom_hline(yintercept=0.95, linetype=2) +
  scale_size_manual(values = c(0.25, 1)) +
  facet_wrap(~ horizon) +
  theme_bw()

ggplotly(coverage_by_horizon)


coverage_by_horizon <- ggplot(data = scores_w_horizon |>
                dplyr::filter(model %in% models_to_keep) |>
                dplyr::mutate(is_umass = grepl("UMass", model))) +
  geom_line(mapping = aes(x = reference_date, y = coverage_95, color = model, size = factor(is_umass))) +
  geom_point(mapping = aes(x = reference_date, y = coverage_95, color = model, size = factor(is_umass))) +
  geom_hline(yintercept=0.95, linetype=2) +
  scale_size_manual(values = c(0.25, 1)) +
  facet_wrap(~ horizon) +
  theme_bw()

ggplotly(coverage_by_horizon)



scoringutils::set_forecast_unit(temp, c("model", "reference_date", "target", "horizon", "target_end_date", "location"))

first_mon <- as.Date("2023-10-14")


last_mon <- lubridate::floor_date(Sys.Date(), unit = "week") + 1 - 7
fcast_dates <- seq.Date(from = first_mon, to = last_mon, by = 7)
fcasts <- load_forecasts(
  source = "local_hub_repo",
  dates = fcast_dates,
  targets = c(paste0(rep(1:4), " wk ahead inc flu hosp")),
  data_processed_subpath = "data-forecasts/",
  hub_repo_path = "../Flusight-forecast-data/",
  hub = "FluSight")

fcasts <- fcasts %>% filter(forecast_date >= "2022-07-01")

fcasts %>%
  dplyr::count(model, forecast_date)

truth <- load_truth(hub = "FluSight")

scores <- score_forecasts(fcasts,truth = truth,metrics = c("abs_error","wis", "wis_components", "interval_coverage"),use_median_as_point=TRUE)

mean_wis_horizon_1 <- scores %>%
    filter(horizon == 1) %>%
    group_by(model) %>%
    summarize(mean_wis = mean(wis)) %>%
    arrange(mean_wis)

ggplot(scores %>% dplyr::mutate(model = factor(model, levels = mean_wis_horizon_1$model)),
        aes(x=model,y=log(wis))) +
    geom_boxplot() +
    geom_hline(yintercept = median(log(scores[scores$model == "Flusight-baseline",]$wis))) +
    facet_wrap(~ horizon) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

p <- scores %>%
  dplyr::group_by(model, forecast_date) %>%
  dplyr::summarize(
    wis = mean(wis),
    is_umass_or_ensemble = (model %in% c("UMass-trends_ensemble", "UMass-gbq", "Flusight-ensemble"))
  ) %>%
  ggplot() +
    geom_line(mapping = aes(x = forecast_date, y = wis, color = model, size = factor(is_umass_or_ensemble))) +
    scale_y_log10() +
    scale_size_manual(values = c(0.5, 3.0)) +
    theme_bw()
#p
library(plotly)
ggplotly(p)

p <- scores %>%
  dplyr::group_by(model, forecast_date) %>%
  dplyr::summarize(
    mae = mean(abs_error),
    is_umass_or_ensemble = (model %in% c("UMass-trends_ensemble", "UMass-gbq", "Flusight-ensemble"))
  ) %>%
  ggplot() +
    geom_line(mapping = aes(x = forecast_date, y = mae, color = model, size = factor(is_umass_or_ensemble))) +
    scale_y_log10() +
    scale_size_manual(values = c(0.5, 3.0)) +
    theme_bw()
ggplotly(p)


p <- scores %>%
  dplyr::group_by(model, forecast_date) %>%
  dplyr::summarize(
    coverage_95 = mean(coverage_95),
    is_umass_or_ensemble = (model %in% c("UMass-trends_ensemble", "UMass-gbq", "Flusight-ensemble"))
  ) %>%
  ggplot() +
    geom_line(mapping = aes(x = forecast_date, y = coverage_95, color = model, size = factor(is_umass_or_ensemble))) +
    geom_hline(yintercept = 0.95) +
    scale_size_manual(values = c(0.5, 3.0)) +
    theme_bw()
ggplotly(p)

p <- scores %>%
  dplyr::group_by(model, forecast_date) %>%
  dplyr::summarize(
    coverage_50 = mean(coverage_50),
    is_umass_or_ensemble = (model %in% c("UMass-trends_ensemble", "UMass-gbq", "Flusight-ensemble"))
  ) %>%
  ggplot() +
    geom_line(mapping = aes(x = forecast_date, y = coverage_50, color = model, size = factor(is_umass_or_ensemble))) +
    geom_hline(yintercept = 0.50) +
    scale_size_manual(values = c(0.5, 3.0)) +
    theme_bw()
ggplotly(p)


scores %>%
  group_by(model) %>%
  summarize(
    n = n(),
    wis = mean(wis),
    mae = mean(abs_error),
    coverage_50 = mean(coverage_50),
    coverage_80 = mean(coverage_80),
    coverage_95 = mean(coverage_95),
    wis_dispersion = mean(dispersion),
    wis_over = mean(overprediction),
    wis_under = mean(underprediction)
  ) %>%
  arrange(wis) %>%
  as.data.frame()
