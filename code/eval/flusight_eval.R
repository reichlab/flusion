library(hubUtils)
library(scoringutils)

library(lubridate)
library(dplyr)
library(ggplot2)
library(plotly)

library(here)
setwd(here::here())

current_ref_date <- lubridate::ceiling_date(Sys.Date(), "week") - lubridate::days(1)

hub_path <- "../FluSight-forecast-hub"

hub_con <- connect_hub(hub_path)
forecasts <- hub_con |>
  dplyr::filter(
    output_type == "quantile"
  ) |>
  dplyr::collect() |>
  as_model_out_tbl() |>
  dplyr::filter(horizon >= 0, reference_date >= "2023-10-07", location != "US", location != "78") #|>

# ens_fc <- forecasts |>
#   dplyr::filter(model_id == "FluSight-ensemble",
#                 horizon >= 0, reference_date >= "2023-10-07", location != "US", location != "78") #|>
  # dplyr::mutate(quantile = format(quantile, 3)) %>%
  # dplyr::filter(quantile == "0.500")



target_data <- readr::read_csv("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/target-data/target-hospital-admissions.csv")
head(target_data)


data_for_su <- forecasts |>
  dplyr::left_join(
    target_data |> dplyr::select(target_end_date = date, location, true_value = value),
    by = c("location", "target_end_date")
  ) |>
  dplyr::rename(model=model_id, quantile=output_type_id, prediction=value) |>
  dplyr::mutate(quantile = as.numeric(quantile))

# data_for_su %>%
#   dplyr::filter(model == "FluSight-ensemble") %>%
#   dplyr::mutate(quantile = format(quantile, 3)) %>%
#   dplyr::filter(quantile == "0.500") %>%
#   nrow()

# data_for_su %>%
#   dplyr::filter(model == "FluSight-ensemble") %>%
#   dplyr::mutate(quantile = format(quantile, 3)) %>%
#   dplyr::filter(quantile == "0.500") %>%
#   dplyr::mutate(ae = abs(prediction - true_value)) %>%
#   dplyr::filter(!is.na(ae)) %>%
#   dplyr::summarize(mae = mean(ae)) %>%
#   dplyr::pull(mae)

# data_for_su %>%
#   dplyr::filter(model == "FluSight-ensemble") %>%
#   dplyr::mutate(quantile = format(quantile, 3)) %>%
#   dplyr::filter(quantile %in% c("0.025", "0.975")) %>%
#   tidyr::pivot_wider(names_from = quantile, values_from = prediction) %>%
#   dplyr::mutate(in_interval = ((true_value >= `0.025`) & (true_value <= `0.975`))) %>%
#   dplyr::filter(!is.na(in_interval)) %>%
#   dplyr::summarize(cov_95 = mean(in_interval)) %>%
#   dplyr::pull(cov_95)

#   dplyr::mutate(cov_95 = abs(prediction - true_value)) %>%
#   dplyr::filter(!is.na(ae)) %>%
#   dplyr::summarize(mae = mean(ae)) %>%
#   dplyr::pull(mae)



# temp <- data_for_su %>%
#   dplyr::filter(model == "FluSight-ensemble", horizon >= 0, location != "US", location != "78") %>%
#   # dplyr::pull(location) %>%
#   # unique() %>%
#   # length()
#   dplyr::mutate(quantile = format(quantile, 3)) %>%
#   dplyr::filter(quantile == "0.500") #%>%
# #  nrow()

# data_for_su %>%
#   dplyr::filter(model == "FluSight-ensemble", horizon >= 0) %>%
#   dplyr::distinct(location, horizon, reference_date)


#   dplyr::mutate(ae = abs(prediction - true_value)) %>%
#   dplyr::filter(!is.na(ae)) %>%
#   dplyr::summarize(mae = mean(ae))


# data_for_su |>
#   scoringutils::check_forecasts()

scores_raw <- data_for_su |>
  scoringutils::score()

# scores_raw |>
#   add_coverage(ranges = c(50, 80, 95), by = c("model")) |>
#   summarise_scores(by = c("model"),
#                    relative_skill = TRUE,
#                    baseline = "FluSight-baseline")


# scores <- scores_raw |>
#   add_coverage(ranges = c(50, 80, 95), by = c("model", "reference_date")) |>
#   summarise_scores(by = c("model", "reference_date"),
#                    relative_skill = TRUE,
#                    baseline = "FluSight-baseline")

n_locs <- dplyr::distinct(forecasts, model_id, location) |>
  group_by(model_id) |>
  summarize(n_locs = n())

models_to_keep <- n_locs |>
  dplyr::filter(n_locs > 50) |>
  dplyr::pull(model_id)


# ggplot(data = scores |>
#                 dplyr::filter(model %in% models_to_keep) |>
#                 dplyr::mutate(is_umass = grepl("UMass", model))) +
#   geom_line(mapping = aes(x = reference_date, y = interval_score, color = model, size = factor(is_umass))) +
#   scale_size_manual(values = c(0.25, 1)) +
#   theme_bw()





scores_w_horizon <- scores_raw |>
  add_coverage(ranges = c(50, 80, 95), by = c("model", "reference_date", "horizon")) |>
  summarise_scores(by = c("model", "reference_date", "horizon"))#,
                  #  relative_skill = TRUE,
                  #  relative_skill_metric = "interval_score",
                  #  baseline = "FluSight-baseline")

p_by_horizon <- ggplot(data = scores_w_horizon |>
                dplyr::filter(model %in% models_to_keep) |>
                dplyr::mutate(is_umass = grepl("UMass", model))) +
  geom_line(mapping = aes(x = reference_date, y = interval_score, color = model, size = factor(is_umass))) +
  geom_point(mapping = aes(x = reference_date, y = interval_score, color = model, size = factor(is_umass))) +
  scale_size_manual(values = c(0.25, 1)) +
  facet_wrap(~ horizon) +
  theme_bw()

ggplotly(p_by_horizon)


p_by_horizon <- ggplot(data = scores_w_horizon |>
                dplyr::filter(model %in% models_to_keep) |>
                dplyr::mutate(is_umass = grepl("UMass", model))) +
  geom_line(mapping = aes(x = reference_date, y = ae_median, color = model, size = factor(is_umass))) +
  geom_point(mapping = aes(x = reference_date, y = ae_median, color = model, size = factor(is_umass))) +
  scale_size_manual(values = c(0.25, 1)) +
  facet_wrap(~ horizon) +
  theme_bw()

ggplotly(p_by_horizon)



coverage_by_horizon <- ggplot(data = scores_w_horizon |>
                dplyr::filter(model %in% models_to_keep) |>
                dplyr::mutate(is_umass = grepl("UMass", model))) +
  geom_line(mapping = aes(x = reference_date, y = coverage_80, color = model, size = factor(is_umass))) +
  geom_point(mapping = aes(x = reference_date, y = coverage_80, color = model, size = factor(is_umass))) +
  geom_hline(yintercept=0.8, linetype=2) +
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
