#' Combine forecast and target data 
merge_data_for_su <- function(forecasts, target_data, models_to_keep) {
  data_for_su <- forecasts |>
    dplyr::left_join(
      target_data |>
        dplyr::select(target_end_date = date, location, observed = value),
      by = c("location", "target_end_date")
    ) |>
    dplyr::rename(quantile_level = output_type_id, predicted = value,
                  model = "model_id") |>
    dplyr::mutate(quantile_level = as.numeric(quantile_level))
  
  return(data_for_su)
}

#' Compute raw scores
compute_raw_scores <- function(data_for_su, scoring_funs) {
  raw_scores <- data_for_su |>
    scoringutils::set_forecast_unit(
      c("location", "reference_date", "horizon", "target_end_date", "target",
        "model")) |>
    scoringutils::as_forecast() |>
    # scoringutils::add_coverage() |>
    scoringutils::score(metrics = scoring_funs)
  return(raw_scores)
}

#' Compute summaries of scores within groups
compute_summarized_scores <- function(raw_scores, by, data_for_su,
                                      submission_freq) {
  q_coverage <- data_for_su |>
    dplyr::mutate(quantile_level_str = format(quantile_level, nsmall = 3)) |>
    dplyr::group_by(across(all_of(c(by, "quantile_level_str")))) |>
    dplyr::summarize(coverage_rate = mean(observed <= predicted,
                                          na.rm = TRUE)) |>
    tidyr::pivot_wider(names_from = quantile_level_str,
                       values_from = coverage_rate,
                       names_prefix = "q_coverage_")

  summarized_scores <- raw_scores |>
    scoringutils::add_pairwise_comparison(
      by = by,
      baseline = "FluSight-baseline") |>
    scoringutils::add_pairwise_comparison(
      by = by,
      baseline = "FluSight-baseline",
      metric = "ae_median") |>
    scoringutils::summarise_scores(by = by) |>
    scoringutils::summarise_scores(
      by = by,
      fun = signif,
      digits = 3
    ) |>
    dplyr::arrange(wis_scaled_relative_skill) |>
    dplyr::left_join(q_coverage, by = by) |>
    dplyr::left_join(submission_freq, by = join_by(model == model_id)) |>
    dplyr::select(all_of(c(by, "prop", "wis", "wis_scaled_relative_skill",
                  "ae_median", "ae_median_scaled_relative_skill",
                  "interval_coverage_50", "interval_coverage_95")),
                  starts_with("q_coverage_"))

  return(summarized_scores)
}



#' Helper function to compute score summaries
compute_scores <- function(
    forecasts, target_data,
    submission_threshold,
    scoring_funs = list(
      wis = scoringutils::wis,
      interval_coverage_50 = scoringutils::interval_coverage,
      interval_coverage_95 = function(...) {
        scoringutils::run_safely(..., interval_range = 95,
                                 fun = scoringutils::interval_coverage)
      },
      ae_median = scoringutils::ae_median_quantile),
    by = list(
      "model",
      c("model", "reference_date", "horizon")
    )) {
  # apply inclusion criteria
  submission_freq <- forecasts |>
    dplyr::count(model_id) |>
    dplyr::mutate(prop = n / max(n))

  models_to_keep <- submission_freq |>
    dplyr::filter(prop >= submission_threshold) |>
    dplyr::pull(model_id)

  forecasts <- forecasts |>
    dplyr::filter(model_id %in% models_to_keep)

  # merge forecast and target data, get into format uesd in scoringutils
  data_for_su <- merge_data_for_su(forecasts, target_data, models_to_keep)

  # compute raw scores
  raw_scores <- compute_raw_scores(data_for_su, scoring_funs)

  # compute score summaries
  summarized_scores <- purrr::map(
    by,
    compute_summarized_scores,
    raw_scores = raw_scores,
    data_for_su = data_for_su,
    submission_freq = submission_freq
  )

  return(summarized_scores)
}
