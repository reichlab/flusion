library(tidyverse)
library(cdcfluview)
library(here)
setwd(here())

#' Convert mmwr week to season week
#' 
#' @param mmwr_week integer vector of weeks in year
#' @param mmwr_year either a single (four digit) integer year or a vector of
#'   integer years with the same length as year_week
#' @param first_season_week number of week in year corresponding to the first
#'   week in the season.  For example, our code takes this value to be 31:
#'   a new influenza season starts on the 31st week of each year.
#' 
#' @return vector of the same length as year_week with the week of the season
#'   that each observation falls on
#' 
#' @export
mmwr_week_to_season_week <- function(
  mmwr_week,
  mmwr_year,
  first_season_week = 31) {
  last_season_week <- first_season_week - 1
  season_week <- ifelse(
    mmwr_week <= last_season_week,
    mmwr_week + MMWRweek::MMWRweek(MMWRweek:::start_date(mmwr_year) - 1)$MMWRweek - last_season_week,
    mmwr_week - last_season_week
  )

  return(season_week)
}


mmwr_week_to_season <- function(
  mmwr_week,
  mmwr_year,
  first_season_week = 31) {
  last_season_week <- first_season_week - 1
  season <- ifelse(
    mmwr_week <= last_season_week,
    paste0(as.character(mmwr_year - 1), "/", substr(as.character(mmwr_year), 3, 4)),
    paste0(as.character(mmwr_year), "/", substr(as.character(mmwr_year + 1), 3, 4))
  )

  return(season)
}


# load data
virology_national <- cdcfluview::who_nrevss(region = "national")
virology_hhs <- cdcfluview::who_nrevss(region = "hhs")
virology_state <- cdcfluview::who_nrevss(region = "state")

# a_h1, a_h3, and b breakdown, prior to 2015/16
prior_2015_16 <- dplyr::bind_rows(
    virology_national$combined_prior_to_2015_16,
    virology_hhs$combined_prior_to_2015_16,
    virology_state$combined_prior_to_2015_16 %>%
      dplyr::mutate(
        total_specimens = as.integer(total_specimens),
        percent_positive = as.numeric(percent_positive),
        a_2009_h1n1 = as.integer(a_2009_h1n1),
        a_h1 = as.integer(a_h1),
        a_h3 = as.integer(a_h3),
        a_subtyping_not_performed = as.integer(a_subtyping_not_performed),
        a_unable_to_subtype = as.integer(a_unable_to_subtype),
        h3n2v = as.integer(h3n2v),
        b = as.integer(b)
      )
  ) %>%
  dplyr::mutate(
    a = a_2009_h1n1 + a_h1 + a_h3 + h3n2v +
      a_subtyping_not_performed + a_unable_to_subtype,
    a_h1 = a_2009_h1n1 + a_h1,
    a_h3 = a_h3 + h3n2v,
    prop_a = a / (a + b),
    prop_b = b / (a + b),
    prop_a_h1 = ifelse(a_h1 + a_h3 > 0, a_h1 / (a_h1 + a_h3), NA_real_) * prop_a,
    prop_a_h3 = ifelse(a_h1 + a_h3 > 0, a_h3 / (a_h1 + a_h3), NA_real_) * prop_a,
    season_week = mmwr_week_to_season_week(week, year),
    season = mmwr_week_to_season(week, year)
  )

# a_h1, a_h3, and b breakdown, 2015/16 on
clinical_labs <- dplyr::bind_rows(
    virology_national$clinical_labs,
    virology_hhs$clinical_labs,
    virology_state$clinical_labs %>%
      dplyr::mutate(
        total_specimens = as.integer(total_specimens),
        total_a = as.integer(total_a),
        total_b = as.integer(total_b),
        percent_positive = as.numeric(percent_positive),
        percent_a = as.numeric(percent_a),
        percent_b = as.numeric(percent_b)
      )
  )
public_health_labs <- dplyr::bind_rows(
    virology_national$public_health_labs,
    virology_hhs$public_health_labs,
    virology_state$public_health_labs %>%
      dplyr::mutate(
        total_specimens = as.integer(total_specimens),
        a_h3 = as.integer(a_h3),
        a_2009_h1n1 = as.integer(a_2009_h1n1),
        a_subtyping_not_performed = as.integer(a_subtyping_not_performed),
        b = as.numeric(b),
        bvic = as.numeric(bvic),
        byam = as.numeric(byam),
        h3n2v = as.numeric(h3n2v)
      )
  )
post_2015_16 <- clinical_labs %>%
  dplyr::mutate(
    region_type, region, year, week,
    a = total_a,
    b = total_b,
    prop_a = total_a / (total_a + total_b),
    prop_b = total_b / (total_a + total_b)
  ) %>%
  dplyr::left_join(
    public_health_labs %>%
      dplyr::mutate(a_h1 = a_2009_h1n1, a_h3 = a_h3 + h3n2v) %>%
      dplyr::select(region_type, region, year, week, a_h1, a_h3),
    by = c("region_type", "region", "year", "week")) %>%
  dplyr::mutate(
    prop_a_h1 = ifelse(
      a_h1 + a_h3 > 0,
      a_h1 / (a_h1 + a_h3),
      NA_real_),
    prop_a_h3 = ifelse(
      a_h1 + a_h3 > 0,
      a_h3 / (a_h1 + a_h3),
      NA_real_),
    season_week = mmwr_week_to_season_week(week, year),
    season = mmwr_week_to_season(week, year)
  )

combined <- dplyr::bind_rows(prior_2015_16, post_2015_16) %>%
  dplyr::select(region_type, region, year, week, season, season_week, wk_date,
    percent_positive, a, b, a_h1, a_h3, prop_a, prop_b, prop_a_h1, prop_a_h3)

readr::write_csv(combined, 'data-raw/influenza-who-nrevss/who-nrevss.csv')

ggplot() +
  geom_line(data = combined, mapping = aes(x = season_week, y = prop_a_h1), color = "red") +
  geom_line(data = combined, mapping = aes(x = season_week, y = prop_a_h3), color = "orange") +
  geom_line(data = combined, mapping = aes(x = season_week, y = prop_b), color = "cornflowerblue") +
  facet_grid(region ~ season) +
  theme_bw()
