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
ili <- cdcfluview::ilinet(region = c("national")) %>%
  # add variable for week of season starting on epi week 31
  dplyr::mutate(
    season_week = mmwr_week_to_season_week(week, year),
    season = mmwr_week_to_season(week, year)
  )

readr::write_csv(ili, 'data-raw/influenza-ilinet/ilinet.csv')

ili_hhs <- cdcfluview::ilinet(region = c("hhs")) %>%
  # add variable for week of season starting on epi week 31
  dplyr::mutate(
    season_week = mmwr_week_to_season_week(week, year),
    season = mmwr_week_to_season(week, year)
  )

readr::write_csv(ili_hhs, 'data-raw/influenza-ilinet/ilinet_hhs.csv')


ili_state <- cdcfluview::ilinet(region = c("state")) %>%
  # add variable for week of season starting on epi week 31
  dplyr::mutate(
    season_week = mmwr_week_to_season_week(week, year),
    season = mmwr_week_to_season(week, year)
  )

readr::write_csv(ili_state, 'data-raw/influenza-ilinet/ilinet_state.csv')
