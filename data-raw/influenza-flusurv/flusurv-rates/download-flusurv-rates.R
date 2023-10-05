library(tidyverse)

# Note: as of Sep 26 2023, need to use the version of cdcfluview at
# https://github.com/vpnagraj/cdcfluview
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


# load data
all_hosp <- cdcfluview::surveillance_areas() %>%
  # subset to regions of interest with a reasonable reporting history
  dplyr::filter(
    !(region %in% c("Idaho", "Iowa", "Oklahoma", "Rhode Island", "South Dakota")),
    !(region == "Entire Network" & surveillance_area %in% c("EIP", "IHSP"))
  ) %>%
  purrr::pmap_dfr(cdcfluview::hospitalizations) %>%
  dplyr::filter(name == surveillance_area) %>%
  # add variables for epi week and week of season starting on epi week 31
  dplyr::mutate(
    epiweek = paste0(year, sprintf("%02d", weeknumber)),
    season_week = mmwr_week_to_season_week(weeknumber, year)
  ) %>%
  dplyr::filter(!is.na(age_label), !is.na(season_label))

readr::write_csv(all_hosp, 'data-raw/influenza-flusurv/flusurv-rates/flusurv-rates.csv')


library(epidatr)
all_hosp <- pub_flusurv(
    locations = c("CA", "CO", "CT", "GA", "IA", "ID", "MD", "MI", "MN", "NM",
                  "NY_albany", "NY_rochester", "OH", "OK", "OR", "RI", "SD",
                  "TN", "UT", "network_all"),
    epiweeks = epirange(200901, 202340)
)
