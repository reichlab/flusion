library(covidData)
library(dplyr)

hosps <- covidData::load_data(
    spatial_resolution = c("state", "national"),
    temporal_resolution = "daily",
    measure = "flu hospitalizations")
hosps$date <- hosps$date + 1

max(hosps$date)

hosps_wk <- hosps %>%
    dplyr::mutate(
        sat_date = lubridate::ceiling_date(
            lubridate::ymd(date), unit = "week") - 1
    ) %>%
    dplyr::group_by(location) %>%
    # if the last week is not complete, drop all observations from the
    # previous Saturday in that week
    dplyr::filter(
        if (max(date) < max(sat_date)) date <= max(sat_date) - 7 else TRUE
    ) %>%
    dplyr::ungroup() %>%
    dplyr::select(-date) %>%
    dplyr::rename(date = sat_date) %>%
    dplyr::group_by(location, date) %>%
    dplyr::summarize(inc = sum(inc, na.rm = FALSE), .groups = "drop")

hosps_wk <- hosps_wk %>%
    dplyr::filter(date >= "2022-09-01", !(location %in% c("60", "78")))

readr::write_csv(hosps_wk, 'data-raw/influenza-hhs/hhs.csv')
readr::write_csv(hosps_wk, paste0('data-raw/influenza-hhs/hhs-', Sys.Date(), '.csv'))

hosps_wk_prev_draft2 <- readr::read_csv('data-raw/influenza-hhs/hhs-2023-10-18-draft2.csv') |>
    dplyr::mutate(as_of = "2023-10-18-draft2")
# hosps_wk_now_draft1 <- readr::read_csv('data-raw/influenza-hhs/hhs-2023-10-18-draft1.csv') |>
#     dplyr::mutate(as_of = "2023-10-18-draft1")

ggplot(data = dplyr::bind_rows(
  hosps_wk_prev_draft2 |>
    dplyr::filter(date >= "2023-09-01"),
  # hosps_wk_now_draft1 |>
  #   dplyr::filter(date >= "2023-09-01"),
  hosps_wk |>
    dplyr::filter(date >= "2023-09-01") |>
    dplyr::mutate(as_of = "2023-10-25-draft2"))) +
  geom_line(mapping = aes(x = date, y = inc, color = as_of, linetype = as_of)) +
  facet_wrap(~ location, scales="free_y") +
  theme_bw()
