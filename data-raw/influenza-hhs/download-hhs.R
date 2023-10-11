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



ggplot(data = hosps |> dplyr::filter(location == "US", date >= "2022-09-01")) +
    geom_line(mapping = aes(x = date, y = inc))
