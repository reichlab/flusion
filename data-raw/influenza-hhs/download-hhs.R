library(covidData)
library(dplyr)

hosps <- covidData::load_data(
    spatial_resolution = c("state", "national"),
    temporal_resolution = "weekly",
    measure = "flu hospitalizations")

hosps$cum <- NULL

hosps <- hosps %>%
    dplyr::filter(date >= "2022-09-01")

readr::write_csv(hosps, 'data-raw/influenza-hhs/hhs.csv')



ggplot(data = hosps |> dplyr::filter(location == "US", date >= "2022-09-01")) +
  geom_line(mapping = aes(x = date, y = inc))


