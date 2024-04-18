# HHS Protect/NHSN influenza data

This folder contains weekly snapshots of influenza hospital admissions as reported by NHSN, formerly known as HHS Protect.  Data files are created by `download-hhs.R`, which uses the [covidData](https://github.com/reichlab/covidData) package.

Weekly snapshots are stored in files with format `hhs-YYYY-MM-DD.csv`, where the date in the file name is the date forecasts were produced, which is typically the Wednesday before the Saturday reference date.  In the first two weeks of the season, I created files with preliminary `draft`s of the data representing intermediate or partial data releases. Informal explorations revealed that this was not necessary as data updates were not substantial.  The most recent data release is stored in `hhs.csv`.

The above data files filter out some dates and locations with unreliable data.  In case it is helpful to see those data, they can be accessed in `hhs_complete.csv`, which is updated each week.
