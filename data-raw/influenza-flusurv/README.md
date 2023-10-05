# FluSurv-NET: Influenza Hospitalizations Data for US

## Hospitalizations by virus type

We downloaded national level counts of hospitalizations with a laboratory test diagnosis of influenza A or B from [Flusurv-NET](https://gis.cdc.gov/grasp/fluview/FluHospChars.html). We downloaded data on February 22, 2022.

The data files and the accompanying query specifications are contained in the zip file `FluViewPhase5Data.zip` which should be unzipped into this folder, resulting in the following file structure:

 - `data-raw/influenza-flusurv/FluViewPhase5Data/Characteristics.csv`
 - `data-raw/influenza-flusurv/FluViewPhase5Data/Medical_Conditions.csv`
 - `data-raw/influenza-flusurv/FluViewPhase5Data/Weekly_Data_Counts_by_Virus.csv`
 - `data-raw/influenza-flusurv/FluViewPhase5Data/Weekly_Data_Percent_by_Virus.csv`

The last two of these files contain the data by virus type.

## Hospitalization rates

Estimates of hospitalization rates per 100,000 population were downloaded using
the `cdcfluview` R package on March 1, 2022, minimally pre-processed, and saved
in the `data-raw/influenza-flusurv/flusurv-rates/flusurv-rates.csv` file. The
script to do this is `data-raw/influenza-flusurv/flusurv-rates/download-flusurv-rates.R`.
