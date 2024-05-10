# Create flusion ensembles for all combinations of
# reference date and subsets of component models selected from
# UMass-gbq_qr, UMass-gbq_qr_no_level, and UMass-sarix
# Run with code/flusion/retrospective-experiments as you working directory

ref_dates <- seq.Date(from = as.Date("2023-10-14"),
                      to = as.Date("2024-04-27"),
                      by = 7)
ref_dates <- as.character(ref_dates)

model_combinations <- c(
  "UMass-gbq_qr UMass-gbq_qr_no_level UMass-sarix",
  "UMass-gbq_qr_no_level UMass-sarix",
  "UMass-gbq_qr UMass-sarix",
  "UMass-gbq_qr UMass-gbq_qr_no_level"
)

for (ref_date in ref_dates) {
  for (models in model_combinations) {
    system(paste(
      "Rscript --vanilla flusion-retrospective.R",
      ref_date,
      models
    ))
  }
}
