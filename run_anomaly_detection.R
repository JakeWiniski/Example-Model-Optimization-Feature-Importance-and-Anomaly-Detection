# -------------------------------------------------------------------------
# Anomaly Detection Settings:
#   • Less liberal (stricter): 
#       alpha = 0.05   # 95% conformal interval (few anomalies)
#       q_scale = 1.0  # no scaling of quantile threshold
#       z_cut   = 3.0  # stricter MAD-based cutoff
#
#   • More liberal (looser):
#       alpha = 0.15   # 85% conformal interval (more anomalies flagged)
#       q_scale = 0.92 # shrink threshold slightly to increase flags
#       z_cut   = 2.0  # looser MAD-based cutoff
#
# Adjust these knobs depending on whether you want to minimize false positives
# (stricter) or maximize sensitivity to potential outliers (more liberal).
# -------------------------------------------------------------------------

# deps (only needed if you’re actually composing plots with /)
suppressPackageStartupMessages(library(patchwork))

# choose your model: uncalibrated or calibrated
model_to_use <- final_model  # or conv_rate_model if you kept that name

anom_raw <- detect_anomalies_from_model(
  model    = model_to_use,
  train_df = conv.rate.train,
  test_df  = conv.rate.test,
  alpha    = 0.15,
  q_scale  = 0.92,
  z_cut    = 2.0,
  tail     = "both",
  use_mad_rule = TRUE
)

anom_cal <- detect_anomalies_with_calibration(
  model    = model_to_use,
  train_df = conv.rate.train,
  test_df  = conv.rate.test,
  alpha_conformal = 0.15,
  q_scale  = 0.92,
  z_cut    = 2.0,
  tail     = "both",
  use_mad_rule = TRUE
)

# plots + export (generic filename)
(p1 <- anom_raw$plot + ggtitle("Residuals Without Calibration"))
(p2 <- anom_cal$plot + ggtitle("Residuals With Post-Hoc Calibration"))
p1 / p2

utils::write.csv(anom_cal$results_table, "anomaly_results_calibrated.csv", row.names = FALSE)
