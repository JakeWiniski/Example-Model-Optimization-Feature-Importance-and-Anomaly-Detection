# Ensure target type matches task:
#   • Regression → leave numeric (default)
#   • Classification → wrap in factor(), e.g. train_y <- factor(target)

# ================================================================
# Model training (ranger, PCA path) + post-hoc calibration
# ================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(vip)
  library(ggplot2)
})

# ─── Step 0: Inputs from prior blocks ───────────────────────────────────
if (!exists("rfe_result")) stop("Missing 'rfe_result' from PCA RFE block.")
if (!exists("features") || !exists("target"))
  stop("Expected 'features' and 'target' objects from prior block.")

# ─── Step 1: Retrieve RFE-selected features ─────────────────────────────
selected_features <- caret::predictors(rfe_result)
if (length(selected_features) == 0) stop("No features selected by RFE.")

# ─── Step 2: Prepare training data ──────────────────────────────────────
train_X <- features %>%
  dplyr::select(dplyr::all_of(selected_features)) %>%
  dplyr::mutate(across(where(is.character), as.factor))

train_y <- target
model_df <- data.frame(train_X, target = train_y)

# ─── Step 3: 5-fold CV control (save predictions for calibration) ───────
set.seed(42)
train_ctrl <- caret::trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  savePredictions = "final"
)

# ─── Step 4: Train model (ranger, permutation importance) ───────────────
final_model <- caret::train(
  target ~ .,
  data = model_df,
  method = "ranger",
  trControl = train_ctrl,
  importance = "permutation",
  tuneLength = 1
)

# Attach RFE-selected features for convenience
attr(final_model, "selected_features") <- selected_features

# ─── Step 5: Print performance ──────────────────────────────────────────
cat("\n✅ Final Model Performance (5-Fold CV):\n")
print(final_model$results)
cv_r2 <- final_model$results$Rsquared[1]

# ─── Step 6: VIP plot (Permutation Importance) ──────────────────────────
vip::vip(final_model$finalModel,
         num_features = length(selected_features)) +
  ggtitle(paste0(
    "Permutation Importance (ranger)\n",
    "5-Fold CV R² = ", round(cv_r2, 3)
  )) +
  theme_minimal()

# ─── Step 7: Extract CV predictions (filter to best tune if needed) ─────
cv_preds <- final_model$pred
if (!is.null(final_model$bestTune) && nrow(final_model$bestTune)) {
  for (col in names(final_model$bestTune)) {
    cv_preds <- cv_preds[cv_preds[[col]] == final_model$bestTune[[col]], , drop = FALSE]
  }
}

# ─── Step 8: Post-hoc linear calibration on CV predictions ──────────────
# Fit a simple linear model: observed ~ predicted
calibration_lm <- lm(obs ~ pred, data = cv_preds)
cat("\n✅ Post-hoc Calibration (lm):\n")
print(summary(calibration_lm))

# Attach calibration model to the trained model
attr(final_model, "calibration_model") <- calibration_lm

# ─── Step 9: Save calibrated model ──────────────────────────────────────
saveRDS(final_model, file = "model_ranger_rfe_calibrated.rds")

# ─── Step 10: Visualize calibration relationship ────────────────────────
coefs <- coef(calibration_lm)
calib_intercept <- unname(coefs["(Intercept)"])
calib_slope     <- unname(coefs["pred"])

ggplot(cv_preds, aes(x = obs, y = pred)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = FALSE) +                # calibration line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +  # ideal line
  labs(
    title = "Cross-Validated Predictions vs Observed (Post-hoc Calibration)",
    subtitle = paste0(
      "CV R² = ", round(cv_r2, 3),
      " | Calib: pred ≈ ", round(calib_intercept, 3),
      " + ", round(calib_slope, 3), "·obs"
    ),
    x = "Observed",
    y = "Predicted"
  ) +
  coord_equal() +
  theme_minimal()
