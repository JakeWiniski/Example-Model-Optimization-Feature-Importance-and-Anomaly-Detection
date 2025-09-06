# Ensure target type matches task:
#   • Regression → leave numeric (default)
#   • Classification → wrap in factor(), e.g. train_y <- factor(target)

# ============================
# Train final model
# ============================

suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
  library(vip)
  library(ggplot2)
})

# ─── Step 1: Retrieve RFE-selected features (from existing rfe_result) ──
if (!exists("rfe_result")) stop("Missing 'rfe_result' from Block 2.")
selected_features <- caret::predictors(rfe_result)
if (length(selected_features) == 0) stop("No features selected by RFE.")

# ─── Step 2: Prepare data from the already-cleaned objects ──────────────
# 'features' and 'target' are produced in Block 2
if (!exists("features") || !exists("target")) {
  stop("Expected 'features' and 'target' from Block 2.")
}

train_X <- features %>%
  dplyr::select(dplyr::all_of(selected_features)) %>%
  dplyr::mutate(across(where(is.character), as.factor))

train_y <- target
model_df <- data.frame(train_X, target = train_y)

# ─── Step 3: 5-fold CV control ──────────────────────────────────────────
set.seed(42)
train_ctrl <- caret::trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE
)

# ─── Step 4: Train final model (ranger, permutation importance) ─────────
final_model <- caret::train(
  target ~ .,
  data = model_df,
  method = "ranger",
  trControl = train_ctrl,
  importance = "permutation",
  tuneLength = 1  # increase to tune more hyperparams if desired
)

# Attach the selected features for convenience when predicting later
attr(final_model, "selected_features") <- selected_features

# Persist to disk so you can recall it later in a fresh R session
saveRDS(final_model, file = "model_ranger_rfe.rds")

# ─── Step 5: Print performance ──────────────────────────────────────────
cat("\n✅ Final Model Performance (5-Fold CV):\n")
print(final_model$results)

# Extract cross-validated R² (single row if tuneLength = 1)
cv_r2 <- final_model$results$Rsquared[1]

# ─── Step 6: VIP plot (Permutation Importance) ──────────────────────────
vip::vip(final_model$finalModel,
         num_features = length(selected_features)) +
  ggtitle(paste0(
    "Permutation Importance\n",
    "5-Fold CV R² = ", round(cv_r2, 3)
  )) +
  theme_minimal()
