# Note: If the target is categorical, wrap it as factor() so RFE runs classification. 
# caret will then report Accuracy/Kappa instead of RMSE.

# ============================
# Block 2 — RFE (generalized)
# ============================

# Minimal dependencies used below
suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
})

# ---- Source (from Block 1 output) ----
if (!exists("splits")) stop("Expected 'splits' from Block 1 (data prep).")

X_train <- splits$X_train
y_train <- splits$y_train
target_name <- splits$target %||% "target"

# Build a combined df in the expected layout:
# [optional id | features | target]
if (!is.null(splits$id_col) && splits$id_col %in% names(splits$train)) {
  df <- cbind(
    run = splits$train[[splits$id_col]],
    X_train,
    !!target_name := y_train
  )
} else {
  # Fallback synthetic ID for consistency with original code's "run" column
  df <- cbind(
    run = seq_len(nrow(X_train)),
    X_train,
    !!target_name := y_train
  )
}

# ---- Split by position: run | features | target ----
p_total <- ncol(df)
if (p_total < 3) stop("Expected at least 3 columns: id, >=1 feature, target.")

# Features are columns 2 .. (n - 1)
features <- df[, 2:(p_total - 1), drop = FALSE]

# Target is the last column
target <- df[[p_total]]

# Optional: sanity check name (comment out if target name varies)
cat("Target column detected as:", names(df)[p_total], "\n")

# ---- Prepare features ----
# Convert characters to factors (RF-friendly)
features <- features %>% mutate(across(where(is.character), as.factor))

# Align rows with non-missing target
keep <- !is.na(target)
features <- features[keep, , drop = FALSE]
target   <- target[keep]

# Drop all-NA columns
all_na <- vapply(features, function(z) all(is.na(z)), logical(1))
if (any(all_na)) features <- features[, !all_na, drop = FALSE]

# Drop near-zero-variance predictors
nzv_idx <- caret::nearZeroVar(features)
if (length(nzv_idx)) features <- features[, -nzv_idx, drop = FALSE]

# Drop duplicate columns
dup_cols <- duplicated(as.list(features))
if (any(dup_cols)) features <- features[, !dup_cols, drop = FALSE]

cat("# predictors after cleaning:", ncol(features), "\n")

# ---- RFE setup (Random Forest) ----
set.seed(42)
ctrl <- rfeControl(
  functions    = rfFuncs,
  method       = "cv",
  number       = 5,
  verbose      = TRUE,
  returnResamp = "final"
)

# Candidate sizes up to p
p <- ncol(features)
if (p < 1) stop("No usable predictors after cleaning.")
sizes_to_try <- sort(unique(pmin(
  c(5,10,20,30,40,50,60,80,100,150, p,
    floor(p * c(.1,.2,.3,.4,.5,.6,.7,.8,.9,1))),
  p)))
sizes_to_try <- sizes_to_try[sizes_to_try >= 1]
cat("Candidate sizes:", paste(sizes_to_try, collapse = ", "), "\n")

# ---- Run RFE ----
rfe_result <- rfe(
  x = features,
  y = target,
  sizes = sizes_to_try,
  rfeControl = ctrl
)

# ---- Outputs ----
cat("\n✅ Best Features Selected:\n")
print(predictors(rfe_result))

plot(rfe_result, type = c("g", "o"))

rfe_result$results %>%
  arrange(RMSE) %>%
  head()
