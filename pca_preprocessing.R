# ===========================================
# Dim-Red Workflow — Block 1: PCA preparation
# ===========================================

suppressPackageStartupMessages({
  library(dplyr)
  library(caret)
})

if (!exists("splits")) stop("Expected 'splits' from Block 1 (data prep).")

# --- Inputs (override as needed) ---
# Use previously selected features by default; override with a custom vector if desired
if (!exists("pca_vars") || is.null(pca_vars)) pca_vars <- splits$features

# Use the ID column (if any) plus any extra context columns you want to carry along
id_col        <- splits$id_col            # may be NULL
context_cols  <- character(0)             # e.g., c("group","site") if present in your data
id_and_context <- c(id_col, context_cols)
id_and_context <- id_and_context[!is.null(id_and_context) & nzchar(id_and_context)]

# Target column defaults to Block 1 target
if (!exists("target_col") || is.null(target_col)) target_col <- splits$target

# Export toggles
export_cos2   <- TRUE
export_scores <- TRUE

# ------------------- VALIDATION -------------------
# Ensure required columns exist in both train and test frames
needed_cols <- unique(c(id_and_context, pca_vars, target_col))
missing_train <- setdiff(needed_cols, names(splits$train))
missing_test  <- setdiff(needed_cols, names(splits$test))
if (length(missing_train))
  stop("Missing columns in training data: ", paste(missing_train, collapse = ", "))
if (length(missing_test))
  stop("Missing columns in test data: ", paste(missing_test, collapse = ", "))

train_raw <- splits$train[, needed_cols, drop = FALSE]
test_raw  <- splits$test[,  needed_cols, drop = FALSE]

# Ensure PCA inputs are numeric
x_train <- train_raw[, pca_vars, drop = FALSE]
x_test  <- test_raw[,  pca_vars, drop = FALSE]
non_num <- names(x_train)[!vapply(x_train, is.numeric, logical(1))]
if (length(non_num))
  stop("PCA requires numeric predictors. Non-numeric columns: ", paste(non_num, collapse = ", "))

# ------------------- PCA MODEL -------------------
set.seed(42)
pp <- caret::preProcess(
  x_train,
  method = c("medianImpute", "center", "scale", "pca"),
  thresh = 0.90
)

# Scores
pcs_train <- predict(pp, x_train)
pcs_test  <- predict(pp, x_test)

# Combine with ID/context + TARGET LAST
train_pca <- dplyr::bind_cols(
  if (length(id_and_context)) train_raw[, id_and_context, drop = FALSE],
  pcs_train,
  train_raw[, target_col, drop = FALSE]
)
test_pca <- dplyr::bind_cols(
  if (length(id_and_context)) test_raw[, id_and_context, drop = FALSE],
  pcs_test,
  test_raw[, target_col, drop = FALSE]
)

# Name harmonization for clarity
names(train_pca)[ncol(train_pca)] <- target_col
names(test_pca)[ncol(test_pca)]   <- target_col

# Attach the preProcess object for downstream use
attr(train_pca, "pp") <- pp

# Make objects available for downstream blocks (mirrors earlier naming)
conv.rate.train <- train_pca
conv.rate.test  <- test_pca

# ------------------- cos² EXPORT -------------------
if (export_cos2) {
  rotation_mat <- pp$rotation  # loadings matrix
  if (!is.null(rotation_mat)) {
    cos2_df <- as.data.frame(rotation_mat^2)
    cos2_df$Variable <- rownames(rotation_mat)
    cos2_df <- dplyr::relocate(cos2_df, Variable)
    utils::write.csv(cos2_df, "pca_cos2_by_pc.csv", row.names = FALSE)
    message("✔ Exported: pca_cos2_by_pc.csv")
  } else {
    warning("No rotation matrix found in PCA object — cannot export cos2.")
  }
}

# ------------------- PCA SCORES EXPORT -------------------
if (export_scores) {
  id_name <- if (length(id_and_context)) id_and_context[1] else "id"
  utils::write.csv(
    dplyr::bind_cols(!!id_name := train_raw[[id_name]], pcs_train),
    "train_pca_scores.csv",
    row.names = FALSE
  )
  utils::write.csv(
    dplyr::bind_cols(!!id_name := test_raw[[id_name]],  pcs_test),
    "test_pca_scores.csv",
    row.names = FALSE
  )
  message("✔ Exported: train_pca_scores.csv and test_pca_scores.csv")
}

# ------------------- INFO -------------------
cat("Train PCs:", paste(names(pcs_train), collapse = ", "), "\n")
cat("Test PCs match train?:", identical(names(pcs_train), names(pcs_test)), "\n")
cat("Target last in train?:", names(conv.rate.train)[ncol(conv.rate.train)] == target_col, "\n")
cat("Train rows:", nrow(conv.rate.train), " | Test rows:", nrow(conv.rate.test), "\n")
