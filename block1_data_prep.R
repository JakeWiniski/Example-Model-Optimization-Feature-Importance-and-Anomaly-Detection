# ============================
# Block 1 — Data prep & splits
# ============================

# ---- USER INPUTS (edit these) -----------------------------------------------
df           <- YOUR_DATAFRAME            # e.g., my_data
id_col       <- "run_id"                  # unique identifier column; set to NULL if none
target_col   <- "target"                  # prediction target column
feature_cols <- c("feat_a","feat_b","feat_c")  # your selected features

# Option A: explicit split by IDs (leave NULL to skip)
train_ids    <- NULL                      # e.g., c("RUN01","RUN02",...)
test_ids     <- NULL                      # e.g., c("RUN10","RUN11",...)

# Option B: random split (used only if train_ids/test_ids are NULL)
holdout_frac <- 0.20                      # 20% of IDs (or rows) to test
split_seed   <- 42

# Data hygiene
drop_na_rows <- TRUE                      # drop rows with NA across selected columns?
# -----------------------------------------------------------------------------

make_train_test <- function(df,
                            id_col,
                            target_col,
                            feature_cols,
                            train_ids = NULL,
                            test_ids = NULL,
                            holdout_frac = 0.2,
                            seed = 42,
                            drop_na = TRUE) {
  stopifnot(is.data.frame(df))
  if (!is.null(id_col) && !(id_col %in% names(df))) {
    stop("id_col '", id_col, "' not found in df. Set id_col = NULL if you don't have one.")
  }
  # Validate columns
  needed <- unique(c(if (!is.null(id_col)) id_col, target_col, feature_cols))
  missing <- setdiff(needed, names(df))
  if (length(missing)) stop("Missing required columns: ", paste(missing, collapse = ", "))

  # Subset & optionally drop NAs
  data <- df[, needed, drop = FALSE]
  if (drop_na) data <- stats::na.omit(data)

  # Helper splitters
  split_by_ids <- function(data, id_col, train_ids, test_ids) {
    if (!is.null(test_ids) && length(test_ids)) {
      test  <- data[data[[id_col]] %in% test_ids, , drop = FALSE]
      train <- data[!data[[id_col]] %in% test_ids, , drop = FALSE]
      if (!is.null(train_ids) && length(train_ids)) {
        train <- data[data[[id_col]] %in% train_ids, , drop = FALSE]
      }
    } else if (!is.null(train_ids) && length(train_ids)) {
      train <- data[data[[id_col]] %in% train_ids, , drop = FALSE]
      test  <- data[!data[[id_col]] %in% train_ids, , drop = FALSE]
    } else {
      set.seed(seed)
      ids <- unique(data[[id_col]])
      if (holdout_frac <= 0 || holdout_frac >= 1) stop("holdout_frac must be in (0,1).")
      n_test <- max(1, floor(length(ids) * holdout_frac))
      test_ids <- sample(ids, n_test)
      test  <- data[data[[id_col]] %in% test_ids, , drop = FALSE]
      train <- data[!data[[id_col]] %in% test_ids, , drop = FALSE]
    }
    list(train = train, test = test)
  }

  if (!is.null(id_col)) {
    out <- split_by_ids(data, id_col, train_ids, test_ids)
  } else {
    # Row-wise random split if no ID column is available
    set.seed(seed)
    idx <- sample.int(nrow(data))
    n_test <- max(1, floor(nrow(data) * holdout_frac))
    test_idx <- idx[seq_len(n_test)]
    test  <- data[test_idx, , drop = FALSE]
    train <- data[setdiff(idx, test_idx), , drop = FALSE]
    out <- list(train = train, test = test)
  }

  # Return convenient pieces
  list(
    train    = out$train,
    test     = out$test,
    X_train  = out$train[, feature_cols, drop = FALSE],
    y_train  = out$train[[target_col]],
    X_test   = out$test[, feature_cols, drop = FALSE],
    y_test   = out$test[[target_col]],
    features = feature_cols,
    target   = target_col,
    id_col   = id_col
  )
}

# ---- RUN (no edits needed below) --------------------------------------------
splits <- make_train_test(
  df           = df,
  id_col       = id_col,
  target_col   = target_col,
  feature_cols = feature_cols,
  train_ids    = train_ids,
  test_ids     = test_ids,
  holdout_frac = holdout_frac,
  seed         = split_seed,
  drop_na      = drop_na_rows
)

# Quick sanity check
cat(
  sprintf("Rows: train=%d, test=%d | Features=%d | Target=%s\n",
          nrow(splits$train), nrow(splits$test), length(splits$features), splits$target)
)
