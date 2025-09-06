# RFE on PCA-transformed training data (generalized, minimal edits)

suppressPackageStartupMessages({ library(dplyr); library(caret) })

df <- conv.rate.train
if (!exists("splits")) stop("Expected 'splits' from Block 1.")
target_col <- splits$target
id_col <- splits$id_col %||% "id"

# Checks
need <- c(target_col, id_col)
miss <- setdiff(need, names(df))
if (length(miss)) stop("Missing required columns: ", paste(miss, collapse=", "))

# Detect PCs and optional context
pc_cols <- grep("^PC\\d+$", names(df), value = TRUE)
if (length(pc_cols) == 0) stop("No PC columns found (expected PC1, PC2, ...).")

non_feature_cols <- unique(c(id_col, target_col))
context_cols <- setdiff(names(df), c(non_feature_cols, pc_cols))  # optional
feature_cols <- c(context_cols, pc_cols)

# Split
features <- df[, feature_cols, drop = FALSE] %>%
  mutate(across(where(is.character), as.factor))
target <- df[[target_col]]
keep <- !is.na(target)
features <- features[keep, , drop = FALSE]
target <- target[keep]

# Clean
all_na <- vapply(features, function(z) all(is.na(z)), logical(1))
if (any(all_na)) features <- features[, !all_na, drop = FALSE]
nzv_idx <- caret::nearZeroVar(features)
if (length(nzv_idx)) features <- features[, -nzv_idx, drop = FALSE]
dup_cols <- duplicated(as.list(features))
if (any(dup_cols)) features <- features[, !dup_cols, drop = FALSE]

# RFE (same as original)
set.seed(42)
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5, verbose = TRUE, returnResamp = "final")
p <- ncol(features); if (p < 1) stop("No usable predictors after cleaning.")
sizes_to_try <- sort(unique(pmin(c(5,10,20,30,40,50,60,80,100,150, p, floor(p * c(.1,.2,.3,.4,.5,.6,.7,.8,.9,1))), p)))
sizes_to_try <- sizes_to_try[sizes_to_try >= 1]

rfe_result <- rfe(x = features, y = target, sizes = sizes_to_try, rfeControl = ctrl)
print(predictors(rfe_result))
plot(rfe_result, type = c("g", "o"))
rfe_result$results %>% arrange(RMSE) %>% head()
