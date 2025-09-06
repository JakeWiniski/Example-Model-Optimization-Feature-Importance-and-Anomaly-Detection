# Anomaly detection helper (non-calibrated model)
detect_anomalies_from_model <- function(
  model,
  train_df,
  test_df,
  alpha = 0.05,           # conformal (1 - alpha) quantile on |residual|
  use_mad_rule = TRUE,    # also apply MAD-based z rule
  seed = 42,
  verbose = TRUE,
  # --- new liberal-friendly knobs (backward compatible defaults) ---
  q_scale = 1.00,         # multiply conformal threshold (e.g., 0.90–0.98 to flag more)
  z_cut = 2.5,            # MAD z cutoff (smaller => more flags; 2.0 is liberal)
  tail = c("both","upper","lower")  # one/two-sided flagging
) {
  tail <- match.arg(tail)
  stopifnot(ncol(train_df) >= 3, ncol(test_df) >= 3)

  # Pull RFE-selected feature names and factor level map from the model
  sel_feats <- attr(model, "selected_features")
  tr_levels <- attr(model, "factor_levels")
  if (is.null(sel_feats)) {
    stop("Model is missing 'selected_features' attribute. Attach it before saving.")
  }

  # Positional split (run | features | target)
  p_tr <- ncol(train_df); p_te <- ncol(test_df)
  if (p_tr != p_te && verbose) {
    message("Train/Test column counts differ; proceeding positionally.")
  }
  train_features_raw <- train_df[, 2:(p_tr - 1), drop = FALSE]
  train_y            <- train_df[[p_tr]]
  test_features_raw  <- test_df[,  2:(p_te - 1), drop = FALSE]
  test_y             <- test_df[[p_te]]

  # Select only model features, ensure factor types
  train_X <- train_features_raw %>%
    dplyr::select(all_of(sel_feats)) %>%
    mutate(across(where(is.character), as.factor))
  test_X  <- test_features_raw %>%
    dplyr::select(all_of(sel_feats)) %>%
    mutate(across(where(is.character), as.factor))

  # Align factor levels in test to training (avoids "new level" errors)
  if (!is.null(tr_levels)) {
    for (nm in names(test_X)) {
      if (!is.null(tr_levels[[nm]])) {
        test_X[[nm]] <- factor(as.character(test_X[[nm]]), levels = tr_levels[[nm]])
      }
    }
  }

  # Calibrate residual thresholds from OOF CV on training
  set.seed(seed)
  calib_ctrl <- caret::trainControl(method = "cv", number = 5, savePredictions = "final", verboseIter = FALSE)
  calib_df <- data.frame(train_X, target = train_y)

  calib_fit <- caret::train(
    target ~ .,
    data = calib_df,
    method = model$method,          # reuse learner type (e.g., "ranger")
    trControl = calib_ctrl,
    tuneGrid  = model$bestTune %||% NULL,  # use best tune if available
    tuneLength = if (is.null(model$bestTune)) 1 else 1, # keep same spec
    importance = "none"
  )

  # Keep predictions for the winning tuning row (if any)
  calib_pred <- calib_fit$pred
  if (!is.null(calib_fit$bestTune)) {
    for (nm in names(calib_fit$bestTune)) {
      calib_pred <- calib_pred[calib_pred[[nm]] == calib_fit$bestTune[[nm]], , drop = FALSE]
    }
  }
  calib_resid <- calib_pred$pred - calib_pred$obs
  abs_calib_resid <- abs(calib_resid)

  # Conformal-style absolute residual threshold (with optional scaling)
  q_abs_base <- as.numeric(stats::quantile(abs_calib_resid, probs = 1 - alpha, na.rm = TRUE))
  q_abs <- q_scale * q_abs_base

  # Robust sigma from MAD
  sigma_hat <- NA_real_
  if (use_mad_rule) {
    mad_resid <- stats::mad(calib_resid, center = 0, constant = 1, na.rm = TRUE)
    sigma_hat <- 1.4826 * mad_resid
  }

  # Score the test set with the provided final model
  test_pred <- predict(model, newdata = test_X)
  test_resid <- test_pred - test_y
  abs_test_resid <- abs(test_resid)

  # Flags (support both- / one-sided)
  if (tail == "both") {
    anomaly_conformal <- abs_test_resid > q_abs
  } else if (tail == "upper") {
    anomaly_conformal <- test_resid > q_abs
  } else {
    anomaly_conformal <- test_resid < -q_abs
  }

  anomaly_3sigma <- if (use_mad_rule && is.finite(sigma_hat) && sigma_hat > 0) {
    z <- test_resid / sigma_hat
    if (tail == "both") abs(z) > z_cut else if (tail == "upper") z > z_cut else z < -z_cut
  } else {
    rep(FALSE, length(test_resid))
  }
  anomaly_flag <- anomaly_conformal | anomaly_3sigma

  # Summaries & metrics
  holdout_metrics <- caret::postResample(pred = test_pred, obs = test_y)
  if (verbose) {
    message(sprintf("Conformal threshold: %.4f (alpha=%.2f, q_scale=%.2f)", q_abs, alpha, q_scale))
    if (use_mad_rule && is.finite(sigma_hat)) message(sprintf("MAD sigma_hat: %.4f; z_cut: %.2f; tail=%s", sigma_hat, z_cut, tail))
    message(sprintf("Holdout RMSE: %.4f | R^2: %.3f | MAE: %.4f",
                    holdout_metrics["RMSE"], holdout_metrics["Rsquared"], holdout_metrics["MAE"]))
  }

  # Percentile rank of test residuals vs calib residuals
  residual_percentile_rank <- vapply(abs_test_resid, function(x) {
    mean(abs_calib_resid <= x, na.rm = TRUE)
  }, numeric(1))

  # Output table (include run if present)
  out_tbl <- tibble::tibble(
    run                      = test_df$run %||% seq_len(nrow(test_df)),
    observed                 = as.numeric(test_y),
    predicted                = as.numeric(test_pred),
    residual                 = as.numeric(test_resid),
    abs_residual             = as.numeric(abs_test_resid),
    residual_percentile_rank = residual_percentile_rank,
    anomaly_conformal        = anomaly_conformal,
    anomaly_3sigma           = anomaly_3sigma,
    anomaly_any              = anomaly_flag
  )

  # Plot
  plt <- ggplot2::ggplot(out_tbl, ggplot2::aes(x = run, y = residual)) +
    ggplot2::geom_hline(yintercept =  q_abs, linetype = "dashed") +
    ggplot2::geom_hline(yintercept = -q_abs, linetype = "dashed") +
    ggplot2::geom_point(ggplot2::aes(shape = anomaly_any), size = 3) +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Holdout Residuals with Conformal Thresholds",
      subtitle = paste0("|residual| >", round(q_abs, 3),
                        if (use_mad_rule && is.finite(sigma_hat) && sigma_hat > 0)
                          paste0("  (z_cut=", z_cut, ", 3σ≈", round(3*sigma_hat, 3), ")"),
                        "  tail=", tail),
      x = "run", y = "residual"
    ) +
    ggplot2::theme_minimal()

  list(
    results_table   = out_tbl,
    thresholds      = list(conformal_abs = q_abs,
                           alpha = alpha,
                           q_scale = q_scale,
                           sigma_hat = sigma_hat,
                           z_cut = z_cut,
                           tail = tail),
    holdout_metrics = holdout_metrics,
    plot            = plt
  )
}

# A tiny infix helper for defaults (x %||% y)
`%||%` <- function(x, y) if (is.null(x)) y else x
