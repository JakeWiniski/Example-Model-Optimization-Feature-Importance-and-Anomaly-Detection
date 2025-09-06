# Anomaly detection helper incorporating post-hoc linear calibration
detect_anomalies_with_calibration <- function(
  model,
  train_df,
  test_df,
  # --- liberal-friendly knobs ---
  alpha_conformal = 0.10,  # was alpha=0.05; higher alpha => lower quantile => more flags
  q_scale = 0.95,          # multiply conformal threshold (e.g., 0.90–0.98 to be looser)
  z_cut = 2.0,             # MAD z cutoff (2.0 looser than 2.5/3.0)
  tail = c("both","upper","lower"),  # one/two-sided flagging
  use_mad_rule = TRUE,
  seed = 42,
  verbose = TRUE
) {
  tail <- match.arg(tail)
  stopifnot(ncol(train_df) >= 3, ncol(test_df) >= 3)

  # --- Model metadata ---
  sel_feats <- attr(model, "selected_features")
  tr_levels <- attr(model, "factor_levels")
  cal_model <- attr(model, "calibration_model")
  if (is.null(sel_feats)) stop("Model missing 'selected_features' attribute.")
  if (is.null(cal_model)) stop("Model missing 'calibration_model' attribute.")

  # --- Positional split (run | features | target) ---
  p_tr <- ncol(train_df); p_te <- ncol(test_df)
  train_features_raw <- train_df[, 2:(p_tr - 1), drop = FALSE]
  train_y            <- train_df[[p_tr]]
  test_features_raw  <- test_df[,  2:(p_te - 1), drop = FALSE]
  test_y             <- test_df[[p_te]]

  # --- Subset + types ---
  train_X <- train_features_raw %>%
    dplyr::select(all_of(sel_feats)) %>%
    mutate(across(where(is.character), as.factor))
  test_X <- test_features_raw %>%
    dplyr::select(all_of(sel_feats)) %>%
    mutate(across(where(is.character), as.factor))

  # --- Align factor levels for test ---
  if (!is.null(tr_levels)) {
    for (nm in names(test_X)) {
      if (!is.null(tr_levels[[nm]])) {
        test_X[[nm]] <- factor(as.character(test_X[[nm]]), levels = tr_levels[[nm]])
      }
    }
  }

  # --- Calibrate thresholds from CV (on calibrated preds) ---
  set.seed(seed)
  calib_ctrl <- caret::trainControl(method = "cv", number = 5, savePredictions = "final")
  calib_df <- data.frame(train_X, target = train_y)
  calib_fit <- caret::train(
    target ~ ., data = calib_df,
    method = model$method,
    trControl = calib_ctrl,
    tuneGrid  = model$bestTune %||% NULL,
    tuneLength = if (is.null(model$bestTune)) 1 else 1,
    importance = "none"
  )
  calib_pred <- calib_fit$pred
  if (!is.null(calib_fit$bestTune)) {
    for (nm in names(calib_fit$bestTune)) {
      calib_pred <- calib_pred[calib_pred[[nm]] == calib_fit$bestTune[[nm]], , drop = FALSE]
    }
  }
  calib_pred$pred_calibrated <- predict(cal_model, newdata = data.frame(pred = calib_pred$pred))
  calib_resid <- calib_pred$pred_calibrated - calib_pred$obs
  abs_calib_resid <- abs(calib_resid)

  # --- Thresholds (liberalizable) ---
  q_abs_base <- as.numeric(stats::quantile(abs_calib_resid, probs = 1 - alpha_conformal, na.rm = TRUE))
  q_abs <- q_scale * q_abs_base  # shrink a bit to flag more
  sigma_hat <- if (use_mad_rule) 1.4826 * stats::mad(calib_resid, center = 0, constant = 1, na.rm = TRUE) else NA_real_

  # --- Score test (apply calibration) ---
  test_pred_raw <- predict(model, newdata = test_X)
  test_pred <- predict(cal_model, newdata = data.frame(pred = test_pred_raw))
  test_resid <- test_pred - test_y
  abs_test_resid <- abs(test_resid)

  # --- Flagging rules ---
  # Conformal (two- or one-sided)
  if (tail == "both") {
    anomaly_conformal <- abs_test_resid > q_abs
  } else if (tail == "upper") {
    anomaly_conformal <- test_resid > q_abs
  } else { # "lower"
    anomaly_conformal <- test_resid < -q_abs
  }

  # MAD z cutoff
  anomaly_3sigma <- if (use_mad_rule && is.finite(sigma_hat) && sigma_hat > 0) {
    z <- test_resid / sigma_hat
    if (tail == "both") abs(z) > z_cut else if (tail == "upper") z > z_cut else z < -z_cut
  } else {
    rep(FALSE, length(test_resid))
  }

  anomaly_flag <- anomaly_conformal | anomaly_3sigma

  # Percentile rank (for context)
  residual_percentile_rank <- vapply(abs_test_resid, function(x) {
    mean(abs_calib_resid <= x, na.rm = TRUE)
  }, numeric(1))

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

  plt <- ggplot2::ggplot(out_tbl, ggplot2::aes(x = run, y = residual)) +
    ggplot2::geom_hline(yintercept =  q_abs, linetype = "dashed") +
    ggplot2::geom_hline(yintercept = -q_abs, linetype = "dashed") +
    ggplot2::geom_point(ggplot2::aes(shape = anomaly_any), size = 3) +
    ggplot2::coord_flip() +
    ggplot2::labs(
      title = "Holdout Residuals with Conformal Thresholds",
      subtitle = paste0("|residual| >", round(q_abs, 3),
                        if (use_mad_rule && is.finite(sigma_hat))
                          paste0("  (z_cut=", z_cut, ", 3σ≈", round(3*sigma_hat, 3), ")"),
                        "  tail=", tail),
      x = "run", y = "residual"
    ) +
    ggplot2::theme_minimal()

  holdout_metrics <- caret::postResample(pred = test_pred, obs = test_y)
  if (verbose) {
    message(sprintf("Conformal |resid| threshold: %.4f  (alpha=%.2f, q_scale=%.2f)", q_abs, alpha_conformal, q_scale))
    if (use_mad_rule && is.finite(sigma_hat)) message(sprintf("MAD sigma_hat: %.4f; z_cut: %.2f", sigma_hat, z_cut))
    message(sprintf("Holdout RMSE: %.4f | R^2: %.3f | MAE: %.4f",
                    holdout_metrics["RMSE"], holdout_metrics["Rsquared"], holdout_metrics["MAE"]))
  }

  list(
    results_table   = out_tbl,
    thresholds      = list(conformal_abs = q_abs,
                           alpha_conformal = alpha_conformal,
                           q_scale = q_scale,
                           sigma_hat = sigma_hat,
                           z_cut = z_cut,
                           tail = tail),
    holdout_metrics = holdout_metrics,
    plot            = plt
  )
}
