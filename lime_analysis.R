# LIME analysis

stopifnot(exists("train_df"), exists("test_df"), exists("final_model"))

n_features <- 10
k_plots    <- 6

lime_out <- explain_with_lime(
  model      = final_model,
  train_df   = train_df,
  test_df    = test_df,
  n_features = n_features
)

for (i in seq_len(min(k_plots, length(lime_out$plots)))) {
  print(lime_out$plots[[i]])
}
