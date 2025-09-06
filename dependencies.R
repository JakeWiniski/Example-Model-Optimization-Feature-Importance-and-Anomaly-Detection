# ------------------------------------------------
# Loads all required packages for the workflow
# ================================================================

# Utility to install missing packages automatically
load_or_install <- function(pkgs) {
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
    library(pkg, character.only = TRUE)
  }
}

# Core tidyverse utilities
core_pkgs <- c(
  "dplyr",
  "ggplot2",
  "tibble"
)

# Modeling and preprocessing
ml_pkgs <- c(
  "caret",      # modeling framework, RFE, CV
  "ranger",     # fast random forest engine
  "vip"         # variable importance plots
)

# Explainability
explain_pkgs <- c(
  "lime"        # local model explanations
)

# Plotting helpers (optional)
plot_pkgs <- c(
  "patchwork"   # combining plots
)

# Load all groups
all_pkgs <- c(core_pkgs, ml_pkgs, explain_pkgs, plot_pkgs)
load_or_install(all_pkgs)

# Confirm loaded
cat("All dependencies loaded.\n")
