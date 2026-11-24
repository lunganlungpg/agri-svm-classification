# agri_svm_classification.R
# -------------------------
# Support Vector Machine for Agricultural Crop Stress Classification in R
#
# This file is part of the `agrisvm` R package:
#   remotes::install_github("lunganlungpg/agri-svm-classification")
#   agrisvm::svm_run_pipeline()
#
# It can also still be run as a plain script from the repository root:
#   Rscript R/agri_svm_classification.R

#' Public URL of the sample crop phenomics dataset
#'
#' Used when the package is installed without a local copy of the repository.
#' @keywords internal
svm_sample_url <- function() {
  "https://raw.githubusercontent.com/lunganlungpg/agri-svm-classification/main/data/crop_disease_phenomics.csv"
}

#' Locate the sample dataset
#'
#' Looks for `data/crop_disease_phenomics.csv` in (or one level above) the
#' working directory, i.e. inside a cloned copy of the repository. If it is not
#' found, the public GitHub copy is used instead.
#'
#' @return A file path or URL that [utils::read.csv()] can read.
#' @export
#' @examples
#' svm_sample_data_path()
svm_sample_data_path <- function() {
  candidates <- c(file.path("data", "crop_disease_phenomics.csv"),
                  file.path("..", "data", "crop_disease_phenomics.csv"))
  found <- candidates[file.exists(candidates)]
  if (length(found)) found[[1]] else svm_sample_url()
}

#' Load the sample crop phenomics dataset
#'
#' Columns: `ndvi`, `ndre`, `canopy_temp_c`, `chlorophyll_spad`,
#' `leaf_moisture_pct` and the class label `condition` (as a factor).
#'
#' @param path Optional CSV path or URL. Defaults to [svm_sample_data_path()].
#' @return A data frame.
#' @export
#' @examples
#' \donttest{
#' df <- svm_load_sample_data()
#' table(df$condition)
#' }
svm_load_sample_data <- function(path = svm_sample_data_path()) {
  df <- utils::read.csv(path)
  df$condition <- as.factor(df$condition)
  df
}

#' Fit an RBF-kernel SVM crop condition classifier
#'
#' Splits the data 80/20, fits [e1071::svm()] on the training part (features
#' are scaled automatically) and evaluates it on the held-out part.
#'
#' @param df Data frame as returned by [svm_load_sample_data()].
#' @param cost,gamma SVM hyper-parameters for the radial kernel.
#' @param train_frac Fraction of rows used for training.
#' @param seed Random seed for the train/test split.
#' @return A list with `model`, `confusion` (table) and `accuracy`.
#' @export
#' @examples
#' \donttest{
#' fit <- svm_fit_classifier(svm_load_sample_data())
#' fit$accuracy
#' }
svm_fit_classifier <- function(df = svm_load_sample_data(), cost = 2.0, gamma = 0.2,
                               train_frac = 0.8, seed = 42) {
  # Train-test split
  set.seed(seed)
  n <- nrow(df)
  train_indices <- sample(seq_len(n), size = round(train_frac * n))
  train_data <- df[train_indices, ]
  test_data <- df[-train_indices, ]

  # e1071 scales features automatically by default
  svm_model <- e1071::svm(
    condition ~ ndvi + ndre + canopy_temp_c + chlorophyll_spad + leaf_moisture_pct,
    data = train_data,
    kernel = "radial",
    cost = cost,
    gamma = gamma,
    probability = TRUE
  )

  test_preds <- stats::predict(svm_model, newdata = test_data)
  conf_matrix <- table(Actual = test_data$condition, Predicted = test_preds)
  list(model = svm_model,
       confusion = conf_matrix,
       accuracy = sum(diag(conf_matrix)) / sum(conf_matrix))
}

#' Run the full SVM demo and print the results
#'
#' Prints the model summary, the confusion matrix on the held-out test set and
#' the test accuracy (same output as the original script).
#'
#' @param path CSV path or URL. Defaults to [svm_sample_data_path()].
#' @return Invisibly, the list returned by [svm_fit_classifier()].
#' @export
#' @examples
#' \donttest{
#' svm_run_pipeline()
#' }
svm_run_pipeline <- function(path = svm_sample_data_path()) {
  fit <- svm_fit_classifier(svm_load_sample_data(path))

  cat("======================================================\n")
  cat("SVM Classification Model Summary (R / e1071):\n")
  cat("======================================================\n")
  print(summary(fit$model))

  cat("\nConfusion Matrix on Held-Out Test Set:\n")
  print(fit$confusion)
  cat(sprintf("\nTest Accuracy: %.2f%%\n", fit$accuracy * 100))
  invisible(fit)
}

# Script mode: `Rscript R/agri_svm_classification.R` runs the demo. When the
# file is loaded as part of the installed package, sys.nframe() is > 0.
if (sys.nframe() == 0L) {
  if (!requireNamespace("e1071", quietly = TRUE)) {
    install.packages("e1071", repos = "https://cloud.r-project.org")
  }
  svm_run_pipeline()
}
