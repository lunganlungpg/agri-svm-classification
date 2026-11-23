# agri_svm_classification.R
# -------------------------
# Support Vector Machine for Agricultural Crop Stress Classification in R

suppressPackageStartupMessages({
  if (!requireNamespace("e1071", quietly = TRUE)) {
    install.packages("e1071", repos = "https://cloud.r-project.org")
  }
  library(e1071)
})

# 1. Load Data
data_path <- file.path("..", "data", "crop_disease_phenomics.csv")
if (!file.exists(data_path)) {
  data_path <- file.path("data", "crop_disease_phenomics.csv")
}
df <- read.csv(data_path)
df$condition <- as.factor(df$condition)

# 2. Train-Test Split (80/20)
set.seed(42)
n <- nrow(df)
train_indices <- sample(1:n, size = round(0.8 * n))
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

# 3. Fit SVM with Radial Basis Function (RBF) Kernel
# e1071 scales features automatically by default
svm_model <- svm(
  condition ~ ndvi + ndre + canopy_temp_c + chlorophyll_spad + leaf_moisture_pct,
  data = train_data,
  kernel = "radial",
  cost = 2.0,
  gamma = 0.2,
  probability = TRUE
)

# 4. Model Diagnostics & Support Vectors
cat("======================================================\n")
cat("SVM Classification Model Summary (R / e1071):\n")
cat("======================================================\n")
print(summary(svm_model))

# 5. Evaluate on Test Set
test_preds <- predict(svm_model, newdata = test_data)
conf_matrix <- table(Actual = test_data$condition, Predicted = test_preds)

cat("\nConfusion Matrix on Held-Out Test Set:\n")
print(conf_matrix)

acc <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat(sprintf("\nTest Accuracy: %.2f%%\n", acc * 100))
