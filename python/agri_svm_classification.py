"""
agri_svm_classification.py
--------------------------
Support Vector Machine (SVM) pipeline for agricultural crop health and
stress classification using spectral and phenomic indicators.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def run_svm_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "crop_disease_phenomics.csv")

    df = pd.read_csv(data_path)
    feature_cols = ["ndvi", "ndre", "canopy_temp_c", "chlorophyll_spad", "leaf_moisture_pct"]
    target_col = "condition"

    X = df[feature_cols].values
    y = df[target_col].values

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Feature Scaling is critical for distance-based SVM margin optimization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compare Kernel Formulations
    kernels = {
        "Linear Kernel": SVC(kernel="linear", C=1.0, random_state=42),
        "RBF Kernel (Gaussian)": SVC(kernel="rbf", C=2.0, gamma="scale", random_state=42),
        "Polynomial Kernel (deg=3)": SVC(kernel="poly", degree=3, C=1.0, random_state=42)
    }

    print("=" * 70)
    print("Agricultural Crop Condition Classification: SVM Kernel Comparison")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in kernels.items():
        scores = cross_validate(
            model, X_train_scaled, y_train, cv=cv,
            scoring=["accuracy", "f1_macro"]
        )
        print(f"\nModel: {name}")
        print(f"  - 5-Fold Mean Accuracy: {scores['test_accuracy'].mean():.4f} (±{scores['test_accuracy'].std():.4f})")
        print(f"  - 5-Fold Macro F1:     {scores['test_f1_macro'].mean():.4f} (±{scores['test_f1_macro'].std():.4f})")

    # Fit best model (RBF Kernel) on training set and evaluate on held-out test set
    best_svm = kernels["RBF Kernel (Gaussian)"]
    best_svm.fit(X_train_scaled, y_train)
    test_preds = best_svm.predict(X_test_scaled)

    print("\n" + "=" * 70)
    print("Held-Out Test Set Performance (RBF Kernel SVM):")
    print("=" * 70)
    print(f"Overall Test Accuracy: {accuracy_score(y_test, test_preds) * 100:.2f}%\n")
    print(classification_report(y_test, test_preds))

    # Support Vector Diagnostics
    print("Support Vector Analysis:")
    print(f"  - Total Support Vectors: {len(best_svm.support_)} out of {len(X_train)} training points")
    print(f"  - Support Vector Ratio:  {len(best_svm.support_) / len(X_train) * 100:.2f}%")
    for cls_name, count in zip(best_svm.classes_, best_svm.n_support_):
        print(f"    * Class '{cls_name}': {count} support vectors")


if __name__ == "__main__":
    run_svm_pipeline()
