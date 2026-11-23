"""
generate_plots.py
-----------------
Renders publication-quality static visualization figures for Support Vector
Machine (SVM) classification of agricultural crop health and stress conditions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


def generate_all_plots():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "crop_disease_phenomics.csv")
    plots_dir = os.path.join(base_dir, "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    feature_cols = ["ndvi", "ndre", "canopy_temp_c", "chlorophyll_spad", "leaf_moisture_pct"]
    target_col = "condition"

    X = df[feature_cols].values
    y = df[target_col].values
    classes = np.unique(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # Plot 1: 2D Decision Boundaries (Linear vs. RBF Kernel)
    # Using 2 key agronomic indicators: NDVI and Canopy Temp
    # -------------------------------------------------------------
    idx_ndvi = 0
    idx_temp = 2
    X_2d_train = X_train[:, [idx_ndvi, idx_temp]]
    y_2d_train = y_train

    scaler_2d = StandardScaler()
    X_2d_scaled = scaler_2d.fit_transform(X_2d_train)

    # Map labels to integers for contour plotting
    label_map = {name: i for i, name in enumerate(classes)}
    y_2d_int = np.array([label_map[item] for item in y_2d_train])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Meshgrid for contour
    x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
    y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    cmap_bg = plt.get_cmap("Pastel2", 3)
    cmap_pts = ["#047857", "#d97706", "#dc2626"]

    for ax, kernel_name, model in [
        (ax1, "Linear Kernel SVM", SVC(kernel="linear", C=1.0)),
        (ax2, "RBF Kernel (Gaussian) SVM", SVC(kernel="rbf", C=2.0, gamma="scale"))
    ]:
        model.fit(X_2d_scaled, y_2d_int)
        Z = model.predict(grid_points).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)
        ax.contour(xx, yy, Z, colors="#475569", linewidths=0.8, alpha=0.7)

        # Plot training points by class
        for i, cls_name in enumerate(classes):
            mask = y_2d_int == i
            ax.scatter(
                X_2d_scaled[mask, 0], X_2d_scaled[mask, 1],
                color=cmap_pts[i], label=cls_name, edgecolors="#0f172a",
                s=40, alpha=0.85
            )

        # Highlight Support Vectors
        sv = model.support_vectors_
        ax.scatter(
            sv[:, 0], sv[:, 1], s=90, facecolors="none",
            edgecolors="#1e1b4b", linewidths=1.5,
            label=f"Support Vectors (n={len(sv)})"
        )

        ax.set_title(f"{kernel_name}\n({len(sv)} Support Vectors)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Standardized NDVI", fontsize=9, fontweight="bold")
        ax.set_ylabel("Standardized Canopy Temperature", fontsize=9, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", fontsize=8, loc="upper right")

    plt.suptitle("SVM Decision Boundaries & Support Vector Margins in Precision Agriculture", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "svm_decision_boundaries.png"))
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: Confusion Matrix Heatmap
    # -------------------------------------------------------------
    svm_full = SVC(kernel="rbf", C=2.0, gamma="scale", random_state=42)
    svm_full.fit(X_train_scaled, y_train)
    y_pred = svm_full.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    im = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized Recall Rate", fontsize=9, fontweight="bold")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=15, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.1%}\n(n={cm[i, j]})", ha="center", va="center", color=color, fontsize=9, fontweight="bold")
    ax.set_title("SVM Crop Stress Classification: Normalized Confusion Matrix", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Condition", fontsize=10, fontweight="bold")
    ax.set_ylabel("Actual True Condition", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()

    # -------------------------------------------------------------
    # Plot 3: Hyperparameter Tuning Grid Search Heatmap (C vs. Gamma)
    # -------------------------------------------------------------
    param_grid = {
        "C": [0.1, 1.0, 5.0, 20.0, 100.0],
        "gamma": [0.005, 0.02, 0.1, 0.5, 2.0]
    }
    grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train_scaled, y_train)

    scores_matrix = grid.cv_results_["mean_test_score"].reshape(len(param_grid["C"]), len(param_grid["gamma"]))

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    im = ax.imshow(scores_matrix, cmap="YlGnBu", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("5-Fold Cross-Validation Accuracy", fontsize=9, fontweight="bold")
    ax.set_xticks(np.arange(len(param_grid["gamma"])))
    ax.set_yticks(np.arange(len(param_grid["C"])))
    ax.set_xticklabels(param_grid["gamma"], fontsize=9)
    ax.set_yticklabels(param_grid["C"], fontsize=9)
    for i in range(len(param_grid["C"])):
        for j in range(len(param_grid["gamma"])):
            val = scores_matrix[i, j]
            color = "white" if val > 0.85 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)
    ax.set_title("RBF Kernel Hyperparameter Optimization (C vs. Gamma Grid)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Gamma (RBF Kernel Bandwidth)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Regularization Parameter (C)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hyperparameter_gridsearch.png"))
    plt.close()
    plt.title("RBF Kernel Hyperparameter Optimization (C vs. Gamma Grid)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Gamma (RBF Kernel Bandwidth)", fontsize=10, fontweight="bold")
    plt.ylabel("Regularization Parameter (C)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hyperparameter_gridsearch.png"))
    plt.close()

    # -------------------------------------------------------------
    # Plot 4: Multiclass ROC-AUC Curves (One-vs-Rest)
    # -------------------------------------------------------------
    svm_prob = SVC(kernel="rbf", C=grid.best_params_["C"], gamma=grid.best_params_["gamma"], probability=True, random_state=42)
    svm_prob.fit(X_train_scaled, y_train)
    y_probs = svm_prob.predict_proba(X_test_scaled)
    y_test_bin = label_binarize(y_test, classes=classes)

    plt.figure(figsize=(7, 5.5), dpi=300)
    colors = ["#047857", "#d97706", "#dc2626"]

    for i, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2.2, label=f"{cls_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier")
    plt.title("Multiclass One-vs-Rest (OvR) ROC Curves for Crop Conditions", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_ovr_curves.png"))
    plt.close()

    print(f"All 4 static publication plots successfully saved in: {plots_dir}")


if __name__ == "__main__":
    generate_all_plots()
