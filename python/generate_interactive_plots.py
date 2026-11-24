"""
generate_interactive_plots.py
------------------------------
Builds interactive Plotly dashboards and 3D phenomic visualizations
for Support Vector Machine (SVM) agricultural classification.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Works both inside the installed ``agrisvm`` package and as a plain script
# run from the repository (``python python/<script>.py``).
try:
    from .agri_svm_data import resolve_data_path, resolve_output_dir
except ImportError:  # no parent package -> running as a standalone script
    from agri_svm_data import resolve_data_path, resolve_output_dir


def build_interactive_visuals(data_path=None, output_dir=None):
    """Write the Plotly dashboard and 3D scatter HTML files; return their paths.

    ``data_path`` defaults to the sample dataset; ``output_dir`` defaults to the
    repo's ``plots/`` folder (or ``./plots`` when installed with pip).
    """
    plots_dir = resolve_output_dir(output_dir)

    df = pd.read_csv(resolve_data_path(data_path))
    feature_cols = ["ndvi", "ndre", "canopy_temp_c", "chlorophyll_spad", "leaf_moisture_pct"]
    target_col = "condition"

    X = df[feature_cols].values
    y = df[target_col].values
    classes = sorted(list(set(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RBF SVM
    svm = SVC(kernel="rbf", C=2.0, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)

    # -------------------------------------------------------------
    # 1. Main 4-Panel Interactive Dashboard
    # -------------------------------------------------------------
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Interactive Confusion Matrix Heatmap</b>",
            "<b>Class-Wise Performance Metrics (Precision, Recall, F1)</b>",
            "<b>2D Phenomics Clustering (NDVI vs. Canopy Temp)</b>",
            "<b>SPAD Chlorophyll Distribution Across Classes</b>"
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.12
    )

    # Subplot (1,1): Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100).round(1)

    fig.add_trace(
        go.Heatmap(
            z=cm_norm,
            x=classes,
            y=classes,
            colorscale="Blues",
            text=[[f"{count} ({pct}%)" for count, pct in zip(row_c, row_p)] for row_c, row_p in zip(cm, cm_norm)],
            texttemplate="%{text}",
            name="Confusion Matrix",
            hoverongaps=False
        ),
        row=1, col=1
    )

    # Subplot (1,2): Class-Wise Performance Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    metric_names = ["precision", "recall", "f1-score"]
    colors = ["#0284c7", "#059669", "#7c3aed"]

    for i, m_name in enumerate(metric_names):
        vals = [report[cls][m_name] for cls in classes]
        fig.add_trace(
            go.Bar(
                name=m_name.capitalize(),
                x=classes,
                y=vals,
                marker_color=colors[i],
                hovertemplate=f"<b>Class</b>: %{{x}}<br><b>{m_name.capitalize()}</b>: %{{y:.3f}}<extra></extra>"
            ),
            row=1, col=2
        )

    # Subplot (2,1): 2D Phenomics Scatter
    class_colors = {"Healthy": "#059669", "Nutrient_Deficiency": "#d97706", "Fungal_Blight": "#dc2626"}
    for cls in classes:
        sub = df[df["condition"] == cls]
        fig.add_trace(
            go.Scatter(
                x=sub["ndvi"],
                y=sub["canopy_temp_c"],
                mode="markers",
                name=cls,
                marker=dict(size=7, color=class_colors[cls]),
                hovertemplate="<b>Condition</b>: " + cls + "<br><b>NDVI</b>: %{x:.3f}<br><b>Canopy Temp</b>: %{y:.1f}°C<extra></extra>"
            ),
            row=2, col=1
        )

    # Subplot (2,2): SPAD Chlorophyll Box Plot
    for cls in classes:
        sub = df[df["condition"] == cls]
        fig.add_trace(
            go.Box(
                y=sub["chlorophyll_spad"],
                name=cls,
                marker_color=class_colors[cls],
                boxpoints="outliers",
                showlegend=False
            ),
            row=2, col=2
        )

    fig.update_layout(
        title=dict(
            text="🌱 <b>Interactive Support Vector Machine (SVM) Agricultural Phenomics Dashboard</b>",
            font=dict(size=17, color="#0f172a"),
            x=0.04
        ),
        template="plotly_white",
        height=850,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(title_text="Predicted Condition", row=1, col=1)
    fig.update_yaxes(title_text="Actual Condition", row=1, col=1)

    fig.update_xaxes(title_text="Crop Condition Class", row=1, col=2)
    fig.update_yaxes(title_text="Score", range=[0, 1.1], row=1, col=2)

    fig.update_xaxes(title_text="Normalized Difference Vegetation Index (NDVI)", row=2, col=1)
    fig.update_yaxes(title_text="Canopy Temperature (°C)", row=2, col=1)

    fig.update_xaxes(title_text="Crop Condition Class", row=2, col=2)
    fig.update_yaxes(title_text="Chlorophyll SPAD Index", row=2, col=2)

    dashboard_path = os.path.join(plots_dir, "interactive_dashboard.html")
    fig.write_html(dashboard_path, include_plotlyjs="cdn")

    # -------------------------------------------------------------
    # 2. Interactive 3D Phenomics Feature Space
    # -------------------------------------------------------------
    fig_3d = go.Figure()
    for cls in classes:
        sub = df[df["condition"] == cls]
        fig_3d.add_trace(
            go.Scatter3d(
                x=sub["ndvi"],
                y=sub["canopy_temp_c"],
                z=sub["chlorophyll_spad"],
                mode="markers",
                name=cls,
                marker=dict(
                    size=5,
                    color=class_colors[cls],
                    opacity=0.85,
                    line=dict(color="#0f172a", width=0.5)
                ),
                hovertemplate="<b>Condition</b>: " + cls +
                              "<br><b>NDVI</b>: %{x:.3f}" +
                              "<br><b>Canopy Temp</b>: %{y:.1f}°C" +
                              "<br><b>SPAD Chlorophyll</b>: %{z:.1f}<extra></extra>"
            )
        )

    fig_3d.update_layout(
        title="<b>Interactive 3D Agricultural Phenomics Space (NDVI vs. Canopy Temp vs. Chlorophyll)</b>",
        scene=dict(
            xaxis_title="NDVI (Vegetative Vigor)",
            yaxis_title="Canopy Temp (°C)",
            zaxis_title="Chlorophyll SPAD Index"
        ),
        height=720,
        template="plotly_white"
    )

    scatter_3d_path = os.path.join(plots_dir, "interactive_3d_scatter.html")
    fig_3d.write_html(scatter_3d_path, include_plotlyjs="cdn")

    print(f"Interactive SVM plots successfully built:\n  - {dashboard_path}\n  - {scatter_3d_path}")
    return dashboard_path, scatter_3d_path


if __name__ == "__main__":
    build_interactive_visuals()
