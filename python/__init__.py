"""
agrisvm - Support Vector Machine classification of crop health / stress.

Install (Python >= 3.8)::

    pip install "git+https://github.com/lunganlungpg/agri-svm-classification.git"

Quick use::

    import agrisvm
    df = agrisvm.load_sample_data()             # spectral & phenomic crop dataset
    result = agrisvm.run_svm_pipeline()         # kernel comparison + test report
    agrisvm.generate_all_plots(output_dir="plots")         # 300 DPI PNG figures
    agrisvm.build_interactive_visuals(output_dir="plots")  # Plotly HTML

The same functions are available from the command line as ``agri-svm``
(see ``agri-svm --help``).

Note: in the repository this package lives in the ``python/`` folder; the
``pyproject.toml`` maps that folder to the import name ``agrisvm`` so the
existing scripts (``python python/agri_svm_classification.py``) keep working.
"""

from .agri_svm_classification import FEATURE_COLS, TARGET_COL, run_svm_pipeline
from .agri_svm_data import SAMPLE_URL, load_sample_data, resolve_data_path
from .generate_interactive_plots import build_interactive_visuals
from .generate_plots import generate_all_plots

__version__ = "1.0.0"

__all__ = [
    "FEATURE_COLS",
    "TARGET_COL",
    "SAMPLE_URL",
    "load_sample_data",
    "resolve_data_path",
    "run_svm_pipeline",
    "generate_all_plots",
    "build_interactive_visuals",
]
