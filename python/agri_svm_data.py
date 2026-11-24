"""
agri_svm_data.py
----------------
Locates the sample dataset and the output folder for plots.

The same code runs in two situations, and this module hides the difference:

1. From a cloned copy of the repository, e.g. ``python python/generate_plots.py``.
   The CSV is read from ``<repo>/data/`` and figures go to ``<repo>/plots/``.
2. After ``pip install git+https://github.com/lunganlungpg/agri-svm-classification``.
   The repository folders are not installed, so the CSV is read from
   ``./data/`` if present, otherwise straight from GitHub (like
   ``seaborn.load_dataset``), and figures go to ``./plots`` in the current
   working directory.

Passing an explicit ``data_path`` / ``output_dir`` always wins.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

#: File name of the bundled crop phenomics / disease dataset.
SAMPLE_FILE = "crop_disease_phenomics.csv"

#: Public copy of the dataset, used when the package is installed without the repo.
SAMPLE_URL = (
    "https://raw.githubusercontent.com/lunganlungpg/agri-svm-classification/main/data/"
    + SAMPLE_FILE
)

# Repository root = the folder above this file's folder (``python/``).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_data_path(data_path: Optional[str] = None) -> str:
    """Return the CSV location to read.

    Order: explicit path > repo copy next to this file > ``./data`` in the
    current working directory (e.g. inside the Docker image) > GitHub URL.
    """
    if data_path:
        return data_path
    for folder in (os.path.join(_REPO_ROOT, "data"), "data"):
        local = os.path.join(folder, SAMPLE_FILE)
        if os.path.exists(local):
            return local
    return SAMPLE_URL


def resolve_output_dir(output_dir: Optional[str] = None) -> str:
    """Return (and create) the folder where figures are written."""
    if output_dir is None:
        # Inside a cloned repo the ``data/`` folder sits next to ``plots/``;
        # otherwise fall back to ``./plots`` so site-packages is never written to.
        in_repo = os.path.isdir(os.path.join(_REPO_ROOT, "data"))
        output_dir = os.path.join(_REPO_ROOT, "plots") if in_repo else os.path.abspath("plots")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_sample_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load the crop phenomics dataset (5 spectral/phenomic features + ``condition``)."""
    return pd.read_csv(resolve_data_path(data_path))
