# Container image published to GitHub Packages as
#   ghcr.io/lunganlungpg/agri-svm-classification
#
# It bundles BOTH implementations so nothing has to be installed locally:
#   docker run --rm ghcr.io/lunganlungpg/agri-svm-classification                 # Python SVM pipeline
#   docker run --rm ghcr.io/lunganlungpg/agri-svm-classification Rscript -e 'agrisvm::svm_run_pipeline()'
#   docker run --rm -v "$PWD/out:/opt/agri-svm/plots" ghcr.io/lunganlungpg/agri-svm-classification agri-svm plots

# rocker/r-ver = Ubuntu + pinned R, preconfigured for fast binary CRAN installs.
FROM rocker/r-ver:4.4.2

LABEL org.opencontainers.image.title="agri-svm-classification" \
      org.opencontainers.image.description="SVM crop health and stress classification (Python + R)" \
      org.opencontainers.image.licenses="MIT"

# Python runtime (venv is used because Ubuntu blocks system-wide pip installs).
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/agri-svm
COPY . .

# 1) Python package `agrisvm` + CLI `agri-svm`
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir .
ENV PATH="/opt/venv/bin:${PATH}"

# 2) R package `agrisvm` and its dependency e1071
RUN R -q -e 'install.packages("e1071")' \
 && R CMD INSTALL .

# Fail the build early if either package cannot be loaded.
RUN python -c "import agrisvm; print('agrisvm (Python)', agrisvm.__version__)" \
 && Rscript -e 'library(agrisvm); cat("agrisvm (R)", as.character(packageVersion("agrisvm")), "\n")'

CMD ["agri-svm", "classify"]
