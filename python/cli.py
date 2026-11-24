"""
Command-line entry point installed as ``agri-svm``.

Examples::

    agri-svm classify                      # kernel comparison on the sample data
    agri-svm classify --data my_field.csv  # ... or on your own CSV
    agri-svm plots --out figures/          # static PNG figures
    agri-svm interactive --out figures/    # Plotly HTML dashboards
"""

import argparse

from . import __version__
from .agri_svm_classification import run_svm_pipeline
from .generate_interactive_plots import build_interactive_visuals
from .generate_plots import generate_all_plots


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="agri-svm",
        description="SVM classification of crop health and stress from phenomic data.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # Every sub-command accepts an optional CSV; plotting ones also an output folder.
    p_cl = sub.add_parser("classify", help="Compare SVM kernels and report test accuracy")
    p_cl.add_argument("--data", help="CSV file or URL (default: bundled sample data)")

    for name, helptext in (
        ("plots", "Write 300 DPI PNG figures"),
        ("interactive", "Write interactive Plotly HTML dashboards"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--data", help="CSV file or URL (default: bundled sample data)")
        p.add_argument("--out", help="Output folder (default: ./plots)")

    args = parser.parse_args(argv)
    if args.command == "classify":
        run_svm_pipeline(args.data)
    elif args.command == "plots":
        generate_all_plots(args.data, args.out)
    else:
        build_interactive_visuals(args.data, args.out)


if __name__ == "__main__":
    main()
