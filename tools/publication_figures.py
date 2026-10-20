"""Generate PDF/SVG plots, source CSV and LaTeX tables."""

import argparse
from pathlib import Path

from learning.publication.figures import render_figures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--width", type=float, default=3.5, help="Figure width in inches"
    )
    args = parser.parse_args(argv)
    render_figures(args.analysis, args.output, width=args.width)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
