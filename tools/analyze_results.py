"""Compute publication statistics from an explicit selection of completed runs."""

import argparse
from pathlib import Path

from learning.publication.analysis import analyze


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.recipe.resolve(), args.output)
    print(f"Analyzed {len(report['sources'])} runs into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
