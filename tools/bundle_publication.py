"""Package selected results and code into a portable figure-reproduction bundle."""

import argparse
from pathlib import Path

from learning.publication.bundle import create_bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=float, default=3.5)
    args = parser.parse_args(argv)
    create_bundle(args.analysis_dir, args.output, width=args.width)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
