"""Preview or execute a local publication experiment matrix."""

import argparse
import json
from pathlib import Path

from learning.publication.matrix import execute_matrix, job_status, plan_matrix


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument(
        "--run", action="store_true", help="Execute missing/failed jobs"
    )
    args = parser.parse_args(argv)
    jobs = plan_matrix(args.matrix.resolve())
    for job in jobs:
        print(f"{job['id']} {job_status(job)} {json.dumps(job['choices'])}")
    print(f"{len(jobs)} jobs")
    return int(not execute_matrix(jobs)) if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
