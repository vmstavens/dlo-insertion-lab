"""Inspect checkpoint metadata without loading weights or executing its loader."""

import argparse
import json
from pathlib import Path

from learning.policies.checkpoints import read_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    path = args.manifest.expanduser().resolve()
    manifest = read_manifest(path)
    checkpoint = Path(manifest["checkpoint"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = path.parent / checkpoint
    print(
        json.dumps(
            {
                **manifest,
                "resolved_checkpoint": str(checkpoint),
                "checkpoint_exists": checkpoint.exists(),
                "checkpoint_bytes": checkpoint.stat().st_size
                if checkpoint.is_file()
                else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
