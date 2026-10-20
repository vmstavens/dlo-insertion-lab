"""Create a small linear checkpoint for the portable Cartpole example."""

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "linear.npz"
    manifest = args.output / "manifest.json"
    if checkpoint.exists() or manifest.exists():
        parser.error("Example output already exists; choose a new directory.")
    # Nonzero weights make this a working inference example, not a trained policy.
    np.savez(
        checkpoint, weights=np.full((5, 1), 0.1, dtype=np.float32), bias=np.zeros(1)
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_type": "linear_tanh_example",
                "loader": "learning.policies.checkpoints:load_linear_policy",
                "checkpoint": "linear.npz",
                "observation_dim": 5,
                "action_dim": 1,
                "action_semantics": "Cartpole normalized actuator control in [-1, 1]",
                "config": {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
