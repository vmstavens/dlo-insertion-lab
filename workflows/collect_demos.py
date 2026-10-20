"""Collect demonstrations for a configured task."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from learning.collection.collect import collect_demonstrations
from learning.config import execute_workflow, register_configs

register_configs()


@hydra.main(version_base="1.3", config_path="../configs", config_name="collect")
def main(config: DictConfig) -> None:
    execute_workflow(config, collect_demonstrations)


if __name__ == "__main__":
    main()
