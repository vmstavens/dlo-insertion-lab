"""Transform raw demonstrations into a training dataset."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from learning.config import execute_workflow, register_configs
from learning.data.prepare import prepare_dataset

register_configs()


@hydra.main(version_base="1.3", config_path="../configs", config_name="prepare")
def main(config: DictConfig) -> None:
    execute_workflow(config, prepare_dataset)


if __name__ == "__main__":
    main()
