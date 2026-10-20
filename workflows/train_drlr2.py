"""Train DRLR2 using a configured imitation-learning prior."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from learning.config import execute_workflow, register_configs
from learning.reinforcement.drlr2 import train_drlr2

register_configs()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_drlr2")
def main(config: DictConfig) -> None:
    execute_workflow(config, train_drlr2)


if __name__ == "__main__":
    main()
