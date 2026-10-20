"""Train a Diffusion Policy imitation-learning model."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from learning.config import execute_workflow, register_configs
from learning.imitation.diffusion_policy import train_diffusion_policy

register_configs()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_dp")
def main(config: DictConfig) -> None:
    execute_workflow(config, train_diffusion_policy)


if __name__ == "__main__":
    main()
