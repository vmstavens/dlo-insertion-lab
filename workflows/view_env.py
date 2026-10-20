"""Inspect an environment with zero actions and no trained policy."""

import hydra
from omegaconf import DictConfig

from learning.config import execute_workflow, register_configs
from learning.evaluation.rollouts import evaluate_policy

register_configs()


@hydra.main(version_base="1.3", config_path="../configs", config_name="view_env")
def main(config: DictConfig) -> None:
    execute_workflow(config, evaluate_policy)


if __name__ == "__main__":
    main()
