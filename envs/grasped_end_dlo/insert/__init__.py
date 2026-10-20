"""Grasped-end cable insertion task registration."""

from envs.grasped_end_dlo import register_environment
from envs.grasped_end_dlo.insert.config import default_config
from envs.grasped_end_dlo.insert.randomize import domain_randomize
from envs.grasped_end_dlo.insert.task import GraspedEndDloInsert

ENV_NAME = "GraspedEndDloInsert"

register_environment(
    ENV_NAME,
    GraspedEndDloInsert,
    default_config,
    domain_randomizer=domain_randomize,
)

__all__ = ["ENV_NAME", "GraspedEndDloInsert", "default_config", "domain_randomize"]
