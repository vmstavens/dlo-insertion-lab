"""Configuration for the grasped-end cable insertion task."""

from __future__ import annotations

from ml_collections import config_dict

from envs.grasped_end_dlo.config import base_config


def default_config() -> config_dict.ConfigDict:
    """Return a fresh insertion-task configuration."""
    config = base_config()
    config.reset_keyframe = "init"
    config.homo = True
    config.reward_config = config_dict.create(
        position_weight=1.0,
        rotation_weight=1.0,
        homo_false=-0.2,
        homo_true=0.05,
        success_bonus=25.0,
        terminal_penalty=-25.0,
    )
    return config
