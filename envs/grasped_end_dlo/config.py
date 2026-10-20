"""Configuration shared by all grasped-end DLO environments."""

from __future__ import annotations

import numpy as np
from ml_collections import config_dict

from envs.grasped_end_dlo import constants


def base_config() -> config_dict.ConfigDict:
    """Return configuration shared by all grasped-end DLO tasks."""
    return config_dict.create(
        ctrl_dt=0.002,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        sparse_reward=False,
        sparse_reward_scale=100.0,
        a_max=0,
        a_min=0,
        success_threshold_pos=0.02,
        success_threshold_rot=float(np.deg2rad(10.0)),
        reset_pos_min=list(constants.RESET_POS_MIN),
        reset_pos_max=list(constants.RESET_POS_MAX),
        reset_rot_min=list(constants.RESET_ROT_MIN),
        reset_rot_max=list(constants.RESET_ROT_MAX),
        initial_curvature_randomization=False,
        curvature_theta_min=float(np.deg2rad(45.0)),
        curvature_theta_max=float(np.deg2rad(60.0)),
        curvature_phi_min=0.0,
        curvature_phi_max=float(2.0 * np.pi),
        curvature_update_ref=True,
        reset_keyframe=None,
        homo=False,
        naconmax=4500,
        njmax=512,
        impl="warp",
    )
