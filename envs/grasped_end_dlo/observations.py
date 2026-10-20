from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import mjsim as ms
import mujoco as mj
import numpy as np
from mjxsim import ObjType, get_pose
from mujoco import mjx

from envs.grasped_end_dlo.constants import OBSERVATION_LAYOUT


def relative_pose_observation(
    current_rotation: jax.Array,
    current_translation: jax.Array,
    target_rotation: jax.Array,
    target_translation: jax.Array,
) -> jax.Array:
    """Express target position and target z-axis in the current frame."""
    current_rotation_inv = jp.linalg.matrix_transpose(current_rotation)
    return jp.concatenate(
        [
            current_rotation_inv @ (target_translation - current_translation),
            current_rotation_inv @ target_rotation[:, 2],
        ]
    )


def mjx_site_observation(
    model: mjx.Model,
    data: mjx.Data,
    *,
    current_site: str,
    target_site: str,
) -> jax.Array:
    """Build the relative-pose observation from two MJX sites."""
    current_pose = get_pose(model, data, current_site, ObjType.SITE)
    target_pose = get_pose(model, data, target_site, ObjType.SITE)
    return relative_pose_observation(
        current_pose.rotation().as_matrix(),
        current_pose.translation(),
        target_pose.rotation().as_matrix(),
        target_pose.translation(),
    )


def mj_site_observation(
    model: mj.MjModel,
    data: mj.MjData,
    *,
    current_site: str,
    target_site: str,
) -> np.ndarray:
    """NumPy equivalent of :func:`mjx_site_observation`."""
    current_pose = ms.get_pose(model, data, current_site, ms.ObjType.SITE)
    target_pose = ms.get_pose(model, data, target_site, ms.ObjType.SITE)
    current_rotation_inv = current_pose.R.T
    return np.concatenate(
        [
            current_rotation_inv @ (target_pose.t - current_pose.t),
            current_rotation_inv @ target_pose.R[:, 2],
        ]
    ).astype(np.float64)


def pose_errors(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return position error in metres and z-axis angle error in radians."""
    position_error = jp.linalg.norm(observation[:3])
    rotation_error = jp.arccos(jp.clip(observation[5], -1.0, 1.0))
    return position_error, rotation_error


def observation_space() -> dict[str, Any]:
    """Describe the flat pose observation used by demonstration tooling."""
    return {
        "shape": [len(OBSERVATION_LAYOUT)],
        "dtype": "float64",
        "layout": list(OBSERVATION_LAYOUT),
    }
