"""Pure reward functions shared by grasped-end DLO tasks."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp

REWARD_METRIC_NAMES = (
    "position_error_prev",
    "rotation_error_prev",
    "position_error",
    "rotation_error",
    "reward_delta_pose_prev",
    "reward_delta_pose",
    "reward_position_progress",
    "reward_rotation_distance_component",
    "is_success",
    "is_homo",
    "reward_homo",
    "reward_total",
)


@dataclass(frozen=True)
class PoseRewardConfig:
    """Static parameters for normalized pose-progress rewards."""

    position_threshold: float
    rotation_threshold: float
    position_weight: float = 1.0
    rotation_weight: float = 1.0
    homo_false: float = -0.2
    homo_true: float = 0.05
    success_bonus: float = 25.0
    terminal_penalty: float = -25.0


def empty_metrics() -> dict[str, jax.Array]:
    """Return zero-valued reward metrics with stable keys."""
    return {name: jp.asarray(0.0) for name in REWARD_METRIC_NAMES}


def sparse_success_reward(
    success: jax.Array,
    *,
    scale: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return a sparse success reward and complete metric dictionary."""
    reward = success.astype(float) * scale
    metrics = {
        **empty_metrics(),
        "is_success": success.astype(float),
        "reward_total": reward,
    }
    return reward, metrics


def pose_progress_reward(
    *,
    previous_position: jax.Array,
    previous_rotation: jax.Array,
    position: jax.Array,
    rotation: jax.Array,
    success: jax.Array,
    done: jax.Array,
    is_homo: jax.Array,
    config: PoseRewardConfig,
    use_homo_reward: bool = True,
) -> dict[str, jax.Array]:
    """Compose normalized pose progress, task shaping, and terminal rewards."""
    epsilon = jp.asarray(1e-6)
    position_scale = jp.maximum(config.position_threshold, epsilon)
    rotation_scale = jp.maximum(config.rotation_threshold, epsilon)
    normalized_position_prev = previous_position / position_scale
    normalized_position = position / position_scale
    normalized_rotation_prev = previous_rotation / rotation_scale
    normalized_rotation = rotation / rotation_scale

    pose_error_prev = (
        config.position_weight * normalized_position_prev
        + config.rotation_weight * normalized_rotation_prev
    )
    pose_error = (
        config.position_weight * normalized_position
        + config.rotation_weight * normalized_rotation
    )
    position_progress = config.position_weight * (
        normalized_position_prev - normalized_position
    )
    rotation_distance = -config.rotation_weight * normalized_rotation

    success_bonus = jp.where(success, config.success_bonus, 0.0)
    terminal_penalty = jp.where((done > 0.0) & (~success), config.terminal_penalty, 0.0)
    homo_reward = jp.asarray(0.0)
    if use_homo_reward:
        homo_reward = jp.where(is_homo, config.homo_true, config.homo_false)
    reward = (
        pose_error_prev - pose_error + homo_reward + success_bonus + terminal_penalty
    )
    return {
        "position_error_prev": previous_position,
        "rotation_error_prev": previous_rotation,
        "position_error": position,
        "rotation_error": rotation,
        "reward_delta_pose_prev": pose_error_prev,
        "reward_delta_pose": pose_error,
        "reward_position_progress": position_progress,
        "reward_rotation_distance_component": rotation_distance,
        "is_success": success.astype(float),
        "is_homo": is_homo.astype(float),
        "reward_homo": homo_reward,
        "reward_total": reward,
    }
