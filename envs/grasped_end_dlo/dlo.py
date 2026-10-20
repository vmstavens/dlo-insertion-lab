"""JAX helpers for initializing deformable linear objects."""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco as mj
from mujoco import mjx


def cable_ball_joint_qpos_adrs(
    model: mj.MjModel,
    *,
    joint_prefix: str = "cable:",
) -> jax.Array:
    """Return qpos addresses for cable ball joints in model order."""
    addresses = [
        int(model.jnt_qposadr[joint_id])
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mj.mjtJoint.mjJNT_BALL)
        and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or "").startswith(
            joint_prefix
        )
    ]
    if not addresses:
        raise ValueError(f"No cable ball joints found with prefix {joint_prefix!r}.")
    return jp.asarray(addresses, dtype=jp.int32)


def _rotation_vector_to_quaternion(rotation_vector: jax.Array) -> jax.Array:
    angle = jp.linalg.norm(rotation_vector)
    half_angle = 0.5 * angle
    scale = jp.where(
        angle > 1e-8,
        jp.sin(half_angle) / angle,
        0.5 - angle**2 / 48.0,
    )
    quaternion = jp.concatenate([jp.cos(half_angle)[None], rotation_vector * scale])
    return quaternion / jp.maximum(jp.linalg.norm(quaternion), 1e-8)


def _curve_qpos(
    qpos: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
    *,
    joint_qpos_adrs: jax.Array,
) -> jax.Array:
    """Apply a constant total bend across all cable ball joints."""
    joint_theta = theta / joint_qpos_adrs.shape[0]
    rotation_vector = jp.stack(
        [
            jp.asarray(0.0, dtype=qpos.dtype),
            joint_theta * jp.cos(phi),
            joint_theta * jp.sin(phi),
        ]
    )
    quaternion = _rotation_vector_to_quaternion(rotation_vector)
    quaternion_indices = joint_qpos_adrs[:, None] + jp.arange(4)
    return qpos.at[quaternion_indices].set(quaternion)


def apply_random_pre_curvature(
    model: mjx.Model,
    qpos: jax.Array,
    rng: jax.Array,
    *,
    cable_joint_qpos_adrs: jax.Array,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    update_ref: bool = True,
) -> tuple[mjx.Model, jax.Array, jax.Array, jax.Array]:
    """Sample and apply planar pre-curvature to a cable state."""
    theta_key, phi_key = jax.random.split(rng)
    theta = jax.random.uniform(theta_key, (), minval=theta_min, maxval=theta_max)
    phi = jax.random.uniform(phi_key, (), minval=phi_min, maxval=phi_max)
    qpos = _curve_qpos(
        qpos,
        theta,
        phi,
        joint_qpos_adrs=cable_joint_qpos_adrs,
    )
    if update_ref:
        quaternion_indices = cable_joint_qpos_adrs[:, None] + jp.arange(4)
        qpos_spring = model.qpos_spring.at[quaternion_indices].set(
            qpos[quaternion_indices]
        )
        model = model.tree_replace({"qpos_spring": qpos_spring})
    return model, qpos, theta, phi
