from __future__ import annotations

import jax
import jax.numpy as jp
from mjxsim import ObjType, get_pose
from mujoco import mjx


def radial_distance(
    point: jax.Array,
    origin: jax.Array,
    *,
    radial_axes: tuple[int, int] = (0, 2),
) -> jax.Array:
    """Return point-to-axis distance for an axis through ``origin``."""
    offset = point - origin
    return jp.linalg.norm(offset[jp.asarray(radial_axes)])


def site_within_pipe(
    model: mjx.Model,
    data: mjx.Data,
    *,
    site_name: str,
    pipe_origin_site: str,
    inner_radius: float | jax.Array,
    radial_axes: tuple[int, int] = (0, 2),
) -> jax.Array:
    """Return whether a site lies within the pipe's radial clearance."""
    site_pose = get_pose(model, data, site_name, ObjType.SITE)
    origin_pose = get_pose(model, data, pipe_origin_site, ObjType.SITE)
    distance = radial_distance(
        site_pose.translation(),
        origin_pose.translation(),
        radial_axes=radial_axes,
    )
    return distance <= jp.asarray(inner_radius)
