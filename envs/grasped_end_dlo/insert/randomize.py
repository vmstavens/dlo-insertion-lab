"""Domain randomization for grasped-end DLO models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from mjxsim import ObjType
from mujoco import mjx

from envs.grasped_end_dlo import constants

_CABLE_BODY_NAMES = tuple([f"cable:B{i}" for i in range(1, 9)] + ["cable:Blast"])
_CABLE_GEOM_NAMES = (
    "cable:Gfirst",
    *(f"cable:G{i}" for i in range(1, 9)),
    "cable:Glast",
)
_CABLE_JOINT_NAMES = (
    "cable:Jfirst",
    *(f"cable:J{i}" for i in range(2, 9)),
    "cable:Jlast",
)
_PIPE_GEOM_NAMES = tuple(f"pipe:{index}" for index in range(constants.PIPE_RESOLUTION))


@dataclass(frozen=True)
class CablePipeRandomizationIds:
    """Model IDs affected by cable/pipe domain randomization."""

    cable_body_ids: np.ndarray
    cable_geom_ids: np.ndarray
    cable_joint_ids: np.ndarray
    pipe_geom_ids: np.ndarray


def cable_ball_joint_stiffness(
    *,
    radius: float | jax.Array,
    segment_length: float | jax.Array,
    youngs_modulus: float | jax.Array,
    shear_modulus: float | jax.Array,
) -> float | jax.Array:
    """Return the isotropic ball-joint stiffness used by the cable model."""
    polar_moment = jp.pi * radius**4 / 2.0
    area_moment = jp.pi * radius**4 / 4.0
    length = jp.maximum(segment_length, 1e-9)
    twist = polar_moment * shear_modulus / length
    bend = area_moment * youngs_modulus / length
    return (2.0 * bend + twist) / 3.0


def _validate_range(name: str, lower: float, upper: float, *, positive: bool) -> None:
    if positive and (lower <= 0 or upper <= 0):
        raise ValueError(f"{name} randomization bounds must be positive.")
    if not positive and (lower < 0 or upper < 0):
        raise ValueError(f"{name} randomization bounds must be non-negative.")
    if lower > upper:
        raise ValueError(f"{name}_min must be <= {name}_max.")


def _named_ids(
    model: mjx.Model,
    obj_type: ObjType,
    names: tuple[str, ...],
) -> np.ndarray:
    ids = np.asarray(
        [mjx.name2id(model, obj_type.value, name) for name in names],
        dtype=np.int32,
    )
    if (ids < 0).any():
        missing = [name for name, obj_id in zip(names, ids, strict=True) if obj_id < 0]
        raise ValueError(f"Expected model objects were not found: {missing}.")
    return ids


def resolve_randomization_ids(
    model: mjx.Model,
    *,
    cable_body_names: tuple[str, ...] = _CABLE_BODY_NAMES,
    cable_geom_names: tuple[str, ...] = _CABLE_GEOM_NAMES,
    cable_joint_names: tuple[str, ...] = _CABLE_JOINT_NAMES,
    pipe_geom_names: tuple[str, ...] = _PIPE_GEOM_NAMES,
    pipe_geom_ids: np.ndarray | None = None,
) -> CablePipeRandomizationIds:
    """Resolve model-specific IDs once when configuring a training task."""
    if pipe_geom_ids is None:
        pipe_geom_ids = _named_ids(model, ObjType.GEOM, pipe_geom_names)
    pipe_geom_ids = np.asarray(pipe_geom_ids, dtype=np.int32)
    if pipe_geom_ids.size == 0:
        raise ValueError("At least one pipe geom ID is required.")
    if pipe_geom_ids.min() < 0 or pipe_geom_ids.max() >= model.ngeom:
        raise ValueError("Pipe geom IDs are outside the model geom range.")
    return CablePipeRandomizationIds(
        cable_body_ids=_named_ids(model, ObjType.BODY, cable_body_names),
        cable_geom_ids=_named_ids(model, ObjType.GEOM, cable_geom_names),
        cable_joint_ids=_named_ids(model, ObjType.JOINT, cable_joint_names),
        pipe_geom_ids=pipe_geom_ids,
    )


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    *,
    model_ids: CablePipeRandomizationIds | None = None,
    pipe_inner_radius_min: float = constants.PIPE_DR_INNER_RADIUS[0],
    pipe_inner_radius_max: float = constants.PIPE_DR_INNER_RADIUS[1],
    length_min: float = constants.CABLE_DR_LENGTH[0],
    length_max: float = constants.CABLE_DR_LENGTH[1],
    youngs_modulus_min: float = constants.CABLE_DR_YOUNGS_MODULUS[0],
    youngs_modulus_max: float = constants.CABLE_DR_YOUNGS_MODULUS[1],
    shear_modulus_min: float = constants.CABLE_DR_SHEAR_MODULUS[0],
    shear_modulus_max: float = constants.CABLE_DR_SHEAR_MODULUS[1],
    thickness_min: float = constants.CABLE_DR_THICKNESS[0],
    thickness_max: float = constants.CABLE_DR_THICKNESS[1],
) -> tuple[mjx.Model, Any]:
    """Randomize pipe clearance and cable geometry/material parameters."""
    _validate_range(
        "pipe_inner_radius",
        pipe_inner_radius_min,
        pipe_inner_radius_max,
        positive=True,
    )
    if pipe_inner_radius_max >= constants.PIPE_OUTER_RADIUS:
        raise ValueError("Pipe inner radius must be smaller than pipe outer radius.")
    _validate_range("length", length_min, length_max, positive=True)
    _validate_range(
        "youngs_modulus",
        youngs_modulus_min,
        youngs_modulus_max,
        positive=False,
    )
    _validate_range(
        "shear_modulus",
        shear_modulus_min,
        shear_modulus_max,
        positive=False,
    )
    _validate_range("thickness", thickness_min, thickness_max, positive=True)

    model_ids = model_ids or resolve_randomization_ids(model)
    cable_body_ids = model_ids.cable_body_ids
    cable_geom_ids = model_ids.cable_geom_ids
    cable_joint_ids = model_ids.cable_joint_ids
    pipe_geom_ids = model_ids.pipe_geom_ids
    pipe_resolution = len(pipe_geom_ids)
    pipe_angles = jp.linspace(
        0.0,
        2.0 * jp.pi,
        pipe_resolution,
        endpoint=False,
        dtype=jp.float32,
    )
    pipe_radial_dirs = jp.stack([jp.cos(pipe_angles), jp.sin(pipe_angles)], axis=1)
    num_segments = len(cable_geom_ids)

    def randomize_one(key: jax.Array):
        (
            _,
            pipe_inner_key,
            length_key,
            youngs_key,
            shear_key,
            thickness_key,
        ) = jax.random.split(key, 6)
        pipe_inner_radius = jax.random.uniform(
            pipe_inner_key,
            shape=(),
            minval=pipe_inner_radius_min,
            maxval=pipe_inner_radius_max,
        )
        pipe_center_radius = 0.5 * (constants.PIPE_OUTER_RADIUS + pipe_inner_radius)
        pipe_half_width = 0.5 * (constants.PIPE_OUTER_RADIUS - pipe_inner_radius)

        cable_length = jax.random.uniform(
            length_key, shape=(), minval=length_min, maxval=length_max
        )
        segment_length = cable_length / num_segments
        half_segment_length = 0.5 * segment_length
        youngs_modulus = jax.random.uniform(
            youngs_key,
            shape=(),
            minval=youngs_modulus_min,
            maxval=youngs_modulus_max,
        )
        shear_modulus = jax.random.uniform(
            shear_key,
            shape=(),
            minval=shear_modulus_min,
            maxval=shear_modulus_max,
        )
        thickness = jax.random.uniform(
            thickness_key,
            shape=(),
            minval=thickness_min,
            maxval=thickness_max,
        )
        stiffness = cable_ball_joint_stiffness(
            radius=thickness,
            segment_length=segment_length,
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
        )

        geom_pos = model.geom_pos.at[pipe_geom_ids, :2].set(
            pipe_radial_dirs * pipe_center_radius
        )
        body_pos = model.body_pos.at[cable_body_ids, 1].set(segment_length)
        # Pipe boxes use x=tangential, y=radial wall thickness, z=half length.
        geom_size = model.geom_size.at[pipe_geom_ids, 1].set(pipe_half_width)
        geom_size = geom_size.at[cable_geom_ids, 0].set(thickness)
        geom_size = geom_size.at[cable_geom_ids, 1].set(half_segment_length)
        jnt_pos = model.jnt_pos.at[cable_joint_ids, 1].set(-half_segment_length)
        jnt_stiffness = model.jnt_stiffness.at[cable_joint_ids].set(stiffness)
        return geom_pos, body_pos, geom_size, jnt_pos, jnt_stiffness

    batched = rng.ndim > 1
    outputs = jax.vmap(randomize_one)(rng) if batched else randomize_one(rng)
    geom_pos, body_pos, geom_size, jnt_pos, jnt_stiffness = outputs

    randomized_fields = {
        "geom_pos": geom_pos,
        "body_pos": body_pos,
        "geom_size": geom_size,
        "jnt_pos": jnt_pos,
        "jnt_stiffness": jnt_stiffness,
    }
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    if batched:
        in_axes = in_axes.tree_replace({name: 0 for name in randomized_fields})
    return model.tree_replace(randomized_fields), in_axes


def make_domain_randomizer(
    model: mjx.Model,
    **randomization_kwargs: Any,
) -> Callable[[mjx.Model, jax.Array], tuple[mjx.Model, Any]]:
    """Create a registry-ready randomizer with model IDs resolved once."""
    model_ids = resolve_randomization_ids(model)
    return partial(
        domain_randomize,
        model_ids=model_ids,
        **randomization_kwargs,
    )
