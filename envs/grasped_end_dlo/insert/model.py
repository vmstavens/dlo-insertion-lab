"""Procedural MuJoCo model for the grasped-end cable insertion task."""

from __future__ import annotations

import mjsim as ms
import mjxsim as mx
import mujoco as mj
import numpy as np

from envs.grasped_end_dlo import constants
from envs.grasped_end_dlo.insert.keyframes import load_keyframe


def build_model(
    *,
    pipe_inner_radius: float = constants.PIPE_INNER_RADIUS,
    pipe_outer_radius: float = constants.PIPE_OUTER_RADIUS,
    pipe_half_length: float = constants.PIPE_LENGTH,
    pipe_resolution: int = constants.PIPE_RESOLUTION,
    cable_length: float = constants.CABLE_LENGTH,
    cable_thickness: float = constants.CABLE_THICKNESS,
    cable_shear_modulus: float = constants.CABLE_SHEAR_MODULUS,
    cable_youngs_modulus: float = constants.CABLE_YOUNGS_MODULUS,
    keyframe_filename: str | None = "init.json",
) -> mj.MjModel:
    """Build and compile the default insertion model."""
    if not 0 < pipe_inner_radius < pipe_outer_radius:
        raise ValueError("Expected 0 < pipe_inner_radius < pipe_outer_radius.")
    if pipe_half_length <= 0:
        raise ValueError("pipe_half_length must be positive.")
    if pipe_resolution < 3:
        raise ValueError("pipe_resolution must be at least 3.")

    scene = ms.empty_scene(
        sim_name="grasped end DLO insertion",
        memory=None,
        add_floor=False,
        sdf_iterations=10,
        sdf_initpoints=40,
        noslip_iterations=0,
        enable_multiccd=False,
        nativeccd=True,
        option_overrides={
            "disableflags": int(mj.mjtDisableBit.mjDSBL_EULERDAMP),
            "enableflags": 0,
        },
    )

    pipe_body = scene.worldbody.add_body(
        name="pipe",
        pos=[0, 0, 0.1],
        euler=[np.pi / 2, 0, np.pi],
    )
    radial_half_width = 0.5 * (pipe_outer_radius - pipe_inner_radius)
    radial_center = 0.5 * (pipe_outer_radius + pipe_inner_radius)
    tangential_half_width = pipe_outer_radius * np.tan(np.pi / pipe_resolution)
    for segment in range(pipe_resolution):
        angle = 2.0 * np.pi * segment / pipe_resolution
        pipe_body.add_geom(
            name=f"pipe:{segment}",
            type=mj.mjtGeom.mjGEOM_BOX,
            pos=[
                radial_center * np.cos(angle),
                radial_center * np.sin(angle),
                0,
            ],
            euler=[0, 0, angle - np.pi / 2],
            size=[
                tangential_half_width,
                radial_half_width,
                pipe_half_length,
            ],
            rgba=[0.2, 0.2, 0.2, 0.2],
            solref=[0.00001, 2],
        )

    pipe_body.add_site(
        name="target",
        pos=[0, 0, pipe_half_length],
        group=1,
        rgba=[1, 0, 0, 1],
    )
    pipe_body.add_site(
        name="target_align",
        pos=[0, 0, -pipe_half_length],
        group=1,
        rgba=[1, 0, 0, 1],
    )

    cable = mx.cable(
        twist=cable_shear_modulus,
        bend=cable_youngs_modulus,
        size=cable_length,
        segment_size=cable_thickness,
        initial="free",
    )
    cable.body("cable:Bfirst").add_site(
        name="cable_weld",
        pos=[0.0, 0.0, 0.0],
        quat=[np.cos(np.pi / 4), 0.0, 0.0, -np.sin(np.pi / 4)],
        group=1,
        rgba=[0, 1, 1, 1],
    )
    cable.body("cable:Blast").add_site(
        name="keypoint",
        group=1,
        rgba=[1, 0, 0, 1],
        pos=[0, 0, 0],
        quat=[0.7071068, -0.7071068, 0, 0],
    )

    scene.worldbody.add_camera(
        name="cam",
        pos=[1.2, 0.234, 0.156],
        xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
        resolution=[640, 480],
    )

    mocap = scene.worldbody.add_body(
        name="mocap",
        mocap=True,
        pos=[0.0, 0.4, 0.4],
        euler=[0, 0, 0],
    )
    mocap.add_geom(
        name="mocap",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        contype=0,
        conaffinity=0,
    )
    mocap.add_site(
        name="mocap_site",
        pos=[0, 0, 0],
        euler=[0, 0, np.pi / 2],
    )

    scene.worldbody.add_frame(
        pos=[0, 0.2, 0.1],
        euler=[0, 0, 0],
    ).attach_body(cable.worldbody.first_body())

    scene.add_equality(
        name="mocap_cable_weld",
        type=mj.mjtEq.mjEQ_WELD,
        objtype=mj.mjtObj.mjOBJ_SITE,
        name1="mocap_site",
        name2="cable_weld",
        solref=[-50_000, -362.71080706],
        solimp=[0.999, 0.9999, 0.0001, 0.5, 1],
    )
    if keyframe_filename is not None:
        keyframe = load_keyframe(keyframe_filename)
        scene.add_key(**keyframe.as_mjspec_kwargs())
    return scene.compile()
