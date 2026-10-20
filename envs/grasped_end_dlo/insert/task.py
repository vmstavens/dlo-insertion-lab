"""Concrete grasped-end cable insertion environment."""

from __future__ import annotations

import jax
import mujoco as mj
from mujoco import mjx

from envs.grasped_end_dlo import constants
from envs.grasped_end_dlo.base import GraspedEndDloBase
from envs.grasped_end_dlo.insert.model import build_model
from envs.grasped_end_dlo.insert.pipe import site_within_pipe


class GraspedEndDloInsert(GraspedEndDloBase):
    """Insert the grasped end of a deformable cable into a pipe."""

    target_site_name = "target"

    def _init(self) -> mj.MjModel:
        self.pipe_inner_radius = constants.PIPE_INNER_RADIUS
        self.pipe_outer_radius = constants.PIPE_OUTER_RADIUS
        self.pipe_length = constants.PIPE_LENGTH
        self._xml_path = "generated://grasped_end_dlo/insert"
        return build_model(
            pipe_inner_radius=self.pipe_inner_radius,
            pipe_outer_radius=self.pipe_outer_radius,
            pipe_half_length=self.pipe_length,
        )

    def _is_homo_for_model(
        self,
        model: mjx.Model,
        data: mjx.Data,
    ) -> jax.Array:
        return site_within_pipe(
            model,
            data,
            site_name="keypoint",
            pipe_origin_site="target",
            inner_radius=self.pipe_inner_radius,
        )
