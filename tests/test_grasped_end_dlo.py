"""Focused regression tests for the shared grasped-end DLO modules."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jp
import mujoco as mj
import numpy as np
from mujoco import mjx

from envs.grasped_end_dlo.base import GraspedEndDloBase
from envs.grasped_end_dlo.dlo import (
    apply_random_pre_curvature,
    cable_ball_joint_qpos_adrs,
)
from envs.grasped_end_dlo.insert.config import default_config
from envs.grasped_end_dlo.insert.keyframes import load_keyframe
from envs.grasped_end_dlo.insert.model import build_model
from envs.grasped_end_dlo.insert.randomize import (
    cable_ball_joint_stiffness,
    resolve_randomization_ids,
)
from envs.grasped_end_dlo.insert.task import GraspedEndDloInsert
from envs.grasped_end_dlo.observations import (
    pose_errors,
    relative_pose_observation,
)
from envs.grasped_end_dlo.rewards import (
    PoseRewardConfig,
    pose_progress_reward,
)
from envs.grasped_end_dlo.viewer import PassiveViewer


def _cable_model() -> mj.MjModel:
    return mj.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
              <freejoint name="cable:free"/>
              <body>
                <joint name="cable:Jfirst" type="ball" stiffness="1"/>
                <geom size=".01"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )


class GraspedEndDloTest(unittest.TestCase):
    def test_insertion_model_compiles_and_randomization_ids_match_pipe(self) -> None:
        model = build_model()
        mjx_model = mjx.put_model(model, impl="jax")

        ids = resolve_randomization_ids(mjx_model)
        pipe_names = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(geom_id))
            for geom_id in ids.pipe_geom_ids
        ]

        self.assertEqual(
            pipe_names,
            [f"pipe:{index}" for index in range(20)],
        )
        self.assertGreaterEqual(model.site("target").id, 0)
        self.assertGreaterEqual(model.site("keypoint").id, 0)
        self.assertGreaterEqual(model.body("mocap").id, 0)
        self.assertEqual(model.nkey, 1)
        self.assertEqual(model.key("init").id, 0)

    def test_cpu_and_mjx_data_use_configured_keyframe(self) -> None:
        config = default_config()
        config.impl = "jax"
        environment = GraspedEndDloInsert(config)
        keyframe = load_keyframe()

        cpu_data = environment.make_mj_data()
        mjx_data = environment._make_data(environment.mjx_model)

        self.assertTrue(np.allclose(cpu_data.qpos, keyframe.qpos))
        self.assertTrue(np.allclose(cpu_data.qvel, keyframe.qvel))
        self.assertTrue(np.allclose(cpu_data.mocap_pos.ravel(), keyframe.mpos))
        self.assertTrue(np.allclose(cpu_data.mocap_quat.ravel(), keyframe.mquat))
        self.assertTrue(np.allclose(mjx_data.qpos, keyframe.qpos))
        self.assertTrue(np.allclose(mjx_data.qvel, keyframe.qvel))
        self.assertTrue(np.allclose(mjx_data.mocap_pos.ravel(), keyframe.mpos))
        self.assertTrue(np.allclose(mjx_data.mocap_quat.ravel(), keyframe.mquat))

    def test_base_environment_remains_abstract(self) -> None:
        self.assertIn("_init", GraspedEndDloBase.__abstractmethods__)
        with self.assertRaises(TypeError):
            GraspedEndDloBase()

    def test_default_config_returns_independent_configs(self) -> None:
        first = default_config()
        second = default_config()
        first.impl = "jax"
        self.assertEqual(second.impl, "warp")

    def test_ball_joint_stiffness_matches_isotropic_formula(self) -> None:
        stiffness = cable_ball_joint_stiffness(
            radius=0.01,
            segment_length=0.04,
            youngs_modulus=2.0,
            shear_modulus=3.0,
        )
        area_moment = np.pi * 0.01**4 / 4.0
        polar_moment = np.pi * 0.01**4 / 2.0
        expected = (2 * area_moment * 2.0 + polar_moment * 3.0) / (3 * 0.04)
        self.assertTrue(np.isclose(stiffness, expected))

    def test_relative_pose_observation_and_errors(self) -> None:
        observation = relative_pose_observation(
            jp.eye(3),
            jp.zeros(3),
            jp.eye(3),
            jp.asarray([1.0, 2.0, 3.0]),
        )
        position_error, rotation_error = pose_errors(observation)
        self.assertTrue(np.allclose(observation, [1, 2, 3, 0, 0, 1]))
        self.assertTrue(np.isclose(position_error, np.sqrt(14)))
        self.assertTrue(np.isclose(rotation_error, 0.0))

    def test_pose_reward_can_disable_pipe_shaping(self) -> None:
        metrics = pose_progress_reward(
            previous_position=jp.asarray(0.2),
            previous_rotation=jp.asarray(0.2),
            position=jp.asarray(0.1),
            rotation=jp.asarray(0.1),
            success=jp.asarray(False),
            done=jp.asarray(False),
            is_homo=jp.asarray(False),
            config=PoseRewardConfig(
                position_threshold=0.1,
                rotation_threshold=0.1,
            ),
            use_homo_reward=False,
        )
        self.assertTrue(np.isclose(metrics["reward_homo"], 0.0))
        self.assertTrue(np.isclose(metrics["reward_total"], 2.0))

    def test_random_pre_curvature_updates_state_and_spring_reference(self) -> None:
        cable_model = _cable_model()
        model = mjx.put_model(cable_model, impl="jax")
        addresses = cable_ball_joint_qpos_adrs(cable_model)
        qpos = jp.asarray(cable_model.qpos0)

        model, curved_qpos, theta, phi = apply_random_pre_curvature(
            model,
            qpos,
            jax.random.PRNGKey(0),
            cable_joint_qpos_adrs=addresses,
            theta_min=0.4,
            theta_max=0.4,
            phi_min=0.2,
            phi_max=0.2,
        )

        quaternion_indices = addresses[:, None] + jp.arange(4)
        self.assertTrue(np.isclose(theta, 0.4))
        self.assertTrue(np.isclose(phi, 0.2))
        self.assertFalse(
            np.allclose(curved_qpos[quaternion_indices], qpos[quaternion_indices])
        )
        self.assertTrue(
            np.allclose(
                model.qpos_spring[quaternion_indices],
                curved_qpos[quaternion_indices],
            )
        )

    def test_viewer_adapter_copies_mjx_data_and_model_fields(self) -> None:
        cable_model = _cable_model()
        model = mjx.put_model(cable_model, impl="jax")
        source = mj.MjData(cable_model)
        source.qpos[0:3] = [0.1, -0.2, 0.3]
        data = mjx.put_data(cable_model, source, impl="jax")
        model = model.tree_replace({"geom_size": model.geom_size.at[0, 0].set(0.025)})

        adapter = PassiveViewer(cable_model)
        target = adapter.update_data(data, model)

        self.assertTrue(np.allclose(target.qpos[:3], [0.1, -0.2, 0.3]))
        self.assertTrue(np.isclose(cable_model.geom_size[0, 0], 0.025))


if __name__ == "__main__":
    unittest.main()
