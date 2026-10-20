"""Abstract environment lifecycle for tasks manipulating a grasped DLO end."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jp
import jaxlie as jaxl
import mujoco as mj
import numpy as np
from mjxsim import ObjType, get_pose, set_pose, set_state
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from envs.grasped_end_dlo.config import base_config
from envs.grasped_end_dlo.constants import (
    OBSERVATION_LAYOUT,
    RESET_POS_MAX,
    RESET_POS_MIN,
    RESET_ROT_MAX,
    RESET_ROT_MIN,
)
from envs.grasped_end_dlo.dlo import (
    apply_random_pre_curvature,
    cable_ball_joint_qpos_adrs,
)
from envs.grasped_end_dlo.observations import (
    mj_site_observation,
    mjx_site_observation,
    observation_space,
    pose_errors,
    relative_pose_observation,
)
from envs.grasped_end_dlo.rewards import (
    PoseRewardConfig,
    empty_metrics,
    pose_progress_reward,
    sparse_success_reward,
)
from envs.grasped_end_dlo.viewer import PassiveViewer

__all__ = ["GraspedEndDloBase"]


def _as_3_vector(value: Any, *, name: str) -> jax.Array:
    """Convert a scalar or three-element value to a float32 JAX vector."""
    array = jp.asarray(value, dtype=jp.float32)
    if array.ndim == 0:
        array = jp.repeat(array[None], 3)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a scalar or 3-vector, got {array.shape}.")
    return array


class GraspedEndDloBase(mjx_env.MjxEnv, ABC):
    """Shared MJX lifecycle for tasks with a mocap-grasped DLO end."""

    target_site_name = "target"

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, str | int | list[Any]] | None = None,
    ):
        config = base_config() if config is None else config
        super().__init__(config, config_overrides=config_overrides)
        self._validate_config()

        self.impl = self._config.impl
        self._mj_model = self._init()
        self._mjx_model = mjx.put_model(self._mj_model, impl=self.impl)
        self._viewer: PassiveViewer | None = None

        self._cache_config()
        self._post_init()

    @abstractmethod
    def _init(self) -> mj.MjModel:
        """Build and return the concrete task's MuJoCo model."""

    def _validate_config(self) -> None:
        if self._config.impl not in {"jax", "warp", "c"}:
            raise ValueError(
                f"Invalid MJX impl {self._config.impl!r}; expected jax, warp, or c."
            )

        bounds = (
            ("reset_pos", RESET_POS_MIN, RESET_POS_MAX),
            ("reset_rot", RESET_ROT_MIN, RESET_ROT_MAX),
        )
        for prefix, default_min, default_max in bounds:
            lower = _as_3_vector(
                self._config.get(f"{prefix}_min", default_min),
                name=f"{prefix}_min",
            )
            upper = _as_3_vector(
                self._config.get(f"{prefix}_max", default_max),
                name=f"{prefix}_max",
            )
            if bool((upper < lower).any()):
                raise ValueError(f"{prefix}_min must be <= {prefix}_max on every axis.")

    def _cache_config(self) -> None:
        config = self._config
        self._reset_pos_min = _as_3_vector(
            config.get("reset_pos_min", RESET_POS_MIN),
            name="reset_pos_min",
        )
        self._reset_pos_max = _as_3_vector(
            config.get("reset_pos_max", RESET_POS_MAX),
            name="reset_pos_max",
        )
        self._reset_rot_min = _as_3_vector(
            config.get("reset_rot_min", RESET_ROT_MIN),
            name="reset_rot_min",
        )
        self._reset_rot_max = _as_3_vector(
            config.get("reset_rot_max", RESET_ROT_MAX),
            name="reset_rot_max",
        )

        self._initial_curvature_randomization = bool(
            config.get("initial_curvature_randomization", False)
        )
        self._curvature_theta_min = float(
            config.get("curvature_theta_min", np.deg2rad(45.0))
        )
        self._curvature_theta_max = float(
            config.get("curvature_theta_max", np.deg2rad(60.0))
        )
        self._curvature_phi_min = float(config.get("curvature_phi_min", 0.0))
        self._curvature_phi_max = float(config.get("curvature_phi_max", 2.0 * np.pi))
        self._curvature_update_ref = bool(config.get("curvature_update_ref", True))

        self._termination_threshold_pos = float(
            config.get("success_threshold_pos", 0.02)
        )
        self._termination_threshold_rot = float(
            config.get("success_threshold_rot", np.deg2rad(10.0))
        )
        self._termination_threshold = (
            self._termination_threshold_pos,
            self._termination_threshold_rot,
        )

        reward_config = config.get("reward_config", {})
        self._w_pos = float(reward_config.get("position_weight", 1.0))
        self._w_rot = float(reward_config.get("rotation_weight", 1.0))
        self._w_homo_false = float(reward_config.get("homo_false", -0.2))
        self._w_homo_true = float(reward_config.get("homo_true", 0.05))
        self._success_bonus = float(reward_config.get("success_bonus", 25.0))
        self._terminal_penalty = float(reward_config.get("terminal_penalty", -25.0))
        self._reward_config = PoseRewardConfig(
            position_threshold=self._termination_threshold_pos,
            rotation_threshold=self._termination_threshold_rot,
            position_weight=self._w_pos,
            rotation_weight=self._w_rot,
            homo_false=self._w_homo_false,
            homo_true=self._w_homo_true,
            success_bonus=self._success_bonus,
            terminal_penalty=self._terminal_penalty,
        )

        self._homo = bool(config.get("homo", True))
        self._sparse_reward = bool(config["sparse_reward"])
        self._sparse_reward_scale = float(config.get("sparse_reward_scale", 1.0))
        self._episode_length = int(config["episode_length"])

    def _post_init(self) -> None:
        """Cache model-dependent IDs after the concrete model is created."""
        self._cache_model_ids()

    def _cache_model_ids(self) -> None:
        model = self.mj_model
        self.target_id = model.site("target").id
        self.task_target_id = model.site(self.target_site_name).id
        self.keypoint_id = model.site("keypoint").id

        target_align_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "target_align")
        self.target_align_id = target_align_id if target_align_id >= 0 else None

        mocap_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "mocap")
        self._mocap_id: int | None = None
        if mocap_body_id >= 0:
            mocap_id = int(model.body_mocapid[mocap_body_id])
            self._mocap_id = mocap_id if mocap_id >= 0 else None

        self._key_id: int | None = None
        key_name = self._config.get("reset_keyframe")
        if key_name is not None:
            key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, key_name)
            if key_id < 0:
                available = [
                    mj.mj_id2name(model, mj.mjtObj.mjOBJ_KEY, index)
                    for index in range(model.nkey)
                ]
                raise ValueError(
                    f"Configured reset_keyframe {key_name!r} was not found; "
                    f"available keyframes: {available}."
                )
            self._key_id = key_id

        self._cable_free_qpos_adr: int | None = None
        cable_free_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "cable:free")
        if cable_free_id >= 0:
            self._cable_free_qpos_adr = int(model.jnt_qposadr[cable_free_id])

        self._cable_joint_qpos_adrs = cable_ball_joint_qpos_adrs(model)

    # -------------------------------------------------------------------------
    # Reset and step.
    # -------------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        curvature_key = None
        if self._initial_curvature_randomization:
            rng, position_key, rotation_key, curvature_key = jax.random.split(rng, 4)
        else:
            rng, position_key, rotation_key = jax.random.split(rng, 3)
        model = self.mjx_model
        data = self._make_data(model)

        position_delta = jax.random.uniform(
            position_key,
            (3,),
            minval=self._reset_pos_min,
            maxval=self._reset_pos_max,
        )
        rotation_delta = jax.random.uniform(
            rotation_key,
            (3,),
            minval=self._reset_rot_min,
            maxval=self._reset_rot_max,
        )
        reset_rotation = jaxl.SO3.exp(rotation_delta)

        qpos = self._transform_cable_root(data.qpos, position_delta, reset_rotation)
        curvature_info: dict[str, jax.Array] = {}
        if self._initial_curvature_randomization:
            assert curvature_key is not None
            model, qpos, theta, phi = apply_random_pre_curvature(
                model,
                qpos,
                curvature_key,
                cable_joint_qpos_adrs=self._cable_joint_qpos_adrs,
                theta_min=self._curvature_theta_min,
                theta_max=self._curvature_theta_max,
                phi_min=self._curvature_phi_min,
                phi_max=self._curvature_phi_max,
                update_ref=self._curvature_update_ref,
            )
            curvature_info = {"curvature_theta": theta, "curvature_phi": phi}

        data = self._set_reset_pose(model, data, qpos, position_delta, reset_rotation)
        obs, pose_metrics = self._observation_and_pose_metrics(model, data)
        reward_metrics = self._empty_reward_metrics()
        metrics = {**reward_metrics, **pose_metrics}
        info = {
            "rng": rng,
            "step": jp.array(0),
            "mjx_model": model,
            **reward_metrics,
            **pose_metrics,
            **curvature_info,
        }
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _make_data(self, model: mjx.Model) -> mjx.Data:
        data = mjx.make_data(
            self.mj_model,
            impl=self.impl,
            naconmax=int(self._config.get("naconmax", 4500)),
            njmax=int(self._config.get("njmax", 512)),
        )
        if self._key_id is not None:
            data = set_state(
                model,
                data,
                self._key_id,
                ObjType.KEYFRAME,
                forward=False,
            )
        return data

    def _transform_cable_root(
        self,
        qpos: jax.Array,
        position_delta: jax.Array,
        rotation: jaxl.SO3,
    ) -> jax.Array:
        if self._cable_free_qpos_adr is None:
            return qpos
        address = self._cable_free_qpos_adr
        quaternion = jaxl.SO3(wxyz=qpos[address + 3 : address + 7])
        quaternion = rotation.multiply(quaternion).normalize().wxyz
        qpos = qpos.at[address : address + 3].add(position_delta)
        return qpos.at[address + 3 : address + 7].set(quaternion)

    def _set_reset_pose(
        self,
        model: mjx.Model,
        data: mjx.Data,
        qpos: jax.Array,
        position_delta: jax.Array,
        rotation: jaxl.SO3,
    ) -> mjx.Data:
        mocap_pos = data.mocap_pos + position_delta if model.nmocap else None
        mocap_quat = None
        if model.nmocap:
            mocap_quat = jax.vmap(
                lambda quaternion: (
                    rotation.multiply(jaxl.SO3(wxyz=quaternion)).normalize().wxyz
                )
            )(data.mocap_quat)

        data = set_state(
            model,
            data,
            qpos=qpos,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            forward=True,
        )
        return set_state(
            model,
            data,
            qacc=jp.zeros(self.mj_model.nv),
            qfrc_applied=jp.zeros(self.mj_model.nv),
            xfrc_applied=jp.zeros((self.mj_model.nbody, 6)),
            forward=False,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        model = state.info.get("mjx_model", self.mjx_model)
        previous_data = state.data
        data = self._apply_action(model, previous_data, action)

        obs, pose_metrics = self._observation_and_pose_metrics(model, data)
        done = self._get_done_for_model(model, data, state.info)
        reward, reward_metrics = self._compute_reward(model, previous_data, data, done)

        rng, _ = jax.random.split(state.info["rng"])
        metrics = {**state.metrics, **reward_metrics, **pose_metrics}
        info = {
            **state.info,
            "rng": rng,
            "step": state.info["step"] + 1,
            "mjx_model": model,
            **reward_metrics,
            **pose_metrics,
        }
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _apply_action(
        self, model: mjx.Model, data: mjx.Data, action: jax.Array
    ) -> mjx.Data:
        mocap_pose = get_pose(model, data, "mocap", ObjType.BODY)
        target_pose = jaxl.SE3.from_rotation_and_translation(
            rotation=(
                jaxl.SO3.exp(action[3:6]).multiply(mocap_pose.rotation()).normalize()
            ),
            translation=mocap_pose.translation() + action[:3],
        )
        data = set_pose(model, data, "mocap", ObjType.BODY, target_pose)
        return mjx_env.step(model, data, data.ctrl, self.n_substeps)

    # -------------------------------------------------------------------------
    # Observation, termination, and reward.
    # -------------------------------------------------------------------------

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        return self._get_obs_for_model(self.mjx_model, data)

    def _get_obs_for_model(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        return mjx_site_observation(
            model,
            data,
            current_site="keypoint",
            target_site=self.target_site_name,
        )

    def _observation_and_pose_metrics(
        self, model: mjx.Model, data: mjx.Data
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        obs, position_error, rotation_error = self._get_error_metrics_for_model(
            model, data
        )
        weighted_position, weighted_rotation = self._get_weighted_error_metrics(
            position_error, rotation_error
        )
        return obs, {
            "position_error": position_error,
            "rotation_error": rotation_error,
            "weighted_position_error": weighted_position,
            "weighted_rotation_error": weighted_rotation,
        }

    def _get_error_metrics_for_model(
        self, model: mjx.Model, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return observation, position error in metres, and angle error in radians."""
        obs = self._get_obs_for_model(model, data)
        position_error, rotation_error = pose_errors(obs)
        return obs, position_error, rotation_error

    def _get_error_metrics(
        self, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return self._get_error_metrics_for_model(self.mjx_model, data)

    def _get_weighted_error_metrics(
        self, position_error: jax.Array, rotation_error: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return self._w_pos * position_error, self._w_rot * rotation_error

    def _get_success_for_model(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        _, position_error, rotation_error = self._get_error_metrics_for_model(
            model, data
        )
        if isinstance(self._termination_threshold, (tuple, list)):
            position_threshold, rotation_threshold = self._termination_threshold
        else:
            position_threshold = self._termination_threshold
            rotation_threshold = self._termination_threshold
        return (position_error < position_threshold) & (
            rotation_error < rotation_threshold
        )

    def _get_done_for_model(
        self, model: mjx.Model, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        unstable = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        timeout = info["step"] + 1 >= self._episode_length
        success = self._get_success_for_model(model, data)
        done = unstable | timeout
        if not self._sparse_reward:
            done |= success
        return done.astype(float)

    def _compute_reward(
        self,
        model: mjx.Model,
        previous_data: mjx.Data,
        data: mjx.Data,
        done: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        if self._sparse_reward:
            success = self._get_success_for_model(model, data)
            return sparse_success_reward(
                success,
                scale=self._sparse_reward_scale,
            )

        metrics = self._get_reward_for_model(model, previous_data, data, done)
        return metrics["reward_total"], metrics

    def _get_reward_for_model(
        self,
        model: mjx.Model,
        previous_data: mjx.Data,
        data: mjx.Data,
        done: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        _, previous_position, previous_rotation = self._get_error_metrics_for_model(
            model, previous_data
        )
        _, position, rotation = self._get_error_metrics_for_model(model, data)

        success = self._get_success_for_model(model, data)
        done_value = jp.asarray(0.0) if done is None else done
        is_homo = jp.asarray(False)
        if self._homo:
            is_homo = self._is_homo_for_model(model, data)
        return pose_progress_reward(
            previous_position=previous_position,
            previous_rotation=previous_rotation,
            position=position,
            rotation=rotation,
            success=success,
            done=done_value,
            is_homo=is_homo,
            config=self._reward_config,
            use_homo_reward=self._homo,
        )

    def _empty_reward_metrics(self) -> dict[str, jax.Array]:
        return empty_metrics()

    def _is_homo_for_model(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        """Return an optional task condition used for reward shaping."""
        del model, data
        return jp.asarray(False)

    # Compatibility wrappers retained for existing task subclasses.
    def _is_homo(self, data: mjx.Data) -> jax.Array:
        return self._is_homo_for_model(self.mjx_model, data)

    def _get_reward(
        self,
        data0: mjx.Data,
        data1: mjx.Data,
        done: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        return self._get_reward_for_model(self.mjx_model, data0, data1, done)

    def _get_empty_reward_metrics(self) -> dict[str, jax.Array]:
        return self._empty_reward_metrics()

    def _get_reward_vel(self, data0: mjx.Data, data1: mjx.Data) -> jax.Array:
        return self._get_reward(data0, data1)["reward_total"]

    def _get_done(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        return self._get_done_for_model(self.mjx_model, data, info)

    # -------------------------------------------------------------------------
    # CPU observations and interactive rendering.
    # -------------------------------------------------------------------------

    def get_mj_obs(self, data: mj.MjData) -> np.ndarray:
        """Return the NumPy equivalent of the MJX observation vector."""
        return mj_site_observation(
            self.mj_model,
            data,
            current_site="keypoint",
            target_site=self.target_site_name,
        )

    def get_demo_observation_space(self) -> dict[str, Any]:
        return observation_space()

    def make_mj_data(self) -> mj.MjData:
        """Create CPU data initialized from the configured reset keyframe."""
        data = mj.MjData(self.mj_model)
        if self._key_id is None:
            mj.mj_resetData(self.mj_model, data)
        else:
            mj.mj_resetDataKeyframe(self.mj_model, data, self._key_id)
        mj.mj_forward(self.mj_model, data)
        return data

    @staticmethod
    def _data_from_state(state: mjx_env.State | mjx.Data) -> mjx.Data:
        return state.data if isinstance(state, mjx_env.State) else state

    def _model_from_state(self, state: mjx_env.State | mjx.Data) -> mjx.Model:
        if isinstance(state, mjx_env.State):
            return state.info.get("mjx_model", self.mjx_model)
        return self.mjx_model

    def _get_viewer(self) -> PassiveViewer:
        if self._viewer is None:
            self._viewer = PassiveViewer(self.mj_model)
        return self._viewer

    def update_mj_data(
        self,
        state: mjx_env.State | mjx.Data,
        data: mj.MjData | None = None,
    ) -> mj.MjData:
        """Copy an MJX state into viewer-compatible MuJoCo data."""
        return self._get_viewer().update_data(
            self._data_from_state(state),
            self._model_from_state(state),
            target=data,
        )

    def render(
        self,
        state: mjx_env.State | mjx.Data,
        *,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
    ) -> Any:
        """Show or update a state in a passive ``mujoco.viewer`` window."""
        return self._get_viewer().render(
            self._data_from_state(state),
            self._model_from_state(state),
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        )

    def launch_viewer(
        self,
        *,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
    ) -> Any:
        """Open the initial CPU model without compiling an MJX reset."""
        return self._get_viewer().launch(
            self.make_mj_data(),
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        )

    def close(self) -> None:
        """Close the managed passive viewer."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    @staticmethod
    def get_obs(
        R_current: jax.Array,
        t_current: jax.Array,
        R_target: jax.Array,
        t_target: jax.Array,
    ) -> jax.Array:
        """Compute the target position and z-axis in the current frame."""
        return relative_pose_observation(
            R_current,
            t_current,
            R_target,
            t_target,
        )

    # -------------------------------------------------------------------------
    # Standard MuJoCo Playground properties.
    # -------------------------------------------------------------------------

    @property
    def observation_size(self) -> int:
        return len(OBSERVATION_LAYOUT)

    @property
    def action_size(self) -> int:
        if self._mocap_id is not None and self.mj_model.nmocap:
            return 6
        return self.mj_model.nu

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
