"""Inspect or simulate a registered environment in MuJoCo's passive viewer."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence

# Viewing a single static environment does not require a GPU backend and should
# not depend on the workstation's CUDA/cuDNN installation.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco as mj  # noqa: E402

import envs.grasped_end_dlo  # noqa: E402, F401  # registers built-in DLO tasks
from envs import registry  # noqa: E402
from envs.grasped_end_dlo.viewer import PassiveViewer  # noqa: E402

DEFAULT_ENVIRONMENT = "GraspedEndDloInsert"
VIEW_MODES = ("static", "cpu", "mjx")


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or simulate a registered MuJoCo environment."
    )
    parser.add_argument(
        "environment",
        nargs="?",
        default=DEFAULT_ENVIRONMENT,
        help=f"registered environment name (default: {DEFAULT_ENVIRONMENT})",
    )
    parser.add_argument(
        "--mode",
        choices=VIEW_MODES,
        default="static",
        help=(
            "static model, CPU MuJoCo physics, or MJX environment rollout "
            "(default: static)"
        ),
    )
    parser.add_argument(
        "--render-fps",
        type=_positive_float,
        default=60.0,
        help="viewer refresh target; physics may take several steps per frame",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="reset seed used by --mode mjx",
    )
    return parser.parse_args(argv)


def _pace(deadline: float, frame_period: float) -> float:
    deadline += frame_period
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
        return deadline
    return time.monotonic()


def _require_viewer_environment(environment: object, name: str) -> None:
    launch_viewer = getattr(environment, "launch_viewer", None)
    if launch_viewer is None:
        raise TypeError(f"Environment {name!r} does not provide launch_viewer().")


def _view_static(environment: object, render_fps: float) -> None:
    viewer = environment.launch_viewer()
    try:
        deadline = time.monotonic()
        while viewer.is_running():
            viewer.sync()
            deadline = _pace(deadline, 1.0 / render_fps)
    finally:
        environment.close()


def _view_cpu(environment: object, render_fps: float) -> None:
    """Advance the CPU model with mj_step and render near real time."""
    model = environment.mj_model
    data = environment.make_mj_data()

    physics_dt = float(model.opt.timestep)
    steps_per_frame = max(1, round(1.0 / (render_fps * physics_dt)))
    frame_period = steps_per_frame * physics_dt
    adapter = PassiveViewer(model)
    viewer = adapter.launch(data)
    try:
        deadline = time.monotonic()
        while viewer.is_running():
            for _ in range(steps_per_frame):
                mj.mj_step(model, data)
            viewer.sync()
            deadline = _pace(deadline, frame_period)
    finally:
        adapter.close()
        environment.close()


def _view_mjx(environment: object, render_fps: float, seed: int) -> None:
    """Advance the RL environment with zero actions and render MJX state."""
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)
    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)

    print(
        "Compiling MJX reset and step; the first frame may take a while...",
        flush=True,
    )
    state = reset(reset_key)
    jax.block_until_ready(state.data.qpos)
    viewer = environment.render(state)
    print("MJX rollout ready.", flush=True)

    control_dt = float(environment.dt)
    steps_per_frame = max(1, round(1.0 / (render_fps * control_dt)))
    frame_period = steps_per_frame * control_dt
    action = jp.zeros(environment.action_size)

    try:
        deadline = time.monotonic()
        while viewer.is_running():
            for _ in range(steps_per_frame):
                state = step(state, action)
            jax.block_until_ready(state.data.qpos)

            if bool(jax.device_get(state.done)):
                rng, reset_key = jax.random.split(rng)
                state = reset(reset_key)
                jax.block_until_ready(state.data.qpos)

            viewer = environment.render(state)
            deadline = _pace(deadline, frame_period)
    finally:
        environment.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = registry.get_default_config(args.environment)
    if "impl" in config:
        config.impl = "jax"
    environment = registry.load(args.environment, config=config)
    _require_viewer_environment(environment, args.environment)

    try:
        if args.mode == "static":
            _view_static(environment, args.render_fps)
        elif args.mode == "cpu":
            _view_cpu(environment, args.render_fps)
        else:
            _view_mjx(environment, args.render_fps, args.seed)
    except KeyboardInterrupt:
        environment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
