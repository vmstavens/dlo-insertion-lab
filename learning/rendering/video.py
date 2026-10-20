"""MuJoCo observers for unbatched MJX states and streamed MP4 output."""

from __future__ import annotations

import copy
import math
import shutil
import subprocess
import time
from pathlib import Path

import jax
import mujoco as mj
import numpy as np
from mujoco import mjx


class Scene:
    """Render a private model so observation cannot mutate simulation state."""

    def __init__(self, environment):
        self.environment = environment
        self.model = copy.copy(environment.mj_model)
        self.data = mj.MjData(self.model)

    def update(self, state):
        model = state.info.get("mjx_model", self.environment.mjx_model)
        for name in (
            "body_pos",
            "body_quat",
            "geom_pos",
            "geom_quat",
            "geom_size",
            "geom_rgba",
            "site_pos",
            "site_quat",
            "site_size",
            "site_rgba",
        ):
            value = getattr(model, name, None)
            if value is not None:
                np.copyto(getattr(self.model, name), np.asarray(jax.device_get(value)))
        mjx.get_data_into(self.data, self.model, state.data)
        mj.mj_forward(self.model, self.data)


class VideoObserver:
    def __init__(
        self, environment, output: Path, *, fps=30, width=640, height=480, camera=None
    ):
        if fps <= 0 or width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(
                "FPS must be positive; video dimensions must be positive and even."
            )
        if not math.isfinite(float(environment.dt)) or environment.dt <= 0:
            raise ValueError("Environment dt must be finite and positive.")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Video recording requires FFmpeg on PATH.")
        self.scene = Scene(environment)
        self.output = output
        self.fps, self.width, self.height = fps, width, height
        self.camera = camera if camera is not None else -1
        self.dt = float(environment.dt)
        self.frame_count = 0
        self.renderer = None
        self.process = None

    def __enter__(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.scene.model.vis.global_.offwidth = max(
            self.width, self.scene.model.vis.global_.offwidth
        )
        self.scene.model.vis.global_.offheight = max(
            self.height, self.scene.model.vis.global_.offheight
        )
        self.renderer = mj.Renderer(
            self.scene.model, height=self.height, width=self.width
        )
        try:
            self.process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{self.width}x{self.height}",
                    "-r",
                    str(self.fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self.output),
                ],
                stdin=subprocess.PIPE,
            )
        except BaseException:
            self.renderer.close()
            raise
        return self

    def __call__(self, state, step):
        # Sample on video timestamps, avoiding accumulated stride-rounding drift.
        required = math.floor(step * self.dt * self.fps + 1e-8) + 1
        if required <= self.frame_count:
            return
        self.scene.update(state)
        self.renderer.update_scene(self.scene.data, camera=self.camera)
        frame = self.renderer.render().tobytes()
        while self.frame_count < required:
            self.process.stdin.write(frame)
            self.frame_count += 1

    def __exit__(self, exc_type, exc, traceback):
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                if exc_type is None:
                    raise RuntimeError("FFmpeg did not finish encoding.") from None
                return
            if code and exc_type is None:
                raise RuntimeError(f"FFmpeg exited with status {code}.")
        finally:
            self.renderer.close()


class LiveObserver:
    def __init__(self, environment):
        self.scene = Scene(environment)
        self.dt = float(environment.dt)
        self.viewer = None

    def __enter__(self):
        self.started = None
        return self

    def __call__(self, state, step):
        import mujoco.viewer

        if self.viewer is None:
            self.scene.update(state)
            self.viewer = mujoco.viewer.launch_passive(
                self.scene.model, self.scene.data
            )
            self.started = time.monotonic()
        elif not self.viewer.is_running():
            raise KeyboardInterrupt
        else:
            with self.viewer.lock():
                self.scene.update(state)
        self.viewer.sync()
        delay = self.started + step * self.dt - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def __exit__(self, exc_type, exc, traceback):
        if self.viewer is not None:
            self.viewer.close()
