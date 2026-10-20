"""Passive MuJoCo viewer adapter for MJX environment states."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import jax
import mujoco as mj
import numpy as np
from mujoco import mjx

_MODEL_FIELDS = (
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
)


class PassiveViewer:
    """Own a passive viewer and its reusable CPU-side ``MjData``."""

    def __init__(self, model: mj.MjModel):
        self._model = model
        self._data: mj.MjData | None = None
        self._viewer: Any = None

    def _sync_model(self, model: mjx.Model) -> None:
        for field in _MODEL_FIELDS:
            source = getattr(model, field, None)
            destination = getattr(self._model, field, None)
            if source is None or destination is None:
                continue
            source = np.asarray(jax.device_get(source))
            if source.shape != destination.shape:
                raise ValueError(
                    f"Cannot render batched MJX model field {field!r}: "
                    f"expected {destination.shape}, got {source.shape}."
                )
            np.copyto(destination, source)

    def update_data(
        self,
        data: mjx.Data,
        model: mjx.Model,
        target: mj.MjData | None = None,
    ) -> mj.MjData:
        """Copy MJX model/data into persistent viewer-compatible objects."""
        if target is None:
            if self._data is None:
                self._data = mj.MjData(self._model)
            target = self._data
        self._sync_model(model)
        mjx.get_data_into(target, self._model, data)
        return target

    def _launch(
        self,
        data: mj.MjData,
        *,
        show_left_ui: bool,
        show_right_ui: bool,
    ) -> Any:
        mj_viewer = import_module("mujoco.viewer")
        self._viewer = mj_viewer.launch_passive(
            model=self._model,
            data=data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        )
        self._viewer.cam.azimuth = self._model.vis.global_.azimuth
        self._viewer.cam.elevation = self._model.vis.global_.elevation
        self._viewer.cam.lookat = self._model.stat.center
        self._viewer.cam.distance = self._model.stat.extent
        self._viewer.sync()
        return self._viewer

    def launch(
        self,
        data: mj.MjData | None = None,
        *,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
    ) -> Any:
        """Open the viewer on a supplied or freshly initialized CPU state."""
        self.close()
        if data is None:
            data = mj.MjData(self._model)
            mj.mj_forward(self._model, data)
        self._data = data
        return self._launch(
            data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        )

    def render(
        self,
        data: mjx.Data,
        model: mjx.Model,
        *,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
    ) -> Any:
        """Create or update the passive viewer and return its handle."""
        if self._viewer is None or not self._viewer.is_running():
            self.close()
            render_data = self.update_data(data, model)
            return self._launch(
                render_data,
                show_left_ui=show_left_ui,
                show_right_ui=show_right_ui,
            )

        with self._viewer.lock():
            self.update_data(data, model)
        self._viewer.sync()
        return self._viewer

    def close(self) -> None:
        """Close the viewer and release its CPU-side state."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._data = None
