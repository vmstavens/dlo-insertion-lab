"""Loading and validation for insertion-task MuJoCo keyframe assets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources
from typing import Any

_KEYFRAME_PACKAGE = "envs.grasped_end_dlo.insert"
_KEYFRAME_DIRECTORY = ("assets", "keyframes")
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class KeyframeAsset:
    """Serializable fields accepted by :meth:`mujoco.MjSpec.add_key`."""

    name: str
    time: float
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    mpos: tuple[float, ...]
    mquat: tuple[float, ...]

    def as_mjspec_kwargs(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "time": self.time,
            "qpos": self.qpos,
            "qvel": self.qvel,
            "mpos": self.mpos,
            "mquat": self.mquat,
        }


def _finite_vector(payload: dict[str, Any], field: str) -> tuple[float, ...]:
    raw = payload.get(field)
    if not isinstance(raw, list):
        raise ValueError(f"Keyframe field {field!r} must be a JSON array.")
    values = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Keyframe field {field!r} contains a non-finite value.")
    return values


def load_keyframe(filename: str = "init.json") -> KeyframeAsset:
    """Load a validated keyframe from this task's packaged assets."""
    resource = resources.files(_KEYFRAME_PACKAGE)
    for directory in _KEYFRAME_DIRECTORY:
        resource = resource.joinpath(directory)
    resource = resource.joinpath(filename)

    with resource.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Keyframe JSON root must be an object.")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported keyframe schema_version "
            f"{payload.get('schema_version')!r}; expected {_SCHEMA_VERSION}."
        )

    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Keyframe field 'name' must be a non-empty string.")
    time = float(payload.get("time", 0.0))
    if not math.isfinite(time):
        raise ValueError("Keyframe field 'time' must be finite.")

    qpos = _finite_vector(payload, "qpos")
    qvel = _finite_vector(payload, "qvel")
    mpos = _finite_vector(payload, "mpos")
    mquat = _finite_vector(payload, "mquat")
    if len(mpos) % 3:
        raise ValueError("Keyframe 'mpos' length must be divisible by 3.")
    if len(mquat) % 4:
        raise ValueError("Keyframe 'mquat' length must be divisible by 4.")
    if len(mpos) // 3 != len(mquat) // 4:
        raise ValueError("Keyframe mocap position/quaternion counts do not match.")
    for offset in range(0, len(mquat), 4):
        norm = math.sqrt(sum(value * value for value in mquat[offset : offset + 4]))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"Keyframe mocap quaternion at index {offset // 4} "
                f"is not normalized (norm={norm})."
            )

    return KeyframeAsset(
        name=name,
        time=time,
        qpos=qpos,
        qvel=qvel,
        mpos=mpos,
        mquat=mquat,
    )
