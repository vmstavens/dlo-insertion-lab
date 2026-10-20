"""Algorithm-independent policies with explicit project checkpoint factories."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    """Controller emitting environment-native actions.

    Adapters own history, normalization, action queues and filters. Reset clears
    all episode state and seeds inference randomness independently of the env.
    """

    def reset(self, *, seed: int) -> None: ...

    def act(self, observation: Any) -> Any: ...


class ZeroPolicy:
    """Baseline for environment inspection without a checkpoint."""

    def __init__(self, action_size: int):
        self.action_size = action_size

    def reset(self, *, seed: int) -> None:
        pass

    def act(self, observation: Any) -> np.ndarray:
        return np.zeros(self.action_size, dtype=np.float32)


def read_manifest(path: Path) -> dict[str, Any]:
    """Inspect JSON without importing policy code or deserializing weights."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Expected a checkpoint manifest with schema_version=1.")
    for field in ("policy_type", "loader", "checkpoint", "action_semantics"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"Manifest requires a nonempty {field} string.")
    if ":" not in manifest["loader"]:
        raise ValueError("Manifest loader must be module:factory.")
    for field in ("observation_dim", "action_dim"):
        if type(manifest.get(field)) is not int or manifest[field] < 1:
            raise ValueError(f"Manifest {field} must be a positive integer.")
    if not isinstance(manifest.get("config", {}), dict):
        raise ValueError("Manifest config must be an object.")
    return manifest


def load_policy(path: Path, environment: Any, *, deterministic: bool = True) -> Policy:
    manifest = read_manifest(path)
    for field, actual in (
        ("observation_dim", environment.observation_size),
        ("action_dim", environment.action_size),
    ):
        if manifest[field] != actual:
            raise ValueError(
                f"Checkpoint {field}={manifest[field]} != environment {actual}"
            )
    checkpoint = Path(manifest["checkpoint"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = path.resolve().parent / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    module, name = manifest["loader"].split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    policy = factory(
        checkpoint=checkpoint,
        config=manifest.get("config", {}),
        deterministic=deterministic,
    )
    if not callable(getattr(policy, "reset", None)) or not callable(
        getattr(policy, "act", None)
    ):
        raise TypeError("Policy factory must return an object with reset() and act().")
    return policy


class LinearPolicy:
    """Portable example: tanh(observation @ weights + bias)."""

    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        if weights.ndim != 2 or bias.shape != (weights.shape[1],):
            raise ValueError("Expected weights[observation, action] and bias[action].")
        self.weights, self.bias = weights, bias

    def reset(self, *, seed: int) -> None:
        pass

    def act(self, observation: Any) -> np.ndarray:
        return np.tanh(np.asarray(observation) @ self.weights + self.bias)


def load_linear_policy(
    *, checkpoint: Path, config: Mapping[str, Any], deterministic: bool
) -> LinearPolicy:
    """Example factory; algorithm adapters implement the same loader signature."""
    with np.load(checkpoint, allow_pickle=False) as data:
        return LinearPolicy(data["weights"], data["bias"])
