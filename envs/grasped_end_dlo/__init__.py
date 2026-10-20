"""Shared components for grasped-end deformable-linear-object tasks."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from envs import registry

__all__ = [
    "ALL_ENVS",
    "GraspedEndDloBase",
    "base_config",
    "get_default_config",
    "get_domain_randomizer",
    "load",
    "register_environment",
]

_EXPORT_MODULES = {
    "GraspedEndDloBase": "envs.grasped_end_dlo.base",
    "base_config": "envs.grasped_end_dlo.config",
}


def __getattr__(name: str) -> Any:
    if name == "ALL_ENVS":
        return registry.list_environments(suite="grasped_end_dlo")
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name]), name)
    globals()[name] = value
    return value


def register_environment(
    env_name: str,
    env_class: Callable[..., mjx_env.MjxEnv],
    cfg_class: Callable[[], config_dict.ConfigDict],
    *,
    domain_randomizer: Callable[[mjx.Model, jax.Array], tuple[mjx.Model, Any]]
    | None = None,
    replace: bool = False,
) -> None:
    """Register a concrete task in the grasped-end DLO suite."""
    registry.register_environment(
        env_name,
        env_class,
        cfg_class,
        domain_randomizer=domain_randomizer,
        suite="grasped_end_dlo",
        replace=replace,
    )


def get_default_config(env_name: str) -> config_dict.ConfigDict:
    _ensure_suite(env_name)
    return registry.get_default_config(env_name)


def load(
    env_name: str,
    config: config_dict.ConfigDict | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> mjx_env.MjxEnv:
    _ensure_suite(env_name)
    return registry.load(env_name, config, config_overrides)


def get_domain_randomizer(
    env_name: str,
) -> Callable[[mjx.Model, jax.Array], tuple[mjx.Model, Any]] | None:
    _ensure_suite(env_name)
    return registry.get_domain_randomizer(env_name)


def _ensure_suite(env_name: str) -> None:
    spec = registry.get_spec(env_name)
    if spec.suite != "grasped_end_dlo":
        raise ValueError(
            f"Environment {env_name!r} belongs to suite {spec.suite!r}, "
            "not 'grasped_end_dlo'."
        )


# Import built-in tasks after the registration API is defined.
import_module("envs.grasped_end_dlo.insert")
