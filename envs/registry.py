from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import registry as playground_registry
from mujoco_playground._src import dm_control_suite, locomotion, manipulation, mjx_env

EnvironmentFactory = Callable[..., mjx_env.MjxEnv]
ConfigFactory = Callable[[], config_dict.ConfigDict]
DomainRandomizer = Callable[[mjx.Model, jax.Array], tuple[mjx.Model, Any]]


@dataclass(frozen=True)
class EnvironmentSpec:
    """Inspectable environment metadata from the unified registry."""

    name: str
    environment_factory: EnvironmentFactory
    config_factory: ConfigFactory
    domain_randomizer: DomainRandomizer | None = None
    suite: str | None = None
    source: str = "project"


_PROJECT_SPECS: dict[str, EnvironmentSpec] = {}
_DOMAIN_RANDOMIZERS: dict[str, DomainRandomizer] = {}


def __getattr__(name: str) -> Any:
    if name == "ALL_ENVS":
        return list_environments()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _playground_environments() -> tuple[str, ...]:
    """Return a dynamic union, including environments registered at runtime."""
    return tuple(
        dict.fromkeys(
            (
                *playground_registry.ALL_ENVS,
                *dm_control_suite.ALL_ENVS,
                *locomotion.ALL_ENVS,
                *manipulation.ALL_ENVS,
            )
        )
    )


def _upstream_suite(env_name: str) -> str | None:
    if env_name in manipulation.ALL_ENVS:
        return "manipulation"
    if env_name in locomotion.ALL_ENVS:
        return "locomotion"
    if env_name in dm_control_suite.ALL_ENVS:
        return "dm_control_suite"
    return None


def _sync_playground_all_envs() -> None:
    """Refresh Playground's import-time snapshot after runtime registration."""
    playground_registry.ALL_ENVS = tuple(
        dict.fromkeys(
            (
                *dm_control_suite.ALL_ENVS,
                *locomotion.ALL_ENVS,
                *manipulation.ALL_ENVS,
            )
        )
    )


def register_environment(
    env_name: str,
    env_class: EnvironmentFactory,
    cfg_class: ConfigFactory,
    *,
    domain_randomizer: DomainRandomizer | None = None,
    suite: str | None = None,
    replace: bool = False,
) -> None:
    """Register a project environment with MuJoCo Playground.

    Playground owns environment construction and default-config lookup.
    Project-only suite metadata and randomizers remain in this adapter.
    Existing upstream environments cannot be replaced.
    """
    if not env_name or not env_name.strip():
        raise ValueError("Environment name must be a non-empty string.")

    if env_name in _playground_environments() and env_name not in _PROJECT_SPECS:
        raise ValueError(
            f"Environment {env_name!r} is provided by MuJoCo Playground and "
            "cannot be replaced by the project registry."
        )
    if env_name in _PROJECT_SPECS and not replace:
        raise ValueError(
            f"Environment {env_name!r} is already registered. "
            "Pass replace=True to replace it explicitly."
        )

    # This is currently Playground's only registration hook. The top-level
    # registry discovers manipulation.ALL_ENVS dynamically when loading.
    manipulation.register_environment(env_name, env_class, cfg_class)
    _sync_playground_all_envs()

    spec = EnvironmentSpec(
        name=env_name,
        environment_factory=env_class,
        config_factory=cfg_class,
        domain_randomizer=domain_randomizer,
        suite=suite,
    )
    _PROJECT_SPECS[env_name] = spec
    if domain_randomizer is None:
        _DOMAIN_RANDOMIZERS.pop(env_name, None)
    else:
        _DOMAIN_RANDOMIZERS[env_name] = domain_randomizer


def register_domain_randomizer(
    env_name: str,
    domain_randomizer: DomainRandomizer,
) -> None:
    """Attach a randomizer without mutating Playground's private dictionaries."""
    spec = get_spec(env_name)
    _DOMAIN_RANDOMIZERS[env_name] = domain_randomizer
    if env_name in _PROJECT_SPECS:
        _PROJECT_SPECS[env_name] = EnvironmentSpec(
            name=spec.name,
            environment_factory=spec.environment_factory,
            config_factory=spec.config_factory,
            domain_randomizer=domain_randomizer,
            suite=spec.suite,
            source=spec.source,
        )


def get_spec(env_name: str) -> EnvironmentSpec:
    """Return metadata for project or built-in Playground environments."""
    if env_name in _PROJECT_SPECS:
        return _PROJECT_SPECS[env_name]
    if env_name not in _playground_environments():
        raise ValueError(
            f"Environment {env_name!r} not found. "
            f"Available environments: {list_environments()}"
        )
    return EnvironmentSpec(
        name=env_name,
        environment_factory=partial(playground_registry.load, env_name),
        config_factory=partial(playground_registry.get_default_config, env_name),
        domain_randomizer=_DOMAIN_RANDOMIZERS.get(env_name),
        suite=_upstream_suite(env_name),
        source="mujoco_playground",
    )


def list_environments(*, suite: str | None = None) -> tuple[str, ...]:
    """List built-in and project environments through one interface."""
    environments = _playground_environments()
    if suite is None:
        return environments

    selected = []
    for env_name in environments:
        if env_name in _PROJECT_SPECS:
            env_suite = _PROJECT_SPECS[env_name].suite
        else:
            env_suite = _upstream_suite(env_name)
        if env_suite == suite:
            selected.append(env_name)
    return tuple(selected)


def get_default_config(env_name: str) -> config_dict.ConfigDict:
    """Delegate default configuration lookup to MuJoCo Playground."""
    get_spec(env_name)
    return playground_registry.get_default_config(env_name)


def load(
    env_name: str,
    config: config_dict.ConfigDict | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> mjx_env.MjxEnv:
    """Delegate construction of built-in or project environments to Playground."""
    get_spec(env_name)
    return playground_registry.load(
        env_name,
        config=config,
        config_overrides=config_overrides,
    )


def get_domain_randomizer(env_name: str) -> DomainRandomizer | None:
    """Return a project sidecar randomizer or Playground's built-in one."""
    get_spec(env_name)
    if env_name in _DOMAIN_RANDOMIZERS:
        return _DOMAIN_RANDOMIZERS[env_name]
    return playground_registry.get_domain_randomizer(env_name)
