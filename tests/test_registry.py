"""Tests for the central environment registry."""

from __future__ import annotations

import unittest
from unittest import mock

import jax
from ml_collections import config_dict
from mujoco_playground import registry as playground_registry
from mujoco_playground._src import manipulation, mjx_env

from envs import grasped_end_dlo, registry


class _FakeEnvironment:
    def __init__(self, config, config_overrides=None):
        self.config = config
        self.config_overrides = config_overrides


def _config():
    return config_dict.create(value=1)


def _randomizer(model, rng):
    del rng
    return model, None


class RegistryTest(unittest.TestCase):
    def test_insertion_task_is_registered(self) -> None:
        name = "GraspedEndDloInsert"
        self.assertIn(name, grasped_end_dlo.ALL_ENVS)
        self.assertEqual(registry.get_spec(name).suite, "grasped_end_dlo")

    def test_register_load_and_inspect(self) -> None:
        name = "RegistryTestEnvironment"
        registry.register_environment(
            name,
            _FakeEnvironment,
            _config,
            domain_randomizer=_randomizer,
            suite="test",
            replace=True,
        )

        self.assertIn(name, registry.list_environments())
        self.assertEqual(registry.list_environments(suite="test"), (name,))
        self.assertEqual(registry.get_default_config(name).value, 1)
        self.assertIs(registry.get_domain_randomizer(name), _randomizer)
        self.assertIn(name, manipulation.ALL_ENVS)
        self.assertIn(name, playground_registry.ALL_ENVS)

        # Both the project adapter and established Playground interface resolve
        # the same runtime registration.
        with mock.patch.object(mjx_env, "ensure_menagerie_exists"):
            environment = playground_registry.load(
                name,
                config_overrides={"value": 2},
            )
        self.assertEqual(environment.config.value, 1)
        self.assertEqual(environment.config_overrides, {"value": 2})

        with mock.patch.object(
            playground_registry,
            "load",
            return_value=environment,
        ) as load:
            self.assertIs(registry.load(name), environment)
            load.assert_called_once_with(
                name,
                config=None,
                config_overrides=None,
            )

    def test_duplicate_registration_requires_explicit_replace(self) -> None:
        name = "RegistryDuplicateTestEnvironment"
        registry.register_environment(
            name,
            _FakeEnvironment,
            _config,
            replace=True,
        )
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register_environment(name, _FakeEnvironment, _config)

    def test_unknown_environment_error_lists_available_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Available environments"):
            registry.get_spec("EnvironmentThatDoesNotExist")

    def test_builtin_playground_environments_are_visible(self) -> None:
        self.assertIn("PandaPickCube", registry.list_environments())
        spec = registry.get_spec("PandaPickCube")
        self.assertEqual(spec.source, "mujoco_playground")
        self.assertEqual(spec.suite, "manipulation")

    def test_upstream_environment_cannot_be_replaced(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided by MuJoCo Playground"):
            registry.register_environment(
                "PandaPickCube",
                _FakeEnvironment,
                _config,
                replace=True,
            )

    def test_family_registration_sets_suite(self) -> None:
        name = "GraspedEndDloRegistryTestEnvironment"
        grasped_end_dlo.register_environment(
            name,
            _FakeEnvironment,
            _config,
            domain_randomizer=_randomizer,
            replace=True,
        )
        self.assertIn(name, grasped_end_dlo.ALL_ENVS)
        self.assertEqual(registry.get_spec(name).suite, "grasped_end_dlo")

    def test_family_loader_rejects_other_suites(self) -> None:
        name = "OtherSuiteRegistryTestEnvironment"
        registry.register_environment(
            name,
            _FakeEnvironment,
            _config,
            suite="other",
            replace=True,
        )
        with self.assertRaisesRegex(ValueError, "belongs to suite"):
            grasped_end_dlo.load(name)

    def test_randomizer_can_be_attached_after_registration(self) -> None:
        name = "LateRandomizerTestEnvironment"
        registry.register_environment(
            name,
            _FakeEnvironment,
            _config,
            replace=True,
        )
        registry.register_domain_randomizer(name, _randomizer)
        self.assertIs(registry.get_domain_randomizer(name), _randomizer)

        model = object()
        randomized_model, in_axes = _randomizer(model, jax.random.PRNGKey(0))
        self.assertIs(randomized_model, model)
        self.assertIsNone(in_axes)


if __name__ == "__main__":
    unittest.main()
