from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException

from learning.config import register_configs
from learning.config.runtime import resolved_mapping
from learning.data.schema import DatasetManifest

CONFIG_DIR = Path(__file__).parents[1] / "configs"


def compose_workflow(name: str, overrides: list[str] | None = None) -> dict:
    register_configs()
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(CONFIG_DIR.resolve()),
    ):
        config = compose(config_name=name, overrides=overrides or [])
    return resolved_mapping(config)


@pytest.mark.parametrize(
    ("config_name", "stage"),
    [
        ("collect", "collect_demos"),
        ("prepare", "prepare_dataset"),
        ("train_dp", "train_dp"),
        ("train_drlr2", "train_drlr2"),
        ("evaluate", "evaluate"),
        ("render_rollout", "render_rollout"),
        ("view_policy", "view_policy"),
        ("view_env", "view_env"),
    ],
)
def test_workflow_configs_compose_and_resolve(
    config_name: str,
    stage: str,
) -> None:
    config = compose_workflow(config_name)

    assert config["stage"] == stage
    assert config["task"]["name"] == "grasped_end_dlo_insert"
    assert config["task"]["env"]["registry_name"] == "GraspedEndDloInsert"
    assert config["runtime"]["dry_run"] is True
    assert config["runtime"]["notifications"]["enabled"] is False
    assert config["runtime"]["notifications"]["topic_env"] == "NTFY_TOPIC"
    assert "token" not in config["runtime"]["notifications"]
    if stage in {"collect_demos", "prepare_dataset", "train_dp", "train_drlr2"}:
        assert config["dataset"]["path"].startswith("data/grasped_end_dlo_insert/")
    else:
        assert "dataset" not in config
        assert "il" not in config


def test_named_experiment_overrides_component_defaults() -> None:
    config = compose_workflow(
        "train_drlr2",
        ["experiment=insert_drlr2_baseline"],
    )

    assert config["rl"]["batch_size"] == 256
    assert config["rl"]["warmup_timesteps"] == 20_000
    assert config["il"]["checkpoint"].endswith("/approved/best.pt")


def test_command_line_override_is_type_checked() -> None:
    with pytest.raises(ConfigCompositionException):
        compose_workflow("train_drlr2", ["rl.batch_size=not_an_integer"])


def make_manifest() -> DatasetManifest:
    return DatasetManifest(
        schema_version=1,
        dataset_id="insert_raw_test",
        kind="raw",
        environment="GraspedEndDloInsert",
        observation_layout=("mocap_x", "mocap_y"),
        observation_dim=2,
        action_dim=2,
        action_semantics="mocap delta position",
        control_dt=0.002,
        episode_count=3,
        git_commit="abc123",
    )


def test_dataset_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = make_manifest()

    manifest.save(path)

    assert DatasetManifest.load(path) == manifest


def test_dataset_manifest_rejects_incompatible_environment_contract() -> None:
    manifest = make_manifest()

    with pytest.raises(ValueError, match="action_dim"):
        manifest.assert_compatible(
            environment="GraspedEndDloInsert",
            observation_dim=2,
            action_dim=6,
            control_dt=0.002,
        )
