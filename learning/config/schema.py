"""Typed project-level schemas registered with Hydra's ConfigStore."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TaskConfig:
    name: str = MISSING
    registration_modules: list[str] = field(
        default_factory=lambda: ["envs.grasped_end_dlo"]
    )
    env: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetExpectedConfig:
    observation_dim: int = MISSING
    action_dim: int = MISSING
    control_dt: float = MISSING


@dataclass
class DatasetConfig:
    id: str = MISSING
    kind: str = MISSING
    path: str = MISSING
    manifest: str = MISSING
    expected: DatasetExpectedConfig = field(default_factory=DatasetExpectedConfig)


@dataclass
class CollectorConfig:
    name: str = MISSING
    controller: str = MISSING
    episodes: int = 10
    max_steps: int = 1000
    translation_scale: float = 1.0
    rotation_scale: float = 1.0


@dataclass
class TransformConfig:
    name: str = MISSING
    trim_idle_prefix: bool = True
    idle_threshold: float = 1e-3
    validation_fraction: float = 0.2
    split_seed: int = 42
    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8


@dataclass
class DiffusionPolicyConfig:
    name: str = "diffusion_policy"
    checkpoint: str | None = None
    device: str = "cuda"
    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    num_diffusion_iters: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 100


@dataclass
class DRLR2Config:
    name: str = "drlr2_sac"
    timesteps: int = 1_000_000
    batch_size: int = 64
    gradient_steps: int = 1
    warmup_timesteps: int = 10_000
    learning_starts: int = 0
    discount_factor: float = 0.99
    polyak: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_learning_rate: float = 3e-4
    actor: str = "both"
    mixed_precision: bool = False


@dataclass
class TrainerConfig:
    name: str = MISSING
    headless: bool = True
    checkpoint_interval: int = 10_000
    write_interval: int = 1_000
    evaluation_interval: int = 10_000


@dataclass
class EvaluationConfig:
    protocol: str | None = None
    split: str = "test"
    condition: str = "nominal"
    method: str = "unspecified"
    training_seed: int | None = None
    episodes: int = 20
    deterministic: bool = True
    max_steps: int = 1000
    record_video: bool = False
    seeds: list[int] = field(default_factory=list)
    policy_manifest: str | None = None
    policy: str = "checkpoint"
    compile_environment: bool = True
    fps: int = 30
    width: int = 640
    height: int = 480
    camera: str | None = None
    interactive: bool = False


@dataclass
class TensorBoardConfig:
    mode: str = "print"
    log_dir: str | None = None
    port: int = 6006
    open_browser: bool = False


@dataclass
class NotificationConfig:
    enabled: bool = False
    provider: str = "ntfy"
    server: str = "https://ntfy.sh"
    topic_env: str = "NTFY_TOPIC"
    token_env: str = "NTFY_TOKEN"
    on_success: bool = True
    on_failure: bool = True
    on_interrupted: bool = True
    minimum_success_duration_seconds: float = 60.0
    timeout_seconds: float = 5.0


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    data_root: str = "data"
    runs_root: str = "runs"
    dry_run: bool = True
    provenance_inputs: dict[str, str] = field(default_factory=dict)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)


@dataclass
class CollectWorkflowConfig:
    stage: str = "collect_demos"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class PrepareWorkflowConfig:
    stage: str = "prepare_dataset"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class TrainDPWorkflowConfig:
    stage: str = "train_dp"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    il: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class TrainDRLR2WorkflowConfig:
    stage: str = "train_drlr2"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    il: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)
    rl: DRLR2Config = field(default_factory=DRLR2Config)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class EvaluateWorkflowConfig:
    stage: str = "evaluate"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


_REGISTERED = False


def register_configs() -> None:
    """Register workflow schemas once, before Hydra composes YAML configs."""
    global _REGISTERED
    if _REGISTERED:
        return
    store = ConfigStore.instance()
    schemas = {
        "collect": CollectWorkflowConfig,
        "prepare": PrepareWorkflowConfig,
        "train_dp": TrainDPWorkflowConfig,
        "train_drlr2": TrainDRLR2WorkflowConfig,
        "evaluate": EvaluateWorkflowConfig,
    }
    for name, node in schemas.items():
        store.store(
            group="schema",
            name=name,
            node=node,
            package="_global_",
            provider="drl-sys-lab",
        )
    _REGISTERED = True
