"""Reproducible episode execution shared by evaluation, viewing and recording."""

from __future__ import annotations

import csv
import importlib
import json
import operator
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from learning.policies.checkpoints import Policy, ZeroPolicy, load_policy, read_manifest
from learning.publication.io import identity, write_json
from learning.publication.protocol import apply_protocol


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    steps: int
    total_reward: float
    done: bool
    step_limit_reached: bool
    success: bool | None
    metrics: dict[str, float]


def rollout_episode(
    environment: Any,
    policy: Policy,
    *,
    seed: int,
    max_steps: int,
    compile_environment: bool = True,
    observer: Callable[[Any, int], None] | None = None,
    success_rule: dict | None = None,
) -> EpisodeResult:
    """Observe reset and every transition without changing control execution.

    Use an unbatched environment without auto-reset. ``done`` retains the
    environment's semantics; the runner does not infer termination vs timeout.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    reset = jax.jit(environment.reset) if compile_environment else environment.reset
    step = jax.jit(environment.step) if compile_environment else environment.step
    env_key, policy_key = jax.random.split(jax.random.PRNGKey(seed))
    policy.reset(seed=int(np.asarray(policy_key)[0]))
    state = reset(env_key)
    if observer is not None:
        observer(state, 0)
    total_reward = 0.0
    steps = 0
    success = None
    while steps < max_steps and not bool(np.asarray(state.done)):
        action = np.asarray(policy.act(state.obs))
        if action.shape != (environment.action_size,) or not np.isfinite(action).all():
            raise ValueError("Policy must emit a finite unbatched environment action.")
        state = step(state, action)
        reward = float(np.asarray(state.reward))
        if not np.isfinite(reward):
            raise ValueError("Environment emitted a non-finite reward.")
        total_reward += reward
        steps += 1
        value = state.metrics.get("is_success", state.info.get("success"))
        if success_rule:
            value = state.metrics[success_rule["metric"]]
            if not np.isfinite(float(np.asarray(value))):
                raise ValueError("Success metric must be finite.")
            compare = {
                ">=": operator.ge,
                "<=": operator.le,
                ">": operator.gt,
                "<": operator.lt,
            }[success_rule["operator"]]
            value = compare(float(np.asarray(value)), success_rule["threshold"])
        if value is not None:
            success = (
                bool(value)
                if success_rule and success_rule["mode"] == "final"
                else bool(success) or bool(np.asarray(value))
            )
        if observer is not None:
            observer(state, steps)
    metrics = {}
    for name, value in state.metrics.items():
        array = np.asarray(value)
        if array.ndim == 0 and np.issubdtype(array.dtype, np.number):
            if np.isfinite(array):
                metrics[name] = float(array)
    done = bool(np.asarray(state.done))
    return EpisodeResult(
        seed,
        steps,
        total_reward,
        done,
        steps == max_steps and not done,
        success,
        metrics,
    )


def create_environment(config: Mapping[str, Any]) -> Any:
    from envs import registry

    task = config["task"]
    for module in task.get("registration_modules", []):
        importlib.import_module(module)
    overrides = dict(task["env"])
    name = overrides.pop("registry_name")
    if config.get("runtime", {}).get("device") == "cpu" and "impl" in overrides:
        overrides["impl"] = "jax"
    return registry.load(name, config_overrides=overrides)


def write_results(results: list[EpisodeResult], output_dir: Path) -> None:
    if not results:
        raise ValueError("Cannot summarize an empty evaluation.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    rows = []
    for result in results:
        row = asdict(result)
        metrics = row.pop("metrics")
        row.update({f"metric/{key}": value for key, value in metrics.items()})
        rows.append(row)
    with (output_dir / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    rewards = [r.total_reward for r in results]
    successes = [r.success for r in results if r.success is not None]
    summary = {
        "episodes": len(results),
        "return_mean": float(np.mean(rewards)),
        "return_std": float(np.std(rewards)),
        "steps_mean": float(np.mean([r.steps for r in results])),
        "success_rate": float(np.mean(successes)) if successes else None,
        "success_episodes": len(successes),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def evaluate_policy(config: Mapping[str, Any], output_dir: Path) -> None:
    config = apply_protocol(dict(config), output_dir)
    options = config["evaluation"]
    seeds = options.get("seeds") or list(
        range(
            int(config.get("seed", 0)), int(config.get("seed", 0)) + options["episodes"]
        )
    )
    if not seeds or options["max_steps"] < 1:
        raise ValueError("Evaluation requires episodes and positive max_steps.")
    kind = options.get("policy", "checkpoint")
    manifest_path = options.get("policy_manifest")
    if kind not in {"checkpoint", "zero"}:
        raise ValueError("evaluation.policy must be checkpoint or zero.")
    if kind == "checkpoint" and not manifest_path:
        raise ValueError(
            "Set evaluation.policy_manifest or use evaluation.policy=zero."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "evaluation_context.json",
        {
            "schema_version": 1,
            "task": config.get("task", {}).get("name", "unknown"),
            "method": options.get("method", "unspecified"),
            "training_seed": options.get("training_seed"),
            "split": options.get("split", "test"),
            "condition": options.get("condition", "nominal"),
            "protocol_sha256": options.get("protocol_sha256"),
            "seeds": seeds,
            "evaluation_signature": identity(
                {
                    "task": config.get("task", {}),
                    "seeds": seeds,
                    "max_steps": options["max_steps"],
                    "deterministic": options.get("deterministic", True),
                    "success_rule": options.get("success_rule"),
                }
            ),
        },
    )
    with ExitStack() as stack:
        backend = config.get("runtime", {}).get("device", "cpu")
        backend = "gpu" if backend == "cuda" else backend
        stack.enter_context(jax.default_device(jax.devices(backend)[0]))
        environment = create_environment(config)
        close = getattr(environment, "close", None)
        if close is not None:
            stack.callback(close)
        if kind == "zero":
            policy = ZeroPolicy(environment.action_size)
        else:
            path = Path(manifest_path).expanduser().resolve()
            policy = load_policy(
                path, environment, deterministic=options.get("deterministic", True)
            )
            snapshot = read_manifest(path)
            checkpoint = Path(snapshot["checkpoint"]).expanduser()
            snapshot["checkpoint"] = str(
                checkpoint if checkpoint.is_absolute() else path.parent / checkpoint
            )
            (output_dir / "policy_manifest.json").write_text(
                json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
            )
        results = []
        for index, seed in enumerate(seeds):
            with ExitStack() as episode_stack:
                observers = []
                if options.get("record_video"):
                    from learning.rendering.video import VideoObserver

                    video = episode_stack.enter_context(
                        VideoObserver(
                            environment,
                            output_dir / f"episode_{index:04d}_seed{seed}.mp4",
                            fps=options.get("fps", 30),
                            width=options.get("width", 640),
                            height=options.get("height", 480),
                            camera=options.get("camera"),
                        )
                    )
                    observers.append(video)
                if options.get("interactive"):
                    from learning.rendering.video import LiveObserver

                    observers.append(
                        episode_stack.enter_context(LiveObserver(environment))
                    )

                def observe(state, step, observers=observers):
                    for observer in observers:
                        observer(state, step)

                results.append(
                    rollout_episode(
                        environment,
                        policy,
                        seed=seed,
                        max_steps=options["max_steps"],
                        compile_environment=options.get("compile_environment", True),
                        observer=observe if observers else None,
                        success_rule=options.get("success_rule"),
                    )
                )
            write_results(results, output_dir)
            print(
                f"Episode {index + 1}/{len(seeds)}: {asdict(results[-1])}", flush=True
            )
