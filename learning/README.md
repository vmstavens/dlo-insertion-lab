# Learning layer

The `learning` package contains reusable implementation code. It does not own
command-line parsing, experiment names, datasets, or run directories.

- `collection/`: demonstration collection logic and teleoperation/scripted
  controller integration.
- `data/`: dataset manifests, validation, and deterministic transforms.
- `imitation/`: supervised or imitation-learning algorithms, beginning with
  Diffusion Policy.
- `reinforcement/`: reinforcement-learning algorithms, beginning with DRLR2.
- `evaluation/`: policy rollouts and metrics shared by IL and RL.
- `policies/`: controller interfaces and explicit checkpoint loaders.
- `rendering/`: live and video observers of the shared rollout loop.
- `tensorboard.py`: optional monitoring process lifecycle for training runs.
- `publication/`: provenance, experiment matrices, protocols, statistics, figures
  and portable result-reproduction bundles.
- `config/`: typed Hydra schemas and the common workflow/provenance boundary.

Collection, dataset preparation and training functions remain deliberate
placeholders. Evaluation, viewing and video recording support real execution
with `runtime.dry_run=false`, using an explicit policy adapter or the zero-action
baseline. See [the tools guide](../tools/README.md) for the executable example
and checkpoint contract.

Dependency direction is:

```text
workflows -> learning -> envs
                 |
                 v
               data
```

Environment classes should not import `learning` or `workflows`. Algorithms
operate through the environment API, while workflows provide configuration and
filesystem policy.
