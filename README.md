# drl-sys-lab

A template and worked example for scalable reinforcement-learning projects
around MuJoCo and MJX environments.

The repository contains one complete environment family and a deliberately
honest learning scaffold. Configuration composition, dataset contracts,
provenance, notifications, environment simulation, policy evaluation, rollout
videos, checkpoint inspection, and tests work today.
Collectors and learning algorithms remain explicit placeholders until their
implementations are supplied.

## Quick start

Prerequisites:

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Create the locked CPU development environment and run the tests:

```bash
uv sync --frozen
JAX_PLATFORMS=cpu uv run pytest -q
```

The current `mjxsim` revision selects CUDA-enabled JAX and PyTorch wheels on
Linux. CPU execution remains supported by setting `JAX_PLATFORMS=cpu`, as in
the test command above.

See [TEMPLATE.md](TEMPLATE.md) before creating a new project from this
repository.

## Repository boundaries

```text
envs/        task definitions, simulation assets, reset/step semantics
tools/       developer utilities such as the interactive environment viewer
learning/    reusable collection, data, IL, RL, and evaluation code
workflows/   thin Hydra-powered executable entry points
configs/     composable, version-controlled experiment configuration
data/        local raw and processed trajectories (contents ignored)
runs/        local checkpoints, logs, videos, and provenance (contents ignored)
tests/       environment and infrastructure contracts
```

Code, configuration, data, and run artifacts have separate homes because they
change at different rates and have different versioning needs. Environments
remain independent of any particular learning algorithm. Workflow modules are
kept thin so the learning code stays importable and testable.

## Environment viewer

Render and simulate the insertion task with:

```bash
uv run python -m tools.view_env GraspedEndDloInsert
```

The viewer and its available simulation modes are documented by:

```bash
uv run python -m tools.view_env --help
```

## Learning workflows

The initial Hydra scaffold supports:

```bash
uv run python -m workflows.collect_demos
uv run python -m workflows.prepare_dataset
uv run python -m workflows.train_dp
uv run python -m workflows.train_drlr2
uv run python -m workflows.evaluate
```

These commands currently perform validated dry-runs and save their resolved
configuration. See [configs/README.md](configs/README.md) for overrides,
experiments, and sweeps, and [learning/README.md](learning/README.md) for the
architectural contract.

Evaluation, live viewing and MP4 recording share one episode runner. Training
launchers print the TensorBoard command by default and can optionally start a
server and open its URL. See [the tools guide](tools/README.md) for commands,
VS Code usage, policy adapters, and a runnable Cartpole checkpoint example.

For paper experiments, see [the publication guide](docs/publication.md): run
provenance, resumable experiment matrices, evaluation protocols, run-level
statistics, publication figures/LaTeX tables, and portable reproduction bundles.

## Completion notifications

Every real workflow passes through one lifecycle boundary that can publish an
optional [ntfy](https://docs.ntfy.sh/) notification when work succeeds, fails,
or is interrupted. Credentials and topic names are read from environment
variables and are never written into Hydra's resolved configuration:

```bash
export NTFY_TOPIC="choose-a-private-topic"
export NTFY_TOKEN="optional-access-token"

uv run python -m workflows.train_drlr2 \
  runtime.dry_run=false \
  runtime.notifications.enabled=true
```

Notifications are disabled for dry-runs. Successful runs shorter than 60
seconds are ignored by default, while failures are reported immediately.
Delivery is best-effort and can never change the workflow exit status. See
[the configuration guide](configs/README.md#notifications) for all options and
security guidance.

## Project quality

```bash
uv run ruff format --check .
uv run ruff check .
JAX_PLATFORMS=cpu uv run pytest -q
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes.
