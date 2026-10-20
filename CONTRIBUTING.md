# Contributing

## Development setup

Install the locked CPU environment:

```bash
uv sync --frozen
```

The locked `mjxsim` dependency selects CUDA-enabled JAX and PyTorch wheels on
Linux. Use `JAX_PLATFORMS=cpu` for portable tests and development that should
not access a GPU.

## Before opening a change

Run:

```bash
uv run ruff format --check .
uv run ruff check .
JAX_PLATFORMS=cpu uv run pytest -q
```

Tests should be deterministic on CPU. Changes that require a GPU should retain
CPU coverage for configuration, validation, and non-accelerated behavior.

Keep generated datasets under `data/` and run artifacts under `runs/`; both
locations are intentionally ignored except for their contract documentation.
Never commit credentials, ntfy topics or tokens, model checkpoints, trajectory
data, or resolved configuration containing secrets.

New workflows should remain thin and route execution through
`learning.config.execute_workflow`. New notification services should implement
a provider adapter under `learning/notifications/` rather than adding
service-specific behavior to workflow modules.
