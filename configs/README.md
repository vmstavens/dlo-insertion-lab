# Hydra configuration

Top-level files describe workflow shapes:

- `collect.yaml`
- `prepare.yaml`
- `train_dp.yaml`
- `train_drlr2.yaml`
- `evaluate.yaml`

Configuration groups hold reusable choices:

- `task/`, `dataset/`, `collector/`, `transform/`
- `il/`, `rl/`, `trainer/`, `evaluation/`, `runtime/`
- `experiment/` for named, reviewed combinations
- `sweep/` for multirun search spaces

The `schema` default in every top-level config merges a structured dataclass
from `learning.config.schema`. This catches unknown keys and missing mandatory
values during composition. YAML remains the user-facing configuration format.

Use command-line overrides for quick trials:

```bash
uv run python -m workflows.train_drlr2 seed=7 rl.batch_size=128
```

Use an experiment config for a meaningful baseline:

```bash
uv run python -m workflows.train_drlr2 \
  experiment=insert_drlr2_baseline
```

Use Hydra multirun for a sweep:

```bash
uv run python -m workflows.train_drlr2 -m \
  +sweep=insert_drlr2_small
```

All supplied runtime profiles default to `dry_run: true`. Inspect composition
with `--cfg job --resolve`. Enable real execution only after implementing the
selected workflow:

```bash
uv run python -m workflows.train_drlr2 runtime.dry_run=false
```

## Notifications

The `runtime.notifications` block controls optional terminal-state
notifications. It defaults to disabled in every runtime profile.

Enable ntfy for a real workflow:

```bash
export NTFY_TOPIC="a-long-random-or-protected-topic"
export NTFY_TOKEN="optional-access-token"

uv run python -m workflows.train_drlr2 \
  runtime.dry_run=false \
  runtime.notifications.enabled=true
```

Available settings:

| Setting | Default | Purpose |
| --- | --- | --- |
| `enabled` | `false` | Enables notification dispatch for real work. |
| `provider` | `ntfy` | Selects the delivery adapter. |
| `server` | `$NTFY_SERVER` or `https://ntfy.sh` | Supports the hosted or a self-hosted ntfy server. |
| `topic_env` | `NTFY_TOPIC` | Names the environment variable containing the topic. |
| `token_env` | `NTFY_TOKEN` | Names the environment variable containing an optional access token. |
| `on_success` | `true` | Reports successful completion. |
| `on_failure` | `true` | Reports exceptions before re-raising them. |
| `on_interrupted` | `true` | Reports interruptions such as Ctrl-C. |
| `minimum_success_duration_seconds` | `60` | Prevents noise from short successful jobs. It does not suppress failures. |
| `timeout_seconds` | `5` | Bounds network delay during best-effort delivery. |

The topic and token values themselves never enter Hydra configuration, so they
are not copied into `resolved_config.yaml`. Public ntfy topics can be read or
written by anyone who knows their name. Use a hard-to-guess topic or,
preferably, an authenticated topic on a trusted server. Notifications include
only the task, stage, duration, hostname, run directory, and exception class;
they do not include tracebacks or resolved configuration.

Hydra multiruns execute one workflow per job, so notifications are also sent
per job. For large sweeps, suppress successful-job notifications while keeping
failure alerts:

```bash
uv run python -m workflows.train_drlr2 -m \
  runtime.dry_run=false \
  runtime.notifications.enabled=true \
  runtime.notifications.on_success=false \
  +sweep=insert_drlr2_small
```
