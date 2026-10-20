# Reusable experiment tools

For experiment matrices, statistical comparisons, publication plots and
reproduction bundles, see [the publication guide](../docs/publication.md).

Run commands from the repository root. The Hydra workflows default to dry-run;
set `runtime.dry_run=false` to execute. Every workflow saves its resolved
configuration in the run directory. Optional rendering uses MuJoCo and requires
FFmpeg on PATH for MP4 encoding.

## TensorBoard

Training workflows print a shell-quoted TensorBoard command pointing at their
resolved run directory. Trainers should write event files somewhere below this
directory, or set `runtime.tensorboard.log_dir` to their actual logging directory.
This integration does not create training metrics or implement the scaffolded
trainers.

```bash
uv sync --extra monitoring
uv run python -m workflows.train_dp
uv run --extra monitoring python -m workflows.train_dp \
  runtime.dry_run=false runtime.tensorboard.mode=serve
```

Modes are `off`, `print` (default), and `serve`. Serving binds to loopback, prints
the URL, and stops the owned process when the workflow exits, including failures
and interrupts. Missing TensorBoard or an occupied port produces a warning and
does not prevent training. Dry-runs only print the command, even in serve mode.
The template's training implementations remain scaffolds until supplied by a
derived project.

Set `runtime.tensorboard.open_browser=true` to open the URL in the system browser.
In VS Code, use **Simple Browser: Show** with the printed URL. For Remote SSH,
forward `runtime.tensorboard.port` (default 6006) through the Ports panel and open
the forwarded URL. Browser opening occurs on the machine running the workflow;
leave it disabled on remote hosts. No VS Code extension is required.

For sweeps, set `runtime.tensorboard.log_dir` to the sweep parent and run one
TensorBoard command there. Keep workers in print mode to avoid duplicate servers.

## Environment and policy viewing

```bash
uv run python -m workflows.view_env runtime=local runtime.dry_run=false
uv run python -m workflows.view_policy runtime=local runtime.dry_run=false \
  evaluation.policy_manifest=/absolute/path/to/manifest.json
```

Both use the same episode execution as evaluation. Closing the viewer interrupts
the workflow and releases its resources. `tools.view_env` also retains the
original static-model and direct CPU-physics inspection modes.

## Portable example

This creates a working linear policy with illustrative weights, not a trained
controller. It uses Playground's Cartpole environment and needs no dataset.

```bash
uv run python -m tools.create_example_policy runs/example_policy
uv run python -m tools.inspect_checkpoint runs/example_policy/manifest.json
JAX_PLATFORMS=cpu uv run python -m workflows.evaluate \
  task=cartpole runtime=local runtime.dry_run=false \
  evaluation.policy_manifest=runs/example_policy/manifest.json \
  'evaluation.seeds=[0,1,2]' evaluation.max_steps=100
MUJOCO_GL=egl JAX_PLATFORMS=cpu uv run python -m workflows.render_rollout \
  task=cartpole runtime=local runtime.dry_run=false \
  evaluation.policy_manifest=runs/example_policy/manifest.json \
  evaluation.max_steps=100
```

On machines with a display, omit `MUJOCO_GL=egl`. Headless rendering requires a
working EGL driver (or an available alternative MuJoCo GL backend).

Evaluation writes `episodes.json`, `episodes.csv`, and `summary.json`. Video
workflows additionally write one MP4 per episode, sampled according to simulation
time and `evaluation.fps`. Configure `width`, `height`, and `camera` under
`evaluation`; dimensions must be even for H.264. Missing success metrics produce
`null`, not a fabricated failure rate. Final scalar task metrics are preserved.
The environment's `done` flag is reported as-is; reaching the runner's own limit
is recorded separately. Use environments without automatic episode reset.

## Adding an algorithm

Implement a controller with `reset(*, seed)` and `act(observation)`. It must own
all inference state: observation history, RMA state, normalization, action
chunking, filters, and randomness. Actions must already match the environment's
physical or normalized action contract. The runner does not clip or normalize
them. A factory has this signature:

```python
def load_policy(*, checkpoint: Path, config: Mapping, deterministic: bool):
    return MyController(checkpoint, config, deterministic=deterministic)
```

Store a JSON manifest next to the checkpoint:

```json
{
  "schema_version": 1,
  "policy_type": "my_policy",
  "loader": "learning.policies.my_policy:load_policy",
  "checkpoint": "actor.flax",
  "observation_dim": 48,
  "action_dim": 6,
  "action_semantics": "Describe units, bounds, and normalization here",
  "config": {}
}
```

Relative checkpoint paths resolve against the manifest directory. Inspection
reads JSON only; execution imports the specified factory. Dimensions are checked
before loading. Matching dimensions alone do not prove matching observation
layouts or action semantics; adapters must validate their algorithm's contracts.
Existing project checkpoints need an appropriate adapter and manifest; file
extensions are never used to guess their architecture.

Add environment registration modules to `task.registration_modules` and settings
to `task.env`. The rollout engine contains no task-specific imports. Rendering
currently supports unbatched MuJoCo/MJX environments exposing `mj_model`,
`mjx_model`, `dt`, and states with `data` and `info`. A per-state `info.mjx_model`
is used to synchronize randomized visual geometry.

## Ownership

- `workflows/`: thin Hydra entry points and shared run lifecycle.
- `learning/policies/`: checkpoint contracts and algorithm adapters.
- `learning/evaluation/`: episode execution and result export.
- `learning/rendering/`: optional live/video observers.
- `tools/`: small inspection commands and project-specific diagnostics.

Keep paper plots and named experiment sweeps in derived projects. Add adapters
for new policy types instead of duplicating episode loops in recording scripts.
