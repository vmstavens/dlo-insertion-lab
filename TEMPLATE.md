# Using this repository as a template

This repository intentionally keeps `GraspedEndDloInsert` as a working vertical
slice. Copy and adapt it before deleting it; the example documents the
contracts between environments, Hydra configuration, data, workflows, and
tests.

## Create a project

1. Create a new repository from this template without copying its Git history.
2. Replace the project identity in `pyproject.toml`, `README.md`,
   `tests/__init__.py`, and the Hydra provider in
   `learning/config/schema.py`.
3. Rename the example environment package, registry name, task configuration,
   dataset IDs, experiment names, and run paths.
4. Replace or remove the `mjsim` and `mjxsim` source declarations.
5. Regenerate and commit `uv.lock`.
6. Choose the runtime profiles and optional GPU backend supported by the
   project.
7. Replace the license copyright line if project ownership differs.
8. Run the clean-clone acceptance checks below.
9. In the GitHub repository settings, enable **Template repository**.

Useful searches:

```bash
rg -n "drl-sys-lab|GraspedEndDlo|grasped_end_dlo|insert_|vmstavens"
```

## Dependency boundary

`mjsim` and `mjxsim` are fetched from public HTTPS Git repositories. A derived
project can keep those sources, replace them with stable package-index
releases, or remove them when adopting different simulation interfaces.

Do not embed personal access tokens in `pyproject.toml`, `uv.lock`, CI files, or
Hydra configuration.

## Keep the architecture

Reusable evaluation, viewing, video recording, checkpoint inspection, and
TensorBoard launcher settings are documented in [tools/README.md](tools/README.md).
Run its Cartpole example to verify a new clone without a trained checkpoint.
For publication experiments, follow [the publication workflow](docs/publication.md).
Keep evaluation protocols and figure recipes versioned with the project.

The intended dependency direction is:

```text
workflows -> learning -> envs
                 |
                 v
               data
```

- Keep workflow modules thin.
- Keep environment code independent of algorithms.
- Store experiment choices in Hydra configuration.
- Store datasets and run artifacts outside Git while versioning their
  contracts.
- Route real workflow execution through `execute_workflow()` so provenance and
  terminal notifications remain consistent.

## Clean-clone acceptance

A new clone should pass these portable CPU-execution checks:

```bash
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
JAX_PLATFORMS=cpu uv run pytest -q
uv run python -m workflows.collect_demos
uv run python -m workflows.prepare_dataset
uv run python -m workflows.train_dp
uv run python -m workflows.train_drlr2
uv run python -m workflows.evaluate
git status --short
```

The workflow commands default to dry-runs. Evaluation, viewing and recording
support real execution; collection, preparation and training remain scaffolds
until their implementations are supplied.
The final command should show no generated or modified tracked files.
