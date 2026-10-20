# Publication workflow

The template supports a traceable path from experiments to paper figures:

```text
protocol + experiment matrix -> completed runs -> analysis recipe
                             -> figures/tables -> reproduction bundle
```

Install optional analysis dependencies with `uv sync --extra publication`.
TensorBoard remains available through `--extra monitoring`. Training algorithms
remain project implementations; these tools do not train a policy for you.

## 1. Run provenance

Every real Hydra workflow saves `provenance.json` alongside
`resolved_config.yaml`. It contains a unique run ID, terminal status, timestamps,
seed, Git commit and working-tree identity, installed package versions, Python,
CPU information, and available NVIDIA GPU/driver information. Tracked changes
are saved as `source.patch`; non-ignored untracked files are copied into
`source_untracked/`. Git-ignored files are not source snapshots.

Configured dataset directories, dataset manifests, policy manifests and their
checkpoints, protocol files, and dependency manifests are hashed with SHA-256.
Dataset hashing reads all files and can take time for large datasets.
Initial imitation-policy checkpoints are also hashed. Additional model assets,
calibration files or other inputs can be declared in
`runtime.provenance_inputs` as a mapping from names to paths.
Evaluation outputs are hashed when the run finishes. Source hashes establish identity;
they do not guarantee numerical determinism across hardware/backends.

A directory containing an existing run manifest cannot be reused. Dry-runs
still save configuration but do not capture full provenance or claim execution.
If a process is forcibly killed, its manifest remains `running`, and analysis
will not accept it as a completed run.

## 2. Evaluation protocols

See `configs/publication/cartpole_protocol.json` for a working example. A protocol
declares its ID, environment, disjoint validation/test seeds, named environment
conditions, maximum steps, deterministic inference setting, success rule and
checkpoint-selection rule.

```bash
JAX_PLATFORMS=cpu uv run python -m workflows.evaluate \
  task=cartpole runtime=local runtime.dry_run=false \
  evaluation.policy=zero evaluation.method=zero_baseline \
  evaluation.protocol=configs/publication/cartpole_protocol.json \
  evaluation.split=test evaluation.condition=nominal
```

The protocol owns trial seeds, episode budget and inference determinism, taking
precedence over those command-line settings. Each selected condition overrides
the specified environment parameters. The effective configuration and protocol
are copied into the run. Other task settings also enter an evaluation signature;
analysis rejects mismatched signatures within a comparison cell.

`success: null` retains the environment's success reporting. To define a metric
threshold explicitly, use:

```json
"success": {"metric": "position_error", "operator": "<=", "threshold": 0.01, "mode": "final"}
```

Supported operators are `<`, `<=`, `>`, `>=`. `mode` is `any` (at any transition)
or `final` (last transition). The metric must exist and be finite. The generic
runner does not change the environment's stopping rule. `max_steps` is an upper
bound in control steps, not seconds.

Choose checkpoints using training/validation evidence before final testing.
The selection rule is a recorded declaration, not an automatic checkpoint
selector or a technical barrier against manually inspecting test results.

## 3. Experiment matrices

Preview the local example without executing or creating run directories:

```bash
uv run python -m tools.run_matrix configs/publication/cartpole_matrix.json
JAX_PLATFORMS=cpu uv run python -m tools.run_matrix \
  configs/publication/cartpole_matrix.json --run
```

`axes` expands the Cartesian product of Hydra overrides. `overrides` provides
common settings. `bindings` fills dependent string overrides using Python's
`string.Template`; axis dots become underscores in variable names. For example:

```json
{
  "schema_version": 1,
  "project_root": "../..",
  "output_root": "../../runs/paper_evaluation",
  "workflow": "workflows.evaluate",
  "overrides": [
    "task=cartpole", "runtime=local",
    "evaluation.protocol=configs/publication/cartpole_protocol.json"
  ],
  "axes": {
    "evaluation.method": ["baseline", "proposed"],
    "evaluation.training_seed": [0, 1, 2, 3, 4],
    "evaluation.condition": ["nominal"]
  },
  "bindings": {
    "evaluation.policy_manifest": "runs/policies/${evaluation_method}/seed${evaluation_training_seed}/manifest.json"
  }
}
```

These checkpoints must already exist and represent independently trained runs.
Changing `evaluation.training_seed` labels the source training run; it does not
train or randomize the loaded policy. Evaluation seeds come from the protocol.

Matrix paths resolve relative to the matrix file; command overrides resolve
relative to `project_root`. The runner owns `runtime.dry_run=false` and
`hydra.run.dir`. Each job stores `status.json` and separate `attempt_0001`, etc.,
with `terminal.log` and normal workflow artifacts. Repeating `--run` skips
successful jobs and retries failed/incomplete jobs in a new attempt directory.
A nonzero command exit or missing successful run provenance marks a job failed.

Job IDs incorporate configuration, non-ignored source files and explicitly
overridden input artifacts. Use the optional `inputs` list for dependencies
selected indirectly inside configuration; these paths are relative to the
matrix file. Input file changes therefore produce new jobs, not stale resumes.
Do not edit source during an active matrix. Source changes between jobs require
replanning. Keep output directories under ignored `runs/` or outside the source
tree so generated artifacts do not change the source identity.

This is a sequential local runner for POSIX systems. File locks prevent two
launchers from executing the same job simultaneously. It is not a cluster
scheduler and does not claim an interrupted training job can resume optimizer
state; retries execute a fresh workflow unless that workflow implements resume.

## 4. Statistical analysis

Create an explicit JSON recipe selecting completed evaluation directories:

```json
{
  "schema_version": 1,
  "metric": "total_reward",
  "confidence": 0.95,
  "bootstrap_samples": 10000,
  "bootstrap_seed": 0,
  "expected_methods": ["baseline", "proposed"],
  "expected_training_seeds": [0, 1, 2],
  "paired_baseline": "baseline",
  "runs": [
    "../../runs/baseline_seed0", "../../runs/baseline_seed1", "../../runs/baseline_seed2",
    "../../runs/proposed_seed0", "../../runs/proposed_seed1", "../../runs/proposed_seed2"
  ]
}
```

Run paths resolve relative to the recipe. Metrics may be `total_reward`,
`success`, `steps`, or `metric/<name>` for final scalar task metrics.

```bash
uv run --extra publication python -m tools.analyze_results \
  configs/publication/paper_results.json --output runs/paper_analysis
```

Analysis first averages evaluation episodes within each independently trained
policy. It then uses SciPy's percentile bootstrap on those run means, with equal
weight per training run. Individual run scores and episode counts are retained.
It never treats all episodes as independent training repetitions. Intervals
describe variation over sampled training runs, conditional on the evaluation
protocol/trial set; this is not a hierarchical bootstrap over both levels.

At least two independent runs are required per method/task/condition/split.
More independent runs are generally needed for useful estimates; a narrow
interval from a small sample is not evidence that the method is stable.
Repeated seed labels or identical checkpoint hashes within a method are rejected.
Changing labels cannot establish independence: experiment design remains the
researcher's responsibility. For deterministic, non-trained baselines or
independent real-robot trials, use a separately specified sampling design instead
of inventing training seeds to satisfy this run-level analysis.

Enable `paired_baseline` only for an intentionally paired design. It computes
method-minus-baseline differences for matched training seeds and trial sets;
matching numeric seeds alone does not justify pairing. No p-values or automatic
"best method" declarations are generated. Tasks are reported separately without
averaging incompatible reward scales. `expected_methods` and
`expected_training_seeds` detect incomplete comparison groups when supplied.

The tool rejects failed, incomplete, modified, or duplicated runs, incompatible
protocols/settings, missing metrics, and incomplete trial budgets. It writes
`analysis.json` and a resolved `recipe.json` containing the source selection.

## 5. Figures and tables

```bash
uv run --extra publication python -m tools.publication_figures \
  runs/paper_analysis/analysis.json --output runs/paper_figures --width 3.5
```

Outputs include one comparison panel per task/condition/split as PDF, SVG and
PNG, a LaTeX table, individual run data as CSV, and `figure_manifest.json` with
source run IDs, analysis hash, confidence level, style settings and artifact
hashes. Gray points are individual run means, red diamonds are method means,
and teal lines are confidence intervals. Width is specified in inches; choose
the dimensions required by your paper layout. LaTeX labels are escaped and
figures do not require an installed LaTeX engine.

This is a comparison-figure foundation. Task-specific learning curves, contact
plots, trajectories and ablation layouts belong in project plotting recipes.

## 6. Reproduction bundles

```bash
uv run --extra publication python -m tools.bundle_publication \
  runs/paper_analysis --output runs/paper_bundle
cd runs/paper_bundle
uv sync --frozen --extra publication
uv run --extra publication python reproduce.py
```

Bundles include selected results, configurations, protocols, run provenance,
source patches/untracked snapshots, analysis recipe, analysis/plotting code,
dependency lockfile and checksums. Source results must still match the hashes
recorded during analysis. The bundle is relocatable; `reproduce.py` verifies
checksums and regenerates figures/tables into `reproduced/`. Existing bundle
directories are never overwritten.

This reproduces analysis from recorded results, not training or robot trials.
Large datasets and model weights are identified by their provenance hashes and
are not embedded. Publishing a full experimental artifact also requires making
those assets and the referenced source commit available. A bundle contains local
paths, source changes and system metadata; review it as part of artifact release.

## Verification

The test suite exercises protocol leakage checks, run-level and paired
statistics on explicitly synthetic fixtures, source/input hashes, matrix
expansion/retries, plot/table export, bundle relocation and tamper detection.
The Cartpole matrix above exercises real simulation and provenance without
requiring a trained policy. It intentionally does not invent independent
training runs for its zero-action baseline.
