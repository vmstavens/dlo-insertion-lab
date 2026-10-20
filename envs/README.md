# Environments

Environment packages follow the same broad organization as MuJoCo Playground:
the environment class owns the MJX `reset`/`step` lifecycle, while model
constants and optional utilities live in focused sibling modules. The central
`envs.registry` is an adapter over MuJoCo Playground's registry. It registers
project environments with Playground's manipulation suite and keeps only
project metadata and randomizers in a local sidecar.

`grasped_end_dlo/` is organized as follows:

| Module | Responsibility | Why it is separate |
| --- | --- | --- |
| `base.py` | Defines the abstract RL `reset`, `step`, termination, and rendering interface shared by grasped-end tasks. | Common lifecycle behavior is implemented once without choosing a concrete task model. |
| `config.py` | Creates a new mutable configuration for each environment. | Experiments can override parameters without sharing accidental state. |
| `constants.py` | Defines nominal physical values, randomization ranges, reset bounds, and the observation layout. | One source of truth prevents mismatches between model building, training, and evaluation. |
| `dlo.py` | Finds cable joints and applies randomized initial curvature to MJX state. | Initialization math stays independent of episode control flow. |
| `observations.py` | Converts keypoint/target poses into the six-value policy observation and computes pose errors. It supports both MJX/JAX and CPU MuJoCo/NumPy. | Training and demonstration/evaluation code use identical observation semantics. |
| `rewards.py` | Contains pure sparse and dense reward functions plus stable metric names. | Reward changes can be unit-tested without constructing or simulating a model. |
| `viewer.py` | Owns the passive MuJoCo viewer and copies device-side MJX state back to CPU MuJoCo data. | GUI state and host/device synchronization stay out of the JIT-friendly RL path. |
| `__init__.py` | Exposes the public package API and family-level registry helpers. | Callers do not need to depend on the internal file layout. |
| `insert/task.py` | Implements the concrete `GraspedEndDloInsert` task and its pipe-specific shaping condition. | Task semantics do not leak into the abstract family base. |
| `insert/model.py` | Builds the procedural cable, pipe, mocap body, sites, and equality constraint. | Model construction can evolve without growing the RL lifecycle class. |
| `insert/config.py` | Extends the shared configuration with insertion reward settings. | Task defaults remain close to the task that consumes them. |
| `insert/keyframes.py` | Loads and validates versioned keyframe assets before adding them to `MjSpec`. | Authoring files are checked once outside JIT and become native MuJoCo model data. |
| `insert/assets/keyframes/init.json` | Stores the nominal stationary DLO and mocap pose. | Large model state does not live in experiment configuration or Python source. |
| `insert/pipe.py` | Computes whether a point/site is inside the pipe clearance. | Pipe-specific geometry stays inside the insertion task. |
| `insert/randomize.py` | Samples cable/pipe physics and geometry for vectorized training. | Insertion-specific domain randomization is independently configurable. |

The important boundary is:

```text
trainer / evaluator
   |                    |
   v                    v
insert/task.py       insert/randomize.py
   |
   v
base.py
(shared reset/step)
   |    |
   v    v
observations.py  rewards.py
   \    /
    v  v
   MJX state  --------> viewer.py --------> CPU MuJoCo window
```

Not every module participates in every step. Normal training primarily calls
the concrete task through `base.py`; the base delegates numerical details to
the small pure helpers.
The viewer is not involved in headless training, and randomization is not
involved unless a trainer explicitly enables it.

Registered models can be inspected with the task-agnostic viewer tool:

```bash
# Inspect the initial model without advancing time.
uv run python -m tools.view_env GraspedEndDloInsert

# Validate CPU MuJoCo physics, constraints, contacts, and passive settling.
uv run python -m tools.view_env GraspedEndDloInsert --mode cpu

# Exercise reset(), step(), termination, and MJX-to-viewer synchronization.
uv run python -m tools.view_env GraspedEndDloInsert --mode mjx
```

Close the window or press Ctrl-C to exit. `launch_viewer()` shows the initial
CPU model quickly; `render(state)` should be used when inspecting an actual
MJX reset or rollout state. CPU mode calls `mj_step()` with the model timestep.
MJX mode applies zero actions, automatically resets completed episodes, and may
pause while JAX compiles the first reset and step. Both simulation modes batch
physics/control steps to approximately match the requested `--render-fps`.

## Initial keyframes

Concrete tasks select their reset keyframe explicitly in task configuration:

```python
config.reset_keyframe = "init"
```

The insertion model loads `assets/keyframes/init.json`, validates its schema and
finite values, adds it to the procedural `MjSpec`, and compiles it into a native
MuJoCo keyframe. Static and CPU modes restore it with
`mj_resetDataKeyframe()`. The MJX reset path restores the same keyframe before
applying configured reset-position, reset-rotation, and optional curvature
randomization. A missing configured keyframe is an environment-construction
error rather than silently falling back to another state.

`GraspedEndDloBase` cannot be instantiated directly. Concrete tasks implement
the model-building hook and may override task-specific conditions while
inheriting the shared lifecycle and rendering API.

A family with several tasks should use task packages:

```text
grasped_end_dlo/
├── base.py
├── observations.py
├── rewards.py
├── insert/
│   ├── __init__.py
│   ├── task.py
│   ├── model.py
│   ├── config.py
│   ├── keyframes.py
│   ├── assets/
│   │   └── keyframes/
│   │       └── init.json
│   ├── randomize.py
│   └── xmls/
└── align/
    ├── __init__.py
    ├── task.py
    ├── config.py
    ├── randomize.py
    └── xmls/
```

Each concrete task registers itself from its package `__init__.py`:

```python
from envs.grasped_end_dlo import register_environment
from envs.grasped_end_dlo.insert.config import default_config
from envs.grasped_end_dlo.insert.randomize import domain_randomize
from envs.grasped_end_dlo.insert.task import GraspedEndDloInsert

register_environment(
    "GraspedEndDloInsert",
    GraspedEndDloInsert,
    default_config,
    domain_randomizer=domain_randomize,
)
```

Built-in task packages must also be imported by
`envs.grasped_end_dlo.__init__` so their registration runs before a caller
asks the registry to load them. External projects can register tasks during
their own package initialization.

Training code can then remain task-agnostic:

```python
from envs import registry

env = registry.load("GraspedEndDloInsert", config_overrides=overrides)
domain_randomizer = registry.get_domain_randomizer("GraspedEndDloInsert")
```

Because project environments are registered with Playground, established code
can also load them directly:

```python
from mujoco_playground import registry

env = registry.load("GraspedEndDloInsert")
```

Use the project adapter when domain-randomizer lookup or project suite metadata
is needed; Playground does not currently provide a public API for registering
either of those.
