# Local run artifacts

Hydra writes each workflow invocation beneath this directory:

```text
runs/<task>/<stage>/<date>/<time>/
├── .hydra/
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── resolved_config.yaml
├── checkpoints/
├── metrics.*
└── videos/
```

Multiruns use `runs/<task>/<stage>/sweeps/<timestamp>/<job-number>/`.

Run artifacts are ignored by Git. Configuration source files belong in
`configs/`; model checkpoints, logs, videos, and resolved configurations
belong here or in an external experiment tracker/artifact store.

The workflow runtime always writes `resolved_config.yaml` before doing work.
This captures interpolation results and gives each artifact a reproducible
configuration record.
