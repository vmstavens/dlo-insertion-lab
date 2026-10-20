# Local datasets

This directory is the default storage root for trajectory data. Dataset files
are intentionally ignored by Git; only this contract is versioned.

```text
data/
└── grasped_end_dlo_insert/
    ├── raw/
    │   └── insert_raw_v1/
    │       ├── manifest.json
    │       └── episodes/
    └── processed/
        └── insert_processed_v1/
            ├── manifest.json
            ├── train/
            └── validation/
```

Raw datasets contain lossless collector output. Never modify one in place.
Create a new raw dataset ID for a new collection and retain episode boundaries,
timestamps, observations, actions, termination reasons, and collection
metadata.

Processed datasets are deterministic products of a raw dataset plus a
versioned transform configuration. They may contain normalized arrays,
fixed-length sequence windows, and train/validation splits. A processed
manifest must list its source dataset IDs.

Every dataset has a `manifest.json` matching
`learning.data.schema.DatasetManifest`. The manifest is the boundary that
prevents a policy from silently consuming observations, actions, or control
rates with incompatible semantics.

Large or shared datasets should live in artifact/object storage and be
downloaded into this layout. Do not commit binary trajectories to Git.
