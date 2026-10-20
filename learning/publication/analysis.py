"""Run-level uncertainty and explicitly paired comparisons from episode files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from learning.publication.io import (
    digest,
    identity,
    read_json,
    relative_to_file,
    write_json,
)


def interval(
    values: list[float], *, confidence: float, samples: int, seed: int
) -> dict:
    from scipy.stats import bootstrap

    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(
            "Confidence intervals require at least two finite independent runs."
        )
    if not 0 < confidence < 1 or samples < 100:
        raise ValueError(
            "Require 0 < confidence < 1 and at least 100 bootstrap samples."
        )
    result = bootstrap(
        (array,),
        np.mean,
        confidence_level=confidence,
        n_resamples=samples,
        method="percentile",
        rng=np.random.default_rng(seed),
    )
    return {
        "mean": float(array.mean()),
        "low": float(result.confidence_interval.low),
        "high": float(result.confidence_interval.high),
        "runs": len(array),
    }


def analyze(recipe_path: Path, output: Path) -> dict:
    recipe = read_json(recipe_path)
    if recipe.get("schema_version") != 1:
        raise ValueError("Analysis recipe schema_version must be 1.")
    metric = recipe.get("metric", "total_reward")
    groups = defaultdict(list)
    sources = []
    seen = set()
    for entry in recipe["runs"]:
        directory = relative_to_file(entry, recipe_path)
        provenance = read_json(directory / "provenance.json")
        context = read_json(directory / "evaluation_context.json")
        episodes = read_json(directory / "episodes.json")
        run_id = provenance["run_id"]
        if provenance["status"] != "succeeded" or run_id in seen:
            raise ValueError("Analysis requires distinct, successfully completed runs.")
        for name in ("episodes.json", "evaluation_context.json", "protocol.json"):
            if provenance.get("outputs", {}).get(name) != digest(directory / name):
                raise ValueError(
                    f"Run output changed after completion: {directory / name}"
                )
        seen.add(run_id)
        for entry in provenance.get("source", {}).get("untracked", []):
            if entry.get("relative_path") and entry.get("sha256"):
                snapshot = directory / "source_untracked" / entry["relative_path"]
                if digest(snapshot) != entry["sha256"]:
                    raise ValueError(
                        "Untracked source snapshot changed after execution."
                    )
        if (directory / "resolved_config.yaml").exists():
            if digest(directory / "resolved_config.yaml") != provenance.get(
                "config_sha256"
            ):
                raise ValueError("Resolved configuration changed after execution.")
        if (directory / "source.patch").exists():
            if digest(directory / "source.patch") != provenance.get("source", {}).get(
                "patch_sha256"
            ):
                raise ValueError("Source patch changed after execution.")
        if context["training_seed"] is None or context["method"] == "unspecified":
            raise ValueError(
                "Each run must declare method and independent training_seed."
            )
        if not context.get("protocol_sha256"):
            raise ValueError("Publication comparisons require a versioned protocol.")
        if (
            identity(read_json(directory / "protocol.json"))
            != context["protocol_sha256"]
        ):
            raise ValueError("Run protocol does not match its evaluation context.")
        if [e["seed"] for e in episodes] != context["seeds"]:
            raise ValueError("Episode results do not match the declared trial budget.")
        scores = []
        for episode in episodes:
            value = (
                episode["metrics"].get(metric.removeprefix("metric/"))
                if metric.startswith("metric/")
                else episode.get(metric)
            )
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"Missing/nonfinite metric {metric} in {directory}")
            scores.append(float(value))
        if not scores:
            raise ValueError("Empty episode results.")
        key = (
            context["task"],
            context["condition"],
            context["split"],
            context["method"],
        )
        groups[key].append(
            {
                "training_seed": context["training_seed"],
                "score": float(np.mean(scores)),
                "episodes": len(scores),
                "protocol": context["protocol_sha256"],
                "run_id": run_id,
                "evaluation_seeds": context["seeds"],
                "evaluation_signature": context.get("evaluation_signature"),
                "checkpoint": provenance.get("inputs", {}).get("checkpoint"),
            }
        )
        artifact_names = [
            "episodes.json",
            "evaluation_context.json",
            "provenance.json",
            "protocol.json",
            "resolved_config.yaml",
            "evaluation_config.json",
            "policy_manifest.json",
            "source.patch",
        ]
        artifact_names.extend(
            str(p.relative_to(directory))
            for p in sorted((directory / "source_untracked").rglob("*"))
            if p.is_file()
        )
        sources.append(
            {
                "run_id": run_id,
                "directory": str(directory),
                "files": {
                    name: digest(directory / name)
                    for name in artifact_names
                    if (directory / name).is_file()
                },
            }
        )
    settings = {
        "confidence": recipe.get("confidence", 0.95),
        "samples": recipe.get("bootstrap_samples", 10000),
        "seed": recipe.get("bootstrap_seed", 0),
    }
    summaries = []
    protocols = defaultdict(set)
    signatures = defaultdict(set)
    for key, runs in sorted(groups.items()):
        expected_seeds = recipe.get("expected_training_seeds")
        if expected_seeds is not None and {r["training_seed"] for r in runs} != set(
            expected_seeds
        ):
            raise ValueError("A method is missing expected independent training seeds.")
        if len({r["training_seed"] for r in runs}) != len(runs):
            raise ValueError(
                "Repeated training seeds cannot count as independent runs."
            )
        checkpoints = [r["checkpoint"] for r in runs if r["checkpoint"]]
        checkpoint_ids = [
            c.get("sha256") or identity(c.get("files")) for c in checkpoints
        ]
        if len(set(checkpoint_ids)) != len(checkpoint_ids):
            raise ValueError("Repeated checkpoints cannot count as independent runs.")
        signatures[key[:3]].update(r["evaluation_signature"] for r in runs)
        if None in signatures[key[:3]] or len(signatures[key[:3]]) != 1:
            raise ValueError(
                "Compared runs must have matching effective evaluation settings."
            )
        protocols[key[:3]].update(r["protocol"] for r in runs)
        if len(protocols[key[:3]]) != 1:
            raise ValueError("Compared methods must use the same evaluation protocol.")
        summaries.append(
            {
                "task": key[0],
                "condition": key[1],
                "split": key[2],
                "method": key[3],
                "individual_runs": runs,
                **interval([r["score"] for r in runs], **settings),
            }
        )
    comparisons = []
    expected_methods = recipe.get("expected_methods")
    if expected_methods is not None:
        for cell in {key[:3] for key in groups}:
            if {key[3] for key in groups if key[:3] == cell} != set(expected_methods):
                raise ValueError("An evaluation cell is missing expected methods.")
    baseline = recipe.get("paired_baseline")
    if baseline:
        for key, runs in sorted(groups.items()):
            if key[3] == baseline:
                continue
            reference = groups.get((*key[:3], baseline))
            if not reference:
                raise ValueError(f"Missing paired baseline for {key[:3]}")
            left = {r["training_seed"]: r for r in runs}
            right = {r["training_seed"]: r for r in reference}
            if left.keys() != right.keys():
                raise ValueError(
                    "Paired comparisons require identical training seed sets."
                )
            if any(
                left[s]["evaluation_seeds"] != right[s]["evaluation_seeds"]
                for s in left
            ):
                raise ValueError(
                    "Paired comparisons require matching evaluation trials."
                )
            comparisons.append(
                {
                    "task": key[0],
                    "condition": key[1],
                    "split": key[2],
                    "method": key[3],
                    "baseline": baseline,
                    **interval(
                        [left[s]["score"] - right[s]["score"] for s in sorted(left)],
                        **settings,
                    ),
                }
            )
    if not summaries:
        raise ValueError("Recipe contains no runs.")
    report = {
        "schema_version": 1,
        "metric": metric,
        "settings": settings,
        "sampling_unit": "independent training run; episodes averaged within run",
        "interval_method": "percentile bootstrap of run means",
        "groups": summaries,
        "paired_differences": comparisons,
        "sources": sources,
    }
    write_json(output / "analysis.json", report)
    write_json(
        output / "recipe.json", {**recipe, "runs": [s["directory"] for s in sources]}
    )
    return report
