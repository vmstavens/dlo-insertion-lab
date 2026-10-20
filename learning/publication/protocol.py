"""Frozen evaluation conditions and disjoint validation/test trial sets."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

from learning.publication.io import identity, read_json, write_json


def apply_protocol(config: dict, output: Path) -> dict:
    config = deepcopy(config)
    options = config["evaluation"]
    path = options.get("protocol")
    if not path:
        return config
    protocol = read_json(Path(path).expanduser())
    if protocol.get("schema_version") != 1 or not protocol.get("id"):
        raise ValueError("Protocol requires schema_version=1 and an id.")
    splits = protocol["splits"]
    for split in ("validation", "test"):
        seeds = splits[split]
        if not seeds or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(
                "Protocol seeds must be nonempty nonnegative integer lists."
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("Protocol seeds must be unique within each split.")
    if set(splits["validation"]) & set(splits["test"]):
        raise ValueError("Validation and test seeds must be disjoint.")
    if not protocol.get("checkpoint_selection"):
        raise ValueError("Declare the checkpoint selection rule before evaluation.")
    if type(protocol.get("max_steps")) is not int or protocol["max_steps"] < 1:
        raise ValueError("Protocol max_steps must be a positive integer.")
    if type(protocol.get("deterministic")) is not bool:
        raise ValueError("Protocol deterministic must be a boolean.")
    split = options.get("split", "test")
    condition = options.get("condition", "nominal")
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test.")
    if condition not in protocol["conditions"]:
        raise ValueError(f"Unknown protocol condition: {condition}")
    if protocol.get("environment") != config["task"]["env"]["registry_name"]:
        raise ValueError("Protocol environment does not match configured task.")
    overrides = protocol["conditions"][condition]
    if "registry_name" in overrides:
        raise ValueError("Conditions cannot change the protocol environment.")
    config["task"]["env"].update(overrides)
    options.update(
        seeds=splits[split],
        max_steps=protocol["max_steps"],
        deterministic=protocol["deterministic"],
    )
    success = protocol.get("success")
    if success is not None:
        if success.get("mode") not in {"any", "final"}:
            raise ValueError("Success mode must be any or final.")
        if success.get("operator") not in {">=", "<=", ">", "<"}:
            raise ValueError("Unsupported success threshold operator.")
        if not isinstance(success.get("metric"), str):
            raise ValueError("Success requires a metric name.")
        if not isinstance(success["threshold"], (int, float)) or not math.isfinite(
            success["threshold"]
        ):
            raise ValueError("Success threshold must be a finite number.")
    options["success_rule"] = success
    options["protocol_sha256"] = identity(protocol)
    write_json(output / "protocol.json", protocol)
    write_json(output / "evaluation_config.json", config)
    return config
