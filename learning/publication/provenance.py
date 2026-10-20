"""Run identities, source changes, environment metadata and input hashes."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from learning.publication.io import (
    digest,
    identity,
    read_json,
    relative_to_file,
    write_json,
)


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=10
    )


def fingerprint(path: Path) -> dict:
    path = path.expanduser().resolve()
    if path.is_file():
        return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
    if path.is_dir():
        return {
            "path": str(path),
            "files": [
                {
                    "path": str(p.relative_to(path)),
                    "sha256": digest(p),
                    "bytes": p.stat().st_size,
                }
                for p in sorted(path.rglob("*"))
                if p.is_file()
            ],
        }
    return {"path": str(path), "missing": True}


def source_identity(root: Path) -> str:
    names = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return identity(
        {
            os.fsdecode(name): digest(root / os.fsdecode(name))
            for name in sorted(set(names.split(b"\0")))
            if name and (root / os.fsdecode(name)).is_file()
        }
    )


def start_run(config: dict, output: Path) -> dict:
    """Capture inputs before execution; do not collect environment variables."""
    root = Path(__file__).resolve().parents[2]
    if (output / "provenance.json").exists():
        raise FileExistsError(f"Run directory already has provenance: {output}")
    source = {"root": str(root)}
    try:
        source["commit"] = _git(root, "rev-parse", "HEAD").decode().strip()
        source["tree_sha256"] = source_identity(root)
        status = _git(root, "status", "--porcelain=v1").decode()
        source["status"] = status
        source["dirty"] = bool(status)
        patch = _git(root, "diff", "HEAD", "--binary", "--no-ext-diff")
        (output / "source.patch").write_bytes(patch)
        source["patch_sha256"] = digest(output / "source.patch")
        names = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
        source["untracked"] = [
            {
                **fingerprint(root / os.fsdecode(name)),
                "relative_path": os.fsdecode(name),
            }
            for name in names.split(b"\0")
            if name
        ]
        for name in names.split(b"\0"):
            if name and (root / os.fsdecode(name)).is_file():
                destination = output / "source_untracked" / os.fsdecode(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / os.fsdecode(name), destination)
    except (OSError, subprocess.SubprocessError) as error:
        source["unavailable"] = str(error)
    inputs = {}
    initial_checkpoint = config.get("il", {}).get("checkpoint")
    if initial_checkpoint:
        inputs["initial_policy_checkpoint"] = fingerprint(Path(initial_checkpoint))
    for name, path in config.get("runtime", {}).get("provenance_inputs", {}).items():
        inputs[f"extra/{name}"] = fingerprint(Path(path))
    evaluation = config.get("evaluation", {})
    manifest = evaluation.get("policy_manifest")
    if manifest:
        path = Path(manifest).expanduser().resolve()
        inputs["policy_manifest"] = fingerprint(path)
        if path.is_file():
            metadata = read_json(path)
            inputs["checkpoint"] = fingerprint(
                relative_to_file(metadata["checkpoint"], path)
            )
    if evaluation.get("protocol"):
        inputs["protocol"] = fingerprint(Path(evaluation["protocol"]))
    dataset = config.get("dataset", {})
    for key in ("manifest", "path"):
        if dataset.get(key):
            inputs[f"dataset_{key}"] = fingerprint(Path(dataset[key]))
    for name in ("pyproject.toml", "uv.lock"):
        if (root / name).is_file():
            inputs[name] = fingerprint(root / name)
    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        hardware["gpus"] = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        hardware["gpus"] = None
    result = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "seed": config.get("seed"),
        "source": source,
        "inputs": inputs,
        "hardware": hardware,
        "python": platform.python_version(),
        "packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
        "config_sha256": digest(output / "resolved_config.yaml"),
    }
    write_json(output / "provenance.json", result)
    return result


def finish_run(
    output: Path, record: dict, status: str, error: str | None = None
) -> None:
    record.update(status=status, finished_at=datetime.now(UTC).isoformat(), error=error)
    record["outputs"] = {
        name: digest(output / name)
        for name in (
            "episodes.json",
            "evaluation_context.json",
            "protocol.json",
            "evaluation_config.json",
        )
        if (output / name).is_file()
    }
    write_json(output / "provenance.json", record)
