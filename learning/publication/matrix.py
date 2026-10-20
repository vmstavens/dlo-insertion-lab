"""Previewable local experiment matrices with resumable, isolated attempts."""

from __future__ import annotations

import fcntl
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from string import Template

from learning.publication.io import (
    identity,
    read_json,
    relative_to_file,
    write_json,
)
from learning.publication.provenance import fingerprint, source_identity


def plan_matrix(path: Path) -> list[dict]:
    spec = read_json(path)
    if spec.get("schema_version") != 1:
        raise ValueError("Matrix schema_version must be 1.")
    workflow = spec["workflow"]
    if not re.fullmatch(r"workflows\.[a-zA-Z_][a-zA-Z_0-9.]*", workflow):
        raise ValueError("Matrix workflow must be an importable workflows module.")
    root = relative_to_file(spec["project_root"], path)
    output = relative_to_file(spec["output_root"], path)
    source_hash = source_identity(root)
    axes = spec.get("axes", {})
    inputs = [
        fingerprint(relative_to_file(name, path)) for name in spec.get("inputs", [])
    ]
    if any(not isinstance(values, list) or not values for values in axes.values()):
        raise ValueError("Every matrix axis must be a nonempty list.")
    jobs = []
    for values in itertools.product(*axes.values()):
        choices = dict(zip(axes, values, strict=True))
        overrides = [*spec.get("overrides", [])]
        for key, value in choices.items():
            overrides.append(f"{key}={json.dumps(value)}")
        variables = {
            key.replace(".", "_"): str(value) for key, value in choices.items()
        }
        if len(variables) != len(choices):
            raise ValueError(
                "Matrix axis names collide after replacing dots with underscores."
            )
        for key, template in spec.get("bindings", {}).items():
            overrides.append(
                f"{key}={json.dumps(Template(template).substitute(variables))}"
            )
        job_inputs = list(inputs)
        for override in overrides:
            key, _, value = override.partition("=")
            if key.lstrip("+") in {
                "evaluation.policy_manifest",
                "evaluation.protocol",
                "il.checkpoint",
                "dataset.path",
                "dataset.manifest",
            }:
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
                if isinstance(value, str):
                    artifact = Path(value).expanduser()
                    if not artifact.is_absolute():
                        artifact = root / artifact
                    job_inputs.append(fingerprint(artifact))
                    if (
                        key.lstrip("+") == "evaluation.policy_manifest"
                        and artifact.is_file()
                    ):
                        job_inputs.append(
                            fingerprint(
                                relative_to_file(
                                    read_json(artifact)["checkpoint"], artifact
                                )
                            )
                        )
        if any(
            o.split("=", 1)[0].lstrip("+") in {"hydra.run.dir", "runtime.dry_run"}
            for o in overrides
        ):
            raise ValueError("The matrix owns hydra.run.dir and runtime.dry_run.")
        job_id = identity(
            {
                "workflow": workflow,
                "overrides": overrides,
                "source": source_hash,
                "inputs": job_inputs,
            }
        )[:20]
        jobs.append(
            {
                "id": job_id,
                "workflow": workflow,
                "overrides": overrides,
                "choices": choices,
                "root": str(root),
                "source_sha256": source_hash,
                "directory": str(output / job_id),
                "inputs": job_inputs,
            }
        )
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Matrix contains duplicate jobs.")
    return jobs


def job_status(job: dict) -> str:
    path = Path(job["directory"]) / "status.json"
    if not path.exists():
        return "missing"
    status = read_json(path)
    if status["status"] == "succeeded":
        provenance = Path(status["output"]) / "provenance.json"
        if not provenance.exists() or read_json(provenance)["status"] != "succeeded":
            return "incomplete"
    return status["status"]


def execute_matrix(jobs: list[dict]) -> bool:
    all_succeeded = True
    for job in jobs:
        directory = Path(job["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "execution.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"{job['id']}: already running", flush=True)
                all_succeeded = False
                continue
            if job_status(job) == "succeeded":
                print(f"{job['id']}: already succeeded", flush=True)
                continue
            if source_identity(Path(job["root"])) != job["source_sha256"]:
                raise RuntimeError(
                    "Project files changed since planning; re-run the matrix command."
                )
            attempts = sorted(directory.glob("attempt_*"))
            output = directory / f"attempt_{len(attempts) + 1:04d}"
            output.mkdir()
            command = [
                sys.executable,
                "-m",
                job["workflow"],
                *job["overrides"],
                "runtime.dry_run=false",
                f"hydra.run.dir={output}",
            ]
            status = {
                **job,
                "status": "running",
                "output": str(output),
                "command": command,
            }
            write_json(directory / "status.json", status)
            print(f"{job['id']}: running -> {output}", flush=True)
            try:
                with (output / "terminal.log").open("w") as log:
                    result = subprocess.run(
                        command,
                        cwd=job["root"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                provenance = output / "provenance.json"
                complete = (
                    result.returncode == 0
                    and provenance.exists()
                    and read_json(provenance)["status"] == "succeeded"
                )
                status.update(
                    status="succeeded" if complete else "failed",
                    returncode=result.returncode,
                )
            except BaseException:
                status["status"] = "interrupted"
                write_json(directory / "status.json", status)
                raise
            write_json(directory / "status.json", status)
            print(f"{job['id']}: {status['status']}", flush=True)
            all_succeeded &= status["status"] == "succeeded"
    return all_succeeded
