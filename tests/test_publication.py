from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from learning.publication.analysis import analyze, interval
from learning.publication.bundle import create_bundle, reproduce_bundle
from learning.publication.figures import render_figures
from learning.publication.io import digest, identity, read_json, write_json
from learning.publication.matrix import execute_matrix, plan_matrix
from learning.publication.protocol import apply_protocol
from learning.publication.provenance import fingerprint, finish_run, start_run


def protocol():
    return {
        "schema_version": 1,
        "id": "test-v1",
        "environment": "TestEnv",
        "splits": {"validation": [1], "test": [10, 11]},
        "conditions": {"nominal": {}},
        "max_steps": 5,
        "deterministic": True,
        "success": None,
        "checkpoint_selection": "fixed before testing",
    }


def make_run(root, method, seed, score):
    directory = root / f"{method}_{seed}"
    directory.mkdir()
    write_json(directory / "protocol.json", protocol())
    write_json(
        directory / "episodes.json",
        [
            {
                "seed": trial,
                "total_reward": score + offset,
                "success": offset > 0,
                "metrics": {},
                "steps": 5,
            }
            for trial, offset in [(10, -1), (11, 1)]
        ],
    )
    write_json(
        directory / "evaluation_context.json",
        {
            "task": "Synthetic_fixture",
            "condition": "nominal",
            "split": "test",
            "method": method,
            "training_seed": seed,
            "seeds": [10, 11],
            "protocol_sha256": identity(protocol()),
            "evaluation_signature": "same-settings",
        },
    )
    finish_run(
        directory,
        {
            "run_id": f"{method}_{seed}",
            "inputs": {"checkpoint": {"sha256": f"unique-{method}-{seed}"}},
        },
        "succeeded",
    )
    return directory


@pytest.fixture
def recipe(tmp_path):
    runs = [
        make_run(tmp_path, method, seed, seed + offset)
        for method, offset in [("baseline", 1), ("candidate", 3)]
        for seed in range(3)
    ]
    path = tmp_path / "recipe.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "metric": "total_reward",
            "runs": [str(p) for p in runs],
            "bootstrap_samples": 500,
            "paired_baseline": "baseline",
        },
    )
    return path


def test_run_level_statistics_and_paired_difference(recipe, tmp_path):
    report = analyze(recipe, tmp_path / "analysis")
    assert [g["mean"] for g in report["groups"]] == [2, 4]
    assert all(g["runs"] == 3 for g in report["groups"])
    difference = report["paired_differences"][0]
    assert difference["mean"] == difference["low"] == difference["high"] == 2
    assert interval([1, 2, 3], confidence=0.95, samples=500, seed=1) == interval(
        [1, 2, 3], confidence=0.95, samples=500, seed=1
    )


def test_analysis_rejects_modified_results(recipe, tmp_path):
    run = Path(read_json(recipe)["runs"][0])
    (run / "episodes.json").write_text("[]")
    with pytest.raises(ValueError, match="changed after completion"):
        analyze(recipe, tmp_path / "analysis")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("training_seed", 1, "Repeated training seeds"),
        ("evaluation_signature", "changed", "matching effective"),
    ],
)
def test_analysis_rejects_incompatible_or_duplicate_runs(
    recipe, tmp_path, field, value, message
):
    run = Path(read_json(recipe)["runs"][0])
    context = read_json(run / "evaluation_context.json")
    context[field] = value
    write_json(run / "evaluation_context.json", context)
    finish_run(run, read_json(run / "provenance.json"), "succeeded")
    with pytest.raises(ValueError, match=message):
        analyze(recipe, tmp_path / "analysis")


def test_single_run_is_not_many_independent_episodes():
    with pytest.raises(ValueError, match="two finite independent"):
        interval([5], confidence=0.95, samples=500, seed=0)


def test_protocol_applies_conditions_and_rejects_leakage(tmp_path):
    path = tmp_path / "protocol.json"
    spec = protocol()
    spec["conditions"]["nominal"] = {"noise": 0.1}
    write_json(path, spec)
    config = {
        "task": {"env": {"registry_name": "TestEnv"}},
        "evaluation": {"protocol": str(path), "max_steps": 999},
    }
    effective = apply_protocol(config, tmp_path / "run")
    assert effective["evaluation"]["seeds"] == [10, 11]
    assert effective["evaluation"]["max_steps"] == 5
    assert effective["task"]["env"]["noise"] == 0.1
    assert config["evaluation"]["max_steps"] == 999
    spec["splits"]["validation"] = [10]
    write_json(path, spec)
    with pytest.raises(ValueError, match="disjoint"):
        apply_protocol(config, tmp_path / "run")


def test_provenance_hashes_inputs_and_records_terminal_status(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text("seed: 3\n")
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"weights")
    manifest = tmp_path / "policy.json"
    write_json(manifest, {"checkpoint": "weights.bin"})
    record = start_run(
        {"seed": 3, "evaluation": {"policy_manifest": str(manifest)}}, tmp_path
    )
    assert record["inputs"]["checkpoint"]["sha256"] == digest(checkpoint)
    assert record["config_sha256"] == digest(tmp_path / "resolved_config.yaml")
    finish_run(tmp_path, record, "failed", "RuntimeError")
    assert read_json(tmp_path / "provenance.json")["status"] == "failed"
    with pytest.raises(FileExistsError):
        start_run({}, tmp_path)
    assert fingerprint(tmp_path / "missing")["missing"]


def test_bundle_relocates_reproduces_and_detects_tampering(recipe, tmp_path):
    pytest.importorskip("matplotlib")
    analysis = tmp_path / "analysis"
    analyze(recipe, analysis)
    render_figures(analysis / "analysis.json", tmp_path / "figures")
    assert (tmp_path / "figures/results.tex").is_file()
    assert (tmp_path / "figures/comparison_00.pdf").stat().st_size > 1000
    bundle = tmp_path / "bundle"
    create_bundle(analysis, bundle)
    moved = tmp_path / "moved"
    shutil.move(bundle, moved)
    reproduce_bundle(moved)
    subprocess.run(
        [sys.executable, str(moved / "reproduce.py")], cwd=tmp_path, check=True
    )
    reproduced = read_json(moved / "reproduced/analysis.json")
    assert reproduced["groups"] == read_json(analysis / "analysis.json")["groups"]
    (moved / "runs/run_0000/episodes.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        reproduce_bundle(moved)


def test_matrix_expansion_resume_and_failed_attempt(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "source.py").write_text("# source\n")
    spec = tmp_path / "matrix.json"
    write_json(
        spec,
        {
            "schema_version": 1,
            "project_root": "project",
            "output_root": "jobs",
            "workflow": "workflows.evaluate",
            "axes": {"seed": [0, 1]},
            "bindings": {"evaluation.method": "method_${seed}"},
        },
    )
    jobs = plan_matrix(spec)
    assert len(jobs) == 2
    assert not (tmp_path / "jobs").exists()
    assert 'evaluation.method="method_0"' in jobs[0]["overrides"]
    calls = []
    original_run = subprocess.run

    def run(command, **kwargs):
        if command[0] == "git":
            return original_run(command, **kwargs)
        output = Path(command[-1].split("=", 1)[1])
        calls.append(output)
        write_json(output / "provenance.json", {"status": "succeeded"})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("learning.publication.matrix.subprocess.run", run)
    assert execute_matrix(jobs)
    assert execute_matrix(jobs)
    assert len(calls) == 2
    status_path = Path(jobs[0]["directory"]) / "status.json"
    status = read_json(status_path)
    status["status"] = "failed"
    write_json(status_path, status)
    assert execute_matrix(jobs)
    assert calls[-1].name == "attempt_0002"
    (root / "source.py").write_text("# changed\n")
    assert plan_matrix(spec)[0]["id"] != jobs[0]["id"]
