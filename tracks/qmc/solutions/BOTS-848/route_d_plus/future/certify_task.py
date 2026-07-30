"""Create a generic Phase 9--11 task envelope after a domain worker exits."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path
from typing import Any

import jsonschema

from route_d_plus.future.verify import (
    git_revision,
    load_json,
    require_artifact,
    require_isolated_path,
    sha256_file,
    validate_dispatch,
    validate_payload,
    validate_task_certificate,
    write_json_atomic,
)

TASK_VERSION = "challenge-15-route-d-plus-future-task-certificate-v1"


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _memory_mb() -> int:
    direct = os.environ.get("SLURM_MEM_PER_NODE")
    if direct:
        return int(direct)
    per_cpu = int(os.environ.get("SLURM_MEM_PER_CPU", "1"))
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    return per_cpu * cpus


def _slurm_elapsed_seconds() -> float:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        raise RuntimeError("task certification requires a Slurm job")
    completed = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%M"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not completed:
        return 0.0
    parts = completed.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return 60.0 * int(minutes) + int(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return 3600.0 * int(hours) + 60.0 * int(minutes) + int(seconds)
    return 0.0


def certify(
    *,
    repo_root: Path,
    dispatch_path: Path,
    task_id: str,
    run_dir: Path,
    domain_path: Path,
    domain_schema_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    checkpoint_path: Path | None,
    extra_input_paths: list[Path],
    started_at_utc: str,
    output_path: Path,
) -> dict[str, Any]:
    dispatch = load_json(dispatch_path)
    validate_dispatch(dispatch)
    revision, dirty = git_revision(repo_root)
    if dirty or revision != dispatch["source_revision"]:
        raise RuntimeError("task certification requires dispatched clean source")
    tasks = {task["task_id"]: task for task in dispatch["tasks"]}
    if task_id not in tasks:
        raise KeyError(f"task is not registered: {task_id}")
    task = tasks[task_id]
    expected_dir = require_isolated_path(
        Path(dispatch["run_root"]), task["run_dir"]
    )
    if run_dir.resolve() != expected_dir:
        raise RuntimeError("task run directory differs from dispatch")

    domain = load_json(domain_path)
    schema = load_json(domain_schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(domain)
    gates = domain.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
        or domain.get("passed") is not True
    ):
        raise RuntimeError("domain certificate did not pass every gate")
    if not set(task["required_gates"]).issubset(gates):
        raise RuntimeError("domain certificate omits dispatched gates")

    for reference in dispatch["prerequisites"]:
        require_artifact(reference)
    dependency_inputs = []
    for dependency_id in task.get("depends_on", []):
        dependency_task = tasks[dependency_id]
        dependency_dir = require_isolated_path(
            Path(dispatch["run_root"]), dependency_task["run_dir"]
        )
        dependency_path = dependency_dir / "task-certificate.json"
        dependency_payload = load_json(dependency_path)
        validate_task_certificate(
            dependency_payload,
            task=dependency_task,
            dispatch=dispatch,
        )
        dependency_inputs.append(_artifact(dependency_path))
    for path in extra_input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (stdout_path, stderr_path, domain_path, domain_schema_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    cluster = os.environ.get("SLURM_CLUSTER_NAME", "")
    if not job_id or not cluster or visible in {"", "-1", "NoDevFiles"}:
        raise RuntimeError("task certification requires Slurm GPU")

    payload = {
        "schema_version": TASK_VERSION,
        "stage": dispatch["stage"],
        "task_id": task_id,
        "kind": task["kind"],
        "run_dir": str(run_dir.resolve()),
        "source_revision": revision,
        "git_dirty": False,
        "started_at_utc": started_at_utc,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_artifacts": [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in dispatch["prerequisites"]
        ]
        + dependency_inputs
        + [_artifact(path) for path in extra_input_paths],
        "slurm": {
            "job_id": job_id,
            "cluster_name": cluster,
            "node_list": os.environ.get("SLURM_NODELIST", "unknown"),
            "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
            "gpu_devices": [
                part for part in visible.split(",") if part
            ],
            "cpus_per_task": int(
                os.environ.get("SLURM_CPUS_PER_TASK", "1")
            ),
            "memory_mb": _memory_mb(),
            "elapsed_seconds": _slurm_elapsed_seconds(),
            "exit_code": 0,
        },
        "logs": {
            "stdout": _artifact(stdout_path),
            "stderr": _artifact(stderr_path),
        },
        "domain_certificate": {
            **_artifact(domain_path),
            "schema_path": str(domain_schema_path.resolve()),
            "schema_sha256": sha256_file(domain_schema_path),
            "schema_valid": True,
        },
        "checkpoint": (
            None if checkpoint_path is None else _artifact(checkpoint_path)
        ),
        "gates": gates,
        "passed": True,
    }
    validate_payload(payload, "task-certificate.schema.json")
    write_json_atomic(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--domain-schema", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--extra-input", type=Path, action="append", default=[])
    parser.add_argument("--started-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    certify(
        repo_root=arguments.repo_root.resolve(),
        dispatch_path=arguments.dispatch.resolve(),
        task_id=arguments.task_id,
        run_dir=arguments.run_dir.resolve(),
        domain_path=arguments.domain.resolve(),
        domain_schema_path=arguments.domain_schema.resolve(),
        stdout_path=arguments.stdout.resolve(),
        stderr_path=arguments.stderr.resolve(),
        checkpoint_path=(
            None
            if arguments.checkpoint is None
            else arguments.checkpoint.resolve()
        ),
        extra_input_paths=[
            path.resolve() for path in arguments.extra_input
        ],
        started_at_utc=arguments.started_at_utc,
        output_path=arguments.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
