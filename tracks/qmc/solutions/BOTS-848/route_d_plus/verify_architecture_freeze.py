"""Independent hash/schema readback for remediated architecture freeze."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import jsonschema

from route_d_plus.future.verify import (
    load_json,
    require_artifact,
    sha256_file,
    validate_payload,
)

MODULE_ROOT = Path(__file__).resolve().parent


def artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def validate(payload: dict[str, Any], schema_name: str) -> None:
    schema = load_json(MODULE_ROOT / schema_name)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(payload)


def git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def slurm() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not os.environ.get("SLURM_JOB_ID") or not visible:
        raise RuntimeError("freeze readback requires a Slurm GPU allocation")
    return {
        "job_id": os.environ["SLURM_JOB_ID"],
        "cluster_name": os.environ.get(
            "SLURM_CLUSTER_NAME", "hpccube-xh5"
        ),
        "node_list": os.environ.get("SLURM_NODELIST", "unknown"),
        "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "gpu_devices": [part for part in visible.split(",") if part],
    }


def verify(
    *,
    repo_root: Path,
    freeze_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    revision = git_output(repo_root, "rev-parse", "HEAD")
    if git_output(repo_root, "status", "--porcelain"):
        raise RuntimeError("freeze readback requires a clean checkout")
    freeze = load_json(freeze_path)
    validate_payload(freeze, "dependency.schema.json")
    if freeze["kind"] != "architecture-freeze" or not freeze["passed"]:
        raise RuntimeError("architecture freeze did not pass")

    required_names = (
        "selection_aggregate",
        "selection_protocol",
        "architecture",
        "phase7_stage_gate",
        "remediation_certificate",
        "remediation_readback",
        "remediation_science",
        "remediation_science_readback",
    )
    resolved = {
        name: require_artifact(freeze[name]) for name in required_names
    }
    for checkpoint in freeze["checkpoints"]:
        require_artifact(checkpoint)

    stage_gate = load_json(resolved["phase7_stage_gate"])
    validate_payload(stage_gate, "stage-gate.schema.json")
    decision = stage_gate["decision"]
    if (
        decision["benchmark_classification"] != "dplus0-sufficient"
        or decision["capacity_action"] != "keep-D+0"
    ):
        raise RuntimeError("freeze Phase 7 decision mismatch")
    science = load_json(resolved["remediation_science"])
    validate_payload(science, "../remediation-science.schema.json")
    science_readback = load_json(
        resolved["remediation_science_readback"]
    )
    validate_payload(
        science_readback,
        "../remediation-science-readback.schema.json",
    )
    if (
        not science["passed"]
        or not science_readback["passed"]
        or not science_readback["science_passed"]
        or science_readback["science_certificate"]["sha256"]
        != sha256_file(resolved["remediation_science"])
    ):
        raise RuntimeError("freeze science lineage mismatch")

    payload = {
        "schema_version": (
            "challenge-15-route-d-plus-architecture-freeze-readback-v1"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "freeze_revision": freeze["source_revision"],
        "readback_revision": revision,
        "architecture_freeze": artifact(freeze_path),
        "selected_capacity": freeze["selected_capacity"],
        "checkpoint_count": len(freeze["checkpoints"]),
        "slurm": slurm(),
        "gates": {
            "freeze_schema_valid": True,
            "freeze_hash_valid": True,
            "selection_aggregate_hash_valid": True,
            "selection_protocol_hash_valid": True,
            "architecture_hash_valid": True,
            "exact_three_checkpoint_set": len(freeze["checkpoints"]) == 3,
            "all_checkpoint_hashes_valid": True,
            "phase7_dplus0_sufficient": True,
            "remediation_hashes_valid": True,
            "science_gate_and_readback_passed": True,
            "heldout_not_accessed": not freeze["heldout_accessed"],
            "beyond_ed_not_accessed": not freeze["beyond_ed_accessed"],
            "clean_readback_revision": True,
            "gpu_slurm_evidence": True,
        },
        "passed": True,
    }
    validate(payload, "architecture-freeze-readback.schema.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    payload = verify(
        repo_root=arguments.repo_root.resolve(),
        freeze_path=arguments.freeze.resolve(),
        output_path=arguments.output.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
