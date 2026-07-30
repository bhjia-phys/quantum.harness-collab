"""Freeze D+0 after remediated Phase 6 statistics and Phase 7 pass."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from route_d_plus.future.verify import (
    load_json,
    require_artifact,
    sha256_file,
    validate_dependency,
    validate_dispatch,
    validate_payload,
)


def artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_commit(repo_root: Path, revision: str) -> None:
    subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def freeze(
    *,
    repo_root: Path,
    science_path: Path,
    science_readback_path: Path,
    phase7_dispatch_path: Path,
    phase7_stage_gate_path: Path,
    phase7_aggregate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    revision = git_output(repo_root, "rev-parse", "HEAD")
    if git_output(repo_root, "status", "--porcelain"):
        raise RuntimeError("architecture freeze requires a clean checkout")

    science = load_json(science_path)
    validate_payload(science, "../remediation-science.schema.json")
    science_readback = load_json(science_readback_path)
    validate_payload(
        science_readback,
        "../remediation-science-readback.schema.json",
    )
    if (
        not science["passed"]
        or not science_readback["passed"]
        or not science_readback["science_passed"]
        or science_readback["science_certificate"]["sha256"]
        != sha256_file(science_path)
    ):
        raise RuntimeError("remediation scientific gate did not pass")

    dispatch = load_json(phase7_dispatch_path)
    validate_dispatch(dispatch)
    require_commit(repo_root, dispatch["source_revision"])
    dependency = validate_dependency(dispatch["prerequisites"][0])
    if dependency["kind"] != "dplus0-remediation-gate":
        raise RuntimeError("freeze requires remediated D+0 reevaluation")

    stage_gate = load_json(phase7_stage_gate_path)
    validate_payload(stage_gate, "stage-gate.schema.json")
    decision = stage_gate["decision"]
    if (
        not stage_gate["passed"]
        or decision["benchmark_classification"] != "dplus0-sufficient"
        or decision["capacity_action"] != "keep-D+0"
        or decision["capacity_protocol_modified"]
        or decision["checkpoint_modified"]
    ):
        raise RuntimeError("remediated D+0 did not pass Phase 7")
    aggregate = load_json(phase7_aggregate_path)
    validate_payload(aggregate, "aggregate-certificate.schema.json")
    if (
        not aggregate["passed"]
        or aggregate["stage_gate"]["sha256"]
        != sha256_file(phase7_stage_gate_path)
        or aggregate["dispatch"]["sha256"]
        != sha256_file(phase7_dispatch_path)
    ):
        raise RuntimeError("Phase 7 aggregate provenance mismatch")

    remediation_path = require_artifact(
        dependency["remediation_certificate"]
    )
    remediation_readback_path = require_artifact(
        dependency["remediation_readback"]
    )
    if (
        science["remediation_certificate"]["sha256"]
        != sha256_file(remediation_path)
        or science["remediation_readback"]["sha256"]
        != sha256_file(remediation_readback_path)
    ):
        raise RuntimeError("science/remediation lineage mismatch")

    payload = {
        "schema_version": (
            "challenge-15-route-d-plus-future-dependency-v1"
        ),
        "kind": "architecture-freeze",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": revision,
        "selection_aggregate": artifact(phase7_aggregate_path),
        "phase7_stage_gate": artifact(phase7_stage_gate_path),
        "remediation_certificate": dependency[
            "remediation_certificate"
        ],
        "remediation_readback": dependency["remediation_readback"],
        "remediation_science": artifact(science_path),
        "remediation_science_readback": artifact(
            science_readback_path
        ),
        "selection_stage": "phase7",
        "selection_protocol": dependency["capacity_protocol"],
        "selected_capacity": "D+0",
        "architecture": dependency["architecture"],
        "checkpoints": dependency["checkpoints"],
        "heldout_accessed": False,
        "beyond_ed_accessed": False,
        "passed": True,
    }
    validate_payload(payload, "dependency.schema.json")
    write_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--science", required=True, type=Path)
    parser.add_argument("--science-readback", required=True, type=Path)
    parser.add_argument("--phase7-dispatch", required=True, type=Path)
    parser.add_argument("--phase7-stage-gate", required=True, type=Path)
    parser.add_argument("--phase7-aggregate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    payload = freeze(
        repo_root=arguments.repo_root.resolve(),
        science_path=arguments.science.resolve(),
        science_readback_path=arguments.science_readback.resolve(),
        phase7_dispatch_path=arguments.phase7_dispatch.resolve(),
        phase7_stage_gate_path=arguments.phase7_stage_gate.resolve(),
        phase7_aggregate_path=arguments.phase7_aggregate.resolve(),
        output_path=arguments.output.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
