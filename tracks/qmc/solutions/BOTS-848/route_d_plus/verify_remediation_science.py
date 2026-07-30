"""Independent schema/hash readback for the remediation science gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import jsonschema

from route_d_plus.future.verify import load_json, sha256_file

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


def require(reference: dict[str, str]) -> Path:
    path = Path(reference["path"]).resolve()
    if sha256_file(path) != reference["sha256"]:
        raise RuntimeError(f"artifact hash mismatch: {path}")
    return path


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
        raise RuntimeError("science readback requires a Slurm GPU allocation")
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
    certificate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    revision = git_output(repo_root, "rev-parse", "HEAD")
    if git_output(repo_root, "status", "--porcelain"):
        raise RuntimeError("science readback requires a clean checkout")
    certificate = load_json(certificate_path)
    validate(certificate, "remediation-science.schema.json")
    require(certificate["remediation_certificate"])
    require(certificate["remediation_readback"])
    for summary in certificate["seed_summaries"]:
        require(summary["result"])
        require(summary["checkpoint"])
    payload = {
        "schema_version": (
            "challenge-15-route-d-plus-remediation-science-readback-v1"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "producer_revision": certificate["producer_revision"],
        "science_verifier_revision": certificate["verifier_revision"],
        "readback_revision": revision,
        "science_certificate": artifact(certificate_path),
        "science_passed": certificate["passed"],
        "slurm": slurm(),
        "gates": {
            "science_schema_valid": True,
            "science_hash_valid": True,
            "remediation_hash_valid": True,
            "remediation_readback_hash_valid": True,
            "exact_three_seed_set": sorted(
                item["seed"] for item in certificate["seed_summaries"]
            )
            == [848, 1848, 2848],
            "all_result_hashes_valid": True,
            "all_checkpoint_hashes_valid": True,
            "clean_readback_revision": True,
            "gpu_slurm_evidence": True,
        },
        "passed": True,
    }
    validate(payload, "remediation-science-readback.schema.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--certificate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    payload = verify(
        repo_root=arguments.repo_root.resolve(),
        certificate_path=arguments.certificate.resolve(),
        output_path=arguments.output.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
