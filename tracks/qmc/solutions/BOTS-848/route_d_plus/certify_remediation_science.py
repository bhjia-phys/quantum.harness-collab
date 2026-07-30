"""Independent Phase 6 statistical gate for remediated D+0 checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import jsonschema

from route_d_plus.future.verify import load_json, sha256_file

MODULE_ROOT = Path(__file__).resolve().parent
SEEDS = (848, 1848, 2848)
SECTORS = ("final_ground", "final_tower")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
        raise RuntimeError("scientific gate requires a Slurm GPU allocation")
    return {
        "job_id": os.environ["SLURM_JOB_ID"],
        "cluster_name": os.environ.get(
            "SLURM_CLUSTER_NAME", "hpccube-xh5"
        ),
        "node_list": os.environ.get("SLURM_NODELIST", "unknown"),
        "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "gpu_devices": [part for part in visible.split(",") if part],
    }


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True


def _sector_summary(statistics: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean": statistics["mean"],
        "standard_error": statistics["standard_error"],
        "effective_sample_size": statistics["effective_sample_size"],
        "ess_per_second": statistics["ess_per_second"],
        "r_hat": statistics["r_hat"],
        "per_chain_correction_acceptance": statistics[
            "per_chain_correction_acceptance"
        ],
        "per_chain_mother_acceptance": statistics[
            "per_chain_mother_acceptance"
        ],
        "global_rotation_residual": statistics[
            "global_rotation_residual"
        ],
    }


def certify(
    *,
    repo_root: Path,
    remediation_path: Path,
    readback_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    revision = git_output(repo_root, "rev-parse", "HEAD")
    if git_output(repo_root, "status", "--porcelain"):
        raise RuntimeError("scientific gate requires a clean checkout")

    remediation = load_json(remediation_path)
    validate(remediation, "optimization-remediation.schema.json")
    readback = load_json(readback_path)
    validate(
        readback, "optimization-remediation-readback.schema.json"
    )
    if (
        not remediation["passed"]
        or not readback["passed"]
        or readback["remediation_certificate"]["sha256"]
        != sha256_file(remediation_path)
    ):
        raise RuntimeError("remediation/readback provenance did not pass")

    results = []
    summaries = []
    for reference in remediation["seed_results"]:
        result_path = require(reference["result"])
        result = load_json(result_path)
        validate(
            result, "optimization-remediation-seed.schema.json"
        )
        if result["seed"] != reference["seed"]:
            raise RuntimeError("seed result reference mismatch")
        results.append(result)
        summaries.append(
            {
                "seed": result["seed"],
                "result": artifact(result_path),
                "checkpoint": reference["checkpoint"],
                "gap": result["final_gap"],
                "gap_standard_error": result[
                    "final_gap_standard_error"
                ],
                "ground": _sector_summary(result["final_ground"]),
                "tower": _sector_summary(result["final_tower"]),
            }
        )
    results.sort(key=lambda item: item["seed"])
    summaries.sort(key=lambda item: item["seed"])
    if tuple(result["seed"] for result in results) != SEEDS:
        raise RuntimeError("scientific gate requires the exact seed set")

    gates = {
        "finite_statistics": all(_finite(result) for result in results),
        "sampling_acceptance": all(
            all(
                0.05 <= acceptance <= 1.0
                for acceptance in result[sector][
                    "per_chain_correction_acceptance"
                ]
            )
            and all(
                0.25 <= acceptance <= 0.70
                for acceptance in result[sector][
                    "per_chain_mother_acceptance"
                ]
            )
            for result in results
            for sector in SECTORS
        ),
        "effective_samples": all(
            result[sector]["effective_sample_size"] >= 32.0
            and result[sector]["ess_per_second"] > 0.0
            and result[sector]["r_hat"] < 1.2
            for result in results
            for sector in SECTORS
        ),
        "gap_precision": all(
            result["final_gap_standard_error"] <= 5.0e-3
            for result in results
        ),
        "three_seed_consistency": all(
            abs(left["final_gap"] - right["final_gap"])
            <= 3.0
            * math.hypot(
                left["final_gap_standard_error"],
                right["final_gap_standard_error"],
            )
            for left, right in itertools.combinations(results, 2)
        ),
        "rotation_invariance": all(
            result[sector]["global_rotation_residual"] < 1.0e-7
            for result in results
            for sector in SECTORS
        ),
        "all_seed_training_gates": all(
            result["passed"] for result in results
        ),
        "immutable_checkpoint_hashes": all(
            sha256_file(require(reference["checkpoint"]))
            == reference["checkpoint"]["sha256"]
            for reference in remediation["seed_results"]
        ),
        "clean_verifier_revision": True,
        "gpu_slurm_evidence": True,
        "no_ed_gradient": remediation["gates"]["no_ed_gradient"],
        "no_ed_checkpoint_selection": remediation["gates"][
            "no_ed_checkpoint_selection"
        ],
    }
    payload = {
        "schema_version": (
            "challenge-15-route-d-plus-remediation-science-v1"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "producer_revision": remediation["source_revision"],
        "verifier_revision": revision,
        "remediation_certificate": artifact(remediation_path),
        "remediation_readback": artifact(readback_path),
        "thresholds": {
            "correction_acceptance_min": 0.05,
            "correction_acceptance_max": 1.0,
            "mother_acceptance_min": 0.25,
            "mother_acceptance_max": 0.70,
            "effective_sample_size_min": 32.0,
            "r_hat_max_exclusive": 1.2,
            "gap_standard_error_max": 5.0e-3,
            "seed_consistency_sigma": 3.0,
            "rotation_residual_max_exclusive": 1.0e-7,
        },
        "seed_summaries": summaries,
        "slurm": slurm(),
        "gates": gates,
        "passed": all(gates.values()),
    }
    validate(payload, "remediation-science.schema.json")
    write_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--remediation", required=True, type=Path)
    parser.add_argument("--readback", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    payload = certify(
        repo_root=arguments.repo_root.resolve(),
        remediation_path=arguments.remediation.resolve(),
        readback_path=arguments.readback.resolve(),
        output_path=arguments.output.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
