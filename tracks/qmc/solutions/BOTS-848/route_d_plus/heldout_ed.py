"""Read-only held-out ED and overlap evaluation for frozen D+0."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from benchmark_v0.lll_coulomb import coulomb_integrals
from route_d_plus import phase7
from route_d_plus.future.verify import (
    load_json,
    require_artifact,
    require_gpu_slurm_environment,
    sha256_file,
    validate_payload,
)

MODULE_ROOT = Path(__file__).resolve().parent


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate(payload: dict[str, Any], schema_name: str) -> None:
    schema = load_json(MODULE_ROOT / schema_name)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(payload)


def _load_frozen_inputs(
    *,
    architecture_freeze_path: Path,
    calibration_path: Path,
    architecture_path: Path,
    seed_certificate_paths: list[Path],
    revision: str,
    n_electrons: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze = load_json(architecture_freeze_path)
    validate_payload(freeze, "dependency.schema.json")
    if (
        freeze["kind"] != "architecture-freeze"
        or freeze["selected_capacity"] != "D+0"
        or not freeze["passed"]
    ):
        raise RuntimeError("held-out ED requires frozen D+0")
    calibration = load_json(calibration_path)
    _validate(calibration, "scalable-calibration.schema.json")
    if (
        calibration["source_revision"] != revision
        or calibration["n_electrons"] != n_electrons
        or calibration["architecture_freeze"]["sha256"]
        != sha256_file(architecture_freeze_path)
        or calibration["size_architecture"]["sha256"]
        != sha256_file(architecture_path)
    ):
        raise RuntimeError("held-out calibration lineage mismatch")
    architecture = load_json(architecture_path)
    _validate(architecture, "scalable-architecture.schema.json")
    if len(seed_certificate_paths) != 3:
        raise ValueError("held-out ED requires exactly three frozen seeds")
    checkpoints = []
    seeds = set()
    for certificate_path in seed_certificate_paths:
        certificate = load_json(certificate_path)
        _validate(certificate, "scalable-seed.schema.json")
        if (
            not certificate["passed"]
            or certificate["source_revision"] != revision
            or certificate["n_electrons"] != n_electrons
            or certificate["architecture_modified"]
            or certificate[
                "heldout_or_beyond_used_for_structure_selection"
            ]
            or certificate["architecture_freeze"]["sha256"]
            != sha256_file(architecture_freeze_path)
            or certificate["calibration"]["sha256"]
            != sha256_file(calibration_path)
            or certificate["size_architecture"]["sha256"]
            != sha256_file(architecture_path)
        ):
            raise RuntimeError("held-out seed certificate lineage mismatch")
        checkpoint_path = require_artifact(certificate["checkpoint"])
        checkpoint = load_json(checkpoint_path)
        _validate(checkpoint, "scalable-checkpoint.schema.json")
        if (
            checkpoint["n_electrons"] != n_electrons
            or checkpoint["two_q"] != 3 * (n_electrons - 1)
            or checkpoint["source_revision"] != revision
            or checkpoint["architecture_sha256"]
            != sha256_file(architecture_path)
        ):
            raise RuntimeError("held-out checkpoint lineage mismatch")
        checkpoints.append(checkpoint)
        seeds.add(checkpoint["seed"])
    if len(seeds) != 3:
        raise RuntimeError("held-out checkpoint seeds are not unique")
    return architecture, checkpoints


def evaluate(
    *,
    repo_root: Path,
    architecture_freeze_path: Path,
    calibration_path: Path,
    architecture_path: Path,
    seed_certificate_paths: list[Path],
    n_electrons: int,
    integrals_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if n_electrons != 7:
        raise ValueError("the pre-registered held-out system is N=7")
    started = time.monotonic()
    job_id, cluster = require_gpu_slurm_environment()
    revision = _git(repo_root, "rev-parse", "HEAD")
    if _git(repo_root, "status", "--porcelain"):
        raise RuntimeError("held-out ED requires a clean checkout")
    architecture, checkpoints = _load_frozen_inputs(
        architecture_freeze_path=architecture_freeze_path,
        calibration_path=calibration_path,
        architecture_path=architecture_path,
        seed_certificate_paths=seed_certificate_paths,
        revision=revision,
        n_electrons=n_electrons,
    )

    phase7.configure_system(n_electrons)
    two_q = 3 * (n_electrons - 1)
    integrals_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        integrals_path,
        integrals=coulomb_integrals(two_q),
    )
    integrals = np.load(integrals_path)["integrals"]
    results, diagnostics = phase7._overlap_result(
        integrals, architecture, checkpoints
    )
    finite = all(
        math.isfinite(float(results[key]))
        for key in (
            "ed_ground_energy",
            "ed_tower_energy",
            "ed_gap",
            "ground_fidelity_mean",
            "tower_fidelity_mean",
            "dplus_gap_mean",
            "gap_absolute_error",
        )
    )
    gates = {
        "finite": finite,
        "three_frozen_seeds": diagnostics["checkpoint_count"] == 3,
        "fivefold_multiplet": diagnostics["multiplet_splitting"] < 5.0e-10,
        "ground_fidelity": results["ground_fidelity_mean"] >= 0.95,
        "tower_fidelity": results["tower_fidelity_mean"] >= 0.90,
        "gap_accuracy": results["gap_absolute_error"] <= 0.005,
        "read_only_evaluation": True,
        "architecture_unmodified": True,
        "no_structure_selection": True,
    }
    payload = {
        "schema_version": "challenge-15-route-d-plus-heldout-ed-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": revision,
        "n_electrons": n_electrons,
        "two_q": two_q,
        "role": "held-out-evaluation-only",
        "architecture_freeze": _artifact(architecture_freeze_path),
        "calibration": _artifact(calibration_path),
        "size_architecture": _artifact(architecture_path),
        "seed_certificates": [
            _artifact(path) for path in seed_certificate_paths
        ],
        "integrals": _artifact(integrals_path),
        "results": results,
        "diagnostics": diagnostics,
        "heldout_used_for_structure_selection": False,
        "architecture_modified": False,
        "slurm": {
            "job_id": job_id,
            "cluster": cluster,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "elapsed_seconds": time.monotonic() - started,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    _validate(payload, "heldout-ed.schema.json")
    _write(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--architecture-freeze", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument(
        "--seed-certificate",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--n-electrons", type=int, default=7)
    parser.add_argument("--integrals-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = evaluate(
        repo_root=arguments.repo_root.resolve(),
        architecture_freeze_path=arguments.architecture_freeze.resolve(),
        calibration_path=arguments.calibration.resolve(),
        architecture_path=arguments.architecture.resolve(),
        seed_certificate_paths=[
            path.resolve() for path in arguments.seed_certificate
        ],
        n_electrons=arguments.n_electrons,
        integrals_path=arguments.integrals_output.resolve(),
        output_path=arguments.output.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
