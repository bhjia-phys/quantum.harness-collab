"""Post-freeze D+0 calibration and training for held-out/beyond-ED sizes."""

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

from route_d_plus.future.verify import (
    load_json,
    require_artifact,
    require_gpu_slurm_environment,
    sha256_file,
    validate_payload,
)
from route_d_plus.symmetry import verify_checkpoint_symmetry
from route_d_plus.train_dplus0 import (
    calibrate_architecture,
    configure_system,
    train_seed,
    write_checkpoint,
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


def _source_revision(repo_root: Path) -> str:
    revision = _git(repo_root, "rev-parse", "HEAD")
    if _git(repo_root, "status", "--porcelain"):
        raise RuntimeError("scalable D+0 requires a clean source checkout")
    return revision


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _load_freeze(path: Path) -> dict[str, Any]:
    freeze = load_json(path)
    validate_payload(freeze, "dependency.schema.json")
    if (
        freeze["kind"] != "architecture-freeze"
        or freeze["selected_capacity"] != "D+0"
        or not freeze["passed"]
    ):
        raise RuntimeError("scalable runner requires frozen D+0 architecture")
    for key in (
        "selection_aggregate",
        "selection_protocol",
        "architecture",
    ):
        require_artifact(freeze[key])
    for checkpoint in freeze["checkpoints"]:
        require_artifact(checkpoint)
    return freeze


def calibrate(
    *,
    repo_root: Path,
    architecture_freeze_path: Path,
    n_electrons: int,
    calibration_seed: int,
    chains: int,
    samples_per_chain: int,
    workers: int,
    architecture_path: Path,
    certificate_path: Path,
) -> dict[str, Any]:
    job_id, cluster = require_gpu_slurm_environment()
    revision = _source_revision(repo_root)
    freeze = _load_freeze(architecture_freeze_path)
    frozen_architecture_path = require_artifact(freeze["architecture"])
    frozen_architecture = load_json(frozen_architecture_path)

    configure_system(n_electrons)
    architecture = calibrate_architecture(
        calibration_seed,
        source_revision=revision,
        chains=chains,
        samples_per_chain=samples_per_chain,
        raw_amplitude_workers=workers,
    )
    if architecture["two_q"] != 3 * (n_electrons - 1):
        raise RuntimeError("scalable architecture violates Laughlin flux")
    if (
        architecture["retained_generators"]
        != frozen_architecture["retained_generators"]
    ):
        raise RuntimeError(
            "size calibration changed the frozen generator dimension"
        )
    _validate(architecture, "scalable-architecture.schema.json")
    _write_json(architecture_path, architecture)
    certificate = {
        "schema_version": (
            "challenge-15-route-d-plus-scalable-calibration-v1"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": revision,
        "n_electrons": n_electrons,
        "two_q": 3 * (n_electrons - 1),
        "architecture_freeze": _artifact(architecture_freeze_path),
        "frozen_architecture": _artifact(frozen_architecture_path),
        "size_architecture": _artifact(architecture_path),
        "frozen_capacity": "D+0",
        "frozen_generator_dimension": frozen_architecture[
            "retained_generators"
        ],
        "size_generator_dimension": architecture["retained_generators"],
        "structural_selection_performed": False,
        "ed_accessed": False,
        "heldout_or_beyond_used_for_structure_selection": False,
        "slurm": {
            "job_id": job_id,
            "cluster": cluster,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "gates": {
            "architecture_frozen": True,
            "generator_dimension_unchanged": True,
            "no_structural_selection": True,
            "no_ed_access": True,
        },
        "passed": True,
    }
    _validate(certificate, "scalable-calibration.schema.json")
    _write_json(certificate_path, certificate)
    return certificate


def _sector_gates(statistics: dict[str, Any]) -> dict[str, bool]:
    return {
        "finite": all(
            math.isfinite(float(statistics[key]))
            for key in (
                "mean",
                "standard_error",
                "effective_sample_size",
                "r_hat",
                "ess_per_second",
                "correction_acceptance",
                "mother_acceptance",
                "global_rotation_residual",
            )
        ),
        "correction_acceptance": (
            0.05 <= statistics["correction_acceptance"] <= 1.0
        ),
        "mother_acceptance": (
            0.25 <= statistics["mother_acceptance"] <= 0.70
        ),
        "effective_sample_size": (
            statistics["effective_sample_size"] >= 32.0
        ),
        "ess_per_second": statistics["ess_per_second"] > 0.0,
        "r_hat": statistics["r_hat"] < 1.2,
        "global_rotation": (
            statistics["global_rotation_residual"] < 1.0e-7
        ),
    }


def train(
    *,
    repo_root: Path,
    architecture_freeze_path: Path,
    calibration_path: Path,
    architecture_path: Path,
    n_electrons: int,
    seed: int,
    updates: int,
    chains: int,
    samples_per_update: int,
    final_samples_per_chain: int,
    checkpoint_path: Path,
    result_path: Path,
    symmetry_path: Path,
    certificate_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    job_id, cluster = require_gpu_slurm_environment()
    revision = _source_revision(repo_root)
    _load_freeze(architecture_freeze_path)
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
        raise RuntimeError("scalable calibration lineage mismatch")
    architecture = load_json(architecture_path)
    _validate(architecture, "scalable-architecture.schema.json")
    architecture_hash = sha256_file(architecture_path)

    configure_system(n_electrons)
    checkpoint, result = train_seed(
        seed,
        architecture=architecture,
        architecture_sha256=architecture_hash,
        chains=chains,
        updates=updates,
        samples_per_update=samples_per_update,
        final_samples_per_chain=final_samples_per_chain,
        checkpoint_selection="final-update-no-ed-postfreeze",
        progress_every=1,
    )
    _validate(checkpoint, "scalable-checkpoint.schema.json")
    write_checkpoint(checkpoint_path, checkpoint)
    _write_json(result_path, result)
    symmetry = verify_checkpoint_symmetry(architecture, checkpoint)
    _validate(symmetry, "scalable-symmetry.schema.json")
    _write_json(symmetry_path, symmetry)

    ground_gates = _sector_gates(result["final_ground"])
    tower_gates = _sector_gates(result["final_tower"])
    wall_seconds = time.monotonic() - started
    gates = {
        "ground_statistics": all(ground_gates.values()),
        "tower_statistics": all(tower_gates.values()),
        "gap_standard_error": (
            math.isfinite(result["final_gap_standard_error"])
            and result["final_gap_standard_error"] <= 0.005
        ),
        "checkpoint_schema": True,
        "symmetry": symmetry["passed"],
        "operator_cost": math.isfinite(wall_seconds) and wall_seconds > 0.0,
        "architecture_frozen": True,
        "no_ed_gradient": True,
        "no_structure_selection": True,
    }
    certificate = {
        "schema_version": (
            "challenge-15-route-d-plus-scalable-seed-v1"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": revision,
        "n_electrons": n_electrons,
        "two_q": 3 * (n_electrons - 1),
        "seed": seed,
        "architecture_freeze": _artifact(architecture_freeze_path),
        "calibration": _artifact(calibration_path),
        "size_architecture": _artifact(architecture_path),
        "checkpoint": _artifact(checkpoint_path),
        "result": _artifact(result_path),
        "symmetry": _artifact(symmetry_path),
        "initialization": "random-identity-nearby",
        "architecture_modified": False,
        "ed_accessed": False,
        "ed_used_for_gradient": False,
        "heldout_or_beyond_used_for_structure_selection": False,
        "ground_gates": ground_gates,
        "tower_gates": tower_gates,
        "resource_profile": {
            "wall_seconds": wall_seconds,
            "coordinate_backend": "coupled-pair-exact-lll",
            "n_electrons": n_electrons,
            "two_q": 3 * (n_electrons - 1),
            "updates": updates,
            "chains": chains,
            "samples_per_update": samples_per_update,
            "final_samples_per_chain": final_samples_per_chain,
            "ground_ess_per_second": result["final_ground"][
                "ess_per_second"
            ],
            "tower_ess_per_second": result["final_tower"][
                "ess_per_second"
            ],
        },
        "gates": gates,
        "slurm": {
            "job_id": job_id,
            "cluster": cluster,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "passed": all(gates.values()),
    }
    _validate(certificate, "scalable-seed.schema.json")
    _write_json(certificate_path, certificate)
    return certificate


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibration_parser = subparsers.add_parser("calibrate")
    training_parser = subparsers.add_parser("train")
    for child in (calibration_parser, training_parser):
        child.add_argument("--repo-root", type=Path, required=True)
        child.add_argument(
            "--architecture-freeze", type=Path, required=True
        )
        child.add_argument("--n-electrons", type=int, required=True)
    calibration_parser.add_argument(
        "--calibration-seed", type=int, default=60860
    )
    calibration_parser.add_argument("--chains", type=int, default=4)
    calibration_parser.add_argument(
        "--samples-per-chain", type=int, default=32
    )
    calibration_parser.add_argument("--workers", type=int, default=4)
    calibration_parser.add_argument(
        "--architecture-output", type=Path, required=True
    )
    calibration_parser.add_argument("--output", type=Path, required=True)
    training_parser.add_argument(
        "--calibration", type=Path, required=True
    )
    training_parser.add_argument(
        "--architecture", type=Path, required=True
    )
    training_parser.add_argument("--seed", type=int, required=True)
    training_parser.add_argument("--updates", type=int, default=24)
    training_parser.add_argument("--chains", type=int, default=4)
    training_parser.add_argument(
        "--samples-per-update", type=int, default=8
    )
    training_parser.add_argument(
        "--final-samples-per-chain", type=int, default=128
    )
    training_parser.add_argument(
        "--checkpoint-output", type=Path, required=True
    )
    training_parser.add_argument(
        "--result-output", type=Path, required=True
    )
    training_parser.add_argument(
        "--symmetry-output", type=Path, required=True
    )
    training_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "calibrate":
        payload = calibrate(
            repo_root=arguments.repo_root.resolve(),
            architecture_freeze_path=arguments.architecture_freeze.resolve(),
            n_electrons=arguments.n_electrons,
            calibration_seed=arguments.calibration_seed,
            chains=arguments.chains,
            samples_per_chain=arguments.samples_per_chain,
            workers=arguments.workers,
            architecture_path=arguments.architecture_output.resolve(),
            certificate_path=arguments.output.resolve(),
        )
    else:
        payload = train(
            repo_root=arguments.repo_root.resolve(),
            architecture_freeze_path=arguments.architecture_freeze.resolve(),
            calibration_path=arguments.calibration.resolve(),
            architecture_path=arguments.architecture.resolve(),
            n_electrons=arguments.n_electrons,
            seed=arguments.seed,
            updates=arguments.updates,
            chains=arguments.chains,
            samples_per_update=arguments.samples_per_update,
            final_samples_per_chain=arguments.final_samples_per_chain,
            checkpoint_path=arguments.checkpoint_output.resolve(),
            result_path=arguments.result_output.resolve(),
            symmetry_path=arguments.symmetry_output.resolve(),
            certificate_path=arguments.output.resolve(),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
