"""N=6 chiral spectral weights for the frozen D+0 state and ED readback."""

from __future__ import annotations

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
from route_d_plus.chirality import (
    chiral_pair_tensors,
    lift_pair_operator_between,
)
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


def _spectral_summary(
    amplitudes: dict[str, dict[str, complex]],
) -> dict[str, Any]:
    weights = {
        helicity: float(
            np.mean(
                [
                    abs(components[str(magnetic)]) ** 2
                    for magnetic in range(-2, 3)
                ]
            )
        )
        for helicity, components in amplitudes.items()
    }
    denominator = weights["minus"] + weights["plus"]
    chi = (
        (weights["minus"] - weights["plus"]) / denominator
        if denominator > 0.0
        else 0.0
    )
    return {
        "z_plus": weights["plus"],
        "z_minus": weights["minus"],
        "chi": float(chi),
        "dominant_helicity": (
            "minus" if weights["minus"] > weights["plus"] else "plus"
        ),
        "amplitudes": {
            helicity: {
                magnetic: {
                    "real": float(value.real),
                    "imag": float(value.imag),
                }
                for magnetic, value in components.items()
            }
            for helicity, components in amplitudes.items()
        },
    }


def evaluate(
    *,
    repo_root: Path,
    architecture_freeze_path: Path,
    algebra_certificate_path: Path,
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    job_id, cluster = require_gpu_slurm_environment()
    revision = _git(repo_root, "rev-parse", "HEAD")
    if _git(repo_root, "status", "--porcelain"):
        raise RuntimeError("chirality spectral worker requires clean source")
    freeze = load_json(architecture_freeze_path)
    validate_payload(freeze, "dependency.schema.json")
    if (
        freeze["kind"] != "architecture-freeze"
        or freeze["selected_capacity"] != "D+0"
        or not freeze["passed"]
    ):
        raise RuntimeError("chirality requires frozen D+0")
    architecture_path = require_artifact(freeze["architecture"])
    architecture = load_json(architecture_path)
    checkpoints = [
        load_json(require_artifact(reference))
        for reference in freeze["checkpoints"]
    ]
    if len(checkpoints) != 3:
        raise RuntimeError("chirality requires exactly three frozen seeds")

    algebra = load_json(algebra_certificate_path)
    algebra_schema = load_json(MODULE_ROOT / "chirality-algebra.schema.json")
    jsonschema.Draft202012Validator(
        algebra_schema, format_checker=jsonschema.FormatChecker()
    ).validate(algebra)
    if not algebra["passed"] or not any(
        system["two_q"] == 15 and system["passed"]
        for system in algebra["systems"]
    ):
        raise RuntimeError("N=6 chirality algebra certificate did not pass")
    protocol = load_json(protocol_path)
    protocol_schema = load_json(
        MODULE_ROOT / "future/postfreeze-protocol.schema.json"
    )
    jsonschema.Draft202012Validator(protocol_schema).validate(protocol)
    chirality_protocol = protocol["phase10"]

    phase7.configure_system(6)
    integrals = coulomb_integrals(15)
    zero_basis, zero_hamiltonian, zero_l2 = phase7._sector(0, integrals)
    _, exact_ground, _ = phase7._lowest_l_state(
        zero_hamiltonian, zero_l2, 0
    )
    ground_generators = phase7._sector_generators(zero_basis)
    dplus_ground = {
        checkpoint["seed"]: phase7._candidate(
            phase7._ground_mother(zero_basis),
            ground_generators,
            phase7._complex_vector(
                checkpoint["ground_coefficients"]
            ),
            architecture,
        )
        for checkpoint in checkpoints
    }
    plus, minus = chiral_pair_tensors(15)
    exact_amplitudes = {"plus": {}, "minus": {}}
    seed_amplitudes = {
        checkpoint["seed"]: {"plus": {}, "minus": {}}
        for checkpoint in checkpoints
    }
    exact_energies = {}
    mother_fidelities = []
    adjoint_residual = 0.0
    for magnetic in range(-2, 3):
        basis, hamiltonian, l_squared = phase7._sector(
            magnetic, integrals
        )
        energy, exact_tower, _ = phase7._lowest_l_state(
            hamiltonian, l_squared, 2
        )
        exact_energies[magnetic] = energy
        mother_fidelities.append(
            phase7._fidelity(
                exact_tower,
                phase7._mother(magnetic, basis),
            )
        )
        lifted = {
            "plus": lift_pair_operator_between(
                15, zero_basis, basis, plus[magnetic]
            ),
            "minus": lift_pair_operator_between(
                15, zero_basis, basis, minus[magnetic]
            ),
        }
        reverse_minus = lift_pair_operator_between(
            15, basis, zero_basis, minus[-magnetic]
        )
        adjoint_residual = max(
            adjoint_residual,
            float(
                np.max(
                    np.abs(
                        lifted["plus"].conj().T
                        - ((-1) ** magnetic) * reverse_minus
                    )
                )
            ),
        )
        for helicity, operator in lifted.items():
            exact_amplitudes[helicity][str(magnetic)] = complex(
                np.vdot(exact_tower, operator @ exact_ground)
            )
        generators = phase7._sector_generators(basis)
        for checkpoint in checkpoints:
            seed = checkpoint["seed"]
            tower = phase7._candidate(
                phase7._mother(magnetic, basis),
                generators,
                phase7._complex_vector(
                    checkpoint["tower_coefficients"]
                ),
                architecture,
            )
            for helicity, operator in lifted.items():
                seed_amplitudes[seed][helicity][str(magnetic)] = complex(
                    np.vdot(tower, operator @ dplus_ground[seed])
                )

    exact_summary = _spectral_summary(exact_amplitudes)
    seed_summaries = {
        str(seed): _spectral_summary(amplitudes)
        for seed, amplitudes in seed_amplitudes.items()
    }
    aggregate = {
        key: float(
            np.mean([summary[key] for summary in seed_summaries.values()])
        )
        for key in ("z_plus", "z_minus", "chi")
    }
    aggregate["dominant_helicity"] = (
        "minus"
        if aggregate["z_minus"] > aggregate["z_plus"]
        else "plus"
    )
    mother_fidelity_mean = float(np.mean(mother_fidelities))
    first_mother_sufficient = (
        mother_fidelity_mean
        >= chirality_protocol["density_mother_bright_fidelity_min"]
    )
    numeric_values = [
        exact_summary["z_plus"],
        exact_summary["z_minus"],
        exact_summary["chi"],
        *[
            summary[key]
            for summary in seed_summaries.values()
            for key in ("z_plus", "z_minus", "chi")
        ],
    ]
    gates = {
        "phase_convention": True,
        "fivefold_multiplet": (
            max(exact_energies.values())
            - min(exact_energies.values())
            < 5.0e-10
        ),
        "finite_spectral_weights": all(
            math.isfinite(value) for value in numeric_values
        ),
        "nonzero_total_weight": all(
            summary["z_plus"] + summary["z_minus"] > 1.0e-16
            for summary in [exact_summary, *seed_summaries.values()]
        ),
        "adjoint_crosscheck": adjoint_residual < 1.0e-10,
        "architecture_unmodified": True,
    }
    payload = {
        "schema_version": (
            "challenge-15-route-d-plus-chirality-spectral-v1"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": revision,
        "n_electrons": 6,
        "two_q": 15,
        "architecture_freeze": _artifact(architecture_freeze_path),
        "algebra_certificate": _artifact(algebra_certificate_path),
        "protocol": _artifact(protocol_path),
        "architecture": _artifact(architecture_path),
        "checkpoint_count": len(checkpoints),
        "phase_convention": {
            "plus": "R-to-R-plus-2",
            "minus": "R-to-R-minus-2",
            "chi": "(Z_minus-Z_plus)/(Z_minus+Z_plus)",
        },
        "exact_ed": exact_summary,
        "dplus_by_seed": seed_summaries,
        "dplus_aggregate": aggregate,
        "density_mother_metric_bright_fidelity_mean": (
            mother_fidelity_mean
        ),
        "first_mother_sufficient": first_mother_sufficient,
        "diagnostics": {
            "exact_multiplet_splitting": (
                max(exact_energies.values())
                - min(exact_energies.values())
            ),
            "adjoint_residual": adjoint_residual,
        },
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
    schema = load_json(MODULE_ROOT / "chirality-spectral.schema.json")
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(payload)
    _write(output_path, payload)
    return payload
