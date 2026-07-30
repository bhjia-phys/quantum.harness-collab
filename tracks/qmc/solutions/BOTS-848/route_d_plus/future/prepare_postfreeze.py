"""Create immutable Phase 9--11 dispatches after architecture freeze."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import jsonschema

from route_d_plus.future.verify import (
    load_json,
    sha256_file,
    validate_dispatch,
    validate_payload,
    write_json_atomic,
)

DISPATCH_VERSION = "challenge-15-route-d-plus-future-dispatch-v1"
DEPENDENCY_VERSION = "challenge-15-route-d-plus-future-dependency-v1"
MODULE_ROOT = Path(__file__).resolve().parent


def _revision(repo_root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("post-freeze dispatch requires clean source")
    return revision


def _phase9_tasks() -> list[dict[str, Any]]:
    tasks = [
        {
            "task_id": "n7-calibration",
            "kind": "heldout-ed",
            "run_dir": "tasks/n7-calibration",
            "required_gates": [
                "architecture_frozen",
                "generator_dimension_unchanged",
                "no_structural_selection",
                "no_ed_access",
            ],
            "n_electrons": 7,
            "seed": 60860,
        }
    ]
    for seed in (848, 1848, 2848):
        tasks.append(
            {
                "task_id": f"n7-seed-{seed}",
                "kind": "heldout-ed",
                "run_dir": f"tasks/n7-seed-{seed}",
                "required_gates": [
                    "ground_statistics",
                    "tower_statistics",
                    "gap_standard_error",
                    "symmetry",
                    "operator_cost",
                    "architecture_frozen",
                    "no_ed_gradient",
                    "no_structure_selection",
                ],
                "depends_on": ["n7-calibration"],
                "n_electrons": 7,
                "seed": seed,
            }
        )
    tasks.append(
        {
            "task_id": "n7-ed-overlap",
            "kind": "heldout-ed",
            "run_dir": "tasks/n7-ed-overlap",
            "required_gates": [
                "finite",
                "three_frozen_seeds",
                "fivefold_multiplet",
                "ground_fidelity",
                "tower_fidelity",
                "gap_accuracy",
                "read_only_evaluation",
                "architecture_unmodified",
                "no_structure_selection",
            ],
            "depends_on": [
                "n7-seed-848",
                "n7-seed-1848",
                "n7-seed-2848",
            ],
            "n_electrons": 7,
            "seed": None,
        }
    )
    return tasks


def _phase10_tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "pair-algebra",
            "kind": "chirality",
            "run_dir": "tasks/pair-algebra",
            "required_gates": [
                "pair_basis_complete",
                "spherical_adjoint",
                "rank_two_tensor",
                "relative_transition_selection",
                "no_ed_access",
            ],
            "n_electrons": 6,
            "helicity": "both",
        },
        {
            "task_id": "n6-z-chi",
            "kind": "chirality",
            "run_dir": "tasks/n6-z-chi",
            "required_gates": [
                "phase_convention",
                "fivefold_multiplet",
                "finite_spectral_weights",
                "nonzero_total_weight",
                "adjoint_crosscheck",
                "architecture_unmodified",
            ],
            "depends_on": ["pair-algebra"],
            "n_electrons": 6,
            "helicity": "both",
        },
    ]


def _phase11_tasks() -> list[dict[str, Any]]:
    tasks = []
    for n_electrons in (8, 10, 12):
        tasks.append(
            {
                "task_id": f"n{n_electrons}-calibration",
                "kind": "beyond-ed",
                "run_dir": f"tasks/n{n_electrons}-calibration",
                "required_gates": [
                    "architecture_frozen",
                    "generator_dimension_unchanged",
                    "no_structural_selection",
                    "no_ed_access",
                ],
                "n_electrons": n_electrons,
                "seed": 60860,
            }
        )
        for seed in (848, 1848, 2848):
            tasks.append(
                {
                    "task_id": f"n{n_electrons}-seed-{seed}",
                    "kind": "beyond-ed",
                    "run_dir": (
                        f"tasks/n{n_electrons}-seed-{seed}"
                    ),
                    "required_gates": [
                        "ground_statistics",
                        "tower_statistics",
                        "gap_standard_error",
                        "symmetry",
                        "operator_cost",
                        "architecture_frozen",
                        "no_ed_gradient",
                        "no_structure_selection",
                    ],
                    "depends_on": [f"n{n_electrons}-calibration"],
                    "n_electrons": n_electrons,
                    "seed": seed,
                }
            )
    seed_tasks = [
        task["task_id"] for task in tasks if "-seed-" in task["task_id"]
    ]
    tasks.append(
        {
            "task_id": "finite-size-synthesis",
            "kind": "beyond-ed",
            "run_dir": "tasks/finite-size-synthesis",
            "required_gates": [
                "preregistered_sizes_complete",
                "seed_spread_included",
                "mc_errors_included",
                "linear_and_quadratic_fits",
                "minimum_size_cuts",
                "read_only_synthesis",
                "architecture_unmodified",
            ],
            "depends_on": seed_tasks,
            "n_electrons": 12,
            "seed": None,
        }
    )
    return tasks


def prepare(
    *,
    repo_root: Path,
    run_root: Path,
    run_id: str,
    stage: str,
    architecture_freeze_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    revision = _revision(repo_root)
    freeze = load_json(architecture_freeze_path)
    validate_payload(freeze, "dependency.schema.json")
    if freeze["kind"] != "architecture-freeze" or not freeze["passed"]:
        raise RuntimeError("post-freeze dispatch requires architecture freeze")
    task_builders = {
        "phase9": _phase9_tasks,
        "phase10": _phase10_tasks,
        "phase11": _phase11_tasks,
    }
    protocol_path = MODULE_ROOT / "postfreeze-protocol.json"
    protocol_schema = load_json(
        MODULE_ROOT / "postfreeze-protocol.schema.json"
    )
    jsonschema.Draft202012Validator.check_schema(protocol_schema)
    jsonschema.Draft202012Validator(protocol_schema).validate(
        load_json(protocol_path)
    )
    registration_path = run_root / "protocol-registration.json"
    registration = {
        "schema_version": DEPENDENCY_VERSION,
        "kind": "protocol-registration",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": revision,
        "protocol_name": f"{stage}-postfreeze-protocol",
        "protocol_artifact": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256_file(protocol_path),
        },
        "passed": True,
    }
    validate_payload(registration, "dependency.schema.json")
    write_json_atomic(registration_path, registration)
    dispatch = {
        "schema_version": DISPATCH_VERSION,
        "stage": stage,
        "run_id": run_id,
        "run_root": str(run_root.resolve()),
        "source_revision": revision,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prerequisites": [
            {
                "kind": "architecture-freeze",
                "path": str(architecture_freeze_path.resolve()),
                "sha256": sha256_file(architecture_freeze_path),
            },
            {
                "kind": "protocol-registration",
                "path": str(registration_path.resolve()),
                "sha256": sha256_file(registration_path),
            },
        ],
        "tasks": task_builders[stage](),
    }
    validate_dispatch(dispatch)
    write_json_atomic(output_path, dispatch)
    return dispatch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage", choices=("phase9", "phase10", "phase11"), required=True
    )
    parser.add_argument("--architecture-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = prepare(
        repo_root=arguments.repo_root.resolve(),
        run_root=arguments.run_root.resolve(),
        run_id=arguments.run_id,
        stage=arguments.stage,
        architecture_freeze_path=arguments.architecture_freeze.resolve(),
        output_path=arguments.output.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
