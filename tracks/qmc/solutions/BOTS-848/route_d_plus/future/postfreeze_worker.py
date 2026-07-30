"""Execute one registered post-freeze task and seal its generic envelope."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from route_d_plus.certify_chirality_algebra import (
    certify as certify_chirality_algebra,
)
from route_d_plus.chirality_spectral import (
    evaluate as evaluate_chirality_spectral,
)
from route_d_plus.finite_size import synthesize as synthesize_finite_size
from route_d_plus.future.certify_task import certify as certify_task
from route_d_plus.future.verify import (
    load_json,
    require_artifact,
    require_isolated_path,
    validate_dispatch,
)
from route_d_plus.heldout_ed import evaluate as evaluate_heldout
from route_d_plus.scalable_dplus0 import calibrate, train

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _task_map(dispatch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in dispatch["tasks"]}


def _task_dir(
    dispatch: dict[str, Any],
    task_id: str,
) -> Path:
    task = _task_map(dispatch)[task_id]
    return require_isolated_path(
        Path(dispatch["run_root"]), task["run_dir"]
    )


def run(
    *,
    repo_root: Path,
    dispatch_path: Path,
    task_id: str,
    run_dir: Path,
) -> None:
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    dispatch = load_json(dispatch_path)
    validate_dispatch(dispatch)
    task = _task_map(dispatch)[task_id]
    expected_dir = _task_dir(dispatch, task_id)
    if run_dir.resolve() != expected_dir:
        raise RuntimeError("worker run directory differs from dispatch")
    freeze_references = [
        reference
        for reference in dispatch["prerequisites"]
        if reference["kind"] == "architecture-freeze"
    ]
    if len(freeze_references) != 1:
        raise RuntimeError("post-freeze worker requires one architecture freeze")
    freeze_path = require_artifact(freeze_references[0])
    registrations = [
        load_json(require_artifact(reference))
        for reference in dispatch["prerequisites"]
        if reference["kind"] == "protocol-registration"
    ]
    if len(registrations) != 1:
        raise RuntimeError("post-freeze worker requires one protocol")
    protocol_path = require_artifact(registrations[0]["protocol_artifact"])
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    if not stdout_path.exists() or not stderr_path.exists():
        raise RuntimeError("worker requires shell-created stdout/stderr logs")

    domain_path: Path
    schema_path: Path
    checkpoint_path: Path | None = None
    extra_inputs: list[Path] = []
    if task_id.endswith("-calibration"):
        architecture_path = run_dir / "architecture.json"
        domain_path = run_dir / "domain-certificate.json"
        calibrate(
            repo_root=repo_root,
            architecture_freeze_path=freeze_path,
            n_electrons=int(task["n_electrons"]),
            calibration_seed=int(task["seed"]),
            chains=4,
            samples_per_chain=32,
            workers=4,
            architecture_path=architecture_path,
            certificate_path=domain_path,
        )
        schema_path = MODULE_ROOT / "scalable-calibration.schema.json"
        extra_inputs = [architecture_path]
    elif "-seed-" in task_id:
        n_electrons = int(task["n_electrons"])
        calibration_dir = _task_dir(
            dispatch, f"n{n_electrons}-calibration"
        )
        calibration_path = calibration_dir / "domain-certificate.json"
        architecture_path = calibration_dir / "architecture.json"
        checkpoint_path = run_dir / "checkpoint.json"
        result_path = run_dir / "result.json"
        domain_path = run_dir / "domain-certificate.json"
        train(
            repo_root=repo_root,
            architecture_freeze_path=freeze_path,
            calibration_path=calibration_path,
            architecture_path=architecture_path,
            n_electrons=n_electrons,
            seed=int(task["seed"]),
            updates=24,
            chains=4,
            samples_per_update=8,
            final_samples_per_chain=128,
            checkpoint_path=checkpoint_path,
            result_path=result_path,
            certificate_path=domain_path,
        )
        schema_path = MODULE_ROOT / "scalable-seed.schema.json"
        extra_inputs = [
            calibration_path,
            architecture_path,
            result_path,
        ]
    elif task_id == "n7-ed-overlap":
        calibration_dir = _task_dir(dispatch, "n7-calibration")
        calibration_path = calibration_dir / "domain-certificate.json"
        architecture_path = calibration_dir / "architecture.json"
        seed_certificates = [
            _task_dir(dispatch, f"n7-seed-{seed}")
            / "domain-certificate.json"
            for seed in (848, 1848, 2848)
        ]
        integrals_path = run_dir / "coulomb-integrals.npz"
        domain_path = run_dir / "domain-certificate.json"
        evaluate_heldout(
            repo_root=repo_root,
            architecture_freeze_path=freeze_path,
            calibration_path=calibration_path,
            architecture_path=architecture_path,
            seed_certificate_paths=seed_certificates,
            n_electrons=7,
            integrals_path=integrals_path,
            output_path=domain_path,
        )
        schema_path = MODULE_ROOT / "heldout-ed.schema.json"
        extra_inputs = [
            calibration_path,
            architecture_path,
            *seed_certificates,
            integrals_path,
        ]
    elif task_id == "pair-algebra":
        domain_path = run_dir / "domain-certificate.json"
        certify_chirality_algebra(
            repo_root=repo_root,
            two_q_values=[15, 18, 21, 27, 33],
            output_path=domain_path,
        )
        schema_path = MODULE_ROOT / "chirality-algebra.schema.json"
    elif task_id == "n6-z-chi":
        algebra_path = (
            _task_dir(dispatch, "pair-algebra")
            / "domain-certificate.json"
        )
        domain_path = run_dir / "domain-certificate.json"
        evaluate_chirality_spectral(
            repo_root=repo_root,
            architecture_freeze_path=freeze_path,
            algebra_certificate_path=algebra_path,
            protocol_path=protocol_path,
            output_path=domain_path,
        )
        schema_path = MODULE_ROOT / "chirality-spectral.schema.json"
        extra_inputs = [algebra_path, protocol_path]
    elif task_id == "finite-size-synthesis":
        seed_domains = [
            _task_dir(dispatch, f"n{n_electrons}-seed-{seed}")
            / "domain-certificate.json"
            for n_electrons in (8, 10, 12)
            for seed in (848, 1848, 2848)
        ]
        domain_path = run_dir / "domain-certificate.json"
        synthesize_finite_size(
            protocol_path=protocol_path,
            seed_domain_paths=seed_domains,
            output_path=domain_path,
        )
        schema_path = MODULE_ROOT / "finite-size.schema.json"
        extra_inputs = [protocol_path, *seed_domains]
    else:
        raise NotImplementedError(
            f"post-freeze worker is not implemented for {task_id}"
        )

    certify_task(
        repo_root=repo_root,
        dispatch_path=dispatch_path,
        task_id=task_id,
        run_dir=run_dir,
        domain_path=domain_path,
        domain_schema_path=schema_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        checkpoint_path=checkpoint_path,
        extra_input_paths=extra_inputs,
        started_at_utc=started_at,
        output_path=run_dir / "task-certificate.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(
        repo_root=arguments.repo_root.resolve(),
        dispatch_path=arguments.dispatch.resolve(),
        task_id=arguments.task_id,
        run_dir=arguments.run_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
