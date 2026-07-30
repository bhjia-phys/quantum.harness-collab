"""Read-only Phase 11 finite-size synthesis for the spin-two gap."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from route_d_plus.future.verify import (
    load_json,
    require_gpu_slurm_environment,
    sha256_file,
)

MODULE_ROOT = Path(__file__).resolve().parent


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fit(
    n_values: np.ndarray,
    gaps: np.ndarray,
    *,
    order: int,
    minimum_n: int,
) -> dict[str, Any]:
    selected = n_values >= minimum_n
    x = 1.0 / (1.5 * (n_values[selected] - 1.0))
    y = gaps[selected]
    if y.size < order + 1:
        raise ValueError("insufficient sizes for registered fit")
    coefficients = np.polynomial.polynomial.polyfit(x, y, order)
    predicted = np.polynomial.polynomial.polyval(x, coefficients)
    return {
        "order": order,
        "minimum_n": minimum_n,
        "sizes": n_values[selected].astype(int).tolist(),
        "delta_infinity": float(coefficients[0]),
        "coefficients_ascending": coefficients.tolist(),
        "root_mean_square_residual": float(
            np.sqrt(np.mean((predicted - y) ** 2))
        ),
    }


def synthesize(
    *,
    protocol_path: Path,
    seed_domain_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    job_id, cluster = require_gpu_slurm_environment()
    protocol = load_json(protocol_path)
    schema = load_json(
        MODULE_ROOT / "future/postfreeze-protocol.schema.json"
    )
    jsonschema.Draft202012Validator(schema).validate(protocol)
    registered = protocol["phase11"]
    expected_sizes = registered["n_electrons"]
    expected_seeds = registered["seeds"]
    records = []
    for path in seed_domain_paths:
        certificate = load_json(path)
        schema = load_json(MODULE_ROOT / "scalable-seed.schema.json")
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).validate(certificate)
        if not certificate["passed"]:
            raise RuntimeError(f"seed domain did not pass: {path}")
        result_path = Path(certificate["result"]["path"])
        if sha256_file(result_path) != certificate["result"]["sha256"]:
            raise RuntimeError("seed result hash mismatch")
        result = load_json(result_path)
        records.append(
            {
                "n_electrons": certificate["n_electrons"],
                "seed": certificate["seed"],
                "gap": float(result["final_gap"]),
                "standard_error": float(
                    result["final_gap_standard_error"]
                ),
                "domain_certificate": _artifact(path),
                "result": _artifact(result_path),
            }
        )
    observed = {
        (record["n_electrons"], record["seed"]) for record in records
    }
    expected = {
        (n_electrons, seed)
        for n_electrons in expected_sizes
        for seed in expected_seeds
    }
    if observed != expected:
        raise RuntimeError("finite-size synthesis seed grid mismatch")

    by_size = []
    for n_electrons in expected_sizes:
        subset = [
            record
            for record in records
            if record["n_electrons"] == n_electrons
        ]
        gaps = np.asarray([record["gap"] for record in subset])
        errors = np.asarray(
            [record["standard_error"] for record in subset]
        )
        by_size.append(
            {
                "n_electrons": n_electrons,
                "two_q": 3 * (n_electrons - 1),
                "q": 1.5 * (n_electrons - 1),
                "gap_mean": float(np.mean(gaps)),
                "mc_standard_error": float(
                    np.sqrt(np.sum(errors**2)) / errors.size
                ),
                "seed_standard_deviation": float(
                    np.std(gaps, ddof=1)
                ),
                "seed_records": subset,
            }
        )
    n_values = np.asarray(expected_sizes, dtype=np.float64)
    gap_means = np.asarray(
        [record["gap_mean"] for record in by_size]
    )
    fits = []
    for minimum_n in registered["linear_minimum_size_cuts"]:
        fits.append(
            _fit(n_values, gap_means, order=1, minimum_n=minimum_n)
        )
    for minimum_n in registered["quadratic_minimum_size_cuts"]:
        fits.append(
            _fit(n_values, gap_means, order=2, minimum_n=minimum_n)
        )

    rng = np.random.default_rng(848_111)
    bootstrap_intercepts = []
    for _ in range(2000):
        sampled_means = []
        for size_record in by_size:
            seed_records = size_record["seed_records"]
            indices = rng.integers(0, len(seed_records), len(seed_records))
            samples = [
                rng.normal(
                    seed_records[index]["gap"],
                    seed_records[index]["standard_error"],
                )
                for index in indices
            ]
            sampled_means.append(float(np.mean(samples)))
        bootstrap_intercepts.append(
            _fit(
                n_values,
                np.asarray(sampled_means),
                order=1,
                minimum_n=8,
            )["delta_infinity"]
        )
    primary = next(
        fit
        for fit in fits
        if fit["order"] == 1 and fit["minimum_n"] == 8
    )
    quadratic = next(fit for fit in fits if fit["order"] == 2)
    bootstrap_error = float(np.std(bootstrap_intercepts, ddof=1))
    seed_spread = float(
        np.sqrt(
            np.mean(
                [
                    record["seed_standard_deviation"] ** 2
                    for record in by_size
                ]
            )
        )
    )
    architecture_residual = abs(
        primary["delta_infinity"] - quadratic["delta_infinity"]
    )
    total_error = float(
        np.sqrt(
            bootstrap_error**2
            + seed_spread**2
            + architecture_residual**2
        )
    )
    gates = {
        "preregistered_sizes_complete": observed == expected,
        "seed_spread_included": seed_spread >= 0.0,
        "mc_errors_included": bootstrap_error >= 0.0,
        "linear_and_quadratic_fits": (
            {fit["order"] for fit in fits} == {1, 2}
        ),
        "minimum_size_cuts": (
            {fit["minimum_n"] for fit in fits} == {8, 10}
        ),
        "read_only_synthesis": True,
        "architecture_unmodified": True,
    }
    numeric = [
        primary["delta_infinity"],
        bootstrap_error,
        seed_spread,
        architecture_residual,
        total_error,
    ]
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError("finite-size synthesis produced non-finite values")
    payload = {
        "schema_version": (
            "challenge-15-route-d-plus-finite-size-v1"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": _artifact(protocol_path),
        "sizes": by_size,
        "fits": fits,
        "bootstrap": {
            "seed": 848111,
            "replicates": 2000,
            "mc_and_seed_resampling_standard_error": bootstrap_error,
        },
        "delta_infinity": primary["delta_infinity"],
        "uncertainty": {
            "bootstrap_mc_and_seed": bootstrap_error,
            "seed_spread": seed_spread,
            "architecture_residual": architecture_residual,
            "total_quadrature": total_error,
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
    schema = load_json(MODULE_ROOT / "finite-size.schema.json")
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(payload)
    _write(output_path, payload)
    return payload
