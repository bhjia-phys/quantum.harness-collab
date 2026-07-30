"""Write the Phase 10 pre-freeze chirality algebra certificate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import jsonschema

from route_d_plus.chirality import certify_chiral_pair_tensors

MODULE_ROOT = Path(__file__).resolve().parent


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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def certify(
    *,
    repo_root: Path,
    two_q_values: list[int],
    output_path: Path,
) -> dict[str, Any]:
    revision = _git(repo_root, "rev-parse", "HEAD")
    clean = not bool(_git(repo_root, "status", "--porcelain"))
    job_id = os.environ.get("SLURM_JOB_ID", "")
    cluster = os.environ.get("SLURM_CLUSTER_NAME", "")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not job_id or not cluster or visible_gpu in {"", "-1", "NoDevFiles"}:
        raise RuntimeError("chirality certification requires Slurm GPU")
    if not clean:
        raise RuntimeError("chirality certification requires clean source")

    systems = [
        certify_chiral_pair_tensors(two_q)
        for two_q in sorted(set(two_q_values))
    ]
    payload = {
        "schema_version": (
            "challenge-15-route-d-plus-chirality-algebra-v1"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": revision,
        "clean_worktree": clean,
        "slurm": {
            "job_id": job_id,
            "cluster": cluster,
            "cuda_visible_devices": visible_gpu,
        },
        "ed_accessed": False,
        "systems": systems,
        "gates": {
            "pair_basis_complete": all(
                system["gates"]["pair_basis_complete"]
                for system in systems
            ),
            "spherical_adjoint": all(
                system["gates"]["spherical_adjoint"]
                for system in systems
            ),
            "rank_two_tensor": all(
                system["gates"]["rank_two_jz"]
                and system["gates"]["rank_two_raising"]
                for system in systems
            ),
            "relative_transition_selection": all(
                system["gates"]["plus_is_r_to_r_plus_2"]
                and system["gates"]["minus_is_r_to_r_minus_2"]
                for system in systems
            ),
            "no_ed_access": True,
        },
        "passed": all(system["passed"] for system in systems),
    }
    schema = json.loads(
        (MODULE_ROOT / "chirality-algebra.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(payload)
    _write(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--two-q", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = certify(
        repo_root=args.repo_root.resolve(),
        two_q_values=args.two_q,
        output_path=args.output.resolve(),
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
