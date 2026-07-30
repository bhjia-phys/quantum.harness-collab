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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--two-q", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
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
        for two_q in sorted(set(args.two_q))
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
    _write(args.output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
