"""Write the single registered Phase 9, 10, or 11 stage decision."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from route_d_plus.future.verify import (
    load_json,
    require_isolated_path,
    validate_dispatch,
    validate_payload,
    validate_task_certificate,
    write_json_atomic,
)

STAGE_GATE_VERSION = "challenge-15-route-d-plus-future-stage-gate-v1"


def finalize(
    *,
    dispatch_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    dispatch = load_json(dispatch_path)
    validate_dispatch(dispatch)
    domains = {}
    for task in dispatch["tasks"]:
        run_dir = require_isolated_path(
            Path(dispatch["run_root"]), task["run_dir"]
        )
        certificate = load_json(run_dir / "task-certificate.json")
        validate_task_certificate(
            certificate,
            task=task,
            dispatch=dispatch,
        )
        domains[task["task_id"]] = load_json(
            run_dir / "domain-certificate.json"
        )

    stage = dispatch["stage"]
    if stage == "phase9":
        decision = {
            "kind": "phase9-heldout-summary",
            "read_only_evaluation": True,
            "architecture_modified": False,
        }
    elif stage == "phase10":
        spectral = domains["n6-z-chi"]
        decision = {
            "kind": "phase10-chirality-summary",
            "phase_convention_gate": True,
            "second_mother_action": (
                "not-needed"
                if spectral["first_mother_sufficient"]
                else "triggered-by-unified-gate"
            ),
            "architecture_modified": False,
        }
    elif stage == "phase11":
        synthesis = domains["finite-size-synthesis"]
        if not synthesis["passed"]:
            raise RuntimeError("finite-size synthesis did not pass")
        decision = {
            "kind": "phase11-beyond-ed-summary",
            "preregistered_sizes_complete": True,
            "read_only_synthesis": True,
            "architecture_modified": False,
        }
    else:
        raise ValueError(f"not a post-freeze stage: {stage}")
    payload = {
        "schema_version": STAGE_GATE_VERSION,
        "stage": stage,
        "source_revision": dispatch["source_revision"],
        "task_ids": [task["task_id"] for task in dispatch["tasks"]],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": decision,
        "passed": True,
    }
    validate_payload(payload, "stage-gate.schema.json")
    write_json_atomic(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = finalize(
        dispatch_path=arguments.dispatch.resolve(),
        output_path=arguments.output.resolve(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
