from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from route_d_plus import remediate_dplus0

SOLUTION_ROOT = Path(__file__).resolve().parents[2]
ROUTE_ROOT = SOLUTION_ROOT / "route_d_plus"


def test_remediation_protocol_is_frozen_and_schema_valid() -> None:
    schema = json.loads(
        (
            ROUTE_ROOT
            / "optimization-remediation-protocol.schema.json"
        ).read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (
            ROUTE_ROOT / "optimization-remediation-protocol.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(protocol)
    assert protocol["capacity"] == "D+0"
    assert protocol["architecture_modified"] is False
    assert protocol["ed_used_for_gradient"] is False
    assert protocol["ed_used_for_checkpoint_selection"] is False
    assert protocol["run_seeds_concurrently"] is True


def test_remediation_certificate_schemas_are_strict() -> None:
    for name in (
        "remediated-checkpoint.schema.json",
        "optimization-remediation-seed.schema.json",
        "optimization-remediation.schema.json",
        "optimization-remediation-readback.schema.json",
        "remediation-science.schema.json",
        "remediation-science-readback.schema.json",
    ):
        schema = json.loads(
            (ROUTE_ROOT / name).read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False


def test_remediation_science_reapplies_phase6_statistical_gates() -> None:
    source = (
        ROUTE_ROOT / "certify_remediation_science.py"
    ).read_text(encoding="utf-8")
    readback = (
        ROUTE_ROOT / "verify_remediation_science.py"
    ).read_text(encoding="utf-8")

    assert '"sampling_acceptance"' in source
    assert '"effective_samples"' in source
    assert '"gap_precision"' in source
    assert '"three_seed_consistency"' in source
    assert '"rotation_invariance"' in source
    assert "final_gap_standard_error" in source
    assert "itertools.combinations(results, 2)" in source
    assert "no_ed_gradient" in source
    assert "no_ed_checkpoint_selection" in source
    assert 'require(summary["result"])' in readback
    assert 'require(summary["checkpoint"])' in readback


def test_remediation_does_not_import_ed_and_uses_fixed_final_update() -> None:
    source = (ROUTE_ROOT / "remediate_dplus0.py").read_text(
        encoding="utf-8"
    )
    batch = (ROUTE_ROOT / "optimization_remediation.sbatch").read_text(
        encoding="utf-8"
    )
    assert "from benchmark_v0" not in source
    assert "import benchmark_v0" not in source
    assert "ProcessPoolExecutor" in source
    assert "max_workers=3" in source
    assert "initial_checkpoint=request" in source
    assert "nvidia-smi" in batch


def test_remediation_preserves_train_seed_return_order(
    monkeypatch: Any,
) -> None:
    checkpoint = {"kind": "checkpoint"}
    result = {
        "kind": "result",
        "final_training_objective": 1.25,
        "final_gap": 0.125,
    }
    observed: dict[str, Any] = {}

    def fake_train_seed(seed: int, **kwargs: Any) -> tuple[dict, dict]:
        observed.update(seed=seed, **kwargs)
        return checkpoint, result

    monkeypatch.setattr(
        remediate_dplus0, "train_seed", fake_train_seed
    )
    payload = remediate_dplus0._run_one(
        {
            "seed": 848,
            "architecture": {},
            "architecture_sha256": "0" * 64,
            "base_checkpoint": {},
            "protocol": {
                "chains": 4,
                "additional_updates": 48,
                "samples_per_update_per_chain": 16,
                "proposal_sweeps": 2,
                "final_samples_per_chain": 512,
                "learning_rate": 0.03,
                "diagonal_shift": 0.03,
                "trust_radius": 0.02,
                "checkpoint_selection": "fixed-final-update",
            },
        }
    )

    assert payload["checkpoint"] is checkpoint
    assert payload["result"] is result
    assert observed["progress_every"] == 1


def test_independent_remediation_readback_rehashes_every_layer() -> None:
    source = (
        ROUTE_ROOT / "verify_optimization_remediation.py"
    ).read_text(encoding="utf-8")
    assert 'require(certificate["phase7_stage_gate"])' in source
    assert 'require(certificate["protocol"])' in source
    assert 'require(certificate["architecture"])' in source
    assert 'require(reference["result"])' in source
    assert 'require(reference["checkpoint"])' in source
    assert 'require(checkpoint["base_checkpoint"])' in source


def test_original_training_defaults_remain_unchanged() -> None:
    source = (ROUTE_ROOT / "train_dplus0.py").read_text(
        encoding="utf-8"
    )
    assert "learning_rate: float = 0.1" in source
    assert "diagonal_shift: float = 1.0e-2" in source
    assert "trust_radius: float = 0.05" in source
    assert 'checkpoint_selection: str = "final_update"' in source


def test_dplus0_freeze_requires_phase6_and_remediated_phase7() -> None:
    source = (ROUTE_ROOT / "freeze_architecture.py").read_text(
        encoding="utf-8"
    )
    assert "phase6-final-v2.schema.json" in source
    assert "phase6-final-v2-readback.schema.json" in source
    assert '"dplus0-remediation-gate"' in source
    assert '"dplus0-sufficient"' in source
    assert '"selected_capacity": "D+0"' in source
    assert '"heldout_accessed": False' in source
    assert '"beyond_ed_accessed": False' in source


def test_remediated_freeze_requires_science_and_phase7() -> None:
    source = (
        ROUTE_ROOT / "freeze_remediated_architecture.py"
    ).read_text(encoding="utf-8")
    dependency_schema = json.loads(
        (
            ROUTE_ROOT / "future" / "dependency.schema.json"
        ).read_text(encoding="utf-8")
    )
    architecture = dependency_schema["$defs"]["architecture_freeze"]
    properties = architecture["properties"]

    assert "remediation scientific gate did not pass" in source
    assert '!= "dplus0-sufficient"' in source
    assert '!= "keep-D+0"' in source
    assert '"heldout_accessed": False' in source
    assert '"beyond_ed_accessed": False' in source
    assert "remediation_science" in properties
    assert "remediation_science_readback" in properties
