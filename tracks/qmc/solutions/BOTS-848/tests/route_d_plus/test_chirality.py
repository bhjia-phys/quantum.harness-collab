from __future__ import annotations

import numpy as np

from route_d_plus.chirality import (
    certify_chiral_pair_tensors,
    chiral_pair_tensors,
)


def test_chiral_pair_transition_convention() -> None:
    certificate = certify_chiral_pair_tensors(7)
    assert certificate["passed"] is True
    assert certificate["allowed_relative_r"] == [1, 3, 5, 7]
    assert certificate["phase_convention"]["plus"] == "R-to-R-plus-2"


def test_chiral_pair_tensors_obey_spherical_adjoint() -> None:
    plus, minus = chiral_pair_tensors(7)
    for magnetic in range(-2, 3):
        assert np.max(
            np.abs(
                plus[magnetic].conj().T
                - ((-1) ** magnetic) * minus[-magnetic]
            )
        ) < 1.0e-12
