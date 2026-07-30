"""Chiral rank-two pair-transition operators in the spherical LLL.

The convention is fixed algebraically:

``plus``
    increases pair relative angular momentum by two, ``R -> R + 2``;
``minus``
    decreases pair relative angular momentum by two, ``R -> R - 2``.

With this convention the two families obey

``O_plus(m).H = (-1)**m O_minus(-m)``.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np

from route_d_plus.scalar import FockSpace, one_body_fock_matrix
from route_d_plus.tensor import angular_momentum_matrices, canonical_tensor


def _maximum_residual(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


@cache
def pair_angular_momentum_projectors(
    two_q: int,
) -> tuple[FockSpace, dict[int, np.ndarray]]:
    """Return antisymmetric two-particle projectors labelled by total ``J``."""

    if isinstance(two_q, bool) or not isinstance(two_q, int):
        raise TypeError("two_q must be an integer")
    if two_q < 2:
        raise ValueError("two_q must be at least two")
    space = FockSpace.build(two_q + 1, 2)
    one_body_j = angular_momentum_matrices(two_q)
    total_j = [
        one_body_fock_matrix(space, component)
        for component in one_body_j
    ]
    total_j2 = sum(component @ component for component in total_j)
    eigenvalues, eigenvectors = np.linalg.eigh(total_j2)
    projectors: dict[int, np.ndarray] = {}
    for total_j_value in range(two_q + 1):
        target = total_j_value * (total_j_value + 1)
        selected = np.isclose(
            eigenvalues, target, rtol=0.0, atol=1.0e-9
        )
        if not np.any(selected):
            continue
        vectors = eigenvectors[:, selected]
        projector = np.asarray(vectors @ vectors.conj().T)
        projector.flags.writeable = False
        projectors[total_j_value] = projector
    return space, projectors


@cache
def chiral_pair_tensors(
    two_q: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Construct the ``(O_plus, O_minus)`` rank-two pair tensors.

    A projected total one-body quadrupole supplies the reduced matrix elements.
    Projection onto adjacent allowed pair-``J`` sectors removes every
    transition except ``Delta R = +/-2``.
    """

    space, projectors = pair_angular_momentum_projectors(two_q)
    plus: dict[int, np.ndarray] = {}
    minus: dict[int, np.ndarray] = {}
    for magnetic in range(-2, 3):
        quadrupole = one_body_fock_matrix(
            space, canonical_tensor(two_q, 2, magnetic)
        )
        raises_relative = np.zeros_like(quadrupole)
        lowers_relative = np.zeros_like(quadrupole)
        for total_j, source in projectors.items():
            lower_j = projectors.get(total_j - 2)
            if lower_j is None:
                continue
            raises_relative += lower_j @ quadrupole @ source
            lowers_relative += source @ quadrupole @ lower_j
        plus[magnetic] = np.asarray(raises_relative)
        minus[magnetic] = np.asarray(lowers_relative)
        plus[magnetic].flags.writeable = False
        minus[magnetic].flags.writeable = False
    return plus, minus


def certify_chiral_pair_tensors(two_q: int) -> dict[str, object]:
    """Return numerical algebra evidence for the registered phase convention."""

    space, projectors = pair_angular_momentum_projectors(two_q)
    plus, minus = chiral_pair_tensors(two_q)
    one_body_j = angular_momentum_matrices(two_q)
    jx, jy, jz = [
        one_body_fock_matrix(space, component)
        for component in one_body_j
    ]
    raising = jx + 1.0j * jy

    adjoint_residual = max(
        _maximum_residual(
            plus[magnetic].conj().T
            - ((-1) ** magnetic) * minus[-magnetic]
        )
        for magnetic in range(-2, 3)
    )
    jz_residual = max(
        _maximum_residual(
            jz @ family[magnetic]
            - family[magnetic] @ jz
            - magnetic * family[magnetic]
        )
        for family in (plus, minus)
        for magnetic in range(-2, 3)
    )
    raising_residual = 0.0
    for family in (plus, minus):
        for magnetic in range(-2, 2):
            coefficient = math.sqrt(
                (2 - magnetic) * (2 + magnetic + 1)
            )
            raising_residual = max(
                raising_residual,
                _maximum_residual(
                    raising @ family[magnetic]
                    - family[magnetic] @ raising
                    - coefficient * family[magnetic + 1]
                ),
            )

    forbidden_plus = 0.0
    forbidden_minus = 0.0
    allowed_plus = 0.0
    allowed_minus = 0.0
    for target_j, target in projectors.items():
        for source_j, source in projectors.items():
            plus_norm = max(
                float(
                    np.linalg.norm(
                        target @ plus[magnetic] @ source
                    )
                )
                for magnetic in range(-2, 3)
            )
            minus_norm = max(
                float(
                    np.linalg.norm(
                        target @ minus[magnetic] @ source
                    )
                )
                for magnetic in range(-2, 3)
            )
            delta_relative = source_j - target_j
            if delta_relative == 2:
                allowed_plus = max(allowed_plus, plus_norm)
            else:
                forbidden_plus = max(forbidden_plus, plus_norm)
            if delta_relative == -2:
                allowed_minus = max(allowed_minus, minus_norm)
            else:
                forbidden_minus = max(forbidden_minus, minus_norm)

    tolerance = 1.0e-10
    gates = {
        "pair_basis_complete": (
            _maximum_residual(
                sum(projectors.values())
                - np.eye(space.dimension, dtype=np.complex128)
            )
            < tolerance
        ),
        "spherical_adjoint": adjoint_residual < tolerance,
        "rank_two_jz": jz_residual < tolerance,
        "rank_two_raising": raising_residual < tolerance,
        "plus_is_r_to_r_plus_2": (
            forbidden_plus < tolerance and allowed_plus > tolerance
        ),
        "minus_is_r_to_r_minus_2": (
            forbidden_minus < tolerance and allowed_minus > tolerance
        ),
    }
    return {
        "two_q": two_q,
        "pair_dimension": space.dimension,
        "allowed_total_j": sorted(projectors),
        "allowed_relative_r": sorted(
            two_q - total_j for total_j in projectors
        ),
        "phase_convention": {
            "plus": "R-to-R-plus-2",
            "minus": "R-to-R-minus-2",
            "adjoint": "O_plus(m)^dagger=(-1)^m O_minus(-m)",
        },
        "residuals": {
            "adjoint": adjoint_residual,
            "jz_commutator": jz_residual,
            "raising_commutator": raising_residual,
            "forbidden_plus_transition": forbidden_plus,
            "forbidden_minus_transition": forbidden_minus,
        },
        "allowed_transition_norms": {
            "plus": allowed_plus,
            "minus": allowed_minus,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


__all__ = [
    "certify_chiral_pair_tensors",
    "chiral_pair_tensors",
    "pair_angular_momentum_projectors",
]
