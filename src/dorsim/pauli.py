from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

from .circuit import Operation


CLIFFORD_GATES = {"H", "S", "S_DAG", "X", "Y", "Z", "CX", "CY", "CZ", "SWAP"}
SINGLE_QUBIT_GATES = {"H", "S", "S_DAG", "X", "Y", "Z"}
TWO_QUBIT_GATES = {"CX", "CY", "CZ", "SWAP"}


# Local Hermitian Pauli code used by the binary xz representation.
# I=0b00, X=0b01, Z=0b10, Y=0b11.
I_CODE = 0
X_CODE = 1
Z_CODE = 2
Y_CODE = 3


def code_from_bits(x: int, z: int) -> int:
    return int(x) | (int(z) << 1)


def bits_from_code(code: int) -> tuple[int, int]:
    return code & 1, (code >> 1) & 1


# Phase in i**k from multiplying local Hermitian Paulis on the left by rhs.
LOCAL_MUL_PHASE = np.array(
    [
        [0, 0, 0, 0],  # I * I,X,Z,Y
        [0, 0, 3, 1],  # X * I,X,Z,Y
        [0, 1, 0, 3],  # Z * I,X,Z,Y
        [0, 3, 1, 0],  # Y * I,X,Z,Y
    ],
    dtype=np.uint8,
)


def identity_tableau(n: int) -> tuple[np.ndarray, np.ndarray]:
    tableau = np.eye(2*n, dtype=int)
    sign = np.zeros(2 * n, dtype=np.uint8)
    return tableau, sign


def pauli_mul_phase(
    lhs_x: np.ndarray,
    lhs_z: np.ndarray,
    lhs_phase: int,
    rhs_x: np.ndarray,
    rhs_z: np.ndarray,
    rhs_sign: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns i**phase * lhs * rhs, keeping xz bits separate from phase."""

    phase = (int(lhs_phase) + 2 * int(rhs_sign)) & 3
    old_x = lhs_x.copy()
    old_z = lhs_z.copy()
    for q in range(lhs_x.size):
        a = code_from_bits(old_x[q], old_z[q])
        b = code_from_bits(rhs_x[q], rhs_z[q])
        phase = (phase + int(LOCAL_MUL_PHASE[a, b])) & 3
    return old_x ^ rhs_x, old_z ^ rhs_z, phase


def tableau_apply(
    tableau: np.ndarray,
    sign: np.ndarray,
    pauli_x: np.ndarray,
    pauli_z: np.ndarray,
    pauli_sign: int,
) -> tuple[np.ndarray, int]:
    """Applies an inverse tableau to one Hermitian Pauli string."""

    n = pauli_x.size
    out_x = np.zeros(n, dtype=np.uint8)
    out_z = np.zeros(n, dtype=np.uint8)
    phase = (2 * int(pauli_sign)) & 3
    for q in range(n):
        code = code_from_bits(pauli_x[q], pauli_z[q])
        if code == I_CODE:
            continue
        if code == X_CODE:
            out_x, out_z, phase = pauli_mul_phase(
                out_x, out_z, phase, tableau[q, :n], tableau[q, n:], sign[q]
            )
        elif code == Z_CODE:
            out_x, out_z, phase = pauli_mul_phase(
                out_x, out_z, phase, tableau[n + q, :n], tableau[n + q, n:], sign[n + q]
            )
        else:
            # Y = i X Z. The extra i cancels the -i from multiplying X then Z
            # in the identity tableau, producing +Y.
            phase = (phase + 1) & 3
            out_x, out_z, phase = pauli_mul_phase(
                out_x, out_z, phase, tableau[q, :n], tableau[q, n:], sign[q]
            )
            out_x, out_z, phase = pauli_mul_phase(
                out_x, out_z, phase, tableau[n + q, :n], tableau[n + q, n:], sign[n + q]
            )
    return np.concatenate([out_x, out_z]).astype(np.uint8), np.uint8((phase >> 1) & 1)


def kron_all(mats: Iterable[np.ndarray]) -> np.ndarray:
    out = np.array([[1]], dtype=np.complex128)
    for m in mats:
        out = np.kron(out, m)
    return out


@lru_cache(maxsize=None)
def gate_matrix(name: str) -> np.ndarray:
    i = 1j
    i2 = np.eye(2, dtype=np.complex128)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.array([[0, -i], [i, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    s = np.array([[1, 0], [0, i]], dtype=np.complex128)
    if name == "H":
        return h
    if name == "S":
        return s
    if name == "S_DAG":
        return s.conj().T
    if name == "X":
        return x
    if name == "Y":
        return y
    if name == "Z":
        return z
    if name == "CX":
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=np.complex128,
        )
    if name == "CY":
        return np.block([[i2, np.zeros((2, 2))], [np.zeros((2, 2)), y]]).astype(np.complex128)
    if name == "CZ":
        return np.diag([1, 1, 1, -1]).astype(np.complex128)
    if name == "SWAP":
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
    return np.eye(2, dtype=np.complex128)


@lru_cache(maxsize=None)
def pauli_tuple_matrix(codes: tuple[int, ...]) -> np.ndarray:
    i = 1j
    mats = {
        I_CODE: np.eye(2, dtype=np.complex128),
        X_CODE: np.array([[0, 1], [1, 0]], dtype=np.complex128),
        Z_CODE: np.array([[1, 0], [0, -1]], dtype=np.complex128),
        Y_CODE: np.array([[0, -i], [i, 0]], dtype=np.complex128),
    }
    return kron_all(mats[c] for c in codes)


@lru_cache(maxsize=None)
def local_conjugation_map(name: str, arity: int) -> dict[tuple[int, ...], tuple[int, tuple[int, ...]]]:
    """Maps local Pauli codes through G^-1 P G."""

    u = gate_matrix(name)
    out: dict[tuple[int, ...], tuple[int, tuple[int, ...]]] = {}
    for k in range(4**arity):
        codes = []
        v = k
        for _ in range(arity):
            codes.append(v & 3)
            v >>= 2
        codes_t = tuple(codes)
        p = pauli_tuple_matrix(codes_t)
        conjugated = u.conj().T @ p @ u
        for j in range(4**arity):
            out_codes = []
            w = j
            for _ in range(arity):
                out_codes.append(w & 3)
                w >>= 2
            out_t = tuple(out_codes)
            q = pauli_tuple_matrix(out_t)
            if np.allclose(conjugated, q):
                out[codes_t] = (0, out_t)
                break
            if np.allclose(conjugated, -q):
                out[codes_t] = (1, out_t)
                break
    return out


def conjugate_pauli_by_gate(
    x: np.ndarray,
    z: np.ndarray,
    sign: int,
    op: Operation,
) -> tuple[np.ndarray, np.ndarray, np.uint8]:
    """Conjugates one Pauli string as G^-1 P G."""

    if op.name not in CLIFFORD_GATES:
        return x.copy(), z.copy(), np.uint8(sign)
    targets = op.targets
    arity = len(targets)
    mapping = local_conjugation_map(op.name, arity)
    local_in = tuple(code_from_bits(x[q], z[q]) for q in targets)
    flip, local_out = mapping[local_in]
    nx = x.copy()
    nz = z.copy()
    for q, code in zip(targets, local_out):
        nx[q], nz[q] = bits_from_code(code)
    return nx, nz, np.uint8(int(sign) ^ flip)
