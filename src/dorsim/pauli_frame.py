from __future__ import annotations

import numpy as np

from .circuit import Circuit, Operation
from .pauli import CLIFFORD_GATES, bits_from_code, code_from_bits, local_conjugation_map


class PauliFrame:
    """Many-shot Pauli-frame propagation using a (2n, shots) uint8 matrix."""

    def __init__(self, circuit: Circuit, shots: int, seed: int | None = None):
        self.circuit = circuit
        self.n = circuit.num_qubits
        self.shots = int(shots)
        self.rng = np.random.default_rng(seed)
        self.frame = np.zeros((2 * self.n, self.shots), dtype=np.uint8)
        self.frame[self.n :, :] = self.rng.integers(0, 2, size=(self.n, self.shots), dtype=np.uint8)
        self.measurement_flips = np.zeros((circuit.num_measurements, self.shots), dtype=np.uint8)
        self.samples: np.ndarray | None = None
        self._measurement_index = 0

    def _conjugate_frame_by_gate(self, op: Operation) -> None:
        targets = op.targets
        mapping = local_conjugation_map(op.name, len(targets))
        for shot in range(self.shots):
            local_in = tuple(
                code_from_bits(self.frame[q, shot], self.frame[self.n + q, shot])
                for q in targets
            )
            _, local_out = mapping[local_in]
            for q, code in zip(targets, local_out):
                self.frame[q, shot], self.frame[self.n + q, shot] = bits_from_code(code)

    def _multiply_pauli_error(self, q: int, x: np.ndarray | int, z: np.ndarray | int) -> None:
        self.frame[q, :] ^= np.asarray(x, dtype=np.uint8)
        self.frame[self.n + q, :] ^= np.asarray(z, dtype=np.uint8)

    def _apply_noise(self, op: Operation) -> None:
        q = op.targets[0]
        if op.name == "X_ERROR":
            m = self.rng.random(self.shots) < op.p
            self._multiply_pauli_error(q, m, 0)
        elif op.name == "Y_ERROR":
            m = self.rng.random(self.shots) < op.p
            self._multiply_pauli_error(q, m, m)
        elif op.name == "Z_ERROR":
            m = self.rng.random(self.shots) < op.p
            self._multiply_pauli_error(q, 0, m)
        elif op.name == "DEPOLARIZE1":
            r = self.rng.random(self.shots)
            x = ((r < op.p / 3) | ((2 * op.p / 3 <= r) & (r < op.p))).astype(np.uint8)
            z = (((op.p / 3 <= r) & (r < op.p))).astype(np.uint8)
            self._multiply_pauli_error(q, x, z)

    def _measure_z(self, q: int) -> None:
        self.measurement_flips[self._measurement_index] = self.frame[q]
        self._measurement_index += 1
        self.frame[self.n + q] ^= self.rng.integers(0, 2, size=self.shots, dtype=np.uint8)

    def _reset_z(self, q: int) -> None:
        self.frame[q] = 0
        self.frame[self.n + q] = self.rng.integers(0, 2, size=self.shots, dtype=np.uint8)

    def run(self, reference: np.ndarray | None = None) -> "PauliFrame":
        for op in self.circuit.operations:
            if op.name in CLIFFORD_GATES:
                self._conjugate_frame_by_gate(op)
            elif op.name in {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"}:
                self._apply_noise(op)
            elif op.name == "M":
                self._measure_z(op.targets[0])
            elif op.name == "R":
                self._reset_z(op.targets[0])
        if reference is not None:
            self.samples = np.asarray(reference, dtype=np.uint8)[:, None] ^ self.measurement_flips
        return self
