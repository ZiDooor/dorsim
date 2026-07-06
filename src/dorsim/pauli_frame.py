from __future__ import annotations

import numpy as np

from .circuit import Circuit, Operation, RecTarget
from .pauli import SINGLE_QUBIT_GATES, TWO_QUBIT_GATES


class PauliFrame:
    """Many-shot Pauli-frame propagation using a (shots, 2n) uint8 matrix."""

    def __init__(self, circuit: Circuit, shots: int, seed: int | None = None):
        self.circuit = circuit
        self.n = circuit.num_qubits
        self.shots = int(shots)
        self.rng = np.random.default_rng(seed)
        self.frame = np.zeros((self.shots, 2 * self.n), dtype=np.uint8)
        self.frame[:, self.n :] = self.rng.integers(0, 2, size=(self.shots, self.n), dtype=np.uint8)
        self.measurement_flips = np.zeros((self.shots, circuit.num_measurements), dtype=np.uint8)
        self.samples: np.ndarray | None = None
        self._measurement_index = 0

    def update(self, frame: np.ndarray | None = None, *, circuit: Circuit | None = None) -> "PauliFrame":
        if circuit is not None:
            self.circuit = circuit
        if frame is not None:
            new_frame = np.asarray(frame, dtype=np.uint8).copy()
            self.shots = new_frame.shape[0]
            self.frame = new_frame
        assert self.frame.shape[1] == 2 * self.circuit.num_qubits
        self.n = self.circuit.num_qubits
        self.measurement_flips = np.zeros((self.shots, self.circuit.num_measurements), dtype=np.uint8)
        self.samples = None
        self._measurement_index = 0
        return self

    def select_qubits(self, qubits, *, circuit: Circuit) -> "PauliFrame":
        qubits = list(qubits)
        assert len(qubits) == circuit.num_qubits
        selected = self.frame[:, qubits + [self.n + q for q in qubits]]
        return self.update(selected, circuit=circuit)

    @classmethod
    def bunch(cls, frames, *, circuit: Circuit, seed: int | None = None) -> "PauliFrame":
        frames = list(frames)
        shots = min(frame.shots for frame in frames)
        total_n = sum(frame.n for frame in frames)
        assert circuit.num_qubits == total_n

        x_part = np.concatenate([frame.frame[:shots, : frame.n] for frame in frames], axis=1)
        z_part = np.concatenate([frame.frame[:shots, frame.n :] for frame in frames], axis=1)

        out = cls.__new__(cls)
        out.circuit = circuit
        out.n = total_n
        out.shots = shots
        out.rng = np.random.default_rng(seed)
        out.frame = np.concatenate([x_part, z_part], axis=1).astype(np.uint8, copy=True)
        out.measurement_flips = np.zeros((shots, circuit.num_measurements), dtype=np.uint8)
        out.samples = None
        out._measurement_index = 0
        return out

    def _conjugate_frame_by_gate(self, op: Operation) -> None:
        if op.name in {"CX", "CZ"} and isinstance(op.targets[0], RecTarget):
            self._apply_feedback_gate(op)
            return
        if op.name in {"X", "Y", "Z"}:
            return
        if op.name == "H":
            q = op.targets[0]
            self.frame[:, [q, self.n + q]] = self.frame[:, [self.n + q, q]]
        elif op.name in {"S", "S_DAG"}:
            q = op.targets[0]
            self.frame[:, self.n + q] ^= self.frame[:, q]
        elif op.name == "SWAP":
            a, b = op.targets
            self.frame[:, [a, b]] = self.frame[:, [b, a]]
            self.frame[:, [self.n + a, self.n + b]] = self.frame[:, [self.n + b, self.n + a]]
        elif op.name == "CX":
            a, b = op.targets
            self.frame[:, b] ^= self.frame[:, a]
            self.frame[:, self.n + a] ^= self.frame[:, self.n + b]
        elif op.name == "CZ":
            a, b = op.targets
            self.frame[:, self.n + a] ^= self.frame[:, b]
            self.frame[:, self.n + b] ^= self.frame[:, a]
        elif op.name == "CY":
            a, b = op.targets
            xa = self.frame[:, a].copy()
            za = self.frame[:, self.n + a].copy()
            xb = self.frame[:, b].copy()
            zb = self.frame[:, self.n + b].copy()
            self.frame[:, a] = xa
            self.frame[:, self.n + a] = za ^ xb ^ zb
            self.frame[:, b] = xb ^ xa
            self.frame[:, self.n + b] = zb ^ xa

    def _apply_feedback_gate(self, op: Operation) -> None:
        rec, q = op.targets
        flips = self.measurement_flips[:, self._measurement_index + rec.offset]
        if op.name == "CX":
            self._multiply_pauli_error(q, flips, 0)
        elif op.name == "CZ":
            self._multiply_pauli_error(q, 0, flips)

    def _multiply_pauli_error(self, q: int, x: np.ndarray | int, z: np.ndarray | int) -> None:
        self.frame[:, q] ^= np.asarray(x, dtype=np.uint8)
        self.frame[:, self.n + q] ^= np.asarray(z, dtype=np.uint8)

    def _apply_noise_to_target(self, op: Operation, q: int) -> None:
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

    def _apply_depolarize2_to_pair(self, op: Operation, a: int, b: int) -> None:
        # Order: IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ.
        codes_a = np.array([0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2], dtype=np.uint8)
        codes_b = np.array([1, 3, 2, 0, 1, 3, 2, 0, 1, 3, 2, 0, 1, 3, 2], dtype=np.uint8)
        mask = (self.rng.random(self.shots) < op.p).astype(np.uint8)
        choices = self.rng.integers(0, 15, size=self.shots)
        pa = codes_a[choices]
        pb = codes_b[choices]

        self._multiply_pauli_error(a, (pa & 1) * mask, ((pa >> 1) & 1) * mask)
        self._multiply_pauli_error(b, (pb & 1) * mask, ((pb >> 1) & 1) * mask)

    def _apply_noise(self, op: Operation) -> None:
        if op.name == "DEPOLARIZE2":
            assert len(op.targets) % 2 == 0
            for k in range(0, len(op.targets), 2):
                self._apply_depolarize2_to_pair(op, op.targets[k], op.targets[k + 1])
            return
        for q in op.targets:
            self._apply_noise_to_target(op, q)

    def _measure_z(self, q: int) -> None:
        self.measurement_flips[:, self._measurement_index] = self.frame[:, q]
        self._measurement_index += 1
        self.frame[:, self.n + q] ^= self.rng.integers(0, 2, size=self.shots, dtype=np.uint8)

    def _measure_x(self, q: int) -> None:
        self.measurement_flips[:, self._measurement_index] = self.frame[:, self.n + q]
        self._measurement_index += 1
        self.frame[:, q] ^= self.rng.integers(0, 2, size=self.shots, dtype=np.uint8)

    def _reset_z(self, q: int) -> None:
        self.frame[:, q] = 0
        self.frame[:, self.n + q] = self.rng.integers(0, 2, size=self.shots, dtype=np.uint8)

    def _iter_single_qubit_ops(self, op: Operation):
        for q in op.targets:
            yield Operation(op.name, (q,), op.p)

    def _iter_two_qubit_ops(self, op: Operation):
        assert len(op.targets) % 2 == 0
        for k in range(0, len(op.targets), 2):
            yield Operation(op.name, (op.targets[k], op.targets[k + 1]), op.p)

    def run(self, reference: np.ndarray | None = None) -> "PauliFrame":
        for op in self.circuit.operations:
            if op.name in SINGLE_QUBIT_GATES:
                for single in self._iter_single_qubit_ops(op):
                    self._conjugate_frame_by_gate(single)
            elif op.name in TWO_QUBIT_GATES:
                for pair in self._iter_two_qubit_ops(op):
                    self._conjugate_frame_by_gate(pair)
            elif op.name in {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}:
                self._apply_noise(op)
            elif op.name == "M":
                for q in op.targets:
                    self._measure_z(q)
            elif op.name == "MX":
                for q in op.targets:
                    self._measure_x(q)
            elif op.name == "R":
                for q in op.targets:
                    self._reset_z(q)
        if reference is None:
            self.samples = self.measurement_flips.copy()
        else:
            self.samples = self.measurement_flips ^ np.asarray(reference, dtype=np.uint8)[None, :]
        return self
