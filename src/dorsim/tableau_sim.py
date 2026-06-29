from __future__ import annotations

import numpy as np

from .circuit import Circuit, Operation
from .pauli import (
    X_CODE,
    Y_CODE,
    Z_CODE,
    SINGLE_QUBIT_GATES,
    TWO_QUBIT_GATES,
    code_from_bits,
    conjugate_pauli_by_gate,
    identity_tableau,
    pauli_mul_phase,
)


class TableauSim:
    """Reference stabilizer simulator using Stim-style inverse-tableau thinking."""

    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.n = circuit.num_qubits
        self.tableau, self.sign = identity_tableau(self.n)
        self.reference_measurements = np.zeros(circuit.num_measurements, dtype=np.uint8)
        self._measurement_index = 0

    def append_gate_after_circuit(self, op: Operation) -> None:
        """Physical append: U' = G U, so T'(P) = T(G^-1 P G)."""

        old_tableau = self.tableau.copy()
        old_sign = self.sign.copy()
        if op.name == "X":
            q = op.targets[0]
            self.sign[self.n + q] ^= 1
        elif op.name == "Y":
            q = op.targets[0]
            self.sign[q] ^= 1
            self.sign[self.n + q] ^= 1
        elif op.name == "Z":
            q = op.targets[0]
            self.sign[q] ^= 1
        elif op.name == "H":
            q = op.targets[0]
            self.tableau[[q, self.n + q]] = old_tableau[[self.n + q, q]]
            self.sign[[q, self.n + q]] = old_sign[[self.n + q, q]]
        elif op.name == "S":
            q = op.targets[0]
            self._set_row_from_local_paulis(q, old_tableau, old_sign, (q,), (Y_CODE,), sign_flip=1)
        elif op.name == "S_DAG":
            q = op.targets[0]
            self._set_row_from_local_paulis(q, old_tableau, old_sign, (q,), (Y_CODE,))
        elif op.name == "SWAP":
            a, b = op.targets
            self.tableau[[a, b]] = old_tableau[[b, a]]
            self.tableau[[self.n + a, self.n + b]] = old_tableau[[self.n + b, self.n + a]]
            self.sign[[a, b]] = old_sign[[b, a]]
            self.sign[[self.n + a, self.n + b]] = old_sign[[self.n + b, self.n + a]]
        elif op.name == "CX":
            a, b = op.targets
            self._set_row_from_local_paulis(a, old_tableau, old_sign, (a, b), (X_CODE, X_CODE))
            self._set_row_from_local_paulis(self.n + b, old_tableau, old_sign, (a, b), (Z_CODE, Z_CODE))
        elif op.name == "CY":
            a, b = op.targets
            self._set_row_from_local_paulis(a, old_tableau, old_sign, (a, b), (X_CODE, Y_CODE))
            self._set_row_from_local_paulis(b, old_tableau, old_sign, (a, b), (Z_CODE, X_CODE))
            self._set_row_from_local_paulis(self.n + b, old_tableau, old_sign, (a, b), (Z_CODE, Z_CODE))
        elif op.name == "CZ":
            a, b = op.targets
            self._set_row_from_local_paulis(a, old_tableau, old_sign, (a, b), (X_CODE, Z_CODE))
            self._set_row_from_local_paulis(b, old_tableau, old_sign, (a, b), (Z_CODE, X_CODE))

    def _set_row_from_local_paulis(
        self,
        row: int,
        old_tableau: np.ndarray,
        old_sign: np.ndarray,
        targets: tuple[int, ...],
        codes: tuple[int, ...],
        sign_flip: int = 0,
    ) -> None:
        out_x = np.zeros(self.n, dtype=np.uint8)
        out_z = np.zeros(self.n, dtype=np.uint8)
        phase = (2 * int(sign_flip)) & 3
        for q, code in zip(targets, codes):
            if code == X_CODE:
                out_x, out_z, phase = pauli_mul_phase(
                    out_x, out_z, phase, old_tableau[q, : self.n], old_tableau[q, self.n :], old_sign[q]
                )
            elif code == Z_CODE:
                out_x, out_z, phase = pauli_mul_phase(
                    out_x,
                    out_z,
                    phase,
                    old_tableau[self.n + q, : self.n],
                    old_tableau[self.n + q, self.n :],
                    old_sign[self.n + q],
                )
            elif code == Y_CODE:
                phase = (phase + 1) & 3
                out_x, out_z, phase = pauli_mul_phase(
                    out_x, out_z, phase, old_tableau[q, : self.n], old_tableau[q, self.n :], old_sign[q]
                )
                out_x, out_z, phase = pauli_mul_phase(
                    out_x,
                    out_z,
                    phase,
                    old_tableau[self.n + q, : self.n],
                    old_tableau[self.n + q, self.n :],
                    old_sign[self.n + q],
                )
        self.tableau[row] = np.concatenate([out_x, out_z]).astype(np.uint8)
        self.sign[row] = np.uint8((phase >> 1) & 1)

    def prepend_gate_before_circuit(self, op: Operation) -> None:
        """Physical prepend: U' = U G, so T'(P) = G^-1 T(P) G."""

        for row in range(2 * self.n):
            x = self.tableau[row, : self.n]
            z = self.tableau[row, self.n :]
            nx, nz, ns = conjugate_pauli_by_gate(x, z, int(self.sign[row]), op)
            self.tableau[row, : self.n] = nx
            self.tableau[row, self.n :] = nz
            self.sign[row] = ns

    def _iter_single_qubit_ops(self, op: Operation):
        for q in op.targets:
            yield Operation(op.name, (q,), op.p)

    def _iter_two_qubit_ops(self, op: Operation):
        assert len(op.targets) % 2 == 0
        for k in range(0, len(op.targets), 2):
            yield Operation(op.name, (op.targets[k], op.targets[k + 1]), op.p)

    def _row_has_x_or_y(self, row: int) -> bool:
        return bool(np.any(self.tableau[row, : self.n]))

    def _collapse_random_measurement(self, row: int) -> None:
        x_support = np.flatnonzero(self.tableau[row, : self.n])
        if x_support.size == 0:
            return
        pivot = int(x_support[0])

        for other in x_support[1:]:
            self.prepend_gate_before_circuit(Operation("CX", (pivot, int(other))))

        if self.tableau[row, pivot] and self.tableau[row, self.n + pivot]:
            self.prepend_gate_before_circuit(Operation("S", (pivot,)))
        self.prepend_gate_before_circuit(Operation("H", (pivot,)))

        # Fix the random branch to outcome 0 by inserting or omitting the X.
        if self.sign[row]:
            self.prepend_gate_before_circuit(Operation("X", (pivot,)))

    def measure_row(self, row: int, record: bool = True) -> np.uint8:
        if self._row_has_x_or_y(row):
            self._collapse_random_measurement(row)
        result = np.uint8(self.sign[row])
        if record:
            self.reference_measurements[self._measurement_index] = result
            self._measurement_index += 1
        return result

    def measure_z(self, q: int, record: bool = True) -> np.uint8:
        return self.measure_row(self.n + q, record=record)

    def measure_x(self, q: int, record: bool = True) -> np.uint8:
        return self.measure_row(q, record=record)

    def reset_z(self, q: int) -> None:
        self.measure_z(q, record=False)
        self.sign[self.n + q] = 0

    def run(self) -> "TableauSim":
        for op in self.circuit.without_noise().operations:
            if op.name in SINGLE_QUBIT_GATES:
                for single in self._iter_single_qubit_ops(op):
                    self.append_gate_after_circuit(single)
            elif op.name in TWO_QUBIT_GATES:
                for pair in self._iter_two_qubit_ops(op):
                    self.append_gate_after_circuit(pair)
            elif op.name == "M":
                for q in op.targets:
                    self.measure_z(q, record=True)
            elif op.name == "MX":
                for q in op.targets:
                    self.measure_x(q, record=True)
            elif op.name == "R":
                for q in op.targets:
                    self.reset_z(q)
        return self

    def format_tableau(self) -> str:
        lines = []
        for row in range(2 * self.n):
            label = f"T(X{row})" if row < self.n else f"T(Z{row - self.n})"
            lines.append(f"{label:6s} = {self.format_row(row)}")
        return "\n".join(lines)

    def format_row(self, row: int) -> str:
        chars = []
        for q in range(self.n):
            code = code_from_bits(self.tableau[row, q], self.tableau[row, self.n + q])
            chars.append("IXZY"[code])
        return ("-" if self.sign[row] else "+") + "".join(chars)
