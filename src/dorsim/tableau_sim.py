from __future__ import annotations

import numpy as np

from .circuit import Circuit, Operation
from .pauli import (
    CLIFFORD_GATES,
    SINGLE_QUBIT_GATES,
    TWO_QUBIT_GATES,
    code_from_bits,
    conjugate_pauli_by_gate,
    identity_tableau,
    tableau_apply,
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
        for row in range(2 * self.n):
            bx = np.zeros(self.n, dtype=np.uint8)
            bz = np.zeros(self.n, dtype=np.uint8)
            if row < self.n:
                bx[row] = 1
            else:
                bz[row - self.n] = 1
            px, pz, ps = conjugate_pauli_by_gate(bx, bz, 0, op)
            self.tableau[row], self.sign[row] = tableau_apply(old_tableau, old_sign, px, pz, ps)

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
