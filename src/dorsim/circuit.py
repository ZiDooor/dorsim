from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    name: str
    targets: tuple[int, ...]
    p: float = 0.0


class Circuit:
    """Small circuit builder for Clifford, measurement, reset, and Pauli noise ops."""

    def __init__(self, num_qubits: int):
        self.num_qubits = int(num_qubits)
        self.operations: list[Operation] = []

    def append(self, name: str, *targets: int, p: float = 0.0) -> "Circuit":
        self.operations.append(Operation(name.upper(), tuple(map(int, targets)), float(p)))
        return self

    def h(self, *targets: int) -> "Circuit":
        for q in targets:
            self.append("H", q)
        return self

    def s(self, *targets: int) -> "Circuit":
        for q in targets:
            self.append("S", q)
        return self

    def sdg(self, *targets: int) -> "Circuit":
        for q in targets:
            self.append("S_DAG", q)
        return self

    def x(self, *targets: int) -> "Circuit":
        for q in targets:
            self.append("X", q)
        return self

    def y(self, *targets: int) -> "Circuit":
        for q in targets:
            self.append("Y", q)
        return self

    def z(self, *targets: int) -> "Circuit":
        for q in targets:
            self.append("Z", q)
        return self

    def cx(self, control: int, target: int) -> "Circuit":
        return self.append("CX", control, target)

    def cy(self, control: int, target: int) -> "Circuit":
        return self.append("CY", control, target)

    def cz(self, a: int, b: int) -> "Circuit":
        return self.append("CZ", a, b)

    def swap(self, a: int, b: int) -> "Circuit":
        return self.append("SWAP", a, b)

    def m(self, *targets: int) -> "Circuit":
        for q in targets:
            self.append("M", q)
        return self

    def r(self, *targets: int) -> "Circuit":
        for q in targets:
            self.append("R", q)
        return self

    def x_error(self, target: int, p: float) -> "Circuit":
        return self.append("X_ERROR", target, p=p)

    def y_error(self, target: int, p: float) -> "Circuit":
        return self.append("Y_ERROR", target, p=p)

    def z_error(self, target: int, p: float) -> "Circuit":
        return self.append("Z_ERROR", target, p=p)

    def depolarize1(self, target: int, p: float) -> "Circuit":
        return self.append("DEPOLARIZE1", target, p=p)

    @property
    def num_measurements(self) -> int:
        return sum(op.name == "M" for op in self.operations)

    def without_noise(self) -> "Circuit":
        out = Circuit(self.num_qubits)
        out.operations = [
            op
            for op in self.operations
            if op.name not in {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"}
        ]
        return out

    def to_stim_circuit(self):
        import stim

        c = stim.Circuit()
        for op in self.operations:
            if op.name in {"H", "S", "X", "Y", "Z", "M", "R", "CX", "CY", "CZ", "SWAP"}:
                c.append(op.name, op.targets)
            elif op.name == "S_DAG":
                c.append("S_DAG", op.targets)
            elif op.name in {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"}:
                c.append(op.name, op.targets, op.p)
        return c
