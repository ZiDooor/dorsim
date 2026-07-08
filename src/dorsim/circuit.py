from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class RecTarget:
    offset: int

    def __repr__(self) -> str:
        return f"target_rec({self.offset})"

    def __str__(self) -> str:
        return f"rec[{self.offset}]"


def target_rec(offset: int) -> RecTarget:
    return RecTarget(int(offset))


Target = int | RecTarget


@dataclass(frozen=True)
class Operation:
    name: str
    targets: tuple[Target, ...]
    p: float = 0.0


class Circuit:
    """Small circuit builder for Clifford, measurement, reset, and Pauli noise ops."""

    def __init__(self, num_qubits: int):
        self.num_qubits = int(num_qubits)
        self.operations: list[Operation] = []
        # If there is no operation, identity?

    def _target(self, target: Target) -> Target:
        if isinstance(target, RecTarget):
            return target
        return int(target)

    def append(self, name: str, *targets: Target, p: float = 0.0) -> "Circuit":
        self.operations.append(Operation(name.upper(), tuple(self._target(t) for t in targets), float(p)))
        return self

    def h(self, targets: Iterable[int]) -> "Circuit":
        for q in targets:
            self.append("H", q)
        return self

    def s(self, targets: Iterable[int]) -> "Circuit":
        for q in targets:
            self.append("S", q)
        return self

    def sdg(self, targets: Iterable[int]) -> "Circuit":
        for q in targets:
            self.append("S_DAG", q)
        return self

    def x(self, targets: Iterable[int]) -> "Circuit":
        for q in targets:
            self.append("X", q)
        return self

    def y(self, targets: Iterable[int]) -> "Circuit":
        for q in targets:
            self.append("Y", q)
        return self

    def z(self, targets: Iterable[int]) -> "Circuit":
        for q in targets:
            self.append("Z", q)
        return self

    def cx(self, targets: Iterable[Target]) -> "Circuit":
        flat = list(targets)
        assert len(flat) % 2 == 0
        for k in range(0, len(flat), 2):
            self.append("CX", flat[k], flat[k + 1])
        return self

    def cy(self, targets: Iterable[int]) -> "Circuit":
        flat = list(targets)
        assert len(flat) % 2 == 0
        for k in range(0, len(flat), 2):
            self.append("CY", flat[k], flat[k + 1])
        return self

    def cz(self, targets: Iterable[Target]) -> "Circuit":
        flat = list(targets)
        assert len(flat) % 2 == 0
        for k in range(0, len(flat), 2):
            self.append("CZ", flat[k], flat[k + 1])
        return self

    def swap(self, targets: Iterable[int]) -> "Circuit":
        flat = list(targets)
        assert len(flat) % 2 == 0
        for k in range(0, len(flat), 2):
            self.append("SWAP", flat[k], flat[k + 1])
        return self

    def m(self, targets: Iterable[int]) -> "Circuit":
        for q in targets:
            self.append("M", q)
        return self

    def mx(self, targets: Iterable[int]) -> "Circuit":
        for q in targets:
            self.append("MX", q)
        return self

    def r(self, targets: Iterable[int]) -> "Circuit":
        for q in targets:
            self.append("R", q)
        return self

    def x_error(self, targets: Iterable[int], p: float) -> "Circuit":
        for q in targets:
            self.append("X_ERROR", q, p=p)
        return self

    def y_error(self, targets: Iterable[int], p: float) -> "Circuit":
        for q in targets:
            self.append("Y_ERROR", q, p=p)
        return self

    def z_error(self, targets: Iterable[int], p: float) -> "Circuit":
        for q in targets:
            self.append("Z_ERROR", q, p=p)
        return self

    def depolarize1(self, targets: Iterable[int], p: float) -> "Circuit":
        for q in targets:
            self.append("DEPOLARIZE1", q, p=p)
        return self

    def depolarize2(self, targets: Iterable[int], p: float) -> "Circuit":
        flat = list(targets)
        assert len(flat) % 2 == 0
        for k in range(0, len(flat), 2):
            self.append("DEPOLARIZE2", flat[k], flat[k + 1], p=p)
        return self

    @property
    def num_measurements(self) -> int:
        return sum(len(op.targets) for op in self.operations if op.name in {"M", "MX"})

    def without_noise(self) -> "Circuit":
        out = Circuit(self.num_qubits)
        out.operations = [
            op
            for op in self.operations
            if op.name not in {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}
        ]
        return out

    def to_stim_circuit(self):
        import stim

        c = stim.Circuit()
        for op in self.operations:
            targets = [
                stim.target_rec(t.offset) if isinstance(t, RecTarget) else t
                for t in op.targets
            ]
            if op.name in {"H", "S", "X", "Y", "Z", "M", "MX", "R", "CX", "CY", "CZ", "SWAP"}:
                c.append(op.name, targets)
            elif op.name == "S_DAG":
                c.append("S_DAG", targets)
            elif op.name in {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}:
                c.append(op.name, targets, op.p)
        return c
