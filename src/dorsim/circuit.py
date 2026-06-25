from __future__ import annotations

from collections.abc import Iterable
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

    def _targets(self, targets: Iterable[int]) -> tuple[int, ...]:
        return tuple(map(int, targets))

    def _append_targets(self, name: str, targets: Iterable[int], *, p: float = 0.0) -> "Circuit":
        return self.append(name, *self._targets(targets), p=p)

    def _append_pair_targets(self, name: str, targets: Iterable[int]) -> "Circuit":
        flat = self._targets(targets)
        assert len(flat) % 2 == 0
        return self.append(name, *flat)

    def h(self, targets: Iterable[int]) -> "Circuit":
        return self._append_targets("H", targets)

    def s(self, targets: Iterable[int]) -> "Circuit":
        return self._append_targets("S", targets)

    def sdg(self, targets: Iterable[int]) -> "Circuit":
        return self._append_targets("S_DAG", targets)

    def x(self, targets: Iterable[int]) -> "Circuit":
        return self._append_targets("X", targets)

    def y(self, targets: Iterable[int]) -> "Circuit":
        return self._append_targets("Y", targets)

    def z(self, targets: Iterable[int]) -> "Circuit":
        return self._append_targets("Z", targets)

    def cx(self, targets: Iterable[int]) -> "Circuit":
        return self._append_pair_targets("CX", targets)

    def cy(self, targets: Iterable[int]) -> "Circuit":
        return self._append_pair_targets("CY", targets)

    def cz(self, targets: Iterable[int]) -> "Circuit":
        return self._append_pair_targets("CZ", targets)

    def swap(self, targets: Iterable[int]) -> "Circuit":
        return self._append_pair_targets("SWAP", targets)

    def m(self, targets: Iterable[int]) -> "Circuit":
        return self._append_targets("M", targets)

    def mx(self, targets: Iterable[int]) -> "Circuit":
        return self._append_targets("MX", targets)

    def r(self, targets: Iterable[int]) -> "Circuit":
        return self._append_targets("R", targets)

    def x_error(self, targets: Iterable[int], p: float) -> "Circuit":
        return self._append_targets("X_ERROR", targets, p=p)

    def y_error(self, targets: Iterable[int], p: float) -> "Circuit":
        return self._append_targets("Y_ERROR", targets, p=p)

    def z_error(self, targets: Iterable[int], p: float) -> "Circuit":
        return self._append_targets("Z_ERROR", targets, p=p)

    def depolarize1(self, targets: Iterable[int], p: float) -> "Circuit":
        return self._append_targets("DEPOLARIZE1", targets, p=p)

    @property
    def num_measurements(self) -> int:
        return sum(len(op.targets) for op in self.operations if op.name in {"M", "MX"})

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
            if op.name in {"H", "S", "X", "Y", "Z", "M", "MX", "R", "CX", "CY", "CZ", "SWAP"}:
                c.append(op.name, op.targets)
            elif op.name == "S_DAG":
                c.append("S_DAG", op.targets)
            elif op.name in {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"}:
                c.append(op.name, op.targets, op.p)
        return c
