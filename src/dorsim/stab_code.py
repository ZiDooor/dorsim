from dataclasses import dataclass, field

import numpy as np


@dataclass
class StabilizerCode:
    name: str
    n: int
    k: int
    stabilizers: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray
    children: tuple["StabilizerCode", ...] = ()
    parent: "StabilizerCode | None" = None
    _pure_errors: np.ndarray | None = field(default=None, init=False, repr=False)

    def steane():
        return StabilizerCode(
            name="Steane",
            n=7,
            k=1,
            stabilizers=np.array(
                [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]],
                dtype=np.uint8),
            logical_x=np.array([[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                               dtype=np.uint8),
            logical_z=np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]],
                               dtype=np.uint8)
        )

    @property
    def pure_errors(self) -> np.ndarray:
        if self._pure_errors is None:
            operators = np.concatenate([self.stabilizers, self.logical_x, self.logical_z], axis=0)
            check = np.concatenate([operators[:, self.n :], operators[:, : self.n]], axis=1)
            rows = check.shape[0]
            syndrome_rows = self.n - self.k
            target = np.zeros((rows, syndrome_rows), dtype=np.uint8)
            target[:syndrome_rows] = np.eye(syndrome_rows, dtype=np.uint8)
            augmented = np.concatenate([check.copy(), target], axis=1)
            pivots = []
            row = 0
            for col in range(2 * self.n):
                found = np.flatnonzero(augmented[row:, col])
                if found.size == 0:
                    continue
                pivot = row + int(found[0])
                augmented[[row, pivot]] = augmented[[pivot, row]]
                for other in range(rows):
                    if other != row and augmented[other, col]:
                        augmented[other] ^= augmented[row]
                pivots.append(col)
                row += 1
                if row == rows:
                    break
            right_inverse = np.zeros((2 * self.n, syndrome_rows), dtype=np.uint8)
            for row, col in enumerate(pivots):
                right_inverse[col] = augmented[row, 2 * self.n :]
            self._pure_errors = right_inverse.T
        return self._pure_errors


@dataclass
class CSSCode(StabilizerCode):
    def get_lx(self) -> np.ndarray:
        return self.logical_x[:, : self.n]

    def get_lz(self) -> np.ndarray:
        return self.logical_z[:, self.n :]
    
    def c4():
        return CSSCode(
            name="C4",
            n=4,
            k=2,
            stabilizers=np.array(
                [[1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1]],
                dtype=np.uint8),
            logical_x=np.array(
                [[1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0]],
                dtype=np.uint8),
            logical_z=np.array(
                [[0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1]],
                dtype=np.uint8),
        )


    def c6():
        return CSSCode(
            name="C6",
            n=6,
            k=2,
            stabilizers=np.array(
                [[1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]],
                dtype=np.uint8,
            ),
            logical_x=np.array(
                [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
                dtype=np.uint8,
            ),
            logical_z=np.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]],
                dtype=np.uint8,
            ),
        )


    def qp():
        return CSSCode(
            name="Qp",
            n=2,
            k=2,
            stabilizers=np.zeros((0, 4), dtype=np.uint8),
            logical_x=np.array(
                [[1, 0, 0, 0],
                [0, 1, 0, 0]],
                dtype=np.uint8,
            ),
            logical_z=np.array(
                [[0, 0, 1, 0],
                [0, 0, 0, 1]],
                dtype=np.uint8,
            ),
        )


def embed_bsr(row: np.ndarray, offset: int, total_n: int) -> np.ndarray:
    local_n = row.size // 2
    out = np.zeros(2 * total_n, dtype=np.uint8)
    out[offset : offset + local_n] = row[:local_n]
    out[total_n + offset : total_n + offset + local_n] = row[local_n:]
    return out



def _lift_parent_row(
    row: np.ndarray,
    children: tuple[StabilizerCode, ...],
    total_n: int,
) -> np.ndarray:
    parent_n = row.size // 2
    out = np.zeros(2 * total_n, dtype=np.uint8)
    physical_offsets = np.cumsum([0] + [child.n for child in children[:-1]])
    logical_offsets = np.cumsum([0] + [child.k for child in children[:-1]])
    for q in range(parent_n):
        child_index = max(i for i, start in enumerate(logical_offsets) if start <= q)
        child = children[child_index]
        logical = q - int(logical_offsets[child_index])
        offset = int(physical_offsets[child_index])
        if row[q]:
            out ^= embed_bsr(child.logical_x[logical], offset, total_n)
        if row[parent_n + q]:
            out ^= embed_bsr(child.logical_z[logical], offset, total_n)
    return out


def concat_code(parent: StabilizerCode, children) -> StabilizerCode:
    children = tuple(children)
    assert sum(child.k for child in children) == parent.n
    total_n = sum(child.n for child in children)

    stabilizers = []
    offset = 0
    for child in children:
        for row in child.stabilizers:
            stabilizers.append(embed_bsr(row, offset, total_n))
        offset += child.n
    stabilizers.extend(_lift_parent_row(row, children, total_n) for row in parent.stabilizers)
    stabilizers = np.array(stabilizers, dtype=np.uint8) if stabilizers else np.zeros((0, 2 * total_n), dtype=np.uint8)
    code_type = CSSCode if isinstance(parent, CSSCode) and all(isinstance(child, CSSCode) for child in children) else StabilizerCode

    return code_type(
        name=f"{parent.name}(" + ",".join(child.name for child in children) + ")",
        n=total_n,
        k=parent.k,
        stabilizers=stabilizers,
        logical_x=np.array([_lift_parent_row(row, children, total_n) for row in parent.logical_x], dtype=np.uint8),
        logical_z=np.array([_lift_parent_row(row, children, total_n) for row in parent.logical_z], dtype=np.uint8),
        children=children,
        parent=parent,
    )


def get_c4c6_code(level: int) -> CSSCode:
    code = CSSCode.c4()
    for _ in range(1, level):
        code = concat_code(CSSCode.c6(), [code, code, code])
    return code
