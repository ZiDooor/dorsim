from dataclasses import dataclass

import numpy as np


@dataclass
class CSSCode:
    name: str
    n: int
    k: int
    stabilizers: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray
    children: tuple["CSSCode", ...] = ()
    parent: "CSSCode | None" = None

    def get_lx(self) -> np.ndarray:
        return self.logical_x[:, :self.n]

    def get_lz(self) -> np.ndarray:
        return self.logical_z[:, self.n:]


def embed_bsr(row: np.ndarray, offset: int, total_n: int) -> np.ndarray:
    local_n = row.size // 2
    out = np.zeros(2 * total_n, dtype=np.uint8)
    out[offset : offset + local_n] = row[:local_n]
    out[total_n + offset : total_n + offset + local_n] = row[local_n:]
    return out


def get_c4() -> CSSCode:
    return CSSCode(
        name="C4",
        n=4,
        k=2,
        stabilizers=np.array(
            [[1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1]],
            dtype=np.uint8,
        ),
        logical_x=np.array(
            [[1, 1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0, 0]],
            dtype=np.uint8,
        ),
        logical_z=np.array(
            [[0, 0, 0, 0, 1, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1, 1]],
            dtype=np.uint8,
        ),
    )


def get_c6() -> CSSCode:
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


def get_qp() -> CSSCode:
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


def _lift_parent_row(
    row: np.ndarray,
    children: tuple[CSSCode, ...],
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


def concat_code(parent: CSSCode, children) -> CSSCode:
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

    return CSSCode(
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
    code = get_c4()
    for _ in range(1, level):
        code = concat_code(get_c6(), [code, code, code])
    return code
