import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dorsim import Circuit, PauliFrame, target_rec


@dataclass
class CSSCode:
    name: str
    n: int
    k: int
    stabilizers: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray
    children: tuple["CSSCode", ...] = ()


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
            [
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        ),
        logical_x=np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        logical_z=np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 1],
            ],
            dtype=np.uint8,
        ),
    )


def get_c6() -> CSSCode:
    return CSSCode(
        name="C6",
        n=6,
        k=2,
        stabilizers=np.array(
            [
                [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
            ],
            dtype=np.uint8,
        ),
        logical_x=np.array(
            [
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        logical_z=np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            ],
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
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.uint8,
        ),
        logical_z=np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.uint8,
        ),
    )


def _logical_z_for_parent_row(child: CSSCode, logical: int, use_alternate_first_z: bool) -> np.ndarray:
    row = child.logical_z[logical]
    if use_alternate_first_z and logical == 0:
        for stabilizer in child.stabilizers:
            if not np.any(stabilizer[:child.n]):
                return row ^ stabilizer
    return row


def _lift_parent_row(
    row: np.ndarray,
    children: tuple[CSSCode, ...],
    total_n: int,
    *,
    use_alternate_first_z: bool = False,
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
            out ^= embed_bsr(_logical_z_for_parent_row(child, logical, use_alternate_first_z), offset, total_n)
    return out


def concat_code(parent: CSSCode, children) -> CSSCode:
    children = tuple(children)
    assert sum(child.k for child in children) == parent.n
    total_n = sum(child.n for child in children)

    stabilizers = [
        _lift_parent_row(row, children, total_n, use_alternate_first_z=True)
        for row in parent.stabilizers
    ]
    offset = 0
    for child in children:
        for row in child.stabilizers:
            stabilizers.append(embed_bsr(row, offset, total_n))
        offset += child.n
    stabilizers = np.array(stabilizers, dtype=np.uint8) if stabilizers else np.zeros((0, 2 * total_n), dtype=np.uint8)

    return CSSCode(
        name=f"{parent.name}(" + ",".join(child.name for child in children) + ")",
        n=total_n,
        k=parent.k,
        stabilizers=stabilizers,
        logical_x=np.array([_lift_parent_row(row, children, total_n) for row in parent.logical_x], dtype=np.uint8),
        logical_z=np.array(
            [_lift_parent_row(row, children, total_n, use_alternate_first_z=True) for row in parent.logical_z],
            dtype=np.uint8,
        ),
        children=children,
    )


class decoder:
    ERASURE = -1

    def decode_qp(self, measurement_flips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(measurement_flips, dtype=np.uint8)
        assert m.shape[1] == 2
        logical = m.astype(np.int8).copy()
        status = np.ones(m.shape[0], dtype=np.int8)
        return logical, status

    def decode_c4(self, measurement_flips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(measurement_flips, dtype=np.uint8)
        assert m.shape[1] == 4
        logical = np.full((m.shape[0], 2), self.ERASURE, dtype=np.int8)
        status = np.full(m.shape[0], -1, dtype=np.int8)
        parity = np.bitwise_xor.reduce(m, axis=1)
        keep = parity == 0
        logical[keep, 0] = m[keep, 0] ^ m[keep, 2]
        logical[keep, 1] = m[keep, 2] ^ m[keep, 3]
        status[keep] = 1
        return logical, status

    def decode_c6_children(self, child_logicals: np.ndarray, child_status: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(child_logicals, dtype=np.int8).copy()
        status_in = np.asarray(child_status, dtype=np.int8)
        shots = d.shape[0]
        logical = np.full((shots, 2), self.ERASURE, dtype=np.int8)
        status = np.full(shots, -1, dtype=np.int8)
        erased = status_in == -1
        erasure_count = erased.sum(axis=1)

        one = erasure_count == 1
        for erased_child in range(3):
            mask = one & erased[:, erased_child]
            if not np.any(mask):
                continue
            a1 = d[mask, 0, 0]
            a2 = d[mask, 0, 1]
            b1 = d[mask, 1, 0]
            b2 = d[mask, 1, 1]
            c1 = d[mask, 2, 0]
            c2 = d[mask, 2, 1]
            if erased_child == 0:
                d[mask, 0, 0] = b2 ^ c1 ^ c2
                d[mask, 0, 1] = b1 ^ b2 ^ c1
            elif erased_child == 1:
                d[mask, 1, 0] = a1 ^ a2 ^ c2
                d[mask, 1, 1] = a1 ^ c1 ^ c2
            else:
                d[mask, 2, 0] = a2 ^ b1 ^ b2
                d[mask, 2, 1] = a1 ^ a2 ^ b1

        no_erasure = erasure_count == 0
        a1 = d[:, 0, 0]
        a2 = d[:, 0, 1]
        b1 = d[:, 1, 0]
        b2 = d[:, 1, 1]
        c1 = d[:, 2, 0]
        c2 = d[:, 2, 1]
        s1_ok = (a1 ^ b2 ^ c1 ^ c2) == 0
        s2_ok = (a1 ^ a2 ^ b1 ^ c2) == 0
        keep = one | (no_erasure & s1_ok & s2_ok)
        logical[keep, 0] = b1[keep] ^ b2[keep] ^ c2[keep]
        logical[keep, 1] = b2[keep] ^ c1[keep]
        status[keep] = 1
        return logical, status

    def decode_code(self, measurement_flips: np.ndarray, code: CSSCode) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(measurement_flips, dtype=np.uint8)
        assert m.shape[1] == code.n
        if code.name == "Qp" and not code.children:
            return self.decode_qp(m)
        if code.name == "C4" and not code.children:
            return self.decode_c4(m)

        assert len(code.children) == 3
        child_results = []
        offset = 0
        for child in code.children:
            child_results.append(self.decode_code(m[:, offset : offset + child.n], child))
            offset += child.n
        child_logicals = np.stack([result[0] for result in child_results], axis=1)
        child_status = np.stack([result[1] for result in child_results], axis=1)
        return self.decode_c6_children(child_logicals, child_status)

    def decode_c4c6(self, measurement_flips: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(measurement_flips, dtype=np.uint8)
        code = get_c4c6_code(level)
        assert m.shape[1] == code.n
        return self.decode_code(m, code)


def get_c4c6_code(level: int) -> CSSCode:
    code = get_c4()
    for _ in range(1, level):
        code = concat_code(get_c6(), [code, code, code])
    return code

class C4C6Circuit(Circuit):
    def _thirds(self, targets):
        a = list(targets)
        assert len(a) % 3 == 0
        k = len(a) // 3
        return a[:k], a[k : 2 * k], a[2 * k :]

    def h_log(self, level: int, targets) -> "C4C6Circuit":
        a = list(targets)
        if level == 1:
            self.h(a)
            self.swap([a[1], a[2]])
            return self
        for b in self._thirds(a):
            self.h_log(level - 1, b)
            self.u2(level - 1, b)
        return self

    def u(self, level: int, targets) -> "C4C6Circuit":
        a = list(targets)
        if level == 1:
            self.swap([a[1], a[2], a[1], a[3]])
            return self
        for b in self._thirds(a):
            self.u2(level - 1, b)
        return self

    def u2(self, level: int, targets) -> "C4C6Circuit":
        a = list(targets)
        if level == 1:
            self.swap([a[1], a[2], a[2], a[3]])
            return self
        for b in self._thirds(a):
            self.u(level - 1, b)
        return self
    

def get_circuit_c4(err):
    # Prepare C4 states, post-select.
    return (
        C4C6Circuit(8)
        .h([0, 2, 4, 6])
        .cx([0, 1, 2, 3, 4, 5, 6, 7])
        .depolarize2([0, 1, 2, 3, 4, 5, 6, 7], err)
        .cx([1, 2, 3, 4, 5, 6])
        .depolarize2([1, 2, 3, 4, 5, 6], err)
        .cx([7, 0])
        .depolarize2([7, 0], err)
        .m([0, 2, 4, 6])
        .cx([
            target_rec(-4), 1,
            target_rec(-4), 3,
            target_rec(-4), 5,
            target_rec(-3), 3,
            target_rec(-3), 5,
            target_rec(-2), 5,
        ])
    )

def get_circuit_c4_b(err):
    return (
        C4C6Circuit(8)
        .h_log(1, [0, 1, 2, 3])
        .cx([0, 4, 1, 5, 2, 6, 3, 7])
        .depolarize2([0, 4, 1, 5, 2, 6, 3, 7], err)
    )

def get_circuit_c4c6(level, err):
    n_q = 4 * 3 ** (level - 1)
    n_sub = 4 * 3 ** (level - 2)
    ind_q = np.arange(n_q * 2)
    return (
        C4C6Circuit(2 * n_q)
        .cx(np.column_stack((ind_q[n_sub:2 * n_sub], ind_q[2 * n_sub:3 * n_sub])).ravel())
        .depolarize2(np.column_stack((ind_q[n_sub:2 * n_sub], ind_q[2 * n_sub:3 * n_sub])).ravel(), err)
        .cx(np.column_stack((ind_q[5 * n_sub:], ind_q[:n_sub])).ravel())
        .depolarize2(np.column_stack((ind_q[5 * n_sub:], ind_q[:n_sub])).ravel(), err)
        .m(ind_q[:n_sub])
        .m(ind_q[2 * n_sub:3 * n_sub])
        .m(ind_q[4 * n_sub:5 * n_sub])
    )



# c4 = get_c4()
# c6 = get_c6()
# qp = get_qp()
# c10d3 = concat_code(c6, [qp, c4, c4])
# print(f"{c10d3.name}: n={c10d3.n}")
# print(f"stabilizers=\n{c10d3.stabilizers}")
# print(f"logical_x=\n{c10d3.logical_x}")
# print(f"logical_z=\n{c10d3.logical_z}")
