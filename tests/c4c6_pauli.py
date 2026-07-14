import itertools
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import scipy.special

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
    parent: "CSSCode | None" = None


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
                [0, 0, 0, 0, 1, 0, 1, 0],
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


class KnillDecoder:
    ERASURE = -1

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def decode_qp(self, measurement_flips: np.ndarray) -> np.ndarray:
        m = np.asarray(measurement_flips, dtype=np.uint8)
        assert m.shape[1] == 2
        return m.astype(np.int8).copy()

    def decode_c4(self, measurement_flips: np.ndarray) -> np.ndarray:
        m = np.asarray(measurement_flips, dtype=np.uint8)
        assert m.shape[1] == 4
        logical = np.full((m.shape[0], 2), self.ERASURE, dtype=np.int8)
        parity = np.bitwise_xor.reduce(m, axis=1)
        keep = parity == 0
        logical[keep, 0] = m[keep, 0] ^ m[keep, 2]
        logical[keep, 1] = m[keep, 2] ^ m[keep, 3]
        return logical

    def decode_c6_children(self, child_logicals: np.ndarray) -> np.ndarray:
        d = np.asarray(child_logicals, dtype=np.int8).copy()
        shots = d.shape[0]
        logical = np.full((shots, 2), self.ERASURE, dtype=np.int8)
        erased = np.any(d == self.ERASURE, axis=2)
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
        return logical

    def decode_code(
        self,
        measurement_flips: np.ndarray,
        code: CSSCode,
        final: bool = False,
    ) -> np.ndarray:
        m = np.asarray(measurement_flips, dtype=np.uint8)
        assert m.shape[1] == code.n
        if code.name == "Qp" and not code.children:
            logical = self.decode_qp(m)
        elif code.name == "C4" and not code.children:
            logical = self.decode_c4(m)
        else:
            assert len(code.children) == 3
            child_results = []
            offset = 0
            for child in code.children:
                child_results.append(
                    self.decode_code(m[:, offset : offset + child.n], child)
                )
                offset += child.n
            child_logicals = np.stack(child_results, axis=1)
            logical = self.decode_c6_children(child_logicals)

        if final: # 0/1 for 50/50 probability
            erased = np.all(logical == self.ERASURE, axis=1)
            logical[erased] = self.rng.integers(0, 2, size=(erased.sum(), 2))
            logical = logical.astype(np.uint8)
        return logical

    def decode_c4c6(self, measurement_flips: np.ndarray, level: int) -> np.ndarray:
        m = np.asarray(measurement_flips, dtype=np.uint8)
        code = get_c4c6_code(level)
        assert m.shape[1] == code.n
        return self.decode_code(m, code, final=True)


def _bits_to_index(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    weights = (1 << np.arange(bits.shape[-1] - 1, -1, -1, dtype=np.int64))
    return (bits.astype(np.int64) * weights).sum(axis=-1)


class PoulinDecoder:
    def __init__(self, XZ: str, code: CSSCode, p: float = 0.01):
        assert 0 < p < 1
        assert XZ in {"X", "Z"}
        self.p = float(p)
        self.XZ = XZ
        self.code = code
        n = code.n
        self.check = code.stabilizers[:, n:] if self.XZ == "X" else code.stabilizers[:, :n]
        self.check = self.check[np.any(self.check, axis=1)].astype(np.uint8)
        self.decode_check, self.decode_logical = self._side_matrices_for_code(code)
        self._table_cache = {}

    def set_physical_rate(self, p: float) -> None:
        assert 0 < p < 1
        self.p = float(p)
        self._table_cache = {}

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        return self.decode_syndrome(syndrome)

    def decode_syndrome(self, syndrome: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        s = np.asarray(syndrome, dtype=np.uint8)
        assert s.ndim == 2 and s.shape[1] == (self.code.n - self.code.k) // 2

        log_prob, recovery_options = self._decode_syndrome_node(s, self.code)
        best_logical = np.argmax(log_prob, axis=1)
        recovery = recovery_options[np.arange(s.shape[0]), best_logical].astype(np.uint8)
        return recovery, {-1: log_prob}

    def decode_measurement(self, measurement_flips: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        m = np.asarray(measurement_flips, dtype=np.uint8)
        assert m.ndim == 2 and m.shape[1] == self.code.n
        syndrome = (m @ self.check.T) % 2
        recovery, prob = self.decode_syndrome(syndrome)
        return recovery, prob

    def decode_code(
        self,
        measurement_flips: np.ndarray,
        code: CSSCode | None = None,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        if code is not None and code is not self.code:
            return PoulinDecoder(self.XZ, code, self.p).decode_measurement(measurement_flips)
        return self.decode_measurement(measurement_flips)

    def decode_c4c6(self, measurement_flips: np.ndarray, level: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        return PoulinDecoder(self.XZ, get_c4c6_code(level), self.p).decode_measurement(measurement_flips)

    def _decode_syndrome_node(self, syndrome: np.ndarray, code: CSSCode) -> tuple[np.ndarray, np.ndarray]:
        assert syndrome.ndim == 2 and syndrome.shape[1] == (code.n - code.k) // 2
        if not code.children:
            data = self._local_table(code)
            syn = _bits_to_index(syndrome)
            log_prob = data["prob"][syn]
            recovery_options = data["recovery"][syn]
            return log_prob, recovery_options
 
        data = self._local_table(code.parent)
        parent_width = data["check"].shape[0]
        child_probs = []
        child_recoveries = []
        child_offset = 0
        for child in code.children:
            child_width = (child.n - child.k) // 2
            prob, recovery_options = self._decode_syndrome_node(
                syndrome[:, child_offset : child_offset + child_width],
                child,
            )
            child_probs.append(prob)
            child_recoveries.append(recovery_options)
            child_offset += child_width
        parent_syndrome = syndrome[:, child_offset : child_offset + parent_width]
        syn = _bits_to_index(parent_syndrome)
        probs = np.stack(child_probs, axis=1)
        local_ops = data["local_ops"][syn]
        x3 = np.zeros((syndrome.shape[0], data["logical_list"].shape[0], data["check_list"].shape[0]), dtype=np.float64)
        batch = np.arange(syndrome.shape[0])[:, None, None]
        for child_i, child in enumerate(code.children):
            sl = slice(2 * child_i, 2 * child_i + 2)
            child_logical = _bits_to_index(local_ops[..., sl])
            x3 += probs[batch, child_i, child_logical] # x3[syndrome, logical_class, stabilizer_choice] = log probability of child logical class given syndrome and stabilizer choice
        log_prob = scipy.special.logsumexp(x3, axis=2)
        log_prob -= scipy.special.logsumexp(log_prob, axis=1, keepdims=True)

        best_stabilizer = np.argmax(x3, axis=2) # best_stabilizer[syndrome, logical_class] = index of stabilizer choice that maximizes log probability of child logical class given syndrome and stabilizer choice
        recovery_options = np.zeros((syndrome.shape[0], data["logical_list"].shape[0], code.n), dtype=np.uint8)
        for parent_logical in range(data["logical_list"].shape[0]):
            offset = 0
            for child_i, child in enumerate(code.children):
                sl = slice(2 * child_i, 2 * child_i + 2)
                desired = _bits_to_index(local_ops[np.arange(syndrome.shape[0]), parent_logical, best_stabilizer[:, parent_logical], sl])
                recovery_options[:, parent_logical, offset : offset + child.n] = child_recoveries[child_i][
                    np.arange(syndrome.shape[0]), desired
                ]
                offset += child.n
        return log_prob, recovery_options

    def _side_matrices_for_code(self, code: CSSCode) -> tuple[np.ndarray, np.ndarray]:
        n = code.n
        if self.XZ == "X":
            check = code.stabilizers[:, n:]
            logical = code.logical_x[:, :n]
        else:
            check = code.stabilizers[:, :n]
            logical = code.logical_z[:, n:]
        check = check[np.any(check, axis=1)]
        return check.astype(np.uint8), logical.astype(np.uint8)

    def _local_table(self, code: CSSCode) -> dict[str, np.ndarray]:
        key = id(code)
        if key in self._table_cache:
            return self._table_cache[key]
        check, logical = self._side_matrices_for_code(code)
        pure_generators = self._pure_generators(code, check)

        pure = (np.array(list(itertools.product([0, 1], repeat=pure_generators.shape[0])), dtype=np.uint8) @ pure_generators) % 2
        check_list = (np.array(list(itertools.product([0, 1], repeat=check.shape[0])), dtype=np.uint8) @ check) % 2
        logical_list = (np.array(list(itertools.product([0, 1], repeat=logical.shape[0])), dtype=np.uint8) @ logical) % 2
        local_ops = (pure[:, None, None, :] ^ logical_list[None, :, None, :] ^ check_list[None, None, :, :]).astype(np.uint8) # local_ops[syndrome, logical_class, stabilizer_choice, qubit]
        weight = local_ops.sum(axis=3)
        score = np.log(self.p) * weight + np.log1p(-self.p) * (local_ops.shape[3] - weight)
        prob = scipy.special.logsumexp(score, axis=2)
        prob -= scipy.special.logsumexp(prob, axis=1, keepdims=True) # prob[syndrome, logical_class] = log probability of logical class given syndrome
        best_stabilizer = np.argmax(score, axis=2) # best_stabilizer[syndrome, logical_class] = index of stabilizer choice that maximizes score
        recovery = local_ops[np.arange(local_ops.shape[0])[:, None], np.arange(local_ops.shape[1])[None, :], best_stabilizer] # recovery[syndrome, logical_class], the best recovery for each syndrome and logical class
        data = {
            "check": check,
            "logical_list": logical_list,
            "check_list": check_list,
            "local_ops": local_ops,
            "prob": prob,
            "recovery": recovery.astype(np.uint8),
        }
        self._table_cache[key] = data
        return data

    def _pure_generators(self, code: CSSCode, check: np.ndarray) -> np.ndarray:
        if check.shape[0] == 0:
            return np.zeros((0, check.shape[1]), dtype=np.uint8)
        if code.name == "C4":
            return np.array([[0, 1, 0, 0]], dtype=np.uint8) if self.XZ == "X" else np.array([[0, 0, 1, 0]], dtype=np.uint8)
        if code.name == "C6":
            if self.XZ == "X":
                return np.array([[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.uint8)
            return np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1]], dtype=np.uint8)


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

def get_circuit_c4c6_bell(level, err):
    '''
    Entangle circuit for two C4/C6 states with level'''
    n_q = 4 * 3 ** (level - 1)
    ind_q = np.arange(n_q * 2)
    return (
        C4C6Circuit(2 * n_q)
        .h_log(level, ind_q[:n_q])
        .cx(np.column_stack((ind_q[:n_q], ind_q[n_q:])).ravel())
        .depolarize2(np.column_stack((ind_q[:n_q], ind_q[n_q:])).ravel(), err)
    )

def get_circuit_c4c6_p1(level, err):
    '''
    Apply logical transversal CNOTs'''
    n_q = 4 * 3 ** (level - 1)
    n_sub = 4 * 3 ** (level - 2)
    ind_q = np.arange(n_q * 2)
    return (
        C4C6Circuit(2 * n_q)
        .cx(np.column_stack((ind_q[n_sub:2 * n_sub], ind_q[2 * n_sub:3 * n_sub])).ravel())
        .depolarize2(np.column_stack((ind_q[n_sub:2 * n_sub], ind_q[2 * n_sub:3 * n_sub])).ravel(), err)
        .cx(np.column_stack((ind_q[3 * n_sub:4 * n_sub], ind_q[4 * n_sub:5 * n_sub])).ravel())
        .depolarize2(np.column_stack((ind_q[3 * n_sub:4 * n_sub], ind_q[4 * n_sub:5 * n_sub])).ravel(), err)
        .cx(np.column_stack((ind_q[5 * n_sub:], ind_q[:n_sub])).ravel())
        .depolarize2(np.column_stack((ind_q[5 * n_sub:], ind_q[:n_sub])).ravel(), err)
        .m(ind_q[:n_sub])
        .m(ind_q[2 * n_sub:3 * n_sub])
        .m(ind_q[4 * n_sub:5 * n_sub])
    )

def get_circuit_c4c6_p2(level, err):
    '''
    Apply logical u and u2 gates'''
    n_q = 4 * 3 ** (level - 1)
    n_sub = 4 * 3 ** (level - 2)
    ind_q = np.arange(n_q)
    return (
        C4C6Circuit(n_q)
        .u(level-1, ind_q[n_sub:2 * n_sub])
        .u2(level-1, ind_q[2 * n_sub:])
    )

def get_circuit_c4c6_tele(level, err):
    '''
    ECT circuit for preparing 2 bells at level'''
    n_q = 4 * 3 ** (level - 1)
    n_sub = 4 * 3 ** (level - 2)
    ind_q = np.arange(n_q * 2 + n_sub * 12)
    return (
        C4C6Circuit(n_q * 2 + n_sub * 12)
        .cx(np.column_stack((ind_q[:n_sub*6], np.concatenate((ind_q[n_sub*6:n_sub*7], ind_q[n_sub*8:n_sub*9], ind_q[n_sub*10:n_sub*11], ind_q[n_sub*12:n_sub*13], ind_q[n_sub*14:n_sub*15], ind_q[n_sub*16:n_sub*17])))).ravel())
        .depolarize2(np.column_stack((ind_q[:n_sub*6], np.concatenate((ind_q[n_sub*6:n_sub*7], ind_q[n_sub*8:n_sub*9], ind_q[n_sub*10:n_sub*11], ind_q[n_sub*12:n_sub*13], ind_q[n_sub*14:n_sub*15], ind_q[n_sub*16:n_sub*17])))).ravel(), err)
        .h_log(level-1, ind_q[:n_sub])
        .h_log(level-1, ind_q[n_sub:2*n_sub])
        .h_log(level-1, ind_q[2*n_sub:3*n_sub])
        .h_log(level-1, ind_q[3*n_sub:4*n_sub])
        .h_log(level-1, ind_q[4*n_sub:5*n_sub])
        .h_log(level-1, ind_q[5*n_sub:6*n_sub])
        .m(ind_q[:6*n_sub]) # measure Z errors
        .m(np.concatenate((ind_q[n_sub*6:n_sub*7], ind_q[n_sub*8:n_sub*9], ind_q[n_sub*10:n_sub*11], ind_q[n_sub*12:n_sub*13], ind_q[n_sub*14:n_sub*15], ind_q[n_sub*16:n_sub*17]))) # measure X errors
    )

def get_circuit_tele(level, err):
    n_q = 4 * 3 ** (level - 1)
    ind_q = np.arange(n_q * 8)
    return (
        C4C6Circuit(n_q * 8)
        .cx(np.column_stack((ind_q[:n_q], ind_q[4*n_q:5*n_q])).ravel())
        .depolarize2(np.column_stack((ind_q[:n_q], ind_q[4*n_q:5*n_q])).ravel(), err)
        .cx(np.column_stack((ind_q[2*n_q:3*n_q], ind_q[6*n_q:7*n_q])).ravel())
        .depolarize2(np.column_stack((ind_q[2*n_q:3*n_q], ind_q[6*n_q:7*n_q])).ravel(), err)
        .h_log(level, ind_q[:n_q])
        .h_log(level, ind_q[2*n_q:3*n_q])
        .m(ind_q[:n_q]) # measure Z errors
        .m(ind_q[2*n_q:3*n_q])
        .m(ind_q[4*n_q:5*n_q]) # measure X errors
        .m(ind_q[6*n_q:7*n_q])
    )

def get_circuit_meaNdec(level):
    n_q = 4 * 3 ** (level - 1)
    ind_q = np.arange(n_q*4)
    return (
        C4C6Circuit(n_q*4)
        .cx(np.column_stack((ind_q[:n_q], ind_q[n_q:2*n_q])).ravel())
        .cx(np.column_stack((ind_q[2*n_q:3*n_q], ind_q[3*n_q:4*n_q])).ravel())
        .h_log(level, ind_q[:n_q])
        .h_log(level, ind_q[2*n_q:3*n_q])
        .m(ind_q[:2*n_q]) # measure X errors
        .m(ind_q[2*n_q:4*n_q]) # measure Z errors
    )

# Prepare l1
def get_PFrame_l1(shots, circuit): # circuit is cir_l1
    # Get frame after circuit c4
    pframe_l1 = PauliFrame(circuit, shots=shots).run()
    # Post-selection
    keep = (pframe_l1.samples.sum(axis=1) % 2) == 0
    pframe_l1.update(pframe_l1.frame[keep])
    pframe_l1.select_qubits([1, 3, 5, 7])
    return pframe_l1

# Prepare l1 Bell
def get_PFrame_l1_bell(shots, circuits: list[Circuit]): # circuit are cir_l1, cir_l1_bell
    pframe_l1_bell = PauliFrame.bunch([get_PFrame_l1(shots=shots, circuit=circuits[0]), get_PFrame_l1(shots=shots, circuit=circuits[0])], circuit=circuits[1]).run()
    return pframe_l1_bell

# Prepare l2
def get_PFrame_l2(shots, decoder, circuits: list[Circuit]):
    '''
    decoder: GHZ state error detection.

    list[Circuit] are cir_l1, cir_l1_bell, cir_l2_p1, cir_l2_p2.
    '''
    pframe_l2 = PauliFrame.bunch([get_PFrame_l1_bell(shots=shots, circuits=circuits[:2]), get_PFrame_l1_bell(shots=shots, circuits=circuits[:2]), get_PFrame_l1_bell(shots=shots, circuits=circuits[:2])], circuit=circuits[2]).run()
    mea = pframe_l2.samples
    re_list = [decoder.decode_code(mea[:, -12:-8], c4), decoder.decode_code(mea[:, -8:-4], c4), decoder.decode_code(mea[:, -4:], c4)]
    re = np.concatenate(re_list, axis=1)
    # post-select
    mask = np.all(re != -1, axis=1) & ((re[:, [0, 2, 4]].sum(axis=1) % 2) == 0) & ((re[:, [1, 3, 5]].sum(axis=1) % 2) == 0)
    frame_l2_new = pframe_l2.frame[mask]
    pframe_l2.update(frame_l2_new)
    pframe_l2.select_qubits(np.r_[4:8, 12:16, 20:24])
    frame_temp0 = pframe_l2.frame
    # correct
    cor = re[mask].astype(np.uint8)
    frame_temp0[:, :4] ^= ((cor[:, [0]] * logx_l1_1) ^ (cor[:, [1]] * logx_l1_2))
    frame_temp0[:, 4:8] ^= (((cor[:, [0]] ^ cor[:, [2]]) * logx_l1_1) ^ ((cor[:, [1]] ^ cor[:, [3]]) * logx_l1_2))
    pframe_l2.update(frame=frame_temp0, circuit=circuits[3]).run()
    return pframe_l2

def get_PFrame_l2_bell(shots, decoder, circuits: list[Circuit]):
    '''
    decoder: Bell state error detection.

    list[Circuit] are cir_l1, cir_l1_bell, cir_l2_p1, cir_l2_p2, cir_l2_bell, cir_l2_bell_tele.
    '''
    pframe_l2_bell = PauliFrame.bunch([get_PFrame_l2(shots=shots, decoder=decoder, circuits=circuits[:4]), get_PFrame_l2(shots=shots, decoder=decoder, circuits=circuits[:4])], circuit=circuits[4]).run()
    pframe_l1_bell_lst = [get_PFrame_l1_bell(shots=shots, circuits=circuits[:2]) for _ in range(6)]
    n_sub = 4
    pframe_l2_edt = PauliFrame.bunch([pframe_l2_bell] + pframe_l1_bell_lst, circuit=circuits[5]).run()
    mea = pframe_l2_edt.samples
    # post-select
    rez = np.concatenate([decoder.decode_code(mea[:, i*n_sub:(i+1)*n_sub], c4) for i in range(6)], axis=1)
    rex = np.concatenate([decoder.decode_code(mea[:, (6 + i)*n_sub:(7 + i)*n_sub], c4) for i in range(6)], axis=1)
    mask_p = np.all(rex != -1, axis=1) & np.all(rez != -1, axis=1)
    rex_new = rex[mask_p].astype(np.uint8)
    rez_new = rez[mask_p].astype(np.uint8)
    frame_new = pframe_l2_edt.frame[mask_p]
    pframe_l2_edt.update(frame_new)
    pframe_l2_edt.select_qubits(np.r_[7*n_sub:8*n_sub, 9*n_sub:10*n_sub, 11*n_sub:12*n_sub, 13*n_sub:14*n_sub, 15*n_sub:16*n_sub, 17*n_sub:18*n_sub])
    frame_temp = pframe_l2_edt.frame
    # correct
    for _ in range(6):
        frame_temp[:, _*n_sub:(_ + 1)*n_sub] ^= ((rex_new[:, [2*_]]*logx_l1_1) ^ (rex_new[:, [2*_ + 1]]*logx_l1_2))
        frame_temp[:, (_ + 6)*n_sub:(_ + 7)*n_sub] ^= ((rez_new[:, [2*_]]*logz_l1_1) ^ (rez_new[:, [2*_ + 1]]*logz_l1_2))
    pframe_l2_edt.update(frame_temp)
    return pframe_l2_edt



def run_level1(shots, noise):
    shot = shots
    er = noise
    dec = KnillDecoder()
    
    cir_l1_i = get_circuit_c4(0)
    cir_l1_bell_i = get_circuit_c4c6_bell(1, 0)

    cir_l1 = get_circuit_c4(er)
    cir_l1_bell = get_circuit_c4c6_bell(1, er)

    ### level 1
    n_q = 4
    ind_q = np.arange(4*n_q)
    ## Dep error
    cir_a = (
        C4C6Circuit(4*n_q)
        .depolarize2(np.column_stack((ind_q[:n_q], ind_q[2*n_q:3*n_q])).ravel(), er)
    )
    pframe_l1 = PauliFrame.bunch([get_PFrame_l1_bell(shots=shot, circuits=[cir_l1_i, cir_l1_bell_i]), get_PFrame_l1_bell(shots=shot, circuits=[cir_l1_i, cir_l1_bell_i])], circuit=cir_a).run()
    ## ECT
    cir_b = get_circuit_tele(1, er)
    pframe_l1_ect = PauliFrame.bunch([pframe_l1, get_PFrame_l1_bell(shots=shot, circuits=[cir_l1, cir_l1_bell]), get_PFrame_l1_bell(shots=shot, circuits=[cir_l1, cir_l1_bell])], circuit=cir_b).run()
    shot_eff = pframe_l1_ect.frame.shape[0] # effective shots after post-selection
    mea = pframe_l1_ect.samples
    rez = np.concatenate([dec.decode_code(mea[:, i*n_q:(i+1)*n_q], c4, True) for i in range(2)], axis=1)
    rex = np.concatenate([dec.decode_code(mea[:, (2 + i)*n_q:(3 + i)*n_q], c4, True) for i in range(2)], axis=1)
    # correct
    frame_corrected = pframe_l1_ect.frame
    frame_corrected[:, 5*n_q:6*n_q] ^= ((rex[:, [0]]*logx_l1_1) ^ (rex[:, [1]]*logx_l1_2))
    frame_corrected[:, 7*n_q:8*n_q] ^= ((rex[:, [2]]*logx_l1_1) ^ (rex[:, [3]]*logx_l1_2))
    frame_corrected[:, 8*n_q + 5*n_q:8*n_q + 6*n_q] ^= ((rez[:, [0]]*logz_l1_1) ^ (rez[:, [1]]*logz_l1_2))
    frame_corrected[:, 8*n_q + 7*n_q:8*n_q + 8*n_q] ^= ((rez[:, [2]]*logz_l1_1) ^ (rez[:, [3]]*logz_l1_2))
    # select qubits
    pframe_l1_ect.update(frame=frame_corrected)
    pframe_l1_ect.select_qubits(np.r_[n_q:2*n_q, 3*n_q:4*n_q, 5*n_q:6*n_q, 7*n_q:8*n_q])
    # relabel
    circ_relabel = (
        Circuit(4*n_q)
        .swap(np.column_stack((ind_q[2*n_q:3*n_q], ind_q[3*n_q:4*n_q])).ravel())
        .swap(np.column_stack((ind_q[n_q:2*n_q], ind_q[3*n_q:4*n_q])).ravel())
        .swap(np.column_stack((ind_q[:n_q], ind_q[n_q:2*n_q])).ravel())
    )
    pframe_l1_ect.update(circuit=circ_relabel).run()
    ## measure and decode
    mea_end = pframe_l1_ect.update(circuit=get_circuit_meaNdec(1)).run().samples
    dec_end = PoulinDecoder("X", c4, p=er)
    mea_re = mea_end.reshape(-1, n_q)
    syndrome = mea_re @ dec_end.check.T % 2
    recovery, prob_L = dec_end.decode_syndrome(syndrome)
    re = (mea_re ^ recovery) @ c4.logical_z[:, n_q:].T % 2
    re = re.reshape(mea_end.shape[0], -1)
    num_err = re.any(axis=1).sum()

    return float(num_err/shot_eff)

def run_level2(shots, noise):
    shot = shots
    er = noise
    dec = KnillDecoder()

    cir_l1_i = get_circuit_c4(0)
    cir_l1_bell_i = get_circuit_c4c6_bell(1, 0)
    cir_l2_p1_i = get_circuit_c4c6_p1(2, 0)
    cir_l2_p2_i = get_circuit_c4c6_p2(2, 0)
    cir_l2_bell_i = get_circuit_c4c6_bell(2, 0)
    cir_l2_bell_tele_i = get_circuit_c4c6_tele(2, 0)

    cir_l1 = get_circuit_c4(er)
    cir_l1_bell = get_circuit_c4c6_bell(1, er)
    cir_l2_p1 = get_circuit_c4c6_p1(2, er)
    cir_l2_p2 = get_circuit_c4c6_p2(2, er)
    cir_l2_bell = get_circuit_c4c6_bell(2, er)
    cir_l2_bell_tele = get_circuit_c4c6_tele(2, er)

    ### level 2
    n_q = 12
    ind_q = np.arange(48)
    ## Dep error
    cir_a = (
        C4C6Circuit(4*n_q)
        .depolarize2(np.column_stack((ind_q[:n_q], ind_q[2*n_q:3*n_q])).ravel(), er)
    )
    pframe_l2 = PauliFrame.bunch([get_PFrame_l2_bell(shots=shot, decoder=dec, circuits=[cir_l1_i, cir_l1_bell_i, cir_l2_p1_i, cir_l2_p2_i, cir_l2_bell_i, cir_l2_bell_tele_i]), get_PFrame_l2_bell(shots=shot, decoder=dec, circuits=[cir_l1_i, cir_l1_bell_i, cir_l2_p1_i, cir_l2_p2_i, cir_l2_bell_i, cir_l2_bell_tele_i])], circuit=cir_a).run()
    ## ECT
    cir_b = get_circuit_tele(2, er)
    pframe_l2_ect = PauliFrame.bunch([pframe_l2, get_PFrame_l2_bell(shots=shot, decoder=dec, circuits=[cir_l1, cir_l1_bell, cir_l2_p1, cir_l2_p2, cir_l2_bell, cir_l2_bell_tele]), get_PFrame_l2_bell(shots=shot, decoder=dec, circuits=[cir_l1, cir_l1_bell, cir_l2_p1, cir_l2_p2, cir_l2_bell, cir_l2_bell_tele])], circuit=cir_b).run()
    shot_eff = pframe_l2_ect.frame.shape[0]
    print(f"shot_eff: {shot_eff}")
    mea = pframe_l2_ect.samples
    rez = np.concatenate([dec.decode_code(mea[:, i*n_q:(i+1)*n_q], c4c6_l2, True) for i in range(2)], axis=1)
    rex = np.concatenate([dec.decode_code(mea[:, (2 + i)*n_q:(3 + i)*n_q], c4c6_l2, True) for i in range(2)], axis=1)
    # correct
    frame_corrected = pframe_l2_ect.frame
    frame_corrected[:, 5*n_q:6*n_q] ^= ((rex[:, [0]]*logx_l2_1) ^ (rex[:, [1]]*logx_l2_2))
    frame_corrected[:, 7*n_q:8*n_q] ^= ((rex[:, [2]]*logx_l2_1) ^ (rex[:, [3]]*logx_l2_2))
    frame_corrected[:, 8*n_q + 5*n_q:8*n_q + 6*n_q] ^= ((rez[:, [0]]*logz_l2_1) ^ (rez[:, [1]]*logz_l2_2))
    frame_corrected[:, 8*n_q + 7*n_q:8*n_q + 8*n_q] ^= ((rez[:, [2]]*logz_l2_1) ^ (rez[:, [3]]*logz_l2_2))
    # select qubits
    pframe_l2_ect.update(frame=frame_corrected)
    pframe_l2_ect.select_qubits(np.r_[n_q:2*n_q, 3*n_q:4*n_q, 5*n_q:6*n_q, 7*n_q:8*n_q])
    # relabel
    circ_relabel = (
        Circuit(4*n_q)
        .swap(np.column_stack((ind_q[2*n_q:3*n_q], ind_q[3*n_q:4*n_q])).ravel())
        .swap(np.column_stack((ind_q[n_q:2*n_q], ind_q[3*n_q:4*n_q])).ravel())
        .swap(np.column_stack((ind_q[:n_q], ind_q[n_q:2*n_q])).ravel())
    )
    pframe_l2_ect.update(circuit=circ_relabel).run()
    ## measure and decode
    mea_end = pframe_l2_ect.update(circuit=get_circuit_meaNdec(2)).run().samples
    dec_end = PoulinDecoder("X", c4c6_l2, p=er)
    mea_re = mea_end.reshape(-1, n_q)
    syndrome = mea_re @ dec_end.check.T % 2
    recovery, prob_L = dec_end.decode_syndrome(syndrome)
    re = (mea_re ^ recovery) @ c4c6_l2.logical_z[:, n_q:].T % 2
    re = re.reshape(mea_end.shape[0], -1)
    num_err = re.any(axis=1).sum()

    return float(num_err/shot_eff)


shot = 10000

c4 = get_c4()
c4c6_l2 = get_c4c6_code(2)

logx_l1_1, logx_l1_2 = c4.logical_x[:, :4]
logz_l1_1, logz_l1_2 = c4.logical_z[:, 4:]
logx_l2_1, logx_l2_2 = c4c6_l2.logical_x[:, :12]
logz_l2_1, logz_l2_2 = c4c6_l2.logical_z[:, 12:]



err_list = np.linspace(0.002, 0.01, 10)
# err_list = np.linspace(0.01, 0.012, 1)
# ler_list = [run_level1(shot, err) for err in err_list]
ler_list = [run_level2(shot, err) for err in err_list]

print(err_list.tolist())
print(ler_list)
