import itertools

import numpy as np
import scipy.special

from .css_code import CSSCode, get_c4c6_code


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
