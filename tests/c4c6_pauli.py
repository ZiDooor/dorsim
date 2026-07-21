from collections import deque
from dataclasses import dataclass

import numpy as np

from dorsim import (
    Circuit,
    KnillDecoder,
    PauliFrame,
    PoulinDecoder,
    concat_code,
    get_c4,
    get_c4c6_code,
    get_c6,
    target_rec,
)


@dataclass(frozen=True)
class StateCacheKey:
    kind: str
    level: int
    error_rate: float


class PreparedFrameCache:
    """Chunked FIFO cache of accepted, unconsumed Pauli-frame rows."""

    def __init__(
        self,
        batch_size: int = 10000,
        max_batch_size: int = 100000,
        max_empty_batches: int = 10,
    ):
        self.batch_size = int(batch_size)
        self.max_batch_size = int(max_batch_size)
        self.max_empty_batches = int(max_empty_batches)
        self._chunks = {}
        self._offsets = {}
        self._available = {}
        self._generated = {}
        self._accepted = {}

    def available(self, key: StateCacheKey) -> int:
        return self._available.get(key, 0)

    def put(self, key: StateCacheKey, frame: np.ndarray) -> None:
        rows = np.asarray(frame, dtype=np.uint8)
        if rows.shape[0] == 0:
            return
        self._chunks.setdefault(key, deque()).append(rows.copy())
        self._offsets.setdefault(key, 0)
        self._available[key] = self.available(key) + rows.shape[0]

    def take(self, key: StateCacheKey, shots: int, circuit: Circuit) -> PauliFrame:
        shots = int(shots)
        assert self.available(key) >= shots
        remaining = shots
        parts = []
        chunks = self._chunks.setdefault(key, deque())
        offset = self._offsets.get(key, 0)
        while remaining:
            chunk = chunks[0]
            count = min(remaining, chunk.shape[0] - offset)
            parts.append(chunk[offset:offset+count])
            offset += count
            remaining -= count
            if offset == chunk.shape[0]:
                chunks.popleft()
                offset = 0
        self._offsets[key] = offset
        self._available[key] = self.available(key) - shots
        rows = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0].copy()
        return PauliFrame(circuit, shots=0).update(rows)

    def get_or_prepare(self, key, shots, circuit, producer) -> PauliFrame:
        shots = int(shots)
        empty_batches = 0
        last_batch = max(self.batch_size, min(shots, self.max_batch_size))
        while self.available(key) < shots:
            missing = shots - self.available(key)
            generated = self._generated.get(key, 0)
            accepted = self._accepted.get(key, 0)
            if generated and accepted:
                rate = accepted / generated
                batch = int(np.ceil(1.2 * missing / rate))
            elif empty_batches:
                batch = 2*last_batch
            else:
                batch = max(self.batch_size, missing)
            batch = min(batch, self.max_batch_size)
            prepared = producer(batch)
            accepted_now = prepared.shots
            self._generated[key] = generated + batch
            self._accepted[key] = accepted + accepted_now
            if accepted_now:
                self.put(key, prepared.frame)
                empty_batches = 0
            else:
                empty_batches += 1
                if empty_batches >= self.max_empty_batches:
                    raise RuntimeError(f"preparation repeatedly accepted no states for {key}")
            last_batch = batch
        return self.take(key, shots, circuit)

    def stats(self, key: StateCacheKey | None = None):
        keys = {key} if key is not None else (
            set(self._available) | set(self._generated) | set(self._accepted)
        )
        return {
            item: {
                "available": self.available(item),
                "generated": self._generated.get(item, 0),
                "accepted": self._accepted.get(item, 0),
                "acceptance_rate": (
                    self._accepted.get(item, 0) / self._generated[item]
                    if self._generated.get(item, 0)
                    else 0.0
                ),
            }
            for item in keys
        }

    def clear(self, key: StateCacheKey | None = None) -> None:
        if key is None:
            self._chunks.clear()
            self._offsets.clear()
            self._available.clear()
            self._generated.clear()
            self._accepted.clear()
            return
        self._chunks.pop(key, None)
        self._offsets.pop(key, None)
        self._available.pop(key, None)
        self._generated.pop(key, None)
        self._accepted.pop(key, None)


PREPARED_FRAME_CACHE = PreparedFrameCache()


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


def ECT(level, decoder, measurement, frame) -> np.ndarray:
    """Decode ECT measurements and return the corrected eight-block frame."""
    n_q = 4 * 3 ** (level - 1)
    code = code_list[level - 1]
    logical_x = code.get_lx()
    logical_z = code.get_lz()
    rez = np.concatenate([decoder.decode_code(measurement[:, i*n_q:(i+1)*n_q], code, True) for i in range(2)], axis=1).astype(np.uint8)
    rex = np.concatenate([decoder.decode_code(measurement[:, (2+i)*n_q:(3+i)*n_q], code, True) for i in range(2)], axis=1).astype(np.uint8)

    frame_corrected = np.asarray(frame, dtype=np.uint8).copy()
    for i, block in enumerate((5, 7)):
        x_correction = (rex[:, [2*i]] * logical_x[0]) ^ (rex[:, [2*i+1]] * logical_x[1])
        z_correction = (rez[:, [2*i]] * logical_z[0]) ^ (rez[:, [2*i+1]] * logical_z[1])
        frame_corrected[:, block*n_q:(block+1)*n_q] ^= x_correction
        frame_corrected[:, (8+block)*n_q:(9+block)*n_q] ^= z_correction
    return frame_corrected


def get_circuit_relabel(level) -> Circuit:
    """Move the four ECT output blocks into final measurement order."""
    n_q = 4 * 3 ** (level - 1)
    ind_q = np.arange(4*n_q)
    return (
        Circuit(4*n_q)
        .swap(np.column_stack((ind_q[2*n_q:3*n_q], ind_q[3*n_q:4*n_q])).ravel())
        .swap(np.column_stack((ind_q[n_q:2*n_q], ind_q[3*n_q:4*n_q])).ravel())
        .swap(np.column_stack((ind_q[:n_q], ind_q[n_q:2*n_q])).ravel())
    )


# Prepare l1
def get_PFrame_l1(shots, circuit, error_rate, cache=None):
    cache = PREPARED_FRAME_CACHE if cache is None else cache
    key = StateCacheKey("state", 1, float(error_rate))

    def produce(batch):
        pframe = PauliFrame(circuit, shots=batch).run()
        keep = (pframe.samples.sum(axis=1) % 2) == 0
        pframe.update(pframe.frame[keep])
        return pframe.select_qubits([1, 3, 5, 7])

    return cache.get_or_prepare(key, shots, Circuit(4), produce)


def get_PFrame_l1_bell(shots, circuits, error_rate, cache=None):
    cache = PREPARED_FRAME_CACHE if cache is None else cache
    key = StateCacheKey("bell", 1, float(error_rate))

    def produce(batch):
        states = [
            get_PFrame_l1(batch, circuits[0], error_rate, cache)
            for _ in range(2)]
        return PauliFrame.bunch(states, circuit=circuits[1]).run()

    return cache.get_or_prepare(key, shots, Circuit(8), produce)


def get_PFrame_l2(shots, decoder, circuits, error_rate, cache=None):
    cache = PREPARED_FRAME_CACHE if cache is None else cache
    key = StateCacheKey("state", 2, float(error_rate))
    lower_code = code_list[0]
    logical_x = lower_code.get_lx()

    def produce(batch):
        lower_bells = [
            get_PFrame_l1_bell(batch, circuits[:2], error_rate, cache)
            for _ in range(3)]
        pframe = PauliFrame.bunch(lower_bells, circuit=circuits[2]).run()
        measurement = pframe.samples
        results = np.concatenate(
            [decoder.decode_code(measurement[:, i*4:(i+1)*4], lower_code) for i in range(3)],
            axis=1)
        keep = (
            np.all(results != -1, axis=1)
            & ((results[:, [0, 2, 4]].sum(axis=1) % 2) == 0)
            & ((results[:, [1, 3, 5]].sum(axis=1) % 2) == 0))
        pframe.update(pframe.frame[keep])
        pframe.select_qubits(np.r_[4:8, 12:16, 20:24])
        correction = results[keep].astype(np.uint8)
        frame = pframe.frame
        frame[:, :4] ^= (correction[:, [0]]*logical_x[0]) ^ (correction[:, [1]]*logical_x[1])
        frame[:, 4:8] ^= (
            (correction[:, [0]] ^ correction[:, [2]])*logical_x[0]) ^ (
            (correction[:, [1]] ^ correction[:, [3]])*logical_x[1])
        return pframe.update(frame=frame, circuit=circuits[3]).run()

    return cache.get_or_prepare(key, shots, Circuit(12), produce)


def get_PFrame_l2_bell(shots, decoder, circuits, error_rate, cache=None):
    cache = PREPARED_FRAME_CACHE if cache is None else cache
    key = StateCacheKey("bell", 2, float(error_rate))
    lower_code = code_list[0]
    logical_x = lower_code.get_lx()
    logical_z = lower_code.get_lz()

    def produce(batch):
        states = [
            get_PFrame_l2(batch, decoder, circuits[:4], error_rate, cache)
            for _ in range(2)]
        pframe_bell = PauliFrame.bunch(states, circuit=circuits[4]).run()
        lower_bells = [
            get_PFrame_l1_bell(batch, circuits[:2], error_rate, cache)
            for _ in range(6)]
        pframe = PauliFrame.bunch([pframe_bell] + lower_bells, circuit=circuits[5]).run()
        measurement = pframe.samples
        rez = np.concatenate(
            [decoder.decode_code(measurement[:, i*4:(i+1)*4], lower_code) for i in range(6)],
            axis=1)
        rex = np.concatenate(
            [decoder.decode_code(measurement[:, (6+i)*4:(7+i)*4], lower_code) for i in range(6)],
            axis=1)
        keep = np.all(rex != -1, axis=1) & np.all(rez != -1, axis=1)
        rex = rex[keep].astype(np.uint8)
        rez = rez[keep].astype(np.uint8)
        pframe.update(pframe.frame[keep])
        pframe.select_qubits(np.r_[28:32, 36:40, 44:48, 52:56, 60:64, 68:72])
        frame = pframe.frame
        for i in range(6):
            frame[:, i*4:(i+1)*4] ^= (rex[:, [2*i]]*logical_x[0]) ^ (rex[:, [2*i+1]]*logical_x[1])
            frame[:, (6+i)*4:(7+i)*4] ^= (rez[:, [2*i]]*logical_z[0]) ^ (rez[:, [2*i+1]]*logical_z[1])
        return pframe.update(frame)

    return cache.get_or_prepare(key, shots, Circuit(24), produce)


def get_PFrame_l3(shots, decoder, circuits, error_rate, cache=None):
    cache = PREPARED_FRAME_CACHE if cache is None else cache
    key = StateCacheKey("state", 3, float(error_rate))
    lower_code = code_list[1]
    n_sub = lower_code.n
    logical_x = lower_code.get_lx()

    def produce(batch):
        lower_bells = [
            get_PFrame_l2_bell(batch, decoder, circuits[:6], error_rate, cache)
            for _ in range(3)]
        pframe = PauliFrame.bunch(lower_bells, circuit=circuits[6]).run()
        measurement = pframe.samples
        results = np.concatenate(
            [decoder.decode_code(measurement[:, i*n_sub:(i+1)*n_sub], lower_code) for i in range(3)],
            axis=1)
        keep = (
            np.all(results != -1, axis=1)
            & ((results[:, [0, 2, 4]].sum(axis=1) % 2) == 0)
            & ((results[:, [1, 3, 5]].sum(axis=1) % 2) == 0))
        pframe.update(pframe.frame[keep])
        pframe.select_qubits(np.r_[n_sub:2*n_sub, 3*n_sub:4*n_sub, 5*n_sub:6*n_sub])
        correction = results[keep].astype(np.uint8)
        frame = pframe.frame
        frame[:, :n_sub] ^= (correction[:, [0]]*logical_x[0]) ^ (correction[:, [1]]*logical_x[1])
        frame[:, n_sub:2*n_sub] ^= (
            (correction[:, [0]] ^ correction[:, [2]])*logical_x[0]) ^ (
            (correction[:, [1]] ^ correction[:, [3]])*logical_x[1])
        return pframe.update(frame=frame, circuit=circuits[7]).run()

    return cache.get_or_prepare(key, shots, Circuit(36), produce)


def get_PFrame_l3_bell(shots, decoder, circuits, error_rate, cache=None):
    cache = PREPARED_FRAME_CACHE if cache is None else cache
    key = StateCacheKey("bell", 3, float(error_rate))
    lower_code = code_list[1]
    n_sub = lower_code.n
    logical_x = lower_code.get_lx()
    logical_z = lower_code.get_lz()

    def produce(batch):
        states = [
            get_PFrame_l3(batch, decoder, circuits[:8], error_rate, cache)
            for _ in range(2)]
        pframe_bell = PauliFrame.bunch(states, circuit=circuits[8]).run()
        lower_bells = [
            get_PFrame_l2_bell(batch, decoder, circuits[:6], error_rate, cache)
            for _ in range(6)]
        pframe = PauliFrame.bunch([pframe_bell] + lower_bells, circuit=circuits[9]).run()
        measurement = pframe.samples
        rez = np.concatenate(
            [decoder.decode_code(measurement[:, i*n_sub:(i+1)*n_sub], lower_code) for i in range(6)],
            axis=1)
        rex = np.concatenate(
            [decoder.decode_code(measurement[:, (6+i)*n_sub:(7+i)*n_sub], lower_code) for i in range(6)],
            axis=1)
        keep = np.all(rex != -1, axis=1) & np.all(rez != -1, axis=1)
        rex = rex[keep].astype(np.uint8)
        rez = rez[keep].astype(np.uint8)
        pframe.update(pframe.frame[keep])
        pframe.select_qubits(
            np.concatenate([
                np.arange((7+2*i)*n_sub, (8+2*i)*n_sub)
                for i in range(6)]))
        frame = pframe.frame
        for i in range(6):
            frame[:, i*n_sub:(i+1)*n_sub] ^= (rex[:, [2*i]]*logical_x[0]) ^ (rex[:, [2*i+1]]*logical_x[1])
            frame[:, (6+i)*n_sub:(7+i)*n_sub] ^= (rez[:, [2*i]]*logical_z[0]) ^ (rez[:, [2*i+1]]*logical_z[1])
        return pframe.update(frame)

    return cache.get_or_prepare(key, shots, Circuit(72), produce)



def run_level1(shots, noise, cache=None):
    shot = shots
    er = noise
    dec = KnillDecoder()
    cache = PREPARED_FRAME_CACHE if cache is None else cache
    
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
    pframe_l1 = PauliFrame.bunch([
        get_PFrame_l1_bell(shot, [cir_l1_i, cir_l1_bell_i], 0.0, cache),
        get_PFrame_l1_bell(shot, [cir_l1_i, cir_l1_bell_i], 0.0, cache),
    ], circuit=cir_a).run()
    ## ECT
    cir_b = get_circuit_tele(1, er)
    pframe_l1_ect = PauliFrame.bunch([
        pframe_l1,
        get_PFrame_l1_bell(shot, [cir_l1, cir_l1_bell], er, cache),
        get_PFrame_l1_bell(shot, [cir_l1, cir_l1_bell], er, cache),
    ], circuit=cir_b).run()
    shot_eff = pframe_l1_ect.frame.shape[0] # effective shots after post-selection
    print(f"shot_eff: {shot_eff}")
    mea = pframe_l1_ect.samples
    frame_corrected = ECT(1, dec, mea, pframe_l1_ect.frame)
    # select qubits
    pframe_l1_ect.update(frame=frame_corrected)
    pframe_l1_ect.select_qubits(np.r_[n_q:2*n_q, 3*n_q:4*n_q, 5*n_q:6*n_q, 7*n_q:8*n_q])
    # relabel
    pframe_l1_ect.update(circuit=get_circuit_relabel(1)).run()
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

def run_level2(shots, noise, cache=None):
    shot = shots
    er = noise
    dec = KnillDecoder()
    cache = PREPARED_FRAME_CACHE if cache is None else cache

    circits_i = [
        get_circuit_c4(0),
        get_circuit_c4c6_bell(1, 0),
        get_circuit_c4c6_p1(2, 0),
        get_circuit_c4c6_p2(2, 0),
        get_circuit_c4c6_bell(2, 0),
        get_circuit_c4c6_tele(2, 0)]
    
    circuits = [
        get_circuit_c4(er),
        get_circuit_c4c6_bell(1, er),
        get_circuit_c4c6_p1(2, er),
        get_circuit_c4c6_p2(2, er),
        get_circuit_c4c6_bell(2, er),
        get_circuit_c4c6_tele(2, er)]

    ### level 2
    n_q = 12
    ind_q = np.arange(48)
    ## Dep error
    cir_a = (
        C4C6Circuit(4*n_q)
        .depolarize2(np.column_stack((ind_q[:n_q], ind_q[2*n_q:3*n_q])).ravel(), er))
    pframe_l2 = PauliFrame.bunch([
        get_PFrame_l2_bell(shot, dec, circits_i, 0.0, cache),
        get_PFrame_l2_bell(shot, dec, circits_i, 0.0, cache)], circuit=cir_a).run()
    ## ECT
    cir_b = get_circuit_tele(2, er)
    pframe_l2_ect = PauliFrame.bunch([
        pframe_l2,
        get_PFrame_l2_bell(shot, dec, circuits, er, cache),
        get_PFrame_l2_bell(shot, dec, circuits, er, cache)], circuit=cir_b).run()
    shot_eff = pframe_l2_ect.frame.shape[0]
    print(f"shot_eff: {shot_eff}")
    mea = pframe_l2_ect.samples
    frame_corrected = ECT(2, dec, mea, pframe_l2_ect.frame)
    # select qubits
    pframe_l2_ect.update(frame=frame_corrected)
    pframe_l2_ect.select_qubits(np.r_[n_q:2*n_q, 3*n_q:4*n_q, 5*n_q:6*n_q, 7*n_q:8*n_q])
    # relabel
    pframe_l2_ect.update(circuit=get_circuit_relabel(2)).run()
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


def run_level3(shots, noise, cache=None):
    shot = shots
    er = noise
    dec = KnillDecoder()
    code = get_c4c6_code(3)
    cache = PREPARED_FRAME_CACHE if cache is None else cache

    circuits_i = [
        get_circuit_c4(0),
        get_circuit_c4c6_bell(1, 0),
        get_circuit_c4c6_p1(2, 0),
        get_circuit_c4c6_p2(2, 0),
        get_circuit_c4c6_bell(2, 0),
        get_circuit_c4c6_tele(2, 0),
        get_circuit_c4c6_p1(3, 0),
        get_circuit_c4c6_p2(3, 0),
        get_circuit_c4c6_bell(3, 0),
        get_circuit_c4c6_tele(3, 0),
    ]
    circuits = [
        get_circuit_c4(er),
        get_circuit_c4c6_bell(1, er),
        get_circuit_c4c6_p1(2, er),
        get_circuit_c4c6_p2(2, er),
        get_circuit_c4c6_bell(2, er),
        get_circuit_c4c6_tele(2, er),
        get_circuit_c4c6_p1(3, er),
        get_circuit_c4c6_p2(3, er),
        get_circuit_c4c6_bell(3, er),
        get_circuit_c4c6_tele(3, er),
    ]

    n_q = code.n
    ind_q = np.arange(4*n_q)
    noisy_cnot = (
        C4C6Circuit(4*n_q)
        .depolarize2(np.column_stack((ind_q[:n_q], ind_q[2*n_q:3*n_q])).ravel(), er)
    )
    pframe_l3 = PauliFrame.bunch([
        get_PFrame_l3_bell(shot, dec, circuits_i, 0.0, cache),
        get_PFrame_l3_bell(shot, dec, circuits_i, 0.0, cache),
    ], circuit=noisy_cnot).run()

    pframe_l3_ect = PauliFrame.bunch([
        pframe_l3,
        get_PFrame_l3_bell(shot, dec, circuits, er, cache),
        get_PFrame_l3_bell(shot, dec, circuits, er, cache),
    ], circuit=get_circuit_tele(3, er)).run()
    shot_eff = pframe_l3_ect.shots
    print(f"shot_eff: {shot_eff}")
    corrected = ECT(3, dec, pframe_l3_ect.samples, pframe_l3_ect.frame)
    pframe_l3_ect.update(corrected)
    pframe_l3_ect.select_qubits(np.r_[n_q:2*n_q, 3*n_q:4*n_q, 5*n_q:6*n_q, 7*n_q:8*n_q])
    pframe_l3_ect.update(circuit=get_circuit_relabel(3)).run()

    measurement = pframe_l3_ect.update(circuit=get_circuit_meaNdec(3)).run().samples
    final_decoder = PoulinDecoder("X", code, p=er)
    measurement_blocks = measurement.reshape(-1, n_q)
    syndrome = (measurement_blocks @ final_decoder.check.T) % 2
    recovery, _ = final_decoder.decode_syndrome(syndrome)
    logical = ((measurement_blocks ^ recovery) @ code.logical_z[:, n_q:].T) % 2
    logical = logical.reshape(measurement.shape[0], -1)
    return float(logical.any(axis=1).sum() / shot_eff)


c4 = get_c4()
c6 = get_c6()
c4c6_l1 = c4
c4c6_l2 = concat_code(c6, [c4, c4, c4])
c4c6_l3 = concat_code(c6, [c4c6_l2, c4c6_l2, c4c6_l2])
code_list = [c4c6_l1, c4c6_l2, c4c6_l3]

shot = 100000

err_list = np.linspace(0.002, 0.01, 10)
# err_list = np.linspace(0.1, 0.2, 3)
ler_list = [run_level1(shot, err) for err in err_list]
# ler_list = [run_level2(shot, err) for err in err_list]
# ler_list = [run_level3(shot, err) for err in err_list]

print(err_list.tolist())
print(ler_list)
