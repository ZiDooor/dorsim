from __future__ import annotations
import itertools
import numpy as np
from dorsim import (
    BiasedPoulinDecoder,
    CSSCode,
    StabilizerCode,
    concat_code,
)
from dorsim.stab_code import CSSCode as CompatibleCSSCode


def _syndrome(error: np.ndarray, code: StabilizerCode) -> np.ndarray:
    check = np.concatenate(
        [code.stabilizers[:, code.n :], code.stabilizers[:, : code.n]],
        axis=1,
    )
    return (error @ check.T) % 2


def _five_qubit_code() -> StabilizerCode:
    return StabilizerCode(
        name="FiveQubit",
        n=5,
        k=1,
        stabilizers=np.array(
            [[1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
             [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
             [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]],
            dtype=np.uint8,
        ),
        logical_x=np.array(
            [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
            dtype=np.uint8,
        ),
        logical_z=np.array(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],
            dtype=np.uint8,
        ),
    )


def test_stabilizer_code_and_css_compatibility():
    assert issubclass(CSSCode, StabilizerCode)
    assert CompatibleCSSCode is CSSCode
    assert isinstance(CSSCode.c4(), CSSCode)
    assert isinstance(concat_code(CSSCode.c6(), [CSSCode.c4(), CSSCode.c4(), CSSCode.c4()]), CSSCode)

    five = _five_qubit_code()
    concatenated = concat_code(five, [five, five, five, five, five])
    assert type(concatenated) is StabilizerCode
    assert (concatenated.n, concatenated.k) == (25, 1)


def test_pure_errors_generate_unit_syndromes():
    for code in [CSSCode.c4(), CSSCode.c6(), _five_qubit_code()]:
        assert code.pure_errors.shape == (code.n - code.k, 2 * code.n)
        assert np.array_equal(
            _syndrome(code.pure_errors, code),
            np.eye(code.n - code.k, dtype=np.uint8),
        )
        assert code.pure_errors is code.pure_errors


def test_biased_c4_probabilities_match_exhaustive_enumeration():
    code = CSSCode.c4()
    px, py, pz = 0.02, 0.01, 0.03
    decoder = BiasedPoulinDecoder(code, px, py, pz)
    syndromes = np.array(
        list(itertools.product([0, 1], repeat=code.n - code.k)),
        dtype=np.uint8,
    )
    _, result = decoder.decode_syndrome(syndromes)

    errors = np.array(
        list(itertools.product([0, 1], repeat=2 * code.n)),
        dtype=np.uint8,
    )
    pauli = 2 * errors[:, : code.n] + errors[:, code.n :]
    probability = np.array([1 - px - py - pz, pz, px, py])[pauli].prod(axis=1)
    error_syndrome = _syndrome(errors, code)
    delta = errors ^ ((error_syndrome @ code.pure_errors) % 2)
    logical_x = (
        delta[:, : code.n] @ code.logical_z[:, code.n :].T
        + delta[:, code.n :] @ code.logical_z[:, : code.n].T
    ) % 2
    logical_z = (
        delta[:, : code.n] @ code.logical_x[:, code.n :].T
        + delta[:, code.n :] @ code.logical_x[:, : code.n].T
    ) % 2
    logical_bits = np.concatenate([logical_x, logical_z], axis=1)
    syndrome_index = (error_syndrome * (1 << np.arange(code.n - code.k - 1, -1, -1))).sum(axis=1)
    logical_index = (logical_bits * (1 << np.arange(2 * code.k - 1, -1, -1))).sum(axis=1)
    expected = np.zeros((2 ** (code.n - code.k), 4**code.k), dtype=np.float64)
    np.add.at(expected, (syndrome_index, logical_index), probability)
    expected /= expected.sum(axis=1, keepdims=True)

    assert np.allclose(np.exp(result[-1]), expected)


def test_biased_decoder_recovers_requested_syndromes():
    c4 = CSSCode.c4()
    codes = [
        _five_qubit_code(),
        concat_code(CSSCode.c6(), [c4, c4, c4]),
        concat_code(CSSCode.c6(), [CSSCode.qp(), c4, c4]),
    ]
    rng = np.random.default_rng(1)
    for code in codes:
        syndromes = rng.integers(
            0,
            2,
            size=(8, code.n - code.k),
            dtype=np.uint8,
        )
        recovery, result = BiasedPoulinDecoder(code, 0.01, 0.002, 0.03).decode_syndrome(syndromes)
        assert recovery.shape == (8, 2 * code.n)
        assert result[-1].shape == (8, 4**code.k)
        assert np.array_equal(_syndrome(recovery, code), syndromes)
        assert np.allclose(np.exp(result[-1]).sum(axis=1), 1)


def test_zero_probability_channels_do_not_produce_nan():
    code = CSSCode.c4()
    syndromes = np.array(
        list(itertools.product([0, 1], repeat=code.n - code.k)),
        dtype=np.uint8,
    )
    decoder = BiasedPoulinDecoder(code, 0, 0, 0)
    _, result = decoder.decode_syndrome(syndromes)
    assert not np.isnan(result[-1]).any()

    decoder.set_error_model(0.1, 0, 0)
    _, changed = decoder.decode_syndrome(syndromes)
    assert not np.isnan(changed[-1]).any()


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
    print("test_biased_decoder ok")
