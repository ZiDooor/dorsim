from __future__ import annotations
import itertools
import numpy as np
from dorsim import (
    BiasedPoulinDecoder,
    CombinedPoulinDecoder,
    CSSCode,
    StabilizerCode,
    concat_code,
)
from dorsim.decoder import _bits_to_index, _index_to_bits
from dorsim.stab_code import CSSCode as CompatibleCSSCode


def _syndrome(error: np.ndarray, code: StabilizerCode) -> np.ndarray:
    check = np.concatenate(
        [code.stabilizers[:, code.n :], code.stabilizers[:, : code.n]],
        axis=1,
    )
    return (error @ check.T) % 2


def _internal_logical_index(error: np.ndarray, code: StabilizerCode) -> np.ndarray:
    syndrome = _syndrome(error, code)
    delta = error ^ ((syndrome @ code.pure_errors) % 2)
    logical_x = (
        delta[:, : code.n] @ code.logical_z[:, code.n :].T
        + delta[:, code.n :] @ code.logical_z[:, : code.n].T
    ) % 2
    logical_z = (
        delta[:, : code.n] @ code.logical_x[:, code.n :].T
        + delta[:, code.n :] @ code.logical_x[:, : code.n].T
    ) % 2
    logical_bits = np.concatenate([logical_x, logical_z], axis=1)
    weights = 1 << np.arange(logical_bits.shape[1] - 1, -1, -1)
    return (logical_bits * weights).sum(axis=1).astype(np.int64)


def _all_pauli_frames(code: StabilizerCode) -> np.ndarray:
    return np.array(
        list(itertools.product([0, 1], repeat=2 * code.n)),
        dtype=np.uint8,
    )


def _combined_exhaustive(
    code: StabilizerCode,
    probabilities: np.ndarray,
    syndrome_m: np.ndarray,
    syndrome_f: np.ndarray,
    logical_m: int,
) -> np.ndarray:
    frames = _all_pauli_frames(code)
    syndromes = _syndrome(frames, code)
    logical = _internal_logical_index(frames, code)
    m_keep = np.all(syndromes == syndrome_m, axis=1) & (logical == logical_m)
    f_keep = np.all(syndromes == syndrome_f, axis=1)
    m = frames[m_keep]
    f = frames[f_keep]
    m_pauli = m[:, : code.n] + 2 * m[:, code.n :]
    f_pauli = f[:, : code.n] + 2 * f[:, code.n :]
    difference = m_pauli[:, None, :] ^ f_pauli[None, :, :]
    weight = probabilities[difference].prod(axis=2)
    result = np.zeros(4**code.k, dtype=np.float64)
    np.add.at(result, logical[f_keep], weight.sum(axis=0))
    return result / result.sum()


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


def test_index_to_bits_is_inverse_of_bits_to_index():
    assert np.array_equal(
        _index_to_bits(np.array(2), 2),
        np.array([1, 0], dtype=np.uint8),
    )

    indices = np.arange(16, dtype=np.int64)
    bits = _index_to_bits(indices, 4)
    assert bits.shape == (16, 4)
    assert bits.dtype == np.uint8
    assert np.array_equal(_bits_to_index(bits), indices)


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
    pauli = errors[:, : code.n] + 2 * errors[:, code.n :]
    probability = np.array([1 - px - py - pz, px, pz, py])[pauli].prod(axis=1)
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


def test_combined_decoder_matches_exhaustive_enumeration_and_recovery():
    code = CSSCode.c4()
    px, py, pz = 0.07, 0.02, 0.11
    probabilities = np.array([1 - px - py - pz, px, pz, py])
    syndrome_m = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    syndrome_f = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    logical_m = np.array([1, 7], dtype=np.int64)

    recovery, result = CombinedPoulinDecoder(code, px, py, pz).decode_syndrome(
        syndrome_m,
        syndrome_f,
        logical_m,
    )
    expected = np.stack(
        [
            _combined_exhaustive(code, probabilities, sm, sf, lm)
            for sm, sf, lm in zip(syndrome_m, syndrome_f, logical_m)
        ]
    )

    assert recovery.shape == (2, 2 * code.n)
    assert result[-1].shape == (2, 4**code.k)
    assert np.allclose(np.exp(result[-1]), expected)
    assert np.array_equal(_syndrome(recovery, code), syndrome_f)
    recovery_logical = _internal_logical_index(recovery, code)
    assert np.allclose(
        expected[np.arange(expected.shape[0]), recovery_logical],
        expected.max(axis=1),
    )


def test_biased_decoder_returns_shared_internal_logical_index():
    code = _five_qubit_code()
    syndromes = np.array(
        [[0, 0, 0, 0], [1, 0, 1, 0]],
        dtype=np.uint8,
    )
    decoder = BiasedPoulinDecoder(code, 0.07, 0.02, 0.08)
    recovery, probabilities = decoder.decode_syndrome(syndromes)
    recovery_with_logical, logical, probabilities_with_logical = (
        decoder.decode_syndrome_with_logical(syndromes)
    )

    assert np.array_equal(recovery_with_logical, recovery)
    assert probabilities_with_logical is not probabilities
    assert np.array_equal(probabilities_with_logical[-1], probabilities[-1])
    assert np.array_equal(
        logical,
        np.argmax(probabilities[-1], axis=1),
    )

    combined = CombinedPoulinDecoder(code, 0.06, 0.01, 0.09)
    syndrome_f = syndromes[::-1].copy()
    _, combined_result = combined.decode_syndrome(
        syndromes,
        syndrome_f,
        logical,
    )
    difference = BiasedPoulinDecoder(code, 0.06, 0.01, 0.09)
    _, difference_result = difference.decode_syndrome(syndromes ^ syndrome_f)
    logical_f = np.arange(4**code.k)
    expected = difference_result[-1][
        np.arange(syndromes.shape[0])[:, None],
        logical[:, None] ^ logical_f[None, :],
    ]
    assert np.allclose(combined_result[-1], expected)


def test_combined_decoder_scalar_logical_and_conditioning_changes_result():
    code = _five_qubit_code()
    syndrome_m = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    syndrome_f = np.array(
        [[1, 0, 0, 1], [1, 0, 0, 1]],
        dtype=np.uint8,
    )
    decoder = CombinedPoulinDecoder(code, 0.07, 0.02, 0.08)
    _, scalar = decoder.decode_syndrome(syndrome_m, syndrome_f, 0)
    _, varied = decoder.decode_syndrome(
        syndrome_m,
        syndrome_f,
        np.array([0, 1]),
    )

    assert np.allclose(scalar[-1][0], scalar[-1][1])
    assert np.allclose(scalar[-1][0], varied[-1][0])
    assert not np.allclose(varied[-1][0], varied[-1][1])


def test_combined_decoder_recursive_result_matches_exhaustive_enumeration():
    qp = CSSCode.qp()
    code = concat_code(CSSCode.c6(), [qp, qp, qp])
    px, py, pz = 0.06, 0.02, 0.09
    probabilities = np.array([1 - px - py - pz, px, pz, py])
    syndrome_m = np.array([[1, 0, 1, 1]], dtype=np.uint8)
    syndrome_f = np.array([[0, 1, 1, 0]], dtype=np.uint8)
    logical_m = 6

    recovery, result = CombinedPoulinDecoder(code, px, py, pz).decode_syndrome(
        syndrome_m,
        syndrome_f,
        logical_m,
    )
    expected = _combined_exhaustive(
        code,
        probabilities,
        syndrome_m[0],
        syndrome_f[0],
        logical_m,
    )

    assert np.allclose(np.exp(result[-1][0]), expected)
    assert np.array_equal(_syndrome(recovery, code), syndrome_f)
    assert _internal_logical_index(recovery, code)[0] == np.argmax(expected)


def test_combined_decoder_maps_all_root_logicals_from_xor_posterior():
    qp = CSSCode.qp()
    code = concat_code(CSSCode.c6(), [qp, qp, qp])
    logical_count = 4**code.k
    syndrome_m = np.repeat(
        np.array([[1, 0, 1, 1]], dtype=np.uint8),
        logical_count,
        axis=0,
    )
    syndrome_f = np.repeat(
        np.array([[0, 1, 1, 0]], dtype=np.uint8),
        logical_count,
        axis=0,
    )
    logical_m = np.arange(logical_count, dtype=np.int64)
    px, py, pz = 0.06, 0.02, 0.09

    recovery, result = CombinedPoulinDecoder(code, px, py, pz).decode_syndrome(
        syndrome_m,
        syndrome_f,
        logical_m,
    )
    _, difference_result = BiasedPoulinDecoder(
        code,
        px,
        py,
        pz,
    ).decode_syndrome(syndrome_m ^ syndrome_f)
    logical_f = np.arange(logical_count)
    expected = difference_result[-1][
        np.arange(logical_count)[:, None],
        logical_m[:, None] ^ logical_f[None, :],
    ]

    assert np.allclose(result[-1], expected)
    assert np.array_equal(_syndrome(recovery, code), syndrome_f)
    assert np.array_equal(
        _internal_logical_index(recovery, code),
        np.argmax(expected, axis=1),
    )


def test_combined_decoder_rejects_invalid_and_impossible_inputs():
    code = CSSCode.c4()
    with np.testing.assert_raises(ValueError):
        CombinedPoulinDecoder(code, -0.1, 0, 0)
    with np.testing.assert_raises(ValueError):
        CombinedPoulinDecoder(code, 0.5, 0.3, 0.3)

    decoder = CombinedPoulinDecoder(code, 0, 0, 0)
    zero = np.zeros((1, code.n - code.k), dtype=np.uint8)
    nonzero = zero.copy()
    nonzero[0, 0] = 1
    with np.testing.assert_raises(ValueError):
        decoder.decode_syndrome(zero, nonzero, 1)
    with np.testing.assert_raises(ValueError):
        decoder.decode_syndrome(zero, zero[:, :1], 0)
    with np.testing.assert_raises(ValueError):
        decoder.decode_syndrome(zero, zero, 4**code.k)


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
    print("test_biased_decoder ok")
