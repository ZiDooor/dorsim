from __future__ import annotations

import numpy as np
import scipy.special

from dorsim import CSSCode, JointPoulinDecoder, StabilizerCode, concat_code
from dorsim.decoder import _bits_to_index


def _direct_child_logical_mappings(
    decoder: JointPoulinDecoder,
    code: StabilizerCode,
) -> np.ndarray:
    parent = code.parent
    local_ops = decoder._local_decomposition(parent)["local_ops"]
    logical_offsets = np.cumsum([0] + [child.k for child in code.children])
    mappings = []
    for child_i, child in enumerate(code.children):
        start = logical_offsets[child_i]
        stop = logical_offsets[child_i + 1]
        child_bits = np.concatenate(
            [
                local_ops[..., start:stop],
                local_ops[..., parent.n + start : parent.n + stop],
            ],
            axis=-1,
        )
        mappings.append(_bits_to_index(child_bits))
    return np.stack(mappings, axis=1)


def test_child_logical_mappings_match_direct_calculation():
    qp = CSSCode.qp()
    c4 = CSSCode.c4()
    steane = StabilizerCode.steane()
    codes = [
        concat_code(c4, [qp, qp]),
        concat_code(steane, [steane, c4, c4, c4]),
    ]
    for code in codes:
        decoder = JointPoulinDecoder.from_ect_rates(code, 0.03, 0.06, 0.08)
        assert np.array_equal(
            decoder._child_logical_mappings(code),
            _direct_child_logical_mappings(decoder, code),
        )


def test_child_logical_mapping_cache_survives_channel_update():
    qp = CSSCode.qp()
    code = concat_code(CSSCode.c4(), [qp, qp])
    decoder = JointPoulinDecoder.from_ect_rates(code, 0.03, 0.06, 0.08)
    mappings = decoder._child_logical_mappings(code)
    decoder.set_joint_error_model(np.ones((4, 4), dtype=np.float64) / 16)
    assert decoder._child_logical_mappings(code) is mappings
    assert decoder._leaf_table_cache == {}


def test_specialized_stabilizer_logsumexp_matches_scipy():
    rng = np.random.default_rng(7)
    score = rng.normal(size=(3, 4, 4, 5, 6))
    score[..., 0, 0] = -np.inf
    score[0, 1, 2, :, :] = -np.inf
    expected = scipy.special.logsumexp(score, axis=(3, 4))
    actual = JointPoulinDecoder._logsumexp_stabilizers(score.copy())
    assert np.allclose(actual, expected)
    assert np.array_equal(np.isneginf(actual), np.isneginf(expected))
