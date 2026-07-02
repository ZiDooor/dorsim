from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dorsim import Circuit, Operation, PauliFrame, TableauSim, target_rec
from dorsim.pauli import bits_from_code, code_from_bits, local_conjugation_map


def test_flat_stim_style_operation_storage():
    circuit = Circuit(4).cx([0, 1, 2, 3])
    assert circuit.operations == [Operation("CX", (0, 1, 2, 3), 0.0)]

    noise = Circuit(2).depolarize2([0, 1], 0.25)
    assert noise.operations == [Operation("DEPOLARIZE2", (0, 1), 0.25)]
    assert str(noise.to_stim_circuit()).strip() == "DEPOLARIZE2(0.25) 0 1"
    assert noise.without_noise().operations == []

    measured = Circuit(3).m([0, 1]).mx([2])
    assert measured.operations == [Operation("M", (0, 1), 0.0), Operation("MX", (2,), 0.0)]
    assert measured.num_measurements == 3


def test_reference_and_pauli_frame_shapes():
    circuit = Circuit(2).h([0]).cx([0, 1]).m([0]).mx([1])

    tab = TableauSim(circuit).run()
    assert tab.tableau.shape == (4, 4)
    assert tab.sign.shape == (4,)
    assert tab.reference_measurements.shape == (2,)

    frames = PauliFrame(circuit, shots=5, seed=1).run(reference=tab.reference_measurements)
    assert frames.frame.shape == (5, 4)
    assert frames.measurement_flips.shape == (2, 5)
    assert frames.samples.shape == (2, 5)


def test_measurement_record_target_converts_to_stim():
    circuit = Circuit(2).m([0]).cx([target_rec(-1), 1])

    assert circuit.operations == [
        Operation("M", (0,), 0.0),
        Operation("CX", (target_rec(-1), 1), 0.0),
    ]
    assert str(circuit.to_stim_circuit()).strip() == "M 0\nCX rec[-1] 1"


def test_reference_feedback_cx_and_cz():
    x_feedback = Circuit(2).x([0]).m([0]).cx([target_rec(-1), 1]).m([1])
    assert TableauSim(x_feedback).run().reference_measurements.tolist() == [1, 1]

    z_feedback = Circuit(2).x([0]).h([1]).m([0]).cz([target_rec(-1), 1]).mx([1])
    assert TableauSim(z_feedback).run().reference_measurements.tolist() == [1, 1]


def test_pauli_frame_feedback_copies_measurement_sample():
    circuit = Circuit(2).h([0]).m([0]).cx([target_rec(-1), 1]).m([1])

    reference = TableauSim(circuit).run().reference_measurements
    frames = PauliFrame(circuit, shots=64, seed=3).run(reference=reference)

    assert np.array_equal(frames.samples[0], frames.samples[1])


def test_depolarize2_samples_all_non_identity_pair_errors():
    class FakeRng:
        def random(self, size):
            return np.zeros(size)

        def integers(self, low, high=None, size=None, dtype=None):
            values = np.arange(size) % high
            if dtype is not None:
                values = values.astype(dtype)
            return values

    frame = PauliFrame(Circuit(2), shots=15, seed=1)
    frame.frame[:] = 0
    frame.rng = FakeRng()
    frame._apply_depolarize2_to_pair(Operation("DEPOLARIZE2", (0, 1), 1.0), 0, 1)

    observed = {
        (
            code_from_bits(frame.frame[shot, 0], frame.frame[shot, 2]),
            code_from_bits(frame.frame[shot, 1], frame.frame[shot, 3]),
        )
        for shot in range(frame.shots)
    }
    expected = {
        (0, 1),
        (0, 3),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 3),
        (1, 2),
        (3, 0),
        (3, 1),
        (3, 3),
        (3, 2),
        (2, 0),
        (2, 1),
        (2, 3),
        (2, 2),
    }
    assert observed == expected


def test_depolarize2_sampling_distribution_matches_stim():
    circuit = Circuit(2).depolarize2([0, 1], p=0.3).m([0, 1])
    reference = TableauSim(circuit).run().reference_measurements
    ours = PauliFrame(circuit, shots=20000, seed=11).run(reference=reference).samples

    stim_samples = circuit.to_stim_circuit().compile_sampler(seed=11).sample(shots=20000).T

    assert np.all(np.abs(ours.mean(axis=1) - stim_samples.mean(axis=1)) < 0.03)


def test_direct_pauli_frame_gate_rules_match_conjugation_map():
    for gate, arity in [
        ("X", 1),
        ("Y", 1),
        ("Z", 1),
        ("H", 1),
        ("S", 1),
        ("S_DAG", 1),
        ("CX", 2),
        ("CY", 2),
        ("CZ", 2),
        ("SWAP", 2),
    ]:
        mapping = local_conjugation_map(gate, arity)
        for local_in, (_, expected_out) in mapping.items():
            circuit = Circuit(arity)
            frame = PauliFrame(circuit, shots=1, seed=1)
            frame.frame[:] = 0
            for q, code in enumerate(local_in):
                x, z = bits_from_code(code)
                frame.frame[0, q] = x
                frame.frame[0, frame.n + q] = z

            frame._conjugate_frame_by_gate(Operation(gate, tuple(range(arity))))

            got = tuple(
                code_from_bits(frame.frame[0, q], frame.frame[0, frame.n + q])
                for q in range(arity)
            )
            assert got == expected_out


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
    print("test_package ok")
