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
    assert frames.measurement_flips.shape == (5, 2)
    assert frames.samples.shape == (5, 2)


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

    assert np.array_equal(frames.samples[:, 0], frames.samples[:, 1])


def test_pauli_frame_update_resets_outputs_and_optionally_replaces_frame():
    frame = PauliFrame(Circuit(2).m([0, 1]), shots=3, seed=1)
    old_frame = frame.frame.copy()
    frame.measurement_flips[:] = 1
    frame.samples = np.ones_like(frame.measurement_flips)
    frame._measurement_index = 2

    assert frame.update() is frame
    assert np.array_equal(frame.frame, old_frame)
    assert np.array_equal(frame.measurement_flips, np.zeros((3, 2), dtype=np.uint8))
    assert frame.samples is None
    assert frame._measurement_index == 0

    new_circuit = Circuit(2).m([0])
    old_frame = frame.frame.copy()
    frame.measurement_flips[:] = 1
    frame.samples = np.ones_like(frame.measurement_flips)
    frame._measurement_index = 2
    frame.update(circuit=new_circuit)

    assert frame.circuit is new_circuit
    assert np.array_equal(frame.frame, old_frame)
    assert np.array_equal(frame.measurement_flips, np.zeros((3, 1), dtype=np.uint8))
    assert frame.samples is None
    assert frame._measurement_index == 0

    newer_circuit = Circuit(2).m([0, 1])
    newer_frame = np.zeros((4, 4), dtype=np.uint8)
    frame.update(newer_frame, circuit=newer_circuit)

    assert frame.circuit is newer_circuit
    assert frame.shots == 4
    assert np.array_equal(frame.frame, newer_frame)
    assert frame.frame is not newer_frame
    assert np.array_equal(frame.measurement_flips, np.zeros((4, 2), dtype=np.uint8))

    resized_circuit = Circuit(3).m([0, 1, 2])
    resized_frame = np.zeros((4, 6), dtype=np.uint8)
    frame.update(resized_frame, circuit=resized_circuit)

    assert frame.circuit is resized_circuit
    assert frame.n == 3
    assert frame.shots == 4
    assert np.array_equal(frame.frame, resized_frame)
    assert np.array_equal(frame.measurement_flips, np.zeros((4, 3), dtype=np.uint8))

    new_frame = np.ones((3, 4), dtype=np.uint8)
    frame.update(new_frame, circuit=Circuit(2).m([0, 1]))
    frame.measurement_flips[:] = 1
    frame.samples = np.ones_like(frame.measurement_flips)
    frame._measurement_index = 2
    frame.update(new_frame)

    assert np.array_equal(frame.frame, new_frame)
    assert frame.frame is not new_frame
    assert np.array_equal(frame.measurement_flips, np.zeros((3, 2), dtype=np.uint8))
    assert frame.samples is None
    assert frame._measurement_index == 0


def test_pauli_frame_update_to_second_circuit_continues_from_current_frame():
    first = Circuit(2).x_error([0], p=1)
    second = Circuit(2).x_error([1], p=1)

    frame = PauliFrame(first, shots=5, seed=1).run()
    after_first = frame.frame.copy()

    frame.update(circuit=second)

    assert frame.circuit is second
    assert np.array_equal(frame.frame, after_first)
    assert frame.measurement_flips.shape == (5, 0)
    assert frame.samples is None

    frame.run()

    expected = after_first.copy()
    expected[:, 1] ^= 1
    assert frame.frame.shape == (5, 4)
    assert np.array_equal(frame.frame, expected)
    assert frame.samples.shape == (5, 0)

    measured = Circuit(2).m([0, 1])
    frame.update(circuit=measured).run()

    assert frame.circuit is measured
    assert frame.measurement_flips.shape == (5, 2)
    assert frame.samples.shape == (5, 2)


def test_pauli_frame_select_qubits_reorders_frame_and_continues():
    source = PauliFrame(Circuit(3).m([0, 1, 2]), shots=2, seed=1)
    source.frame = np.array(
        [
            [1, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    source.measurement_flips[:] = 1
    source.samples = np.ones_like(source.measurement_flips)
    source._measurement_index = 3

    next_circuit = Circuit(2).x_error([1], p=1).m([0, 1])
    assert source.select_qubits([2, 0], circuit=next_circuit) is source

    expected_selected = np.array(
        [
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert source.n == 2
    assert source.circuit is next_circuit
    assert np.array_equal(source.frame, expected_selected)
    assert np.array_equal(source.measurement_flips, np.zeros((2, 2), dtype=np.uint8))
    assert source.samples is None
    assert source._measurement_index == 0

    source.run()

    assert source.frame.shape == (2, 4)
    assert source.measurement_flips.shape == (2, 2)
    assert source.samples.shape == (2, 2)


def test_pauli_frame_bunch_combines_frames_in_input_order():
    f1 = PauliFrame(Circuit(2).m([0, 1]), shots=3, seed=1)
    f2 = PauliFrame(Circuit(1).m([0]), shots=2, seed=2)
    f1.frame = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    f2.frame = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    old_f1 = f1.frame.copy()
    old_f2 = f2.frame.copy()

    circuit = Circuit(3).x_error([2], p=1).m([0, 1, 2])
    combined = PauliFrame.bunch([f1, f2], circuit=circuit, seed=3)

    expected = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    assert combined.circuit is circuit
    assert combined.n == 3
    assert combined.shots == 2
    assert np.array_equal(combined.frame, expected)
    assert np.array_equal(combined.measurement_flips, np.zeros((2, 3), dtype=np.uint8))
    assert combined.samples is None
    assert combined._measurement_index == 0
    assert np.array_equal(f1.frame, old_f1)
    assert np.array_equal(f2.frame, old_f2)

    combined.run()

    expected_after_run = expected.copy()
    expected_after_run[:, 2] ^= 1
    assert np.array_equal(combined.frame[:, :3], expected_after_run[:, :3])
    assert combined.measurement_flips.shape == (2, 3)
    assert combined.samples.shape == (2, 3)


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

    stim_samples = circuit.to_stim_circuit().compile_sampler(seed=11).sample(shots=20000)

    assert np.all(np.abs(ours.mean(axis=0) - stim_samples.mean(axis=0)) < 0.03)


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
