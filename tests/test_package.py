from __future__ import annotations

from dorsim import Circuit, Operation, PauliFrame, TableauSim
from dorsim.pauli import bits_from_code, code_from_bits, local_conjugation_map


def test_flat_stim_style_operation_storage():
    circuit = Circuit(4).cx([0, 1, 2, 3])
    assert circuit.operations == [Operation("CX", (0, 1, 2, 3), 0.0)]

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
    assert frames.frame.shape == (4, 5)
    assert frames.measurement_flips.shape == (2, 5)
    assert frames.samples.shape == (2, 5)


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
                frame.frame[q, 0] = x
                frame.frame[frame.n + q, 0] = z

            frame._conjugate_frame_by_gate(Operation(gate, tuple(range(arity))))

            got = tuple(
                code_from_bits(frame.frame[q, 0], frame.frame[frame.n + q, 0])
                for q in range(arity)
            )
            assert got == expected_out
