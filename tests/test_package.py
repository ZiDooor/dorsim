from __future__ import annotations

from dorsim import Circuit, Operation, PauliFrame, TableauSim


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
