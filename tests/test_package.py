from __future__ import annotations

from dorsim import Circuit, PauliFrame, TableauSim


def test_reference_and_pauli_frame_shapes():
    circuit = Circuit(2).h(0).cx(0, 1).m(0).m(1)

    tab = TableauSim(circuit).run()
    assert tab.tableau.shape == (4, 4)
    assert tab.sign.shape == (4,)
    assert tab.reference_measurements.shape == (2,)

    frames = PauliFrame(circuit, shots=5, seed=1).run(reference=tab.reference_measurements)
    assert frames.frame.shape == (4, 5)
    assert frames.measurement_flips.shape == (2, 5)
    assert frames.samples.shape == (2, 5)
