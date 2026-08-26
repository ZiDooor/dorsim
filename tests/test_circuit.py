import pytest

from dorsim import Circuit, Operation, TableauSim, target_rec


def test_combine_concatenates_without_mutating_inputs():
    first = Circuit(1).h([0]).m([0])
    second = Circuit(3).x_error([2], p=0.25).cx([target_rec(-1), 1]).m([1])

    combined = first.combine(second)

    assert type(combined) is Circuit
    assert combined.num_qubits == 3
    assert combined.operations == [
        Operation("H", (0,), 0.0),
        Operation("M", (0,), 0.0),
        Operation("X_ERROR", (2,), 0.25),
        Operation("CX", (target_rec(-1), 1), 0.0),
        Operation("M", (1,), 0.0),
    ]
    assert first.operations == [Operation("H", (0,), 0.0), Operation("M", (0,), 0.0)]
    assert second.operations == [
        Operation("X_ERROR", (2,), 0.25),
        Operation("CX", (target_rec(-1), 1), 0.0),
        Operation("M", (1,), 0.0),
    ]
    assert TableauSim(combined).run().reference_measurements.tolist() == [0, 0]

    first.x([0])
    second.z([1])
    assert len(combined.operations) == 5


def test_combine_handles_empty_circuits():
    circuit = Circuit(2).x([1])

    assert Circuit(1).combine(circuit).operations == circuit.operations
    assert circuit.combine(Circuit(4)).operations == circuit.operations
    assert circuit.combine(Circuit(4)).num_qubits == 4


def test_combine_requires_another_circuit():
    with pytest.raises(TypeError, match="other must be a Circuit"):
        Circuit(1).combine(object())
