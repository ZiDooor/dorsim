from __future__ import annotations

import numpy as np

from .circuit import Circuit
from .pauli import identity_tableau
from .pauli_frame import PauliFrame
from .tableau_sim import TableauSim


def stim_inverse_rows(circuit: Circuit) -> tuple[list[str], list[str]]:
    import stim

    sim = stim.TableauSimulator()
    sim.do_circuit(circuit.without_noise().to_stim_circuit())
    t = sim.current_inverse_tableau()
    return [str(t.x_output(q)) for q in range(circuit.num_qubits)], [
        str(t.z_output(q)) for q in range(circuit.num_qubits)
    ]


def run_self_checks() -> None:
    ident = TableauSim(Circuit(2)).run()
    expected, expected_sign = identity_tableau(2)
    assert np.array_equal(ident.tableau, expected)
    assert np.array_equal(ident.sign, expected_sign)

    assert TableauSim(Circuit(1).h(0).m(0)).run().reference_measurements.tolist() == [0]
    assert TableauSim(Circuit(1).x(0).m(0)).run().reference_measurements.tolist() == [1]

    mid = Circuit(2).h(0).cx(0, 1).m(0).h(1).m(1)
    mid_sim = TableauSim(mid).run()
    assert mid_sim.reference_measurements.tolist() == [0, 0]

    c = Circuit(2).h(0).cx(0, 1)
    our = TableauSim(c).run()
    stim_x, stim_z = stim_inverse_rows(c)
    assert [our.format_row(q) for q in range(c.num_qubits)] == [s.replace("_", "I") for s in stim_x]
    assert [our.format_row(c.num_qubits + q) for q in range(c.num_qubits)] == [
        s.replace("_", "I") for s in stim_z
    ]

    pf_circuit = Circuit(1).h(0).m(0)
    ref = TableauSim(pf_circuit).run().reference_measurements
    pf = PauliFrame(pf_circuit, shots=5000, seed=5).run(reference=ref)
    mean = float(pf.samples[0].mean())
    assert 0.45 < mean < 0.55

    noisy = Circuit(1).x_error(0, 0.25).m(0)
    noisy_ref = TableauSim(noisy).run().reference_measurements
    ours = PauliFrame(noisy, shots=20000, seed=9).run(reference=noisy_ref).samples[0]
    stim_samples = noisy.to_stim_circuit().compile_sampler(seed=9).sample(shots=20000)[:, 0]
    assert abs(float(ours.mean()) - float(stim_samples.mean())) < 0.03


def demo() -> None:
    circuit = (
        Circuit(3)
        .h(0)
        .cx(0, 1)
        .m(0)
        .h(1)
        .depolarize1(1, 0.05)
        .m(1)
        .r(2)
        .h(2)
        .cz(1, 2)
        .z_error(2, 0.1)
        .m(2)
    )
    tab = TableauSim(circuit).run()
    frames = PauliFrame(circuit, shots=8, seed=7).run(reference=tab.reference_measurements)

    print("Reference inverse tableau:")
    print(tab.format_tableau())
    print("reference_measurements:", tab.reference_measurements)
    print("measurement_flips:")
    print(frames.measurement_flips)
    print("samples:")
    print(frames.samples)


def main() -> None:
    run_self_checks()
    demo()
