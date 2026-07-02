import sys
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dorsim import Circuit, PauliFrame, TableauSim, target_rec

shots = 2
err = 0.01
### test ler of level-1 C4/C6 code
## stim version
stim_circuit = stim.Circuit()
stim_circuit.append("H", [0, 2, 4, 6])
stim_circuit.append("CX", [0, 1, 2, 3, 4, 5, 6, 7])
stim_circuit.append("DEPOLARIZE2", [0, 1, 2, 3, 4, 5, 6, 7], err)
stim_circuit.append("CX", [1, 2, 3, 4, 5, 6])
stim_circuit.append("DEPOLARIZE2", [1, 2, 3, 4, 5, 6], err)
stim_circuit.append("CX", [7, 0])
stim_circuit.append("DEPOLARIZE2", [7, 0], err)
stim_circuit.append("M", [0, 2, 4, 6])
stim_circuit.append(
    "CX",
    [
        stim.target_rec(-4), 1,
        stim.target_rec(-4), 3,
        stim.target_rec(-4), 5,
        stim.target_rec(-3), 3,
        stim.target_rec(-3), 5,
        stim.target_rec(-2), 5,
    ],
)
stim_circuit.append("M", [1, 3, 5, 7])
stim_circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-3), stim.target_rec(-4)])

sampler = stim_circuit.compile_detector_sampler()
re = sampler.sample(shots=shots)
print(f"stim ler: {re.sum()/shots}")


## dorsim version
dorsim_circuit = (
    Circuit(8)
    .h([0, 2, 4, 6])
    .cx([0, 1, 2, 3, 4, 5, 6, 7])
    .depolarize2([0, 1, 2, 3, 4, 5, 6, 7], err)
    .cx([1, 2, 3, 4, 5, 6])
    .depolarize2([1, 2, 3, 4, 5, 6], err)
    .cx([7, 0])
    .depolarize2([7, 0], err)
    .m([0, 2, 4, 6])
    .cx([
        target_rec(-4), 1,
        target_rec(-4), 3,
        target_rec(-4), 5,
        target_rec(-3), 3,
        target_rec(-3), 5,
        target_rec(-2), 5,
    ])
    .m([1, 3, 5, 7])
)

# reference = TableauSim(dorsim_circuit).run().reference_measurements
# frames = PauliFrame(dorsim_circuit, shots=shots).run(reference=reference)
frames = PauliFrame(dorsim_circuit, shots=shots).run()
# print(f"dorsim pauliframe samples:\n{frames.frame}")
dorsim_detectors = np.bitwise_xor.reduce(frames.samples[-4:], axis=0)
print(f"dorsim ler: {dorsim_detectors.sum()/shots}")
