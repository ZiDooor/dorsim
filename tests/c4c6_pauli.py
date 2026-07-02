import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dorsim import Circuit, PauliFrame, target_rec


shots = 10
err = 0.9

c4 = (
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

frames = PauliFrame(c4, shots=shots).run()
print(f"Pauli frame:\n{frames.frame}")

check = np.sum(frames.measurement_flips[:, 0:4], axis=1) % 2
frames.update(frames.frame[check == 0]) 

print(f"New Pauli frame:\n{frames.frame}")