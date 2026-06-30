# dorsim

Python implementation of an inverse stabilizer tableau reference simulator and a Pauli-frame sampler.

The original standalone script is kept in `inverse_tableau_sim.py`. The reusable package lives under `src/dorsim`.

```python
from dorsim import Circuit, TableauSim, PauliFrame

circuit = Circuit(2).h([0]).cx([0, 1]).m([0, 1])
frames = PauliFrame(circuit, shots=16, seed=1).run()

print(frames.measurement_flips)
print(frames.samples)
```

`PauliFrame.run()` gives measurement shifts directly. To reconstruct full
samples relative to a reference trajectory, first run `TableauSim` and pass its
`reference_measurements` into `PauliFrame.run(reference=...)`.

Run the package demo:

```bash
python -m dorsim
```

More user and simulator documentation is in [`doc/index.md`](doc/index.md).
