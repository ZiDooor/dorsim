# dorsim

Python implementation of an inverse stabilizer tableau reference simulator and a Pauli-frame sampler.

The original standalone script is kept in `inverse_tableau_sim.py`. The reusable package lives under `src/dorsim`.

```python
from dorsim import Circuit, TableauSim, PauliFrame

circuit = Circuit(2).h(0).cx(0, 1).m(0).m(1)
reference = TableauSim(circuit).run().reference_measurements
frames = PauliFrame(circuit, shots=16, seed=1).run(reference=reference)

print(reference)
print(frames.samples)
```

Run the package demo:

```bash
python -m dorsim
```
