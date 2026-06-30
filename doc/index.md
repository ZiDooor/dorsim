# Dorsim Documentation

Dorsim is a small educational simulator for Clifford circuits with measurements,
resets, Pauli noise, and Stim-style measurement-record feedback.

The package has two simulator layers:

1. `TableauSim` builds one reference trajectory using an inverse stabilizer
   tableau.
2. `PauliFrame` runs many shots by tracking Pauli-frame differences from that
   reference trajectory.

The usual workflow is:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

from dorsim import Circuit, TableauSim, PauliFrame

circuit = Circuit(2).h([0]).cx([0, 1]).m([0, 1])

tab = TableauSim(circuit).run()
frames = PauliFrame(circuit, shots=16, seed=1).run(
    reference=tab.reference_measurements
)

print(tab.reference_measurements)
print(frames.measurement_flips.shape)
print(frames.samples.shape)
```

`TableauSim` ignores Pauli noise and produces `reference_measurements`.
`PauliFrame` includes Pauli noise and produces shot-by-shot flips. When a
reference is provided, it also produces samples:

```text
samples = reference[:, None] ^ measurement_flips
```

## Documents

- [Circuit API](circuit_api.md): how to build circuits.
- [TableauSim](tableau_sim.md): how the inverse tableau reference simulator works.
- [PauliFrame](pauli_frame.md): how many-shot Pauli-frame simulation works.
- [Examples](examples.md): copy-paste simulation examples.
- [Update Rules](update_rules.md): math details for appending and prepending gates.

## Public API

The main package imports are:

```python
from dorsim import Circuit, Operation, TableauSim, PauliFrame, target_rec
```

`target_rec(-k)` is used for Stim-style feedback such as:

```python
Circuit(2).m([0]).cx([target_rec(-1), 1])
```

This means: if the previous measurement result is `1`, apply `X` to qubit `1`.
