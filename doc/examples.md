# Examples

These examples assume the package has not been installed. From the repository
root, add `src` to `sys.path`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
```

Then import the public API:

```python
from dorsim import Circuit, TableauSim, PauliFrame, target_rec
```

## Bell Circuit

```python
circuit = Circuit(2).h([0]).cx([0, 1]).m([0, 1])

tab = TableauSim(circuit).run()
frames = PauliFrame(circuit, shots=8, seed=1).run(
    reference=tab.reference_measurements
)

print("reference:", tab.reference_measurements)
print("flips shape:", frames.measurement_flips.shape)
print("samples shape:", frames.samples.shape)
print("equal results:", (frames.samples[0] == frames.samples[1]).all())
```

Expected shapes:

```text
flips shape: (2, 8)
samples shape: (2, 8)
equal results: True
```

## Z And X Measurements

```python
z_circuit = Circuit(1).h([0]).m([0])
x_circuit = Circuit(1).h([0]).mx([0])

z_ref = TableauSim(z_circuit).run().reference_measurements
x_ref = TableauSim(x_circuit).run().reference_measurements

print("Z-basis reference:", z_ref)
print("X-basis reference:", x_ref)
```

`H; M` is random, but the reference branch is fixed to `0`. `H; MX` is
deterministic `0`.

## Noisy Sampling

```python
circuit = Circuit(2).x_error([0, 1], p=0.25).m([0, 1])

reference = TableauSim(circuit).run().reference_measurements
frames = PauliFrame(circuit, shots=2000, seed=9).run(reference=reference)

print("sample means:", frames.samples.mean(axis=1))
```

The two sample means should be close to `0.25`.

Two-qubit depolarizing noise uses flat pairs:

```python
circuit = Circuit(2).depolarize2([0, 1], p=0.3).m([0, 1])

reference = TableauSim(circuit).run().reference_measurements
frames = PauliFrame(circuit, shots=2000, seed=9).run(reference=reference)

print("sample means:", frames.samples.mean(axis=1))
print(circuit.to_stim_circuit())
```

This emits Stim-style pair noise:

```text
DEPOLARIZE2(0.3) 0 1
M 0 1
```

## Reset

```python
circuit = Circuit(1).x([0]).m([0]).r([0]).m([0])
tab = TableauSim(circuit).run()

print(tab.reference_measurements)
```

The first measurement is `1`. After reset, the second measurement is `0`:

```text
[1 0]
```

## Measurement-Record Feedback

```python
circuit = (
    Circuit(2)
    .h([0])
    .m([0])
    .cx([target_rec(-1), 1])
    .m([1])
)

reference = TableauSim(circuit).run().reference_measurements
frames = PauliFrame(circuit, shots=16, seed=3).run(reference=reference)

print("reference:", reference)
print("samples:")
print(frames.samples)
print("copied:", (frames.samples[0] == frames.samples[1]).all())
```

`CX target_rec(-1), 1` conditionally applies `X` to qubit `1` using the previous
measurement result, so measurement `1` copies measurement `0`.

## Compare Dorsim And Stim Circuit Construction

```python
import stim

stim_circuit = stim.Circuit()
stim_circuit.append("H", [0, 2, 4, 6])
stim_circuit.append("CX", [0, 1, 2, 3, 4, 5, 6, 7])
stim_circuit.append("M", list(range(8)))

dorsim_circuit = (
    Circuit(8)
    .h([0, 2, 4, 6])
    .cx([0, 1, 2, 3, 4, 5, 6, 7])
    .m(list(range(8)))
)

assert str(dorsim_circuit.to_stim_circuit()) == str(stim_circuit)
print(dorsim_circuit.to_stim_circuit())
```

You can also use feedback targets:

```python
dorsim_feedback = Circuit(2).m([0]).cx([target_rec(-1), 1])
print(dorsim_feedback.to_stim_circuit())
```

Output:

```text
M 0
CX rec[-1] 1
```
