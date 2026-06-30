# PauliFrame

`PauliFrame` performs many-shot simulation by tracking Pauli corrections relative
to the reference trajectory produced by `TableauSim`.

```python
from dorsim import Circuit, TableauSim, PauliFrame

circuit = Circuit(1).h([0]).m([0])
reference = TableauSim(circuit).run().reference_measurements
frames = PauliFrame(circuit, shots=8, seed=1).run(reference=reference)

print(frames.measurement_flips)
print(frames.samples)
```

## Stored Variables

For `n` qubits and `shots` shots:

```text
frame shape = (2n, shots)
```

Rows are binary Pauli-frame bits:

```text
row 0      ... n-1   = X bits
row n      ... 2n-1  = Z bits
```

Columns are shots.

`measurement_flips` has shape:

```text
(number_of_measurements, shots)
```

Each entry says whether that shot's measurement differs from the reference
measurement.

If a reference is passed into `run(reference=...)`, Dorsim also stores:

```python
samples = reference[:, None] ^ measurement_flips
```

So `samples` has shape:

```text
(number_of_measurements, shots)
```

## Initialization

The frame starts with random `I` or `Z` on each qubit in each shot:

```text
X rows = 0
Z rows = random 0/1
```

This is how random measurement branches are sampled around the fixed reference
branch.

## Clifford Updates

Global phase is ignored in the Pauli frame. Gates update the binary rows
directly.

Important examples:

```text
H(q):       swap Xq and Zq rows
S(q):       Zq ^= Xq
S_DAG(q):   Zq ^= Xq
CX(a, b):   Xb ^= Xa, Za ^= Zb
CZ(a, b):   Za ^= Xb, Zb ^= Xa
SWAP(a, b): swap Xa/Xb and Za/Zb
```

`X`, `Y`, and `Z` gates do not change the Pauli frame because phase is ignored.

## Measurements

For Z-basis measurement:

```python
M q
```

the flip bit is the current X component on qubit `q`:

```python
measurement_flips[i] = frame[q]
```

Then Dorsim randomizes the Z component on that qubit.

For X-basis measurement:

```python
MX q
```

the flip bit is the current Z component on qubit `q`:

```python
measurement_flips[i] = frame[n + q]
```

Then Dorsim randomizes the X component on that qubit.

## Noise

Pauli noise multiplies sampled Pauli terms into the frame:

```text
X_ERROR(q): toggles Xq
Z_ERROR(q): toggles Zq
Y_ERROR(q): toggles Xq and Zq
DEPOLARIZE1(q): randomly toggles X, Y, or Z
DEPOLARIZE2(a, b): randomly toggles a non-identity two-qubit Pauli
```

For `DEPOLARIZE2(p) a b`, the sampled two-qubit Pauli is:

```text
1-p:  II
p/15: IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ
```

Example:

```python
circuit = Circuit(2).x_error([0, 1], p=0.25).m([0, 1])
reference = TableauSim(circuit).run().reference_measurements
frames = PauliFrame(circuit, shots=1000, seed=5).run(reference=reference)

print(frames.samples.mean(axis=1))
```

Two-qubit depolarizing noise uses flat pairs:

```python
circuit = Circuit(2).depolarize2([0, 1], p=0.3).m([0, 1])
reference = TableauSim(circuit).run().reference_measurements
frames = PauliFrame(circuit, shots=1000, seed=5).run(reference=reference)

print(frames.samples.mean(axis=1))
```

## Reset

`R q` clears the frame on qubit `q`, then randomizes the Z component:

```text
Xq = 0
Zq = random 0/1
```

This matches reset to the Z-basis `|0>` state while keeping future X-basis
measurements random when appropriate.

## Feedback

For `CX target_rec(-k), q`, the Pauli-frame simulator uses the previous
measurement flip vector:

```python
flips = measurement_flips[current_measurement_index - k]
```

Then it multiplies `X(q)` into exactly the shots where `flips` is `1`.

For `CZ target_rec(-k), q`, it multiplies `Z(q)` into exactly those shots.

Example:

```python
from dorsim import Circuit, TableauSim, PauliFrame, target_rec

circuit = Circuit(2).h([0]).m([0]).cx([target_rec(-1), 1]).m([1])
reference = TableauSim(circuit).run().reference_measurements
frames = PauliFrame(circuit, shots=16, seed=3).run(reference=reference)

print((frames.samples[0] == frames.samples[1]).all())
# True
```

The second measurement copies the first because the feedback applies `X` to
qubit `1` when the first measurement is `1`.
