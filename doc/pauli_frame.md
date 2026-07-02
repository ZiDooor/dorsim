# PauliFrame

`PauliFrame` performs many-shot simulation by tracking Pauli corrections and
measurement outcome shifts. It can be used directly when you only care about
which measurement results are flipped.

```python
from dorsim import Circuit, PauliFrame

circuit = Circuit(1).h([0]).m([0])
frames = PauliFrame(circuit, shots=8, seed=1).run()

print(frames.measurement_flips)
print(frames.samples)
```

## Stored Variables

For `n` qubits and `shots` shots:

```text
frame shape = (shots, 2n)
```

Rows are shots. Columns are binary Pauli-frame bits:

```text
column 0      ... n-1   = X bits
column n      ... 2n-1  = Z bits
```

So `frame[a][b]` means the `b`-th Pauli-frame bit in the `a`-th shot.

`measurement_flips` has shape:

```text
(shots, number_of_measurements)
```

Each entry is a measurement outcome shift. It says whether the Pauli frame flips
that measurement result. `measurement_flips[a][b]` means measurement `b`'s
shift in shot `a`.

When `run()` is called without a reference, Dorsim stores:

```python
samples = measurement_flips.copy()
```

So shift-only simulation does not need `TableauSim`.

When a reference is passed into `run(reference=...)`, Dorsim stores full samples:

```python
samples = measurement_flips ^ reference[None, :]
```

So `samples` has shape:

```text
(shots, number_of_measurements)
```

`samples[a][b]` means measurement `b`'s sample in shot `a`.

## Updating State

`update()` resets measurement-output bookkeeping for manual or mid-simulation
use:

```python
pf.update()
```

This resets:

```text
measurement_flips
samples
_measurement_index
```

and keeps the current `frame` unchanged.

You can also replace the frame:

```python
pf.update(custom_frame)
```

The custom frame must have shape:

```text
(shots, 2n)
```

## Initialization

The frame starts with random `I` or `Z` on each qubit in each shot:

```text
X columns = 0
Z columns = random 0/1
```

This is how measurement shifts are sampled around the Pauli frame.

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
measurement_flips[:, i] = frame[:, q]
```

Then Dorsim randomizes the Z component on that qubit.

For X-basis measurement:

```python
MX q
```

the flip bit is the current Z component on qubit `q`:

```python
measurement_flips[:, i] = frame[:, n + q]
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
frames = PauliFrame(circuit, shots=1000, seed=5).run()

print(frames.samples.mean(axis=0))
```

Two-qubit depolarizing noise uses flat pairs:

```python
circuit = Circuit(2).depolarize2([0, 1], p=0.3).m([0, 1])
frames = PauliFrame(circuit, shots=1000, seed=5).run()

print(frames.samples.mean(axis=0))
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
measurement shift vector:

```python
shift = measurement_flips[:, current_measurement_index - k]
```

Then it multiplies `X(q)` into exactly the shots where `shift` is `1`.

For `CZ target_rec(-k), q`, it multiplies `Z(q)` into exactly those shots.

This works without reference simulation when you only care about shift
information. A reference is needed only when you want full physical measurement
outcomes.

Example:

```python
from dorsim import Circuit, PauliFrame, target_rec

circuit = Circuit(2).h([0]).m([0]).cx([target_rec(-1), 1]).m([1])
frames = PauliFrame(circuit, shots=16, seed=3).run()

print((frames.samples[:, 0] == frames.samples[:, 1]).all())
# True
```

The second shift copies the first shift because feedback propagates the previous
measurement shift into qubit `1`.

To convert shifts into full samples, pass a reference:

```python
from dorsim import TableauSim

reference = TableauSim(circuit).run().reference_measurements
frames = PauliFrame(circuit, shots=16, seed=3).run(reference=reference)
```
