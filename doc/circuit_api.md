# Circuit API

`Circuit` records an ordered list of operations. It does not simulate by itself.
The simulators read the operations later.

```python
from dorsim import Circuit

circuit = (
    Circuit(4)
    .h([0, 2])
    .cx([0, 1, 2, 3])
    .m([0, 1, 2, 3])
)
```

## Operations

Each operation is stored as an `Operation`:

```python
Operation(name: str, targets: tuple, p: float = 0.0)
```

Examples:

```python
Circuit(4).h([0, 2]).operations
# [
#     Operation(name="H", targets=(0,), p=0.0),
#     Operation(name="H", targets=(2,), p=0.0),
# ]

Circuit(4).cx([0, 1, 2, 3]).operations
# [
#     Operation(name="CX", targets=(0, 1), p=0.0),
#     Operation(name="CX", targets=(2, 3), p=0.0),
# ]
```

The API is iterable-only. Use lists, tuples, ranges, or other iterables.

## Single-Qubit Gates

These helpers apply the same gate to each target:

```python
Circuit(4).h([0, 1, 2, 3])
Circuit(4).s([0, 2])
Circuit(4).sdg([1])
Circuit(4).x([0])
Circuit(4).y([1])
Circuit(4).z([2])
```

`h([0, 1, 2])` stores three operations: `H 0`, `H 1`, and `H 2`.

## Two-Qubit Gates

Two-qubit gates use flat target pairs:

```python
Circuit(4).cx([0, 1, 2, 3])
```

This means:

```text
CX 0 1
CX 2 3
```

The circuit stores these as two separate `Operation("CX", ...)` entries.

Available two-qubit gates:

```python
cx(targets)
cy(targets)
cz(targets)
swap(targets)
```

The target list length must be even.

## Measurements

`m(targets)` measures in the Z basis:

```python
Circuit(2).m([0, 1])
```

`mx(targets)` measures in the X basis:

```python
Circuit(2).mx([0, 1])
```

The number of measurement results is:

```python
circuit.num_measurements
```

It counts all targets in `M` and `MX` operations.

## Reset

`r(targets)` resets qubits to the Z-basis `|0>` state:

```python
Circuit(2).x([0]).m([0]).r([0]).m([0])
```

## Pauli Noise

Noise operations are used by `PauliFrame`, not by `TableauSim`.

```python
Circuit(2).x_error([0, 1], p=0.01)
Circuit(2).y_error([0], p=0.01)
Circuit(2).z_error([1], p=0.01)
Circuit(2).depolarize1([0, 1], p=0.001)
Circuit(4).depolarize2([0, 1, 2, 3], p=0.001)
```

`depolarize2` uses flat pairs, like `cx`:

```text
depolarize2([0, 1, 2, 3], p)
```

means:

```text
DEPOLARIZE2(p) 0 1
DEPOLARIZE2(p) 2 3
```

`TableauSim(circuit).run()` uses `circuit.without_noise()`.

## Measurement-Record Feedback

Dorsim supports Stim-style measurement-record feedback with `target_rec`.

```python
from dorsim import Circuit, target_rec

circuit = (
    Circuit(2)
    .h([0])
    .m([0])
    .cx([target_rec(-1), 1])
    .m([1])
)
```

`target_rec(-1)` means the previous measurement result.

```text
CX rec[-1] 1
```

means:

```text
if previous_measurement == 1:
    apply X to qubit 1
```

Supported feedback forms:

```python
cx([target_rec(-k), q])  # conditional X on q
cz([target_rec(-k), q])  # conditional Z on q
```

## Converting To Stim

If Stim is installed, a Dorsim circuit can be converted into a `stim.Circuit`:

```python
stim_circuit = circuit.to_stim_circuit()
print(stim_circuit)
```

Measurement-record targets convert back into Stim targets:

```python
Circuit(2).m([0]).cx([target_rec(-1), 1]).to_stim_circuit()
```

prints:

```text
M 0
CX rec[-1] 1
```
