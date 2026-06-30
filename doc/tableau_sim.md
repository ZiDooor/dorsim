# TableauSim

`TableauSim` produces one reference trajectory for a circuit. It tracks an
inverse stabilizer tableau, following the same core idea as Stim:

```text
T_U(P) = U^-1 P U
```

Here `U` is the current physical circuit and `P` is an end-of-time Pauli
generator.

```python
from dorsim import Circuit, TableauSim

circuit = Circuit(2).h([0]).cx([0, 1]).m([0, 1])
tab = TableauSim(circuit).run()

print(tab.tableau)
print(tab.sign)
print(tab.reference_measurements)
```

## Stored Variables

For `n` qubits:

```text
tableau shape = (2n, 2n)
sign shape    = (2n,)
```

Rows are tracked Pauli generators:

```text
row 0      ... n-1   = T(X0), T(X1), ...
row n      ... 2n-1  = T(Z0), T(Z1), ...
```

Columns are binary X/Z components:

```text
column 0      ... n-1   = X bits
column n      ... 2n-1  = Z bits
```

For one qubit:

```text
[1, 0] = X
[0, 1] = Z
[1, 1] = Y
[0, 0] = I
```

`sign[row]` stores whether the Pauli row is positive or negative:

```text
0 = +
1 = -
```

`reference_measurements` has shape `(m,)`, where `m` is the number of
measurement targets in the circuit.

## Running

```python
circuit = Circuit(1).x([0]).m([0])
tab = TableauSim(circuit).run()

print(tab.reference_measurements)
# [1]
```

This result is deterministic because `X 0` flips `|0>` to `|1>`, then `M 0`
measures in the Z basis.

## Deterministic And Random Measurements

For `M q`, `TableauSim` inspects row `T(Zq)`.

For `MX q`, it inspects row `T(Xq)`.

If the inspected row has no X component, the result is deterministic and comes
from the row sign.

If the inspected row has X or Y support, the measurement is random. Dorsim fixes
the reference random branch to outcome `0`, then updates the tableau so later
operations see a consistent reference trajectory.

Example:

```python
circuit = Circuit(1).h([0]).m([0])
tab = TableauSim(circuit).run()

print(tab.reference_measurements)
# [0]
```

The physical measurement is random, but the reference trajectory chooses the
zero branch.

## Formatting The Tableau

Use `format_tableau()` to print rows as Pauli strings:

```python
circuit = Circuit(2).h([0]).cx([0, 1])
tab = TableauSim(circuit).run()

print(tab.format_tableau())
```

The output labels rows as `T(Xq)` and `T(Zq)`.

## Reset

`R q` is simulated as a hidden Z-basis measurement followed by forcing the
reference Z result to zero:

```python
circuit = Circuit(1).x([0]).m([0]).r([0]).m([0])
tab = TableauSim(circuit).run()

print(tab.reference_measurements)
# [1 0]
```

The reset measurement is not reported.

## Feedback

For `CX target_rec(-k), q`, the reference simulator reads:

```python
reference_measurements[current_measurement_index - k]
```

If that bit is `1`, it folds an ordinary `X(q)` into the tableau.

For `CZ target_rec(-k), q`, it folds `Z(q)` when the referenced bit is `1`.

Example:

```python
from dorsim import Circuit, TableauSim, target_rec

circuit = Circuit(2).x([0]).m([0]).cx([target_rec(-1), 1]).m([1])
tab = TableauSim(circuit).run()

print(tab.reference_measurements)
# [1 1]
```

The first measurement is `1`, so the feedback applies `X` to qubit `1`.

## Noise

`TableauSim` is the reference simulator, so it ignores Pauli noise by running:

```python
circuit.without_noise()
```

Use `PauliFrame` to include noise in many-shot samples.

For lower-level gate update math, see [Update Rules](update_rules.md).
