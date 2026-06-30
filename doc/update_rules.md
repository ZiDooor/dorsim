In this package, the inverse tableau means:

```
T_U(P)=U^{-1} P U
```


where `U` is the current physical circuit.

The matrix is stored as:

```text
rows    = horizontal arrays = tracked Pauli generators
columns = vertical arrays   = X/Z bits on qubits
```

For `n` qubits:

```text
tableau shape = (2n, 2n)

rows:
  0      ... n-1      = T(X0), T(X1), ...
  n      ... 2n-1     = T(Z0), T(Z1), ...

columns:
  0      ... n-1      = X bits on qubits
  n      ... 2n-1     = Z bits on qubits
```

So for one qubit, row `[1, 0]` means `X`, row `[0, 1]` means `Z`, row `[1, 1]` means `Y`. The separate `sign[row]` stores whether the row is positive or negative.

Example:

```text
tableau =
row 0: [1, 0]   means T(X0) = +X0
row 1: [0, 1]   means T(Z0) = +Z0

sign = [0, 0]
```

---

**Appending A Gate After The Current Circuit**

Suppose current circuit is `U`.

If you append a new physical gate `G` after it:

```text
old circuit:  U
new circuit:  U then G
unitary:      G U
```

Then the new inverse tableau is:

```
T_{GU}(P)=(GU)^{-1}P(GU)=U^{-1}(G^{-1}PG)U
```

So:

```
T_\text{new}(P)=T_\text{old}(G^{-1}PG)
```

This means: first conjugate the end Pauli `P` by the new gate `G`, then express the result using the old tableau.

That is why appending gates usually updates **rows**.

For example, appending `H(q)`:

```text
H^-1 X H = Z
H^-1 Z H = X
```

So:

```text
new T(Xq) = old T(Zq)
new T(Zq) = old T(Xq)
```

Therefore we swap the two horizontal rows:

```python
tableau[[q, n + q]] = old_tableau[[n + q, q]]
sign[[q, n + q]] = old_sign[[n + q, q]]
```

Appending `CX(a, b)`:

```text
CX^-1 Xa CX = Xa Xb
CX^-1 Zb CX = Za Zb
```

So:

```text
new T(Xa) = old T(Xa Xb) = old T(Xa) * old T(Xb)
new T(Zb) = old T(Za Zb) = old T(Za) * old T(Zb)
```

This is why your package uses row multiplication with phase/sign tracking.

---

**Prepending A Gate Before The Current Circuit**

Now suppose current circuit is `U`.

If you insert/prepend a gate `G` before it:

```text
old circuit:  U
new circuit:  G then U
unitary:      U G
```

Then:


```T_{UG}(P)=(UG)^{-1}P(UG)=G^{-1}(U^{-1}PU)G```


So:

```T_\text{new}(P)=G^{-1}T_\text{old}(P)G```

This is different.

Now you already have each row `T_old(P)`. You conjugate every existing row by `G`.

So prepending usually updates **columns/local bits inside every row**, not just replacing generator rows.

For example, prepending `H(q)`:

```text
H^-1 X H = Z
H^-1 Z H = X
H^-1 Y H = -Y
```

For every row:

```python
x_q, z_q = z_q, x_q
sign ^= x_q & z_q   # because Y changes sign
```

So append `H` swaps two horizontal rows.  
Prepend `H` swaps two vertical columns inside every row.

That is the key difference.

---

**Concrete Example**

Start with one qubit identity tableau:

```text
T(X0) = +X0   row [1, 0], sign 0
T(Z0) = +Z0   row [0, 1], sign 0
```

Now append `H` after the empty circuit.

```text
circuit: H
```

Because appending `H` swaps rows:

```text
T(X0) = +Z0   row [0, 1], sign 0
T(Z0) = +X0   row [1, 0], sign 0
```

Now compare two different updates.

Case 1: append `S` after `H`

```text
circuit: H then S
unitary: S H
```

Appending uses:

```T_\text{new}(P)=T_\text{old}(S^{-1} P S)```

Since:

```text
S^-1 X S = -Y
S^-1 Z S = Z
```

we get:

```text
new T(X0) = old T(-Y0)
new T(Z0) = old T(Z0)
```

For the old `H` tableau:

```text
old T(X0) = Z0
old T(Z0) = X0
old T(Y0) = -Y0
```

Therefore:

```text
new T(X0) = +Y0
new T(Z0) = +X0
```

Final tableau:

```text
T(X0) = +Y0   row [1, 1], sign 0
T(Z0) = +X0   row [1, 0], sign 0
```

Case 2: prepend `S` before `H`

```text
circuit: S then H
unitary: H S
```

Prepending uses:

```T_\text{new}(P)=S^{-1}T_\text{old}(P)S```

Old `H` tableau was:

```text
T(X0) = +Z0
T(Z0) = +X0
```

Conjugate each existing row by `S`:

```text
S^-1 Z S = Z
S^-1 X S = -Y
```

So:

```text
new T(X0) = +Z0
new T(Z0) = -Y0
```

Final tableau:

```text
T(X0) = +Z0   row [0, 1], sign 0
T(Z0) = -Y0   row [1, 1], sign 1
```

So appending `S` after `H` and prepending `S` before `H` are not the same update.

---

In short:

```text
Append after current circuit:
  T_new(P) = T_old(G^-1 P G)
  update by replacing/multiplying horizontal rows

Prepend before current circuit:
  T_new(P) = G^-1 T_old(P) G
  update by conjugating every row, usually changing vertical columns
```

That is the main mental model. Rows are the tracked start-of-time Pauli images; columns are the binary X/Z components inside those Pauli strings.