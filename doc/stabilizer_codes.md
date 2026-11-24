# Stabilizer Codes

`stab_code.py` describes codes and their concatenation trees. These objects
contain Pauli operators, not encoding or preparation circuits.

## Binary Pauli Representation

An operator on `n` qubits is a binary row of length `2*n`, ordered
`[x0, ..., x(n-1), z0, ..., z(n-1)]`. On each qubit, `(x, z)` means
`I=(0,0)`, `X=(1,0)`, `Z=(0,1)`, or `Y=(1,1)`. Global phase is omitted.
Use binary NumPy arrays, preferably `dtype=np.uint8`.

For a valid `[[n,k]]` code:

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `stabilizers` | `(n-k, 2*n)` | Independent commuting stabilizer generators |
| `logical_x` | `(k, 2*n)` | Logical X generators |
| `logical_z` | `(k, 2*n)` | Corresponding logical Z generators |
| `pure_errors` | `(n-k, 2*n)` | Representatives of the unit syndromes |
| `children` | tuple | Inner codes; empty for a leaf |
| `parent` | code or `None` | Outer code for a concatenated node |

`StabilizerCode(name, n, k, stabilizers, logical_x, logical_z)` accepts
general stabilizer codes. The caller must supply valid generators and logical
operators; construction does not validate their commutation or independence.
The chosen stabilizer and logical bases determine syndrome and logical-index
ordering in the decoders.

## Built-In Codes

```python
from dorsim import CSSCode, StabilizerCode, get_c4c6_code

steane = StabilizerCode.steane()  # [[7,1,3]], a StabilizerCode object
c4 = CSSCode.c4()                # [[4,2,2]]
c6 = CSSCode.c6()                # [[6,2,2]]
qp = CSSCode.qp()                # Unencoded pair: n=2, k=2, no stabilizers
level3 = get_c4c6_code(3)         # n=36, k=2

assert (level3.n, level3.k) == (36, 2)
assert c4.get_lx().shape == (2, 4)
assert c4.get_lz().shape == (2, 4)
```

`CSSCode` subclasses `StabilizerCode`. Its `get_lx()` returns the X half
of `logical_x`; `get_lz()` returns the Z half of `logical_z`.
`StabilizerCode.steane()` uses CSS operators but returns the general class.

For positive integer `level`, `get_c4c6_code(level)` starts with C4 at level 1,
then places a C6 outer code over three copies of the previous level. Thus
`n = 4 * 3**(level-1)` and `k = 2`. Supply `level >= 1`; the function does
not validate that range.

## Concatenation

`concat_code(parent, children)` replaces the outer code's physical positions
with the children's logical qubits, in child order and then logical-qubit order.
It requires `sum(child.k for child in children) == parent.n`.

```python
from dorsim import CSSCode, StabilizerCode, concat_code

c4 = CSSCode.c4()
c4c6 = concat_code(CSSCode.c6(), [c4, c4, c4])
assert (c4c6.n, c4c6.k) == (12, 2)

steane = StabilizerCode.steane()
steane2 = concat_code(steane, [steane] * 7)
assert (steane2.n, steane2.k) == (49, 1)
```

The result has `n = sum(child.n)` and `k = parent.k`. It remains a
`CSSCode` only when the parent and every child are `CSSCode` objects.
Children may have different sizes as long as their logical qubits fill the
parent's positions.

Stabilizer rows are ordered as all rows of child 0, then child 1, and so on,
followed by the lifted parent stabilizers. Recursive decoders rely on this
order. Pass syndromes in exactly the order of `code.stabilizers`.

The module-level helper `embed_bsr(row, offset, total_n)` embeds a local
`[X|Z]` row into a larger register, placing both halves at the same physical
offset. It is available from `dorsim.stab_code`, not the package root.

## Syndromes And Pure Errors

Two Pauli rows anticommute when their symplectic product is one. For a batch
of errors `E`, compute syndromes as follows:

```python
import numpy as np
from dorsim import CSSCode

code = CSSCode.c4()
errors = np.zeros((2, 2 * code.n), dtype=np.uint8)
errors[1, 0] = 1  # X on physical qubit 0
check = np.concatenate(
    [code.stabilizers[:, code.n:], code.stabilizers[:, :code.n]], axis=1
)
syndrome = (errors @ check.T) % 2
assert syndrome.shape == (2, code.n - code.k)

representatives = (syndrome @ code.pure_errors) % 2
assert np.array_equal((representatives @ check.T) % 2, syndrome)
```

`pure_errors` is computed lazily by binary Gaussian elimination and cached.
Each row has a unit syndrome and commutes with the supplied logical generators.
The rows need not be minimum-weight errors. Avoid changing a code's matrices
after its pure errors or decoder tables have been cached; create a new code
and decoder instead.

See [Decoders](decoders.md) for recovery and logical probabilities.
