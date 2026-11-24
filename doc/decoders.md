# Decoders

`decoder.py` provides four decoders. They operate on batches of binary arrays;
they do not run circuits or apply corrections to a `PauliFrame` automatically.
See [Stabilizer Codes](stabilizer_codes.md) for code construction and Pauli
conventions.

| Decoder | Input | Result | Noise model |
| --- | --- | --- | --- |
| `KnillDecoder` | Single-basis measurement flips | Two logical bits, possibly erased | Hard decisions; no rate parameter |
| `PoulinDecoder` | Single-basis flips or syndrome | Single-basis physical recovery and logical probabilities | Independent identical bit errors |
| `CombinedPoulinDecoder` | Full stabilizer syndrome | Full Pauli recovery and logical probabilities | Independent identical Pauli errors |
| `JointPoulinDecoder` | Measurement and corrected-output syndromes, first recovery's logical index | Output Pauli recovery and logical probabilities | Independent identical joint Pauli pairs |

Rows always represent shots. Binary arithmetic is modulo two; `^` is XOR
for binary integer arrays. A full Pauli recovery has columns `[X bits | Z bits]`.
Applying a recovery means XORing it into a frame. A matching syndrome removes
the syndrome but does not guarantee logical success: the residual can still be
a nontrivial logical Pauli.

## KnillDecoder

This decoder implements the C4/C6 measurement-decoding rules, with
`ERASURE = -1`. It returns logical bits, rather than a physical recovery.

```python
import numpy as np
from dorsim import KnillDecoder, get_c4c6_code

code = get_c4c6_code(2)
measurements = np.zeros((4, code.n), dtype=np.uint8)
decoder = KnillDecoder(seed=1)
logical = decoder.decode_code(measurements, code)
assert logical.shape == (4, 2)

# At the final level, replace unresolved erasures with random binary bits.
final_logical = decoder.decode_code(measurements, code, final=True)
assert np.all((final_logical == 0) | (final_logical == 1))
```

- All decoding methods accept keyword-only `basis="Z"` or `basis="X"`.
  Checks and output parities use stabilizers and logical operators in that
  measurement basis. The default preserves the original Z-basis results.
- `decode_qp(m, code=None)` accepts `(shots, 2)` and computes logical parities
  of the supplied pair code; the standard pair passes through its two bits.
- `decode_c4(m, code=None)` accepts `(shots, 4)`. Failed stabilizer checks
  erase both logical bits. With the standard code and default basis, odd
  total parity erases the pair; even parity gives `(m0 XOR m2, m2 XOR m3)`.
- `decode_c6_children(m, code=None)` accepts `(shots, 3, 2)`, with two logical
  bits per child. Either bit being `-1` erases the whole child. It reconstructs
  one erased child by trying its four possible assignments against the supplied
  checks. Recovery must be unique. Multiple erased children or inconsistent
  parity produce an erased output pair. Optional codes default to the standard
  Qp, C4, and C6 definitions, respectively.
- `decode_code(m, code, final=False)` recursively supports Qp/C4 leaves and
  three-child C6 parents. A bare C6 leaf treats its six measurements as three
  unencoded pairs. It is not a general stabilizer decoder.
- `decode_c4c6(m, level, *, basis="Z")` builds the standard code and uses
  `final=True`. Use `decode_code` for custom definitions.

To change logical definitions, modify the base codes **before** concatenation
and rebuild the tree after any later changes. This keeps the stored physical
operators consistent with the parent and child definitions. For example, swap
both X and Z logical labels to preserve their canonical pairing:

```python
from dorsim import CSSCode, concat_code

c4 = CSSCode.c4()
c6 = CSSCode.c6()
for base in (c4, c6):
    base.logical_x = base.logical_x[::-1].copy()
    base.logical_z = base.logical_z[::-1].copy()
code = concat_code(c6, [c4, c4, c4])
measurements = np.zeros((4, code.n), dtype=np.uint8)
logical_z = decoder.decode_code(measurements, code, basis="Z")
logical_x = decoder.decode_code(measurements, code, basis="X")
```

Outputs have shape `(shots, 2)` and dtype `int8`, with erased pairs `[-1, -1]`.
With `final=True`, erased pairs become random binary bits and the result has
dtype `uint8`; `seed` controls reproducibility.

Decode X- and Z-basis measurement records separately. A Z-basis measurement
is flipped by the X component of an error; an X-basis measurement is flipped
by the Z component.

## PoulinDecoder: One Pauli Component

`PoulinDecoder("X", code, p)` estimates X error bits using Z stabilizers;
`"Z"` estimates Z error bits using X stabilizers. Here `p` is the probability
of the selected binary component on each physical qubit, with `0 < p < 1`.
For a depolarizing channel of total rate `q`, either component has rate
`2*q/3`, since Y contributes to both.

The implementation is specialized to the C4/C6 matrices and decomposition:
its leaf pure-error generators are defined only for names `C4` and `C6`,
and recursion groups two logical qubits per child. The `CSSCode` type
annotation does not imply support for arbitrary CSS codes or Qp leaves.

```python
import numpy as np
from dorsim import PoulinDecoder, get_c4c6_code

code = get_c4c6_code(2)
decoder = PoulinDecoder("X", code, p=0.01)
measurements = np.zeros((3, code.n), dtype=np.uint8)
measurements[0, 0] = 1
recovery, probabilities = decoder.decode_measurement(measurements)
corrected = measurements ^ recovery
assert np.all((corrected @ decoder.check.T) % 2 == 0)
assert probabilities[-1].shape == (3, 2**code.k)
```

`decode_measurement(m)` takes `(shots, n)` and computes
`(m @ decoder.check.T) % 2`. `decode(s)` and `decode_syndrome(s)` instead
take `(shots, (n-k)//2)`. They return a `(shots, n)` recovery and a dictionary
whose `-1` entry is the normalized natural-log logical posterior.

`decode_code(m, code=None)` is a measurement-input wrapper; an alternate code
creates a temporary decoder. `decode_c4c6(m, level)` does likewise for a
C4/C6 level. `set_physical_rate(p)` changes the common physical rate and
clears cached tables. This decoder treats X and Z separately and does not
retain their correlation through Y errors.

## CombinedPoulinDecoder: Full Pauli Errors

This decoder jointly tracks X and Z components, including Y. It accepts
general valid `StabilizerCode` objects, including concatenated trees.

```python
import numpy as np
from dorsim import CombinedPoulinDecoder, StabilizerCode

code = StabilizerCode.steane()
p = 0.01
decoder = CombinedPoulinDecoder(code, px=p/3, py=p/3, pz=p/3)

errors = np.zeros((3, 2 * code.n), dtype=np.uint8)
errors[0, 0] = 1
check = np.concatenate(
    [code.stabilizers[:, code.n:], code.stabilizers[:, :code.n]], axis=1
)
syndrome = (errors @ check.T) % 2
recovery, probabilities = decoder.decode(syndrome)
corrected = errors ^ recovery
assert recovery.shape == errors.shape
assert np.all((corrected @ check.T) % 2 == 0)
assert np.allclose(np.exp(probabilities[-1]).sum(axis=1), 1)
```

The constructor arguments are `(code, px, py, pz)`, with nonnegative rates
whose sum is at most one. Each physical position independently has
`P(I)=1-px-py-pz`, `P(X)=px`, `P(Y)=py`, and `P(Z)=pz`.
Every leaf uses the same physical rates. Different measured syndromes can
still give different logical posteriors for different child blocks. There
is no interface for per-position rates or correlations across positions.

`decode` aliases `decode_syndrome`. Input shape is `(shots, n-k)`; output
recovery shape is `(shots, 2*n)`. `probabilities[-1]` has shape
`(shots, 4**k)` and contains natural-log probabilities.
`set_error_model(px, py, pz)` changes the rates and clears the local tables.

### Logical Classes And Recursion

For a syndrome `s`, errors are enumerated as
`T(s) XOR L XOR S`, where `T(s) = (s @ code.pure_errors) % 2`, `L` is a
logical Pauli, and `S` is a stabilizer. The decoder sums probabilities over
all stabilizers for each logical class, then chooses the most likely class.
This accounts for degeneracy; it does not simply choose the most likely
individual physical error.

For concatenation, each child passes its full logical-class distribution
upward. The parent multiplies the appropriate child probabilities and sums
over parent stabilizers. Computation uses log probabilities and
`logsumexp`. Recovery construction chooses a maximizing stabilizer branch
for each class and assembles child recoveries. This provides a representative
of the chosen class, not a guarantee of a globally minimum-weight recovery.
Local enumeration grows exponentially with constituent code size, so small
constituent codes are important even when the total concatenated code is large.

Logical classes are indexed by big-endian bits
`[logical_X_0, ..., logical_X_(k-1), logical_Z_0, ..., logical_Z_(k-1)]`.
For `k=1`, indices are **I=0, Z=1, X=2, Y=3**. This differs from the
single-physical-qubit channel order **[I, X, Z, Y]**.

`decode_syndrome_with_logical(s)` returns
`(recovery, logical_index, probabilities)`. The index labels the selected
recovery relative to `T(s)`; it is not a measurement of the true logical error.
Classes within `1e-12` of the maximum log likelihood select the first index.
Zero-probability physical channels are supported, but an impossible syndrome
has no normalized posterior (its log probabilities remain `-inf`).

## Teleportation: Input And Two Bell Halves

The ECT circuit used in `tests/steane_concat_vs.py` has three encoded blocks:
A is the input, B is the first Bell half, and C is the output half. Each block
contains `n` physical qubits and encodes `k` logical qubits. The three blocks
are distinct from the children of a concatenated code.

In this circuit the Pauli errors are inserted **before** the Bell-preparation
`CX(B, C)`, followed by `CX(A, B)`, `H(A)`, and Z measurements of B then A.
For each corresponding physical position, propagation gives:

```text
M_X = x_A XOR x_B
M_Z = z_A XOR z_B XOR z_C
F_X = x_B XOR x_C
F_Z = z_C
```

`M` is the effective measurement-flip vector `[B results | A results]`,
and `F` is C's frame before correction. Given a first recovery `R(s_M)`,
the corrected output frame is:

```text
F' = F XOR M XOR R(s_M)
```

`CombinedPoulinDecoder` sees only `s_M` and a supplied effective Pauli
distribution for `M`; it does not separately decode A, B, and C. Their rates
may differ. For independent depolarizing rates `p_a`, `p_b`, and `p_c`, set
`r_a = 1 - 4*p_a/3` and similarly for B and C. The marginal channel of M is:

```text
px_M = py_M = (1 - r_a*r_b) / 4
pz_M = (1 + r_a*r_b - 2*r_a*r_b*r_c) / 4
```

Use these in `CombinedPoulinDecoder(code, px_M, py_M, pz_M)` when matching
this circuit's noise model. They generally differ from the physical rates
on any one block. Noise added after Bell preparation or at circuit gates
requires a model appropriate to that circuit.

## JointPoulinDecoder: Correlated ECT Recovery

This decoder retains the correlation between `M` and `D = M XOR F`.
Its constructor is `JointPoulinDecoder(code, joint_md)`, where `joint_md`
is a normalized, finite, strictly positive `(4, 4)` probability table.
Rows label M and columns label D, both in physical channel order
**[I, X, Z, Y]**. The same joint distribution applies independently at every
physical position. `set_joint_error_model(joint_md)` copies a new table and
clears the leaf probability cache. These inputs are trusted; the class does
not validate them.

`JointPoulinDecoder.from_ect_rates(code, p_a, p_b, p_c)` constructs the
table for the circuit above by enumerating independent depolarizing errors
on A/B/C. Supply finite rates in `[0, 1]` that produce a strictly positive
kernel. Unlike the single-table Combined decoder, this table preserves the
dependence between measurement errors and output errors.

The following complete example shows both decoding stages. In an experiment,
`syndrome_f` must come from available output syndrome information; calculating
it from the simulated frame is a simulation convenience.

```python
import numpy as np
from dorsim import CombinedPoulinDecoder, JointPoulinDecoder, StabilizerCode

code = StabilizerCode.steane()
joint = JointPoulinDecoder.from_ect_rates(code, 0.01, 0.02, 0.03)
marginal_m = joint.joint_md.sum(axis=1)  # Order: I, X, Z, Y
first = CombinedPoulinDecoder(
    code, px=marginal_m[1], py=marginal_m[3], pz=marginal_m[2]
)
check = np.concatenate(
    [code.stabilizers[:, code.n:], code.stabilizers[:, :code.n]], axis=1
)

# Example measurement flips M and output frame F before correction.
m = np.zeros((2, 2 * code.n), dtype=np.uint8)
f = np.zeros_like(m)
m[0, 0] = 1
f[1, code.n] = 1
syndrome_m = (m @ check.T) % 2
r_m, logical_m, _ = first.decode_syndrome_with_logical(syndrome_m)
f_prime = f ^ m ^ r_m
syndrome_f = (f_prime @ check.T) % 2

r_f, probabilities = joint.decode(syndrome_m, syndrome_f, logical_m)
final_frame = f_prime ^ r_f
assert np.all((final_frame @ check.T) % 2 == 0)
```

`decode` aliases `decode_syndrome(syndrome_m, syndrome_f, logical_m)`.
Both syndrome arrays have shape `(shots, n-k)`. `syndrome_f` means the
syndrome of **F' after the first correction**, not F before correction.
`logical_m` is the first recovery's selected logical-class index: a scalar
or `(shots,)` array. Do not substitute the unknown true logical class of M.

Internally, `s_D = s_M XOR s_F'`, since `R(s_M)` has syndrome `s_M`.
The decoder recursively calculates the joint logical posterior for M and D
given their syndromes, sums over M's logical class, then relabels D's classes
by XOR with `logical_m` to predict F'. It does not condition on the first
decision as if that decision were known to be correct.

The returned recovery is `T(s_F')` XOR the predicted logical operator,
with shape `(shots, 2*n)`. `probabilities[-1]` gives normalized log
probabilities for F' with shape `(shots, 4**k)`. Exactly tied classes select
the first index. Recursion contracts paired logical/stabilizer choices in
chunks to limit intermediate allocation, but time and memory still grow
rapidly with constituent code size.
