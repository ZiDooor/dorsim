# Source Modules And Lower-Level Helpers

The user-facing classes are imported from `dorsim`. Lower-level helpers can
be imported from their defining modules; they are primarily used by the
simulators and verification code.

| Module | Purpose | Guide |
| --- | --- | --- |
| `__init__.py` | Package exports | [Index](index.md) |
| `circuit.py` | Circuit, Operation, record targets | [Circuit API](circuit_api.md) |
| `tableau_sim.py` | Inverse tableau reference simulation | [TableauSim](tableau_sim.md) |
| `pauli_frame.py` | Batched frame sampling | [PauliFrame](pauli_frame.md) |
| `stab_code.py` | Code matrices and concatenation | [Stabilizer Codes](stabilizer_codes.md) |
| `decoder.py` | Knill, single-component, combined and joint decoders | [Decoders](decoders.md) |
| `pauli.py` | Pauli arithmetic and Clifford conjugation | Below |
| `cli.py`, `__main__.py` | Demo and reference checks | Below |

## Pauli Arithmetic

`dorsim.pauli` uses Hermitian local Paulis with integer codes
`I=0`, `X=1`, `Z=2`, `Y=3`, computed as `x + 2*z`.
`code_from_bits(x, z)` and `bits_from_code(code)` convert between them.
`CLIFFORD_GATES`, `SINGLE_QUBIT_GATES`, and `TWO_QUBIT_GATES` list the
supported gate names; `S_DAG` is the internal name for `Circuit.sdg`.

```python
import numpy as np
from dorsim.pauli import bits_from_code, code_from_bits, identity_tableau, tableau_apply

assert code_from_bits(1, 1) == 3
assert bits_from_code(2) == (0, 1)
tableau, sign = identity_tableau(1)
row, row_sign = tableau_apply(
    tableau, sign, np.array([1]), np.array([1]), 0
)
assert row.tolist() == [1, 1] and row_sign == 0
```

`identity_tableau(n)` returns the `(2*n, 2*n)` identity and a zero sign
array of length `2*n`. `tableau_apply(tableau, sign, pauli_x, pauli_z,
pauli_sign)` maps a Hermitian Pauli through the inverse tableau, returning
the concatenated `[X|Z]` row and its binary sign.

`pauli_mul_phase(lhs_x, lhs_z, lhs_phase, rhs_x, rhs_z, rhs_sign)`
multiplies two strings in left-to-right order. The left phase is an exponent
of `i` modulo four; the right sign is a binary exponent of `-1`.
The returned `(x, z, phase)` keeps the full exponent modulo four, so it can
represent products such as `X*Z = -iY`. `LOCAL_MUL_PHASE` supplies the
single-qubit multiplication phases. This full phase differs from the binary
sign used for Hermitian tableau rows.

## Matrices And Conjugation

`kron_all(mats)` takes an ordered tensor product. `gate_matrix(name)`
returns a dense matrix for each supported Clifford gate.
`pauli_tuple_matrix(codes)` builds the matrix of a tuple of local Pauli
codes, in tuple order. These matrix helpers and the maps below are intended
for small local operations, not large-register dense simulation.

`local_conjugation_map(name, arity)` enumerates the local Pauli basis and
maps each tuple to `(sign, output_tuple)` under `G^-1 P G`. It is cached,
as are gate and Pauli matrices. `conjugate_pauli_by_gate(x, z, sign, op)`
uses that map to return updated arrays and a binary sign. An operation whose
name is outside `CLIFFORD_GATES` returns copies unchanged.

```python
import numpy as np
from dorsim import Operation
from dorsim.pauli import conjugate_pauli_by_gate

x, z, sign = conjugate_pauli_by_gate(
    np.array([1]), np.array([1]), 0, Operation("H", (0,))
)
assert x.tolist() == [1] and z.tolist() == [1] and sign == 1  # H Y H = -Y
```

For the relationship between these helpers and tableau row/column updates,
see [Update Rules](update_rules.md).

## Command-Line Demo And Stim Checks

After installation, either entry point runs `cli.main()`:

```bash
python -m dorsim
dorsim-demo
```

Both first run `run_self_checks()` and then `demo()`. The checks compare
tableaux and sampling with Stim, so these commands require the optional
Stim dependency:

```bash
pip install -e '.[test]'
```

The demo prints a three-qubit reference tableau, reference measurements,
measurement flips, and samples. To run just the demo without the Stim checks:

```python
from dorsim.cli import demo

demo()
```

`stim_inverse_rows(circuit)` returns Stim's inverse X and Z tableau rows
after the noiseless circuit. `stim_fixed_zero_branch_rows(circuit)` uses
postselection to reproduce the reference simulator's zero branch when
possible, preserving deterministic one outcomes. These are comparison
helpers, not alternatives to the simulator's measurement-record feedback
implementation; the latter helper does not implement record-target feedback.
