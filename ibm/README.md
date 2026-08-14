# DCS circuit weak-simulation example (QrackStabilizer + QrackAceBackend)

## Circuit attribution

`nq70_depth70_checks27_doped.qasm` (70-qubit, unencoded) and
`nq70_depth70_checks27_doped_checks.qasm` (97-qubit, encoded with 27
syndrome ancillas) are from:

Martiel, Chung, Seif, Ghosh, Hincks, Deshpande, Fefferman, Gambetta,
Javadi-Abhari, "Sampling hard circuits with verifiably high fidelity,"
arXiv:2607.25941 (IBM Research / University of Chicago).

Data and circuits released under CC BY 4.0 via Zenodo,
10.5281/zenodo.21633064.

## Verified circuit structure

- Unencoded file: 70 qubits, 2415 CZ gates, 468 `rz(pi/4)` (T) gates,
  matching the paper's stated circuit exactly.
- Encoded file: 97 qubits (70 data + 27 ancilla), 2869 CZ gates (2415 +
  454 for syndrome extraction, matching the paper's stated total
  exactly), same 468 T gates.
- Ancilla qubits are indices 70-96 in the encoded file (confirmed from
  gate-count structure -- data qubits average ~198 gates each, ancillas
  ~20 -- and from the gate pattern on individual ancillas: H, [~20x CZ to
  a single data qubit across the circuit's depth], H, matching the
  paper's space-local, single-data-qubit-support check construction
  exactly).

## Known bug: QrackStabilizer.m_all() above 64 qubits

Verified directly: `m_all()` gives correct results at exactly 64 qubits,
and silently wrong results at 65+ (tested and confirmed at n=65, 70, 97).
`QrackAceBackend.m_all()` does not have this problem. Both scripts here
measure qubits individually via `.m(q)` instead, which is correct at
every size tested. This is a workaround at the call site, not a fix to
the library -- worth fixing at the source.

## Usage

```
python3 dcs_post_select.py nq70_depth70_checks27_doped_checks.qasm --shots 500 --out stab_result.json
python3 dcs_ace_compare.py nq70_depth70_checks27_doped.qasm --shots 500 --stabilizer-result stab_result.json --out ace_result.json
```

## What this does and doesn't show

With no physical noise model applied (no CZ error, no readout error),
post-selection acceptance in simulation is close to 100% -- confirmed at
500/500 shots. This is the *correct* result for a noiseless simulation,
not a discrepancy with the paper: their 5.9e-4 acceptance rate is a
statement about their hardware's physical noise, which isn't modeled
here. The 100% acceptance also incidentally confirms that
`QrackStabilizer`'s stochastic T-gate rounding genuinely respects the
check-commutation property the paper's fidelity bound depends on.

This is not yet a reproduction of the paper's fidelity bound -- that
would need either a real noise model on the CZ/H/S gates (giving
post-selection something to actually filter) or continued use of the
ACE cross-validation as an independent check, since no ground-truth
reference is feasible to compute directly at this qubit count.

It's important to note that a basic "sanity test" tells us what we'd
expect when applying this heuristic to a similar circuit for Shor's
algorithm with error-detection ancillae post-selected on the T-gate
error the heuristic actually has, analytically: we can effectively
"steer" to nominally error-free syndrome and effectively the "closest
Clifford circuit" while losing virtually or entirely all coherence.
