"""
Best-effort benchmark: an error-detection circuit built for QrackStabilizer's
OWN error model (T-gate weak-simulation stochastic rounding), applied to
IBM's published 70-qubit doped-Clifford-sampling circuit, attempting a
"fair" analogue of their reported hardware acceptance rate.

Source circuit: Martiel, Chung, Seif, Ghosh, Hincks, Deshpande, Fefferman,
Gambetta, Javadi-Abhari, "Sampling hard circuits with verifiably high
fidelity," arXiv:2607.25941 (IBM Research / U. Chicago). CC BY 4.0 via
Zenodo, 10.5281/zenodo.21633064. This script uses the UNENCODED circuit
(nq70_depth70_checks27_doped.qasm, 70 qubits, 468 rz(pi/4) gates) and adds
its own ancilla checks -- it does not use IBM's spacetime-code ancillas.

=============================================================================
WHAT THIS ACTUALLY MEASURES, AND WHY IT ISN'T DIRECTLY COMPARABLE TO IBM's
REPORTED 5.9e-4 ACCEPTANCE RATE -- READ BEFORE INTERPRETING ANY NUMBER BELOW
=============================================================================

Mechanism: QrackStabilizer approximates each rz(pi/4) as a stochastic choice
between "no correction" and "an extra S/Sdg quarter-turn," each shot. A
check ancilla, prepared in |+>, coupled via CX(ancilla, target_qubit)
immediately before and after ONE specific T-gate, detects whether that
gate's kick fired -- verified directly (not assumed) on small cases below,
including confirming a Z-type (CZ) coupling does NOT work, since it
commutes with the S/Sdg error and is therefore blind to it by the same
logic that makes IBM's own code-preserving checks blind to T-doping.

Two honest limitations, found by actually running this, not anticipated in
advance:

1. The stochastic decision is already realized (not held in superposition)
   by the time run_qiskit_circuit finishes -- each T-gate's kick either did
   or didn't happen, as an already-determined fact about this specific run.
   So "force all checks to show no error" means conditioning on an event
   with probability ~0.5^468, which underflows to exactly zero in floating
   point -- confirmed directly, not estimated. This is NOT analogous to
   IBM's checks, which detect rare physical faults (~1e-4), not a ~50%
   built-in mechanism.

2. Pushing the same idea "as far as it goes" -- force every check that CAN
   be forced, accept (don't disturb) any check that's already a certain,
   unforceable outcome -- gives a well-defined result for any GIVEN
   processing order, but the result is genuinely ORDER-DEPENDENT: checks
   are correlated through the shared stabilizer state, so which specific
   subset ends up unforceable depends on what's already been conditioned
   on. Confirmed directly: three different orderings gave 81, 94, and 104
   "accepted errors" and accumulated probabilities spanning roughly two
   orders of magnitude. This script deliberately runs multiple orderings
   and reports all of them, rather than picking one and calling it "the"
   number.

Bottom line, stated plainly: this is a best-effort, honestly-limited
attempt at the "fair" analogue IBM's post-selection rate represents for
their error model. It does not resolve into a single clean scalar the way
their physical measurement does, and that's a real, structural finding
about the classical-simulation version of this question -- not a bug to be
patched away. Anyone extending this should look first at whether an
order-independent formulation of "how much of our own error survived" is
even possible, before trusting any single number this script prints.
"""

import argparse
import random
import time

from pyqrack import QrackStabilizer
from qiskit import QuantumCircuit


# -----------------------------------------------------------------------
# Mechanism verification -- run before trusting anything below. These are
# the exact checks that were actually run, interactively, before this
# script was written; included here so the mechanism isn't just asserted.
# -----------------------------------------------------------------------

def verify_mechanism():
    from pyqrack import QrackSimulator

    # 1. Chain-rule joint probability via force_m, on a Bell pair: known
    # exact answer is 0.5.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    sim = QrackStabilizer(2)
    sim.run_qiskit_circuit(qc, shots=0)
    p = 1.0
    for q in [0, 1]:
        p *= 1.0 - sim.prob(q)
        sim.force_m(q, False)
    assert abs(p - 0.5) < 1e-6, f"Bell-pair chain-rule check failed: got {p}"

    # 2. Same, on a GHZ state, checking intermediate conditionals too:
    # known exact answer is P(all 3=0)=0.5, with P(q1=0|q0=0)=1.0 and
    # P(q2=0|q0=0,q1=0)=1.0 (both deterministic given the prior forces).
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    sim = QrackStabilizer(3)
    sim.run_qiskit_circuit(qc, shots=0)
    conds = []
    p = 1.0
    for q in [0, 1, 2]:
        c = 1.0 - sim.prob(q)
        conds.append(c)
        p *= c
        sim.force_m(q, False)
    assert abs(p - 0.5) < 1e-6, f"GHZ chain-rule check failed: got {p}"
    assert abs(conds[1] - 1.0) < 1e-6 and abs(conds[2] - 1.0) < 1e-6, (
        f"GHZ intermediate conditionals wrong: {conds}"
    )

    # 3. A Z-type (CZ) ancilla check does NOT detect an S-gate kick
    # (diagonal operators commute -- this is expected to fail, and its
    # failure is exactly why IBM's own checks are blind to T-doping).
    def cz_check(apply_kick):
        sim = QrackSimulator(2)
        sim.h(0)
        sim.h(1)
        sim.mcz([0], 1)
        if apply_kick:
            sim.s(0)
        sim.mcz([0], 1)
        sim.h(1)
        return sim.prob(1)

    p_no_kick = cz_check(False)
    p_kick = cz_check(True)
    assert abs(p_no_kick - p_kick) < 1e-6, (
        "CZ check unexpectedly distinguished the kick -- mechanism "
        "understanding may be wrong, investigate before trusting anything else"
    )

    # 4. An X-type (CX, ancilla as control) check DOES detect the kick:
    # no-kick -> P(ancilla=1)=0.0 exactly, kick -> P(ancilla=1)=0.5 exactly.
    def cx_check(apply_kick):
        sim = QrackSimulator(2)
        sim.h(1)
        sim.mcx([1], 0)
        if apply_kick:
            sim.s(0)
        sim.mcx([1], 0)
        sim.h(1)
        return sim.prob(1)

    p_no_kick = cx_check(False)
    p_kick = cx_check(True)
    assert abs(p_no_kick - 0.0) < 1e-6, f"CX no-kick check failed: got {p_no_kick}"
    assert abs(p_kick - 0.5) < 1e-6, f"CX kick check failed: got {p_kick}"

    print("all mechanism checks passed (Bell-pair, GHZ, CZ-blind, CX-sensitive)")


# -----------------------------------------------------------------------
# Build the checked circuit: one fresh ancilla per rz(pi/4) instance,
# surgically bracketing just that gate (avoiding confusion with normal,
# intended Clifford evolution elsewhere in the circuit).
# -----------------------------------------------------------------------

def build_checked_circuit(qasm_path):
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    n_data = qc.num_qubits

    t_gate_count = sum(1 for inst in qc.data if inst.operation.name == "rz")
    new_qc = QuantumCircuit(n_data + t_gate_count)
    ancilla_idx = n_data
    ancilla_qubits = []

    for inst in qc.data:
        name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name == "rz":
            a = ancilla_idx
            ancilla_idx += 1
            ancilla_qubits.append(a)
            new_qc.h(a)
            new_qc.cx(a, qubits[0])
            new_qc.append(inst.operation, qubits)
            new_qc.cx(a, qubits[0])
        else:
            new_qc.append(inst.operation, qubits)

    for a in ancilla_qubits:
        new_qc.h(a)

    return new_qc, ancilla_qubits


# -----------------------------------------------------------------------
# Greedy "force as far as it goes" pass: force every check that has
# nonzero probability of reading no-error; accept (don't disturb) any
# check that's already a certain, unforceable outcome. Result is
# order-dependent -- see module docstring -- so this is run under several
# orderings rather than reported as a single number.
# -----------------------------------------------------------------------

def greedy_force_pass(qc, ancilla_qubits, order):
    sim = QrackStabilizer(qc.num_qubits)
    sim.run_qiskit_circuit(qc, shots=0)

    p_accumulated = 1.0
    n_forced = 0
    n_accepted_error = 0

    for a in order:
        p1 = sim.prob(a)
        p0 = 1.0 - p1
        if p0 < 1e-9:
            n_accepted_error += 1
            continue
        sim.force_m(a, False)
        p_accumulated *= p0
        n_forced += 1

    return n_forced, n_accepted_error, p_accumulated


def run_benchmark(qasm_path, n_orderings=3, seed=0):
    print(f"loading and building checked circuit from {qasm_path}...")
    t0 = time.perf_counter()
    qc, ancilla_qubits = build_checked_circuit(qasm_path)
    t1 = time.perf_counter()
    print(
        f"circuit built: {qc.num_qubits} total qubits "
        f"({qc.num_qubits - len(ancilla_qubits)} data + {len(ancilla_qubits)} check ancillas), "
        f"{t1 - t0:.2f}s"
    )

    random.seed(seed)
    orderings = {"original": list(ancilla_qubits), "reversed": list(reversed(ancilla_qubits))}
    for i in range(n_orderings - 2):
        shuffled = list(ancilla_qubits)
        random.shuffle(shuffled)
        orderings[f"shuffled_{i}"] = shuffled

    print(f"\n{'ordering':>14s} {'forced':>8s} {'accepted_error':>15s} {'accumulated_p':>15s}")
    results = {}
    for name, order in orderings.items():
        n_f, n_e, p = greedy_force_pass(qc, ancilla_qubits, order)
        results[name] = {"n_forced": n_f, "n_accepted_error": n_e, "accumulated_p": p}
        print(f"{name:>14s} {n_f:>8d} {n_e:>15d} {p:>15.3e}")

    print(
        "\nNote the spread across orderings above -- that spread is the actual "
        "finding here, not noise to average away. See module docstring for why "
        "no single one of these numbers should be quoted as 'the' result."
    )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("qasm_path", nargs="?", default="nq70_depth70_checks27_doped.qasm")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--orderings", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.skip_verify:
        print("running mechanism verification...")
        verify_mechanism()
        print()

    run_benchmark(args.qasm_path, args.orderings, args.seed)


if __name__ == "__main__":
    main()
