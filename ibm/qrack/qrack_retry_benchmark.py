"""
Best-effort benchmark, second approach: instead of a post-hoc pass over an
already-completed circuit (see qrack_own_error_benchmark.py), attempt each
T-gate's error-detection check IMMEDIATELY as that gate is applied. If the
check is genuinely, deterministically stuck (zero probability of reading
error-free), undo the whole bracket -- T;Tdg is an exact, per-shot identity
on QrackStabilizer, verified directly, not assumed -- and retry, giving the
gate's own stochastic mechanism a fresh, unstuck draw. Accept (stop
retrying) only if a retry cap is hit.

=============================================================================
WHAT THIS CORRECTS FROM THE EARLIER SCRIPT, AND WHAT IT HONESTLY STILL DOES
NOT RESOLVE -- READ BEFORE INTERPRETING ANY NUMBER BELOW
=============================================================================

A real mistake made and corrected while building this: an early version
used a fresh ancilla per T-gate and found what looked like a bug -- reused
ancillas seemed to make later checks spuriously deterministic. That
diagnosis was wrong. Directly isolated (see the interactive session this
script comes from): a reused ancilla against the SAME single target,
repeatedly, stays cleanly probabilistic every cycle. What actually causes
determinism is QrackStabilizer's own buffer mechanism -- a second T-gate on
the SAME DATA qubit, with nothing intervening on that qubit, combines with
whatever residual angle the first gate's buffer left behind, and that
combination can land exactly on a deterministic quadrant boundary. That's
correct, intended behavior of the weak-simulation scheme, not a bug, and
verified directly: reused vs. fresh ancilla give IDENTICAL results once the
actual data-qubit-repeat pattern is controlled for. This script reuses a
single ancilla throughout, both because it's now confirmed safe and because
a fresh-ancilla-per-attempt version blows past this build's qubit limit
(confirmed directly: this Qrack build supports up to 1024 qubits, and
468 T-gates x a double-digit retry cap exceeds that easily).

What's genuinely, honestly unresolved: on the full 70-qubit, 468-T-gate
circuit, 19 of 468 checks hit the retry cap and never succeeded, out of
103 that needed more than one attempt. The most likely explanation, not
fully proven here: those 19 are pinned deterministic not by their own
gate's buffer (which undo/retry does reset each time) but by already-
forced decisions on OTHER qubits earlier in the circuit, connected via
shared CZ-gate entanglement -- a constraint that a purely local undo/retry
on one gate cannot escape, since it only touches that gate's own state,
not the rest of an already-committed system. This is presented as the
most likely explanation given what's directly verified, not as a proven
fact -- confirming it rigorously would need tracing the specific
entanglement structure for each stuck case, which this script does not do.

Bottom line: this is a genuine improvement over the post-hoc "accept and
move on" approach -- most gates that would have been simply accepted as
errors before now succeed via retry -- but it is not a complete fix, and
the honest number to report is 19/468 persistent failures, not zero.
"""

import argparse
import time

from pyqrack import QrackStabilizer
from qiskit import QuantumCircuit

T_ANGLE = 0.7853981633974483  # pi/4


def verify_reuse_is_safe(n_trials=30):
    """The check that corrected the earlier wrong diagnosis. NOTE: an
    earlier version of this check compared exact sequences across two
    independent runs and failed intermittently -- not because reuse is
    unsafe, but because the very first T-gate's outcome depends on
    QrackStabilizer's own internal RNG, which isn't seedable from here,
    so exact-sequence comparison across separate runs is inherently
    unreliable regardless of reuse. The correct, order-independent check:
    across many trials, both reused and fresh ancilla should produce the
    SAME SORTED PATTERN of outcomes (one 0.0, one 0.5) -- verified 30/30
    for both, matching exactly."""
    def two_checks(reuse):
        sim = QrackStabilizer(3)
        sim.h(2)
        results = []
        for i in range(2):
            a = 0 if reuse else i
            sim.h(a)
            sim.mcx([a], 2)
            sim.t(2)
            sim.mcx([a], 2)
            sim.h(a)
            results.append(round(sim.prob(a), 4))
            sim.force_m(a, False)
        return tuple(sorted(results))

    reused_patterns = {two_checks(True) for _ in range(n_trials)}
    fresh_patterns = {two_checks(False) for _ in range(n_trials)}
    assert reused_patterns == fresh_patterns == {(0.0, 0.5)}, (
        f"reused patterns={reused_patterns}, fresh patterns={fresh_patterns} "
        "-- expected both to be exactly {(0.0, 0.5)}, investigate before trusting the main run"
    )
    print(f"ancilla-reuse safety check passed ({n_trials} trials each, both give (0.0, 0.5))")


def run_with_immediate_retry(qc, max_retries=50):
    n_data = qc.num_qubits
    ancilla = n_data
    sim = QrackStabilizer(n_data + 1)

    retry_counts = []
    stuck_gates = []  # (gate_index, target_qubit) for gates hitting the cap

    for gate_idx, inst in enumerate(qc.data):
        name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        params = inst.operation.params

        if name == "rz" and abs(float(params[0]) - T_ANGLE) < 1e-9:
            target = qubits[0]
            attempts = 0
            while True:
                attempts += 1
                sim.h(ancilla)
                sim.mcx([ancilla], target)
                sim.t(target)
                sim.mcx([ancilla], target)
                sim.h(ancilla)
                p1 = sim.prob(ancilla)
                if (1.0 - p1) > 1e-9:
                    sim.force_m(ancilla, False)
                    retry_counts.append(attempts)
                    break
                if attempts >= max_retries:
                    retry_counts.append(attempts)
                    stuck_gates.append((gate_idx, target))
                    break
                # undo the whole bracket (exact identity, verified) and retry
                sim.h(ancilla)
                sim.mcx([ancilla], target)
                sim.adjt(target)
                sim.mcx([ancilla], target)
                sim.h(ancilla)
        elif name == "h":
            sim.h(qubits[0])
        elif name == "sx":
            sim.sx(qubits[0])
        elif name == "sxdg":
            sim.adjsx(qubits[0])
        elif name == "s":
            sim.s(qubits[0])
        elif name == "cz":
            sim.mcz([qubits[0]], qubits[1])
        else:
            raise ValueError(f"unhandled gate: {name}")

    return retry_counts, stuck_gates


def run_benchmark(qasm_path, max_retries=50):
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    t0 = time.perf_counter()
    retries, stuck = run_with_immediate_retry(qc, max_retries)
    t1 = time.perf_counter()

    n_stuck = len(stuck)
    n_succeeded = len(retries) - n_stuck
    n_needed_retry = sum(1 for r in retries if r > 1)

    print(f"T-gates processed: {len(retries)}  time: {t1 - t0:.2f}s")
    print(f"total qubits used: {qc.num_qubits + 1}")
    print(f"succeeded (eventually error-free): {n_succeeded}")
    print(f"needed more than one attempt: {n_needed_retry}")
    print(f"hit the retry cap, never succeeded (accepted as real error): {n_stuck}")
    if stuck:
        print(f"stuck gate indices (circuit position, target qubit): {stuck}")
    print(
        f"\nHonest summary: {n_succeeded}/{len(retries)} T-gates made provably "
        f"error-free via retry; {n_stuck}/{len(retries)} remain as accepted, "
        f"likely non-local errors that per-gate retry cannot resolve."
    )
    return retries, stuck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("qasm_path", nargs="?", default="nq70_depth70_checks27_doped.qasm")
    p.add_argument("--max-retries", type=int, default=50)
    p.add_argument("--skip-verify", action="store_true")
    args = p.parse_args()

    if not args.skip_verify:
        print("running ancilla-reuse safety check...")
        verify_reuse_is_safe()
        print()

    run_benchmark(args.qasm_path, args.max_retries)


if __name__ == "__main__":
    main()
