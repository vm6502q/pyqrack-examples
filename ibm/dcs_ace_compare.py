# Independent cross-validation of the DCS circuit's data-qubit statistics:
# QrackAceBackend (patch/boundary-consensus approximation) against
# QrackStabilizer's post-selected weak simulation (dcs_post_select.py).
#
# These two methods have genuinely independent error mechanisms -- ACE
# doesn't know or care about Cliffordness at all (its approximation is
# purely geometric, via patch decomposition and boundary-qubit consensus),
# while QrackStabilizer's error is entirely about magic-gate stochastic
# rounding. Agreement between them on the same circuit is real, if
# suggestive rather than conclusive, evidence that both are tracking
# something real -- there's no third, independent, ground-truth reference
# available at this qubit count (that's the whole point of the circuit
# being classically hard), so this cross-check is what's actually
# available here, not a substitute for one.
#
# ACE runs against the UNENCODED 70-qubit circuit (nq70_depth70_checks27_
# doped.qasm), not the 97-qubit encoded one -- ACE has no use for the check
# ancillas, and comparing the same 70 data qubits both ways is what makes
# the comparison apples-to-apples.

import argparse
import json
import time

from pyqrack import QrackAceBackend
from qiskit import QuantumCircuit


def run_ace_samples(qasm_path, shots, long_range_columns=4, long_range_rows=4):
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    n = qc.num_qubits

    samples = []
    t0 = time.perf_counter()
    for _ in range(shots):
        # Same principle as QrackStabilizer: ACE's shadow-consensus
        # mechanism is stochastic too, so each shot needs a fresh,
        # independent, full top-to-bottom execution.
        sim = QrackAceBackend(
            n,
            long_range_columns=long_range_columns,
            long_range_rows=long_range_rows,
            is_1d_chain=True,
        )
        sim.run_qiskit_circuit(qc, shots=0)
        samples.append([sim.m(q) for q in range(n)])
    elapsed = time.perf_counter() - t0

    return samples, elapsed


def marginal_probs(samples, n_qubits):
    counts = [0] * n_qubits
    for s in samples:
        for i, b in enumerate(s):
            counts[i] += b
    return [c / len(samples) for c in counts]


def compare_marginals(ace_samples, stabilizer_samples, n_qubits):
    p_ace = marginal_probs(ace_samples, n_qubits)
    p_stab = marginal_probs(stabilizer_samples, n_qubits)
    diffs = [abs(a - b) for a, b in zip(p_ace, p_stab)]
    return {
        "ace_marginals": p_ace,
        "stabilizer_marginals": p_stab,
        "mean_abs_diff": sum(diffs) / len(diffs),
        "max_abs_diff": max(diffs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("qasm_path")
    p.add_argument("--shots", type=int, default=500)
    p.add_argument(
        "--stabilizer-result",
        default=None,
        help="path to a dcs_post_select.py --out JSON file to compare against",
    )
    p.add_argument("--out", default="dcs_ace_result.json")
    args = p.parse_args()

    samples, elapsed = run_ace_samples(args.qasm_path, args.shots)
    print(f"ACE: {args.shots} shots, {elapsed:.1f}s total, {elapsed/args.shots:.4f}s/shot")

    result = {"qasm_path": args.qasm_path, "shots": args.shots, "samples": samples}

    if args.stabilizer_result:
        with open(args.stabilizer_result) as f:
            stab_result = json.load(f)
        stab_samples = stab_result["accepted_samples"]
        if stab_samples:
            n_qubits = len(stab_samples[0])
            comparison = compare_marginals(samples, stab_samples, n_qubits)
            print(
                f"marginal comparison ({len(stab_samples)} post-selected "
                f"Stabilizer samples vs {len(samples)} ACE samples): "
                f"mean |diff| = {comparison['mean_abs_diff']:.4f}, "
                f"max |diff| = {comparison['max_abs_diff']:.4f}"
            )
            result["comparison"] = comparison
        else:
            print("no accepted Stabilizer samples in the given result file -- nothing to compare")

    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
