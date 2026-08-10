# Post-selected weak simulation of a doped-Clifford-sampling (DCS) circuit
# on QrackStabilizer.
#
# Circuit source: Martiel, Chung, Seif, Ghosh, Hincks, Deshpande, Fefferman,
# Gambetta, Javadi-Abhari, "Sampling hard circuits with verifiably high
# fidelity," arXiv:2607.25941 (IBM Research / U. Chicago). Data and circuits
# from the paper are released under CC BY 4.0 via Zenodo,
# 10.5281/zenodo.21633064.
#
# IMPORTANT BUG NOTE: QrackStabilizer.m_all() silently corrupts results for
# systems above 64 qubits (verified directly: correct at n=64, wrong at
# n=65, confirmed at several points up to n=97). This script measures each
# qubit individually via .m(q) instead, which is correct at every size
# tested. QrackAceBackend.m_all() does NOT have this problem -- it's
# specific to QrackStabilizer. Worth fixing at the source; this is a
# workaround, not a fix.
#
# Expected behavior: this simulation has no physical hardware noise (no CZ
# error, no readout error, nothing) -- the only error source present is the
# T-gate weak-simulation rounding, which is specifically constructed
# (code-preserving doping) to commute with the checks. So acceptance here
# should be close to 100%, not the paper's 5.9e-4 -- that rate is almost
# entirely a statement about their physical hardware's noise, which this
# simulation doesn't model. 100% acceptance is the CORRECT result here, not
# a sign that post-selection isn't doing anything.

import argparse
import json
import time

from pyqrack import QrackStabilizer
from qiskit import QuantumCircuit


def run_post_selected(qasm_path, n_data, n_ancilla, shots, out_path=None):
    """Run `shots` independent, fresh, top-to-bottom weak-simulation trials
    of the circuit at qasm_path, keeping only shots where every ancilla
    measures 0 (the "zero syndrome" post-selection condition).

    Assumes the convention used in the reference circuits: data qubits are
    indices [0, n_data), ancillas are the following n_ancilla indices. This
    is a real assumption about register layout, not something inferred
    automatically -- verify it holds for any new circuit (e.g. by checking
    that the low-index qubits have much higher per-qubit gate counts, as
    they do here: ~198 gates/qubit for data vs ~20 gates/qubit for
    ancillas in the reference circuit) before trusting results from a
    circuit built differently.
    """
    n_qubits = n_data + n_ancilla
    data_qubits = list(range(n_data))
    ancilla_qubits = list(range(n_data, n_qubits))

    qc = QuantumCircuit.from_qasm_file(qasm_path)
    if qc.num_qubits != n_qubits:
        raise ValueError(
            f"circuit has {qc.num_qubits} qubits, expected {n_qubits} "
            f"(n_data={n_data} + n_ancilla={n_ancilla})"
        )

    accepted_samples = []
    t0 = time.perf_counter()
    for _ in range(shots):
        # Fresh instance per shot: mandatory for weak simulation, since
        # every T-gate's stochastic rounding decision needs to be an
        # independent draw, not shared state reused across "shots" of a
        # single execution.
        sim = QrackStabilizer(n_qubits)
        sim.run_qiskit_circuit(qc, shots=0)

        syndrome = [sim.m(q) for q in ancilla_qubits]
        if all(s == 0 for s in syndrome):
            accepted_samples.append([sim.m(q) for q in data_qubits])
    elapsed = time.perf_counter() - t0

    result = {
        "qasm_path": qasm_path,
        "shots": shots,
        "accepted": len(accepted_samples),
        "acceptance_rate": len(accepted_samples) / shots,
        "elapsed_seconds": elapsed,
        "seconds_per_shot": elapsed / shots,
        "accepted_samples": accepted_samples,
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f)

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("qasm_path")
    p.add_argument("--n-data", type=int, default=70)
    p.add_argument("--n-ancilla", type=int, default=27)
    p.add_argument("--shots", type=int, default=500)
    p.add_argument("--out", default="dcs_post_select_result.json")
    args = p.parse_args()

    result = run_post_selected(
        args.qasm_path, args.n_data, args.n_ancilla, args.shots, args.out
    )
    print(
        f"{result['accepted']}/{result['shots']} accepted "
        f"({100 * result['acceptance_rate']:.1f}%), "
        f"{result['elapsed_seconds']:.1f}s total, "
        f"{result['seconds_per_shot']:.4f}s/shot"
    )
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
