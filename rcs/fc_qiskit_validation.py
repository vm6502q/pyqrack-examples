# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from collections import Counter

from scipy.stats import binom

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector


def bench_qrack(width, depth, trials, sdrp):
    # This is a "nearest-neighbor" coupler random circuit.
    control = AerSimulator(method="statevector")
    shots = 1 << (width + 2)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    results = []

    for trial in range(trials):
        circ = QuantumCircuit(width)
        for d in range(depth):
            # Single-qubit gates
            for i in lcv_range:
                th = random.uniform(0, 2 * math.pi)
                ph = random.uniform(0, 2 * math.pi)
                lm = random.uniform(0, 2 * math.pi)
                circ.u(th, ph, lm, i)

            # 2-qubit couplers
            unused_bits = all_bits.copy()
            random.shuffle(unused_bits)
            while len(unused_bits) > 1:
                c = unused_bits.pop()
                t = unused_bits.pop()
                circ.cx(c, t)

            experiment = QrackSimulator(width, isOpenCL=False, isSparse=True)
            if sdrp > 0:
                experiment.set_sdrp(sdrp)
            experiment.run_qiskit_circuit(circ)

            circ_aer = transpile(circ, backend=control)
            circ_aer.save_statevector()
            job = control.run(circ_aer)

            experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))
            control_probs = Statevector(job.result().get_statevector()).probabilities()

            stats = calc_stats(control_probs, experiment_counts, d + 1, shots)

            if trial == 0:
                results.append(stats)
            else:
                results[d]["xeb"] += stats["xeb"]
                results[d]["hog_prob"] += stats["hog_prob"]
                results[d]["p-value"] *= stats["p-value"]

            if trial == (trials - 1):
                results[d]["xeb"] /= trials
                results[d]["hog_prob"] /= trials
                results[d]["p-value"] = results[d]["p-value"] ** (1 / trials)
                print(results[d])


def calc_stats(ideal_probs, counts, depth, shots):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    sum_hog_counts = 0
    for i in range(n_pow):
        count = counts[i] if i in counts else 0
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * ((count / shots) - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    hog_prob = sum_hog_counts / shots
    xeb = numer / denom
    # p-value of heavy output count, if method were actually 50/50 chance of guessing
    p_val = (
        (1 - binom.cdf(sum_hog_counts - 1, shots, 1 / 2)) if sum_hog_counts > 0 else 1
    )

    return {
        "qubits": n,
        "depth": depth,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "p-value": float(p_val),
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth] [trials]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    trials = 1
    sdrp = 0
    if len(sys.argv) > 3:
        trials = int(sys.argv[3])
    if len(sys.argv) > 4:
        sdrp = float(sys.argv[4])

    # Run the benchmarks
    bench_qrack(width, depth, trials, sdrp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
