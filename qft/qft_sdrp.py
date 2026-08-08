import math
import random
import statistics
import sys
import time

from collections import Counter

from pyqrack import QrackSimulator


def calc_stats(ideal_probs, exp_probs, sdrp, fidelity):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    mean_guess = 1 / n_pow
    model = 1 / 2
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    sqr_diff = 0
    m_sqr_diff = 0
    for i in range(n_pow):
        exp = exp_probs[i]
        ideal = ideal_probs[i]

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * (exp - u_u)

        # L2 norm
        sqr_diff += (ideal - exp) ** 2
        m_sqr_diff += (ideal - mean_guess) ** 2

        # QV / HOG
        if ideal > threshold:
            hog_prob += exp

    xeb = numer / denom
    rss = math.sqrt(sqr_diff)
    mf_rss = math.sqrt(m_sqr_diff)

    return {
        "qubits": n,
        "sdrp": sdrp,
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "inner_product_lower_bound": fidelity,
        "l2_diff": float(rss),
        "mf_l2_diff": float(mf_rss)
    }


def u(sim, q, th, ph, lm):
    sim.u(q, th, ph, lm)


def h(sim, n):
    sim.h(n)


def cp(sim, th, q, n):
    sim.mcu([q], n, 0, th, 0)


def cx(sim, c, t):
    sim.mcx([c], t)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def reverse(num_qubits, circ):
    start = 0
    end = num_qubits - 1
    while start < end:
        circ.append((swap, start, end))
        start += 1
        end -= 1


# Implementation of the Quantum Fourier Transform
# (See https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html)
def qft(n, circuit):
    if n == 0:
        return circuit
    n -= 1

    circuit.append((h, n))
    for qubit in range(n):
        circuit.append((cp, math.pi / 2 ** (n - qubit), qubit, n))

    # Recursive QFT is very similiar to a ("classical") FFT
    qft(n, circuit)


def run_circuit(sim, circ):
    for g in circ:
        g[0](sim, *(g[1:]))


def bench_qrack(n, sdrp):
    circ = []

    # GHZ state
    circ.append((h, 0))
    for i in range(1, n):
        circ.append((cx, i - 1, i))

    # Random U3 initialization
    # for i in range(0, n):
    #     th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
    #     # Keep it Haar-random towards the poles:
    #     th = math.asin(th / math.pi)
    #     circ.append((u, i, th, ph, lm))

    qft(n, circ)
    reverse(n, circ)

    control = QrackSimulator(n)
    run_circuit(control, circ)
    control = control.out_probs()

    experiment = QrackSimulator(n)
    if sdrp > 0:
        experiment.set_sdrp(sdrp)
    run_circuit(experiment, circ)
    fidelity = experiment.get_unitary_fidelity()
    experiment = experiment.out_probs()

    return calc_stats(control, experiment, sdrp, fidelity)


def main():
    bench_qrack(1, 0)

    n = 16
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    sdrp = (1-1/math.sqrt(2))/2
    if len(sys.argv) > 2:
        sdrp = float(sys.argv[2])

    print(bench_qrack(n, sdrp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
