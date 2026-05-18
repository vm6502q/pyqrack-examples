# How good are Google's own "patch circuits" and "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)

import math
import random
import statistics
import sys

from pyqrack import QrackSimulator


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def cx(sim, q1, q2):
    sim.mcx([q1], q2)


def cy(sim, q1, q2):
    sim.mcy([q1], q2)


def cz(sim, q1, q2):
    sim.mcz([q1], q2)


def acx(sim, q1, q2):
    sim.macx([q1], q2)


def acy(sim, q1, q2):
    sim.macy([q1], q2)


def acz(sim, q1, q2):
    sim.macz([q1], q2)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.adjiswap(q1, q2)


def pswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def nswap(sim, q1, q2):
    sim.mcz([q1], q2)
    sim.swap(q1, q2)
    sim.mcz([q1], q2)


def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    ace_qb = (width + 3) >> 2
    print(f"Maximum entangled subsystem qubit footprint: {ace_qb}")
    control = QrackSimulator(width)
    experiment = QrackSimulator(width)
    experiment.set_ace_max_qb(ace_qb)

    lcv_range = range(width)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            control.u(i, th, ph, lm)
            experiment.u(i, th, ph, lm)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row = temp_row + row_len
                if temp_col < 0:
                    temp_col = temp_col + col_len
                if temp_row >= row_len:
                    temp_row = temp_row - row_len
                if temp_col >= col_len:
                    temp_col = temp_col - col_len

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(control, b1, b2)
                g(experiment, b1, b2)

        print(calc_stats(control.out_ket(), experiment.out_ket(), d + 1, ace_qb))


def calc_stats(ideal_ket, split_ket, depth, ace_qb):
    ideal_prob = [(x * x.conjugate()).real for x in ideal_ket]
    split_prob = [(x * x.conjugate()).real for x in split_ket]
    n_pow = len(ideal_ket)
    n = math.log2(n_pow)
    u_u = 1.0 / n_pow
    threshold = statistics.median(ideal_prob)
    numer = 0.0
    denom = 0.0
    l2 = 0.0
    prob_diff = 0.0
    hog_prob = 0.0

    for i in range(n_pow):
        l2 += split_ket[i] * ideal_ket[i].conjugate()
        p_i = ideal_prob[i]
        q_i = split_prob[i]
        numer += (p_i - u_u) * (q_i - u_u)
        denom += (p_i - u_u) ** 2
        prob_diff += (p_i - q_i) ** 2
        if p_i > threshold:
            hog_prob += q_i

    xeb = numer / denom if denom > 0 else 0.0
    l2 = (l2 * l2.conjugate()).real

    return {
        "qubits": n,
        "ace_qb_limit": ace_qb,
        "depth": depth,
        "l2": float(l2),
        "xeb": float(xeb),
        "hog_prob": float(hog_prob),
        "prob_diff": float(prob_diff)
    }


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_qiskit_validation.py [width] [depth]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    bench_qrack(width, depth)

    return 0


if __name__ == "__main__":
    sys.exit(main())
