# Orbifolded random circuit sampling
# How good are Google's own "elided circuits" as a direct XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
# (This is actually a different "elision" concept, but allow that it works.)

import math
import random
import sys
import time

from pyqrack import QrackSimulator, Pauli


def factor_width(width):
    row_len = math.floor(math.sqrt(width))
    while (((width // row_len) * row_len) != width):
        row_len -= 1
    col_len = width // row_len
    if row_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


def ct_pair_prob(sim, q1, q2):
    p1 = sim.prob(q1)
    p2 = sim.prob(q2)

    if p1 < p2:
        return p2, q1

    return p1, q2


def cz_shadow(sim, q1, q2):
    prob_max, t = ct_pair_prob(sim, q1, q2)
    if prob_max > 0.5:
        sim.z(t)


def anti_cz_shadow(sim, q1, q2):
    sim.x(q1)
    cz_shadow(sim, q1, q2)
    sim.x(q1)


def cx_shadow(sim, c, t):
    sim.h(t)
    cz_shadow(sim, c, t)
    sim.h(t)


def anti_cx_shadow(sim, c, t):
    sim.x(t)
    cx_shadow(sim, c, t)
    sim.x(t)


def cy_shadow(sim, c, t):
    sim.adjs(t)
    cx_shadow(sim, c, t)
    sim.s(t)


def anti_cy_shadow(sim, c, t):
    sim.x(t)
    cy_shadow(sim, c, t)
    sim.x(t)


def unpack(lq, reverse = False):
    return [3 * lq + 2, 3 * lq + 1, 3 * lq] if reverse else [3 * lq, 3 * lq + 1, 3 * lq + 2]


def encode(sim, hq, reverse = False):
    if reverse:
        cx_shadow(sim, hq[0], hq[1])
        sim.mcx([hq[1]], hq[2])
    else:
        sim.mcx([hq[0]], hq[1])
        cx_shadow(sim, hq[1], hq[2])


def decode(sim, hq, reverse = False):
    if reverse:
        sim.mcx([hq[1]], hq[2])
        cx_shadow(sim, hq[0], hq[1])
    else:
        cx_shadow(sim, hq[1], hq[2])
        sim.mcx([hq[0]], hq[1])


def u(sim, th, ph, lm, lq):
    hq = unpack(lq)
    decode(sim, hq)
    sim.u(hq[0], th, ph, lm)
    encode(sim, hq)


def s(sim, lq):
    hq = unpack(lq)
    decode(sim, hq)
    sim.s(hq[0])
    encode(sim, hq)


def adjs(sim, lq):
    hq = unpack(lq)
    decode(sim, hq)
    sim.adjs(hq[0])
    encode(sim, hq)


def cpauli(sim, lq1, lq2, anti, pauli):
    gate = None
    if pauli == Pauli.PauliX:
        gate = sim.macx if anti else sim.mcx
    elif pauli == Pauli.PauliY:
        gate = sim.macy if anti else sim.mcy
    elif pauli == Pauli.PauliZ:
        gate = sim.macz if anti else sim.mcz
    else:
        return

    if (lq2 == (lq1 + 1)) or (lq1 == (lq2 + 1)):
        hq1 = unpack(lq1, True)
        hq2 = unpack(lq2, False)
        decode(sim, hq1, True)
        decode(sim, hq2, False)
        gate([hq1[0]], hq2[0])
        encode(sim, hq2, False)
        encode(sim, hq1, True)
    else:
        hq1 = unpack(lq1)
        hq2 = unpack(lq2)
        gate([hq1[0]], hq2[0])
        gate([hq1[1]], hq2[1])
        gate([hq1[2]], hq2[2])


def cx(sim, lq1, lq2):
    cpauli(sim, lq1, lq2, False, Pauli.PauliX)


def cy(sim, lq1, lq2):
    cpauli(sim, lq1, lq2, False, Pauli.PauliY)


def cz(sim, lq1, lq2):
    cpauli(sim, lq1, lq2, False, Pauli.PauliZ)


def acx(sim, lq1, lq2):
    cpauli(sim, lq1, lq2, True, Pauli.PauliX)


def acy(sim, lq1, lq2):
    cpauli(sim, lq1, lq2, True, Pauli.PauliY)


def acz(sim, lq1, lq2):
    cpauli(sim, lq1, lq2, True, Pauli.PauliZ)


def swap(sim, lq1, lq2):
    cx(sim, lq1, lq2)
    cx(sim, lq2, lq1)
    cx(sim, lq1, lq2)


def iswap(sim, lq1, lq2):
    swap(sim, lq1, lq2)
    cz(sim, lq1, lq2)
    s(sim, lq1)
    s(sim, lq2)


def adjiswap(sim, lq1, lq2):
    adjs(sim, lq2)
    adjs(sim, lq1)
    cz(sim, lq1, lq2)
    swap(sim, lq1, lq2)


def m(sim, lq):
    hq = unpack(lq)
    syndrome = 0
    bits = []
    for q in hq:
        bits.append(sim.m(q))
        if bits[-1]:
            syndrome += 1
    result = True if (syndrome > 1) else False
    for i in range(len(hq)): 
        if bits[i] != result:
            sim.x(hq[i])

    return result


def m_all(sim):
    result = 0
    for lq in range(sim.num_qubits() // 3):
        result <<= 1
        if m(sim, lq):
            result |= 1

    return result

def bench_qrack(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()
    experiment = QrackSimulator(3 * width)

    lcv_range = range(width)

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            u(experiment, th, ph, lm, i)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1);
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
                g(experiment, b1, b2)

    # Terminal measurement
    sample = m_all(experiment)
    seconds = time.perf_counter() - start

    return seconds, sample


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 rcs_nn_elided_time.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    # Run the benchmarks
    seconds, sample = bench_qrack(width, depth)

    # Print the results
    print({ 'width': width,  'depth': depth, 'seconds': seconds, 'sample': sample })

    return 0


if __name__ == '__main__':
    sys.exit(main())
