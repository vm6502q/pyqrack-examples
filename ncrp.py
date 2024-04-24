import math
import random
import sys
import time

from pyqrack import QrackSimulator, Pauli


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


def bench_qrack(width, depth, magic, ncrp):
    # This is a "nearest-neighbor" coupler random circuit.
    start = time.perf_counter()

    sim = QrackSimulator(width, isTensorNetwork=False, isSchmidtDecompose=False)
    sim.set_ncrp(ncrp)

    magic_fraction = 3 * width * depth / magic

    # Nearest-neighbor couplers:
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    col_len = math.floor(math.sqrt(width))
    while (((width // col_len) * col_len) != width):
        col_len -= 1
    row_len = width // col_len

    for i in range(depth):
        # Single bit gates
        for j in range(width):
            for _ in range(3):
                # We're trying to cover 3 Pauli axes
                # with Euler angle axes x-z-x.
                sim.h(j)

                # We can trace out a quarter rotations around the Bloch sphere with stabilizer.
                rnd = random.randint(0, 3)
                if rnd & 1:
                    sim.s(j)
                if rnd & 2:
                    sim.z(j)

                # For each axis, there is a chance of "magic."
                if (magic > 0) and ((magic_fraction * random.random()) < 1):
                    angle = random.uniform(0, math.pi / 2)
                    sim.r(Pauli.PauliZ, angle, j)
                    magic -= 1

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

                if (temp_row < 0) or (temp_col < 0) or (temp_row >= row_len) or (temp_col >= row_len):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(sim, b1, b2)

    # Terminal measurement
    sim.m_all()
    fidelity = sim.get_unitary_fidelity()

    return (time.perf_counter() - start, fidelity)


def main():
    bench_qrack(1, 1, 1, 0.5)

    width = 36
    depth = 6
    magic = 6
    samples = 1
    if len(sys.argv) < 6:
        raise RuntimeError('Usage: python3 sdrp.py [ncrp] [width] [depth] [magic] [samples]')

    ncrp = float(sys.argv[1])

    if len(sys.argv) > 2:
        width = int(sys.argv[2])

    if len(sys.argv) > 3:
        depth = int(sys.argv[3])

    if len(sys.argv) > 4:
        magic = int(sys.argv[4])

    if len(sys.argv) > 5:
        samples = int(sys.argv[5])

    # Run the benchmarks
    width_results = []
    for i in range(samples):
        width_results.append(bench_qrack(width, depth, magic, ncrp))

    time_result = sum(r[0] for r in width_results) / samples
    fidelity_result = sum(r[1] for r in width_results) / samples
    print("Width=", width, ", Depth=", depth, ": ", time_result, " seconds, ", fidelity_result, " out of 1.0 fidelity")

    return 0


if __name__ == '__main__':
    sys.exit(main())
