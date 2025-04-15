# Demonstrate mirror circuit simplification

import math
import random
import sys
import time

from pyqrack import QrackSimulator, QrackCircuit


def bench_qrack(n):
    circ = QrackCircuit()

    lcv_range = range(n)
    all_bits = list(lcv_range)
    x_op = [0, 1, 1, 0]
    gateSequence = [ 0, 3, 2, 1, 2, 1, 0, 3 ]
    row_len = math.ceil(math.sqrt(n))

    for _ in lcv_range:
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            cos0 = math.cos(th / 2);
            sin0 = math.sin(th / 2);
            u_op = [
                cos0 + 0j, sin0 * (-math.cos(lm) + -math.sin(lm) * 1j),
                sin0 * (math.cos(ph) + math.sin(ph) * 1j), cos0 * (math.cos(ph + lm) + math.sin(ph + lm) * 1j)
            ]
            circ.mtrx(u_op, i)

        # Nearest-neighbor couplers:
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(row_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1);
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if (temp_row < 0) or (temp_col < 0) or (temp_row >= row_len) or (temp_col >= row_len):
                    continue

                b1 = row * row_len + col
                b2 = temp_row * row_len + temp_col

                if (b1 >= n) or (b2 >= n):
                    continue

                if random.uniform(0, 1) < 0.5:
                    tmp = b1
                    b1 = b2
                    b2 = tmp

                circ.ucmtrx([b1], x_op, b2, 1)

    shots = 100
    start = time.perf_counter()
    sim = QrackSimulator(n)
    circ.run(sim)
    circ.inverse().run(sim)
    results = sim.measure_shots(all_bits, shots)
    seconds = time.perf_counter() - start
    fidelity = results.count(0) / shots

    return (seconds, fidelity)


def main():
    n = 50
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    results = bench_qrack(n)

    print(n, "qubits,",
        results[0], "seconds,",
        results[1], "fidelity"
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
