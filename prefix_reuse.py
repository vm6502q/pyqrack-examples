# Demonstrates an inexpensive pattern for prefix reuse

import math
import random
import sys
import time

from pyqrack import QrackSimulator


def random_circuit(qsim, width, depth):
    # This is a "fully-connected" coupler random circuit.
    lcv_range = range(width)
    all_bits = list(lcv_range)

    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            qsim.u(i, th, ph, lm)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            qsim.mcx([c], t)


def main():
    qsim = QrackSimulator(16)

    # Prefix to reuse
    print("Running 16-by-8 RCS prefix...")
    start = time.perf_counter()
    random_circuit(qsim, 16, 8)

    # Flush and isolate prefix
    # The ranges are arbitrary for the example, but this flushes the caches.
    qsim.are_factorized(list(range(8)), list(range(8, 16)), True)
    end = time.perf_counter()
    print(f"(Took {end - start} seconds.")

    # You can create as many clones as you like, or daisy-chain the process.
    print("Cloning prefix...")
    prefix = qsim.clone()
    print("(Done.)")

    # Reuse prefix
    print("Running two different 16-by-8 continuations on the same prefix...")
    start = time.perf_counter()
    random_circuit(qsim, 16, 8)
    random_circuit(prefix, 16, 8)
    qsim.m_all()
    prefix.m_all()
    end = time.perf_counter()
    print(f"(Took {end - start} seconds.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
