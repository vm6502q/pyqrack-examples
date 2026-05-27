# Fully-connected RCS: Automatic circuit elision
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    if col_len == 1:
        raise Exception("ERROR: Can't simulate prime number width!")

    return (row_len, col_len)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, exp_probs, n_pow):
    u_u   = 1.0 / n_pow
    p_c   = ideal_probs - u_u
    q_c   = exp_probs   - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_probs[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0, trials=1):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 4
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    results = {
        "width":         width,
        "depth":         depth,
        "sdrp":          sdrp,
        "trials":        trials,
        "xeb_ace":       0.0,
        "hog_ace":       0.0,
    }

    # Number of candidate outcomes to probe via the sieve.
    # width**2 keeps cost polynomial; sqrt(n_pow) is the natural "heavy
    # output" count for a Porter-Thomas distributed circuit.
    n_candidates = max(width ** 2, int(math.sqrt(n_pow) + 0.5))

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    row_len, col_len = factor_width(width)

    for _ in range(trials):
        # -----------------------------------------------------------------------
        # Build n_inst independent random circuits (same single-qubit angles,
        # different coupler orderings — identical to fc_ace_consensus.py)
        # -----------------------------------------------------------------------
        t_circ = time.perf_counter()
        qc = [QuantumCircuit(width) for _ in range(n_inst)]

        for _ in range(depth):
            for i in lcv_range:
                th, ph, lm = (random.uniform(0, 2*math.pi) for _ in range(3))
                for c in qc:
                    c.u(th, ph, lm, i)

            gate = gateSequence.pop(0)
            gateSequence.append(gate)
            cl = []
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

                    if random.randint(0, 1):
                        b1, b2 = b2, b1

                    cl.append(((b1, b2), [random.uniform(0, 2*math.pi) for _ in range(4)]))

            for c in qc:
                random.shuffle(cl)
                for g in cl:
                    b, p = g
                    c.cu(p[0], p[1], p[2], p[3], b[0], b[1])

        t_build = time.perf_counter()
        if trials == 1:
            print(f"circuit_build_seconds: {t_build - t_circ}")

        # -----------------------------------------------------------------------
        # Ideal ground truth (full state vector via Qrack)
        # -----------------------------------------------------------------------
        sim_ideal = QrackSimulator(width)
        sim_ideal.run_qiskit_circuit(qc[0], shots=0)
        ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
        del sim_ideal


        t_ideal = time.perf_counter()
        if trials == 1:
            print(f"ideal_seconds: {t_ideal - t_build}")

        # -----------------------------------------------------------------------
        # ACE consensus instances
        # -----------------------------------------------------------------------
        ace_sims = []
        for inst in range(n_inst):
            sim = QrackSimulator(width)
            if sdrp > 0.0:
                sim.set_sdrp(sdrp)
            sim.set_ace_max_qb((width + 1) >> 1)
            sim.run_qiskit_circuit(qc[inst], shots=0)
            ace_sims.append(sim)

        t_ace = time.perf_counter()
        if trials == 1:
            print(f"ace_seconds: {t_ace - t_ideal}")

        # -----------------------------------------------------------------------
        # Compute statistics
        # -----------------------------------------------------------------------
        ace_probs = np.empty(n_pow, dtype=np.float64)
        for inst in ace_sims:
            ace_probs += np.array(inst.out_probs())
        ace_probs /= n_inst

        # q_bits = list(range(width))
        # for outcome in range(n_pow):
            # It's possible to access every amplitude without materializing the state vector:
            # bits  = [(outcome >> b) & 1 for b in range(width)]
            # ace_probs[outcome] = sum(s.prob_perm(q_bits, bits) for s in ace_sims) / n_inst

        xeb, hog = calc_stats(ideal_probs, ace_probs, n_pow)

        if math.isfinite(xeb) and math.isfinite(hog):
            results["xeb_ace"]      += xeb
            results["hog_ace"]      += hog
        else:
            trials -= 1

    results["xeb_ace"]       /= trials
    results["hog_ace"]       /= trials

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_ace_consensus.py [width] [depth] [sdrp=0] [trials=1]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0  # ((1 - 1 / math.sqrt(2)) / 2)
    trials = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    result = bench_qrack(width, depth, sdrp, trials)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
