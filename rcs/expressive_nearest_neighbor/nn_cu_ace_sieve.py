# Nearest-neighbor RCS: ACE consensus sieve for heavy outputs.
#
# Candidate outcomes are drawn by sampling. Each candidate's
# probability is estimated solely as the average prob_perm across n_inst
# ACE instances. Candidates are routed into heavy/light buckets relative
# to the uniform distribution, mixed equally with raw ACE, and XEB / HOG
# are computed against a full ideal Qrack simulation.
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
# Statistics (sparse heavy/light representation)
# ---------------------------------------------------------------------------

def route_heavy_light(prob_dict, u_u):
    """
    Split a {outcome: p} dict into (heavy, light) dicts centered on u_u.
    heavy: outcomes where p > u_u, values normalised to sum 1.
    light: outcomes where p < u_u, values (stored positive) normalised to sum 1.
    """
    heavy_raw = {}
    light_raw = {}
    for outcome, p in prob_dict.items():
        c = p - u_u
        if c > 0:
            heavy_raw[outcome] = c
        elif c < 0:
            light_raw[outcome] = -c          # store as positive

    s_h = sum(heavy_raw.values())
    s_l = sum(light_raw.values())
    heavy = {k: v / s_h for k, v in heavy_raw.items()} if s_h > 0 else {}
    light = {k: v / s_l for k, v in light_raw.items()} if s_l > 0 else {}
    return heavy, light


def calc_stats_sparse(ideal_probs, exp_probs_sparse, exp_probs, n_pow):
    """
    Reconstruct a dense approximate distribution from the heavy/light split
    and compute XEB and HOG against ideal_probs.

    Dense reconstruction:
        exp_mixed[k] = 0.5 * heavy[k]  +  0.5 * u_u * (1 - light[k])
    Heavy outcomes sit above u_u; light outcomes are suppressed below u_u.
    Everything not sampled sits exactly at u_u in the light half, which is
    conservative and avoids phantom signal from unvisited outcomes.
    """
    u_u = 1.0 / n_pow
    heavy, light = exp_probs_sparse

    h_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in heavy.items():
        h_dense[k] = v

    l_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in light.items():
        l_dense[k] = v
    exp_mixed = 0.5 * (0.5 * (h_dense + u_u * (1.0 - l_dense)) + exp_probs)

    p_c   = ideal_probs - u_u
    q_c   = exp_mixed   - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_mixed[ideal_probs > float(np.median(ideal_probs))].sum())
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
        "n_candidates":  0.0,
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
        candidates = set()
        for inst in range(n_inst):
            sim = QrackSimulator(width)
            if sdrp > 0.0:
                sim.set_sdrp(sdrp)
            sim.set_ace_max_qb((width + 1) >> 1)
            sim.run_qiskit_circuit(qc[inst], shots=0)
            ace_sims.append(sim)
            candidates.add(sim.highest_prob_perm())

        t_ace = time.perf_counter()
        if trials == 1:
            print(f"ace_seconds: {t_ace - t_ideal}")

        # -----------------------------------------------------------------------
        # Sampling sieve.
        #
        # Sample n_candidates outcomes from the full 2^n Hilbert space. Each run
        # uses a different ACE cut, reducing biased variance across instances.
        # -----------------------------------------------------------------------
        shots = n_candidates - len(candidates)
        remainder = shots % n_inst
        shots //= n_inst
        all_qubits = list(range(width))
        for sim in ace_sims:
            s = shots
            if remainder:
                s += 1
                remainder -= 1
            candidates.update(sim.measure_shots(all_qubits, s))

        candidates = list(candidates)
        q_bits = list(range(width))
        ace_probs = {}
        for idx in candidates:
            bits  = [(idx >> b) & 1 for b in range(width)]
            p_ace = sum(s.prob_perm(q_bits, bits) for s in ace_sims) / n_inst
            if p_ace > 0:
                ace_probs[idx] = p_ace

        t_sieve = time.perf_counter()
        if trials == 1:
            print(f"sieve_seconds: {t_sieve - t_ace}")

        # -----------------------------------------------------------------------
        # Route into heavy / light and compute statistics
        # -----------------------------------------------------------------------
        sparse = route_heavy_light(ace_probs, u_u)

        ace_probs = np.empty(n_pow, dtype=np.float64)
        for inst in ace_sims:
            ace_probs += np.array(inst.out_probs())
        ace_probs /= n_inst

        for s in ace_sims:
            del s

        xeb, hog = calc_stats_sparse(ideal_probs, sparse, ace_probs, n_pow)

        if math.isfinite(xeb) and math.isfinite(hog):
            results["n_candidates"] += len(candidates)
            results["xeb_ace"]      += xeb
            results["hog_ace"]      += hog
        else:
            trials -= 1

    results["n_candidates"]  /= trials
    results["xeb_ace"]       /= trials
    results["hog_ace"]       /= trials

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 nn_cu_ace_sieve.py [width] [depth] [sdrp=0] [trials=1]")
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
