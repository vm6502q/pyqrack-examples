# Fully-connected RCS with built-in greedy ACE consensus.
#
# Two QrackSimulator instances run the same random circuit, but receive
# the 2-qubit couplers in REVERSED order within each layer.  ACE's greedy
# boundary selection is order-dependent, so the two instances develop
# approximately orthogonal separability structures — the same idea as the
# explicit H/V patch split, but achieved for free by reordering commuting
# gates rather than running shadow circuits.
#
# The equal mixture of the two statevectors is the consensus heuristic state.
# Statistics are computed against that mixture as both "ideal" and "experiment,"
# and also cross-ways (instance A as ideal, mixture as experiment, and vice versa).
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time

import numpy as np
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_ket, split_ket):
    n_pow = len(ideal_ket)
    u_u = 1.0 / n_pow
    numer = 0.0
    denom = 0.0
    l2 = 0.0
    prob_diff = 0.0

    for i in range(n_pow):
        c = ideal_ket[i]
        e = split_ket[i]
        l2 += e * c.conjugate()
        p_i = (c * c.conjugate()).real
        q_i = (e * e.conjugate()).real
        numer += (p_i - u_u) * (q_i - u_u)
        denom += (p_i - u_u) ** 2
        prob_diff += (p_i - q_i) ** 2

    xeb = numer / denom if denom > 0 else 0.0
    l2 = (l2 * l2.conjugate()).real

    return xeb, l2, prob_diff


def calc_stats_np(ideal_ket, split_ket):
    """Vectorised version of calc_stats for the mixture (which is a numpy array)."""
    n_pow = len(ideal_ket)
    u_u   = 1.0 / n_pow

    ideal  = np.asarray(ideal_ket,  dtype=np.complex128)
    split  = np.asarray(split_ket,  dtype=np.complex128)

    l2        = complex(np.dot(split, ideal.conj()))
    l2        = (l2 * l2.conjugate()).real
    p         = (ideal * ideal.conj()).real
    q         = (split * split.conj()).real
    p_c       = p - u_u
    q_c       = q - u_u
    numer     = float(np.dot(p_c, q_c))
    denom     = float(np.dot(p_c, p_c))
    xeb       = numer / denom if denom > 0 else 0.0
    prob_diff = float(np.sum((p - q) ** 2))

    return xeb, l2, prob_diff


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0):
    lcv_range = range(width)
    all_bits  = list(lcv_range)

    # Instance A: couplers applied in the shuffled order (natural greedy)
    # Instance B: couplers applied in the REVERSED shuffled order
    # Both see identical single-qubit gates and the same set of coupler pairs —
    # only the presentation order to ACE differs, biasing its greedy
    # boundary choices toward approximately orthogonal separability structures.
    sim_a = QrackSimulator(width)
    sim_b = QrackSimulator(width)
    sim_i = QrackSimulator(width)
    if sdrp > 0.0:
        sim_a.set_sdrp(sdrp)
        sim_b.set_sdrp(sdrp)
    # Split into 4 subsystems, to demonstrate ~30-32 qubits per segment
    # for 4 segments on a GPU, in proof-of-concept.
    sim_a.set_ace_max_qb((width + 3) >> 2)
    sim_b.set_ace_max_qb((width + 3) >> 2)

    rng_state = random.getstate()   # snapshot so both instances replay identically

    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Instance "Ideal" — natural coupler order
    # -----------------------------------------------------------------------
    random.setstate(rng_state)
    for _ in range(depth):
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            sim_i.u(i, th, ph, lm)

        pairs = []
        unused = all_bits.copy()
        random.shuffle(unused)
        while len(unused) > 1:
            c = unused.pop()
            t = unused.pop()
            sim_i.mcx([c], t)
            

    ket_i = sim_i.out_ket()

    # -----------------------------------------------------------------------
    # Instance A — natural coupler order
    # -----------------------------------------------------------------------
    random.setstate(rng_state)
    for _ in range(depth):
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            sim_a.u(i, th, ph, lm)

        pairs = []
        unused = all_bits.copy()
        random.shuffle(unused)
        while len(unused) > 1:
            c = unused.pop()
            t = unused.pop()
            sim_a.mcx([c], t)

    ket_a = sim_a.out_ket()

    # -----------------------------------------------------------------------
    # Instance B — every-other coupler order within each layer
    # -----------------------------------------------------------------------
    random.setstate(rng_state)
    for _ in range(depth):
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            sim_b.u(i, th, ph, lm)

        pairs = []
        unused = all_bits.copy()
        random.shuffle(unused)
        coupler_order = []
        while len(unused) > 1:
            pairs.append((unused.pop(), unused.pop()))

        for i in range(1, width >> 1, 2):
            pair = pairs[i]
            sim_b.mcx([pair[0]], pair[1])

        for i in range(0, width >> 1, 2):
            pair = pairs[i]
            sim_b.mcx([pair[0]], pair[1])

        # Reversed presentation order — same gates, different ACE greedy path
        for c, t in reversed(pairs):
            sim_b.mcx([c], t)

    ket_b = sim_b.out_ket()

    t_elapsed = time.perf_counter() - t_start

    # -----------------------------------------------------------------------
    # Consensus: equal mixture of the two statevectors
    # (as a coherent superposition, since we would normalize the average)
    # -----------------------------------------------------------------------
    ki = np.asarray(ket_i, dtype=np.complex128)
    ka = np.asarray(ket_a, dtype=np.complex128)
    kb = np.asarray(ket_b, dtype=np.complex128)
    mix = (ka + kb) / 2.0
    mix_norm = float(np.sqrt((mix * mix.conj()).real.sum()))
    if mix_norm > 0:
        mix /= mix_norm

    # A vs B (how different are the two instances?)
    xeb_ab, l2_ab, pd_ab = calc_stats(ket_a, ket_b)

    # A vs consensus
    xeb_ac, l2_ac, pd_ac = calc_stats_np(mix, ka)

    # B vs consensus
    xeb_bc, l2_bc, pd_bc = calc_stats_np(mix, kb)

    # Ideal vs consensus
    xeb_ic, l2_ac, pd_ac = calc_stats_np(ki, mix)

    # Consensus self-consistency: treat A as ideal, consensus as experiment,
    # then B as ideal, consensus as experiment — average XEB is the headline figure
    xeb_mean = (xeb_ac + xeb_bc) / 2.0

    return {
        "width":          width,
        "depth":          depth,
        "seconds":        t_elapsed,
        # Cross-instance agreement (low = orthogonal, high = redundant)
        "xeb_A_vs_B":     xeb_ab,
        "l2_A_vs_B":      l2_ab,
        "prob_diff_A_B":  pd_ab,
        # Consensus quality
        "xeb_A_vs_cons":  xeb_ac,
        "xeb_B_vs_cons":  xeb_bc,
        "xeb_I_vs_cons":  xeb_ic,
        "xeb_consensus":  xeb_mean,
        "l2_A_vs_cons":   l2_ac,
        "l2_B_vs_cons":   l2_bc,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_consensus.py [width] [depth] [sdrp=0]"
        )
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    result = bench_qrack(width, depth, sdrp)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
