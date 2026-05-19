# Fully-connected RCS with 4-instance greedy ACE consensus.
#
# Four QrackSimulator instances run the same random circuit, receiving
# the 2-qubit couplers in four different orderings within each layer.
# ACE's greedy boundary selection is order-dependent, so the four instances
# develop approximately mutually orthogonal separability structures.
#
# Ordering strategy: quarter-rotation of the coupler pair list.
# Instance i starts at offset i*k//4 in the shuffled pair list (wrapping
# around), so each instance "leads" with a different quarter of the couplers.
# This maximally separates the greedy fusion entry points across ACE's four
# subsystems and gives uniform pairwise Kendall distances between orderings.
#
# Additionally, instances 2 and 3 apply an odd/even stride interleave on top
# of their quarter-rotation — giving a two-level hierarchy (rotation at the
# block level, stride at the element level) that mirrors the exponential
# structure of the statevector subsystem decomposition.
#
# The coherent superposition of all four is the consensus heuristic state.
# For four mutually orthogonal unit vectors, the sum has norm 2, so:
#   mix = (ka + kb + kc + kd) / 2.0   (exact normalization, no sqrt needed)
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time

import numpy as np
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Coupler ordering strategies
# ---------------------------------------------------------------------------

def _order_pairs(pairs, inst):
    """
    Return pairs in the ordering for instance `inst` (0-3).

    Bit 1 of inst: quarter-rotation (0 = no rotation, 1 = rotate by k//2)
    Bit 0 of inst: stride interleave (0 = contiguous, 1 = odds-then-evens)

    This gives four orderings with a two-level structure:
      inst 0: natural              [0,1,2,3,4,5,6,7]
      inst 1: stride               [1,3,5,7,0,2,4,6]
      inst 2: half-rotation        [4,5,6,7,0,1,2,3]
      inst 3: half-rotation+stride [5,7,1,3,4,6,0,2]
    """
    k = len(pairs)
    if k == 0:
        return pairs

    # Bit 1: half-rotation (offset by k//2)
    offset = (k >> 1) if (inst & 2) else 0
    rotated = pairs[offset:] + pairs[:offset]

    # Bit 0: odd/even stride interleave
    if inst & 1:
        rotated = [rotated[i] for i in range(1, k, 2)] + \
                  [rotated[i] for i in range(0, k, 2)]

    return rotated


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats_np(ideal_ket, split_ket):
    """Vectorised XEB, L2 inner-product fidelity, prob_diff."""
    n_pow  = len(ideal_ket)
    u_u    = 1.0 / n_pow
    ideal  = np.asarray(ideal_ket, dtype=np.complex128)
    split  = np.asarray(split_ket, dtype=np.complex128)
    l2     = complex(np.dot(split, ideal.conj()))
    l2     = (l2 * l2.conjugate()).real
    p      = (ideal * ideal.conj()).real
    q      = (split * split.conj()).real
    p_c    = p - u_u
    q_c    = q - u_u
    denom  = float(np.dot(p_c, p_c))
    xeb    = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    pd     = float(np.sum((p - q) ** 2))
    return xeb, l2, pd


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0):
    lcv_range = range(width)
    all_bits  = list(lcv_range)
    n_inst    = 4

    sims = [QrackSimulator(width) for _ in range(n_inst)]
    for s in sims:
        if sdrp > 0.0:
            s.set_sdrp(sdrp)
        # ~width/4 qubits per ACE segment — four segments per GPU.
        s.set_ace_max_qb((width + 3) >> 2)

    rng_state = random.getstate()
    t_start   = time.perf_counter()

    kets = []
    for inst in range(n_inst):
        random.setstate(rng_state)
        sim = sims[inst]

        for _ in range(depth):
            # Identical single-qubit gates
            for i in lcv_range:
                th = random.uniform(0, 2 * math.pi)
                ph = random.uniform(0, 2 * math.pi)
                lm = random.uniform(0, 2 * math.pi)
                sim.u(i, th, ph, lm)

            # Build pair list (same shuffle for all instances via RNG replay)
            pairs = []
            unused = all_bits.copy()
            random.shuffle(unused)
            while len(unused) > 1:
                c = unused.pop()
                t = unused.pop()
                pairs.append((c, t))

            # Apply couplers in the instance-specific order
            for c, t in _order_pairs(pairs, inst):
                sim.mcx([c], t)

        kets.append(np.asarray(sim.out_ket(), dtype=np.complex128))

    t_elapsed = time.perf_counter() - t_start

    # -----------------------------------------------------------------------
    # Consensus: average vectors (which might not be perfectly orthogonal)
    # -----------------------------------------------------------------------
    mix = sum(kets) / len(kets)
    mix_norm = float(np.sqrt((mix * mix.conj()).real.sum()))
    if mix_norm > 0:
        mix /= mix_norm

    # Cross-instance statistics (should approach zero for orthogonal instances)
    from itertools import combinations
    cross = {}
    for i, j in combinations(range(n_inst), 2):
        xeb_ij, l2_ij, _ = calc_stats_np(kets[i], kets[j])
        cross[f"xeb_{i}_vs_{j}"] = xeb_ij
        cross[f"l2_{i}_vs_{j}"]  = l2_ij

    # Each instance vs consensus
    xeb_vs  = []
    l2_vs   = []
    for i, k in enumerate(kets):
        xeb_i, l2_i, _ = calc_stats_np(k, mix)
        xeb_vs.append(xeb_i)
        l2_vs.append(l2_i)

    result = {
        "width":         width,
        "depth":         depth,
        "seconds":       t_elapsed,
        "xeb_consensus": float(np.mean(xeb_vs)),
    }
    for i in range(n_inst):
        result[f"xeb_{i}_vs_cons"] = xeb_vs[i]
        result[f"l2_{i}_vs_cons"]  = l2_vs[i]
    result.update(cross)

    return result


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
