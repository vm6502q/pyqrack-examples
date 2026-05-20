# Fully-connected RCS: uniform-random MPS sieve + ACE prob_perm consensus.
#
# Three methods compared:
#
# Method 1 — Uniform random MPS sieve:
#   Pick width**3 candidates uniformly at random from [0, 2^n).
#   Query MPS amplitude for each via trie contraction.
#   Route by p_mps vs u_u: heavy above, light below.
#
# Method 2 — ACE prob_perm consensus:
#   Two ACE instances (sequential + stride coupler order).
#   measure_shots to identify candidate set.
#   prob_perm queries averaged over both instances.
#   Route by p_ace vs u_u.
#
# Method 3 — Equal 50/50 mixture of Methods 1 and 2.
#
# XEB and HOG computed against a full ideal simulator (small scale only).
#
# By Dan Strano and (Anthropic) Claude.

import math
import random
import sys
import time
from collections import defaultdict

import numpy as np
import jax.numpy as jnp
import quimb.tensor as tn
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Coupler ordering
# ---------------------------------------------------------------------------

def _order_pairs(pairs, inst):
    k = len(pairs)
    if k == 0 or inst == 0:
        return pairs
    return [pairs[i] for i in range(1, k, 2)] + \
           [pairs[i] for i in range(0, k, 2)]


# ---------------------------------------------------------------------------
# Trie-based MPS amplitude contraction
# ---------------------------------------------------------------------------

def _int_to_bittuple(integer, length):
    return tuple((integer >> b) & 1 for b in range(length))


def batch_amplitudes_trie(mps_psi, bitstrings):
    tensors = [np.array(t.data) for t in mps_psi.tensors]
    n       = len(tensors)
    results = {}

    def _recurse(site, env, group):
        if site == n:
            scalar = complex(env.flat[0]) if hasattr(env, 'flat') else complex(env)
            for bs in group:
                results[bs] = scalar
            return
        t = tensors[site]
        by_bit = defaultdict(list)
        for bs in group:
            by_bit[bs[site]].append(bs)
        for bit, subgroup in by_bit.items():
            if site == 0:
                new_env = t[:, bit].copy()
            elif site == n - 1:
                new_env = env @ t[:, bit]
            else:
                new_env = env @ t[:, :, bit]
            _recurse(site + 1, new_env, subgroup)

    _recurse(0, None, bitstrings)
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, exp_probs, n_pow):
    u_u   = 1.0 / n_pow
    model = 0.5
    p_c   = ideal_probs - u_u
    q_c   = exp_probs   - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_probs[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog


def calc_stats_sparse(ideal_probs, exp_probs_sparse, n_pow):
    u_u   = 1.0 / n_pow
    h_probs, l_probs = exp_probs_sparse
    h_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in h_probs.items():
        h_dense[k] = v
    l_dense = h_dense.copy()
    for k, v in l_probs.items():
        l_dense[k] = v
    exp_mixed = (h_dense + u_u * (l_dense + 1.0)) / 2.0
    p_c   = ideal_probs - u_u
    q_c   = exp_mixed   - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_mixed[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog


def route_heavy_light(prob_dict, u_u):
    heavy_raw = {}; light_raw = {}
    for outcome, p in prob_dict.items():
        p -= u_u
        if p > 0:
            heavy_raw[outcome] = p
        elif p < 0:
            light_raw[outcome] = p
    s_h = sum(heavy_raw.values())
    heavy = {k:v/s_h for k,v in heavy_raw.items()} if s_h > 0 else {}
    s_l = sum(light_raw.values())
    light = {k:-v/s_l for k,v in light_raw.items()} if s_l < 0 else {}

    return (heavy, light)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0, chi=None):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 2
    n_candidates = width ** 3
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    if chi is None:
        chi = min(width ** 2, 1 << (width // 2))

    # -----------------------------------------------------------------------
    # Build circuit once in Qiskit + quimb MPS from same RNG
    # -----------------------------------------------------------------------
    qc      = QuantumCircuit(width)
    mps_sim = tn.CircuitMPS(width, max_bond=chi, to_backend=jnp.array)

    rng_state = random.getstate()
    for _ in range(depth):
        for i in lcv_range:
            th, ph, lm = (random.uniform(0, 2*math.pi) for _ in range(3))
            qc.u(th, ph, lm, i)
            mps_sim.apply_gate('U3', th, ph, lm, i)
        shuffled = all_bits[:]
        random.shuffle(shuffled)
        while len(shuffled) > 1:
            c, t = shuffled.pop(), shuffled.pop()
            qc.cx(c, t)
            mps_sim.apply_gate('CX', c, t)

    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Ideal ground truth
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    random.setstate(rng_state)
    sim_ideal.run_qiskit_circuit(qc, shots=0)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    # -----------------------------------------------------------------------
    # Method 1: Uniform random MPS sieve
    # Pick n_candidates uniformly, query MPS, route by p vs u_u
    # -----------------------------------------------------------------------
    uniform_candidates = random.sample(range(n_pow), min(n_candidates, n_pow))
    candidate_tuples   = [_int_to_bittuple(i, width) for i in uniform_candidates]
    amp_map = batch_amplitudes_trie(mps_sim.psi, candidate_tuples)

    mps_probs = {}
    for idx, bs_tup in zip(uniform_candidates, candidate_tuples):
        amp = amp_map.get(bs_tup, 0.0+0.0j)
        p   = amp.real**2 + amp.imag**2
        if p > 0:
            mps_probs[int(idx)] = p

    mps_combined = route_heavy_light(mps_probs, u_u)
    xeb_mps, hog_mps = calc_stats_sparse(ideal_probs, mps_combined, n_pow)

    # -----------------------------------------------------------------------
    # Method 2: ACE prob_perm over full Hilbert space.
    # Since the ideal simulation is already materialized for ground truth,
    # we can afford to walk all 2^n permutations with prob_perm — giving
    # the complete ACE probability distribution, not just sampled candidates.
    # Two ACE instances (sequential + stride); average their prob_perm values.
    # -----------------------------------------------------------------------
    ace_sims = []
    for inst in range(n_inst):
        random.setstate(rng_state)
        sim = QrackSimulator(width)
        if sdrp > 0.0:
            sim.set_sdrp(sdrp)
        sim.set_ace_max_qb((width + 1) >> 1)
        sim.run_qiskit_circuit(qc, shots=0)
        ace_sims.append(sim)

    q_bits = list(range(width))
    ace_probs = np.empty(n_pow, dtype=np.float64)
    for outcome in range(n_pow):
        bits  = [(outcome >> b) & 1 for b in range(width)]
        ace_probs[outcome] = sum(s.prob_perm(q_bits, bits) for s in ace_sims) / n_inst
    for s in ace_sims: del s

    xeb_ace, hog_ace = calc_stats(ideal_probs, ace_probs, n_pow)

    return {
        "width":        width,
        "depth":        depth,
        "chi":          chi,
        "n_candidates": len(uniform_candidates),
        "n_ace_probs":  len(ace_probs),
        "seconds":      t_elapsed,
        "xeb_mps":      xeb_mps,
        "hog_mps":      hog_mps,
        "xeb_ace":      xeb_ace,
        "hog_ace":      hog_ace,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_mps_uniform_consensus.py [width] [depth] [sdrp=0] [chi=auto]")
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    chi   = int(sys.argv[4])   if len(sys.argv) > 4 else None
    result = bench_qrack(width, depth, sdrp, chi)
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
