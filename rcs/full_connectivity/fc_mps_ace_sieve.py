# Fully-connected RCS: MPS sieve + ACE consensus mixture.
#
# For each sieved candidate, probability estimate is 50% MPS amplitude
# squared + 50% ACE prob_perm average. This allows MPS to run at lower
# chi (width**2) while ACE fills in approximation gaps.
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

def calc_stats_sparse(ideal_probs, exp_probs_sparse, n_pow):
    u_u = 1.0 / n_pow
    h_probs, l_probs = exp_probs_sparse
    # Heavy tail: normalized to sum 1, placed above uniform.
    h_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in h_probs.items():
        h_dense[k] = v
    # Light tail: normalized to sum 1 (stored positive), placed below uniform.
    # Concatenate with heavy tail so it sums to 0.
    # l_dense encodes the shape of the light distribution.
    l_dense = h_dense.copy()
    for k, v in l_probs.items():
        l_dense[k] = v
    # Final distribution:
    #   50% heavy tail (sums to 1, so contributes 0.5 total mass)
    #   50% mean field modulated by light tail:
    #       u_u * (1 - l_dense) sums to u_u*(n_pow - 1) = 1
    #       so u_u*(l_dense + 1)/2 contributes ~0.5 total mass
    # Combined exp_mixed sums to ~1 and is normalized.
    exp_mixed = (h_dense + u_u * (1.0 - l_dense)) / 2.0
    p_c   = ideal_probs - u_u
    q_c   = exp_mixed   - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_mixed[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog


def route_heavy_light(prob_dict, u_u):
    # Work in the centered basis (p - u_u).
    # Heavy: centered weight > 0, normalized to sum 1.
    # Light: centered weight < 0, normalized so abs values sum to 1
    #        (returned as negative values summing to -1).
    # Returns (heavy, light) tuple for use in calc_stats_sparse.
    heavy_raw = {}; light_raw = {}
    for outcome, p in prob_dict.items():
        c = p - u_u
        if c > 0:
            heavy_raw[outcome] = c
        elif c < 0:
            light_raw[outcome] = c
    s_h = sum(heavy_raw.values())
    heavy = {k: v/s_h for k, v in heavy_raw.items()} if s_h > 0 else {}
    s_l = sum(light_raw.values())
    light = {k: -v/s_l for k, v in light_raw.items()} if s_l < 0 else {}
    return (heavy, light)


def bench_qrack(width, depth, sdrp=0.0, chi=None):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 2
    n_pow        = 1 << width
    n_candidates = min(width ** 2, int(math.sqrt(n_pow) + 0.5))
    u_u          = 1.0 / n_pow

    if chi is None:
        chi = min(width ** 2, int(math.sqrt(n_pow) + 0.5))

    # -----------------------------------------------------------------------
    # Build circuit once in Qiskit + quimb MPS from same RNG
    # -----------------------------------------------------------------------
    t_circ = time.perf_counter()
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

    t_ideal = time.perf_counter()
    print(f"mps_circuit_seconds: {t_ideal - t_circ}")

    # -----------------------------------------------------------------------
    # ACE consensus instances (two orthogonal coupler orderings)
    # -----------------------------------------------------------------------
    def _order_pairs(pairs, inst):
        k = len(pairs)
        if k == 0 or inst == 0: return pairs
        return [pairs[i] for i in range(1, k, 2)] +                [pairs[i] for i in range(0, k, 2)]

    ace_sims = []
    for inst in range(n_inst):
        random.setstate(rng_state)
        sim = QrackSimulator(width)
        sim.set_ace_max_qb((width + 1) >> 1)
        sim.run_qiskit_circuit(qc, shots=0)
        ace_sims.append(sim)

    # -----------------------------------------------------------------------
    # Ideal ground truth
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    random.setstate(rng_state)
    sim_ideal.run_qiskit_circuit(qc, shots=0)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_start = time.perf_counter()
    print(f"ideal_seconds: {t_start - t_ideal}")

    # -----------------------------------------------------------------------
    # Prefix-maximizing MPS sieve via trie.
    # Fix suffix_bits = floor(log2(n_candidates)) suffix qubits as free;
    # draw a random prefix over the remaining prefix_bits qubits.
    # Enumerate all 2^suffix_bits suffixes — trie shares the prefix
    # contraction and fans out only at the suffix level.
    # -----------------------------------------------------------------------
    suffix_bits = int(math.log2(n_candidates))
    prefix_bits = width - suffix_bits
    prefix_val  = random.randrange(1 << prefix_bits)

    uniform_candidates = [prefix_val | (suffix << prefix_bits)
                          for suffix in range(1 << suffix_bits)]
    candidate_tuples   = [_int_to_bittuple(idx, width)
                          for idx in uniform_candidates]
    amp_map = batch_amplitudes_trie(mps_sim.psi, candidate_tuples)

    q_bits = list(range(width))
    mixed_probs = {}
    for idx, bs_tup in zip(uniform_candidates, candidate_tuples):
        # MPS amplitude squared
        amp   = amp_map.get(bs_tup, 0.0+0.0j)
        p_mps = amp.real**2 + amp.imag**2
        # ACE prob_perm average
        bits  = [(idx >> b) & 1 for b in range(width)]
        p_ace = sum(s.prob_perm(q_bits, bits) for s in ace_sims) / n_inst
        # 50/50 mixture
        p = 0.5 * p_mps + 0.5 * p_ace
        if p > 0:
            mixed_probs[int(idx)] = p
    for s in ace_sims: del s

    mps_combined = route_heavy_light(mixed_probs, u_u)
    xeb_mps, hog_mps = calc_stats_sparse(ideal_probs, mps_combined, n_pow)

    t_elapsed = time.perf_counter() - t_start

    return {
        "width":         width,
        "depth":         depth,
        "chi":           chi,
        "n_candidates":  len(uniform_candidates),
        "sieve_seconds": t_elapsed,
        "xeb_mix":       xeb_mps,
        "hog_mix":       hog_mps,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_mps_ace_sieve.py [width] [depth] [sdrp=0] [chi=width**2]")
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
