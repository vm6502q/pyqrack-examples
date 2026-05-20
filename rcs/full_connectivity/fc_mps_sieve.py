# Fully-connected RCS: uniform-random MPS sieve.
#
# Prefix-maximizing MPS sieve:
#   Pick candidates that maximally share trie prefixes, minimizing
#   redundant computation. Enumerate all suffixes beneath a random prefix
#   so the trie evaluates the shared prefix once and fans out only at
#   the suffix level. Route by p_mps vs u_u: heavy above, light below.
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
# Helper
# ---------------------------------------------------------------------------

def _int_to_bittuple(integer, length):
    return tuple((integer >> b) & 1 for b in range(length))


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
        chi = int(math.sqrt(n_pow) + 0.5)

    # -----------------------------------------------------------------------
    # Prefix-fixing via post-hoc MPS projection.
    #
    # Build the full circuit MPS on all `width` qubits (forward pass),
    # then project out the prefix qubits by contracting each fixed site
    # tensor against a computational basis bra vector [1,0] or [0,1].
    # This collapses those sites out of the tensor network, leaving a
    # reduced MPS over just the `suffix_bits` free qubits — equivalent
    # to running the adjoint circuit from the fixed output state, but
    # expressed directly in quimb's tensor network language without
    # manually reversing gates.
    #
    # The projection is O(prefix_bits * chi^2) — negligible vs construction.
    # -----------------------------------------------------------------------
    suffix_bits = int(math.log2(n_candidates))
    prefix_bits = width - suffix_bits
    n_free      = suffix_bits

    # Draw a random prefix value
    prefix_val  = random.randrange(1 << prefix_bits)

    t_circ    = time.perf_counter()
    qc        = QuantumCircuit(width)
    mps_sim   = tn.CircuitMPS(width, max_bond=chi, to_backend=jnp.array)

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

    # Project out the prefix qubits by contracting each fixed site against
    # its basis vector. quimb MPS site indices are 'k0', 'k1', ..., 'k{n-1}'.
    # We contract site q against bra vector e_{b} = [1-b, b] (little-endian).
    psi = mps_sim.psi.copy()
    for q in range(prefix_bits):
        b     = (prefix_val >> q) & 1
        bra_v = jnp.array([1.0 - b, float(b)], dtype=jnp.complex128)
        site_ind = f'k{q}'
        psi.isel_({site_ind: b})   # project site q onto basis state b in-place

    # psi is now a tensor network over the free suffix qubits only.
    # Re-index sites to 0..n_free-1 for the trie.
    for i, q in enumerate(range(prefix_bits, width)):
        psi.reindex_({f'k{q}': f'k{i}'})

    t_ideal = time.perf_counter()
    print(f"mps_circuit_seconds: {t_ideal - t_circ}")

    # -----------------------------------------------------------------------
    # Ideal ground truth (full width, Qrack)
    # -----------------------------------------------------------------------
    sim_ideal = QrackSimulator(width)
    random.setstate(rng_state)
    sim_ideal.run_qiskit_circuit(qc, shots=0)
    ideal_probs = np.asarray(sim_ideal.out_probs(), dtype=np.float64)
    del sim_ideal

    t_start = time.perf_counter()
    print(f"ideal_seconds: {t_start - t_ideal}")

    # -----------------------------------------------------------------------
    # Sieve: enumerate all 2^suffix_bits suffixes under the fixed prefix.
    # -----------------------------------------------------------------------
    uniform_candidates = [prefix_val | (suffix << prefix_bits)
                          for suffix in range(1 << suffix_bits)]
    candidate_tuples   = [_int_to_bittuple(suffix, n_free)
                          for suffix in range(1 << suffix_bits)]
    # Densify the projected MPS over the free qubits.
    # n_free = suffix_bits is small (log2 of n_candidates), so this is cheap.
    # isel_ leaves irregular bond/physical index structure; to_dense is clean.
    free_inds = [f'k{i}' for i in range(n_free)]
    psi_dense = np.array(psi.to_dense(free_inds)).ravel()

    mps_probs = {}
    for idx, bs_tup in zip(uniform_candidates, candidate_tuples):
        suffix = sum(bs_tup[i] << i for i in range(n_free))
        amp = complex(psi_dense[suffix])
        p   = amp.real**2 + amp.imag**2
        if p > 0:
            mps_probs[int(idx)] = p

    mps_combined = route_heavy_light(mps_probs, u_u)
    xeb_mps, hog_mps = calc_stats_sparse(ideal_probs, mps_combined, n_pow)

    t_elapsed = time.perf_counter() - t_start

    return {
        "width":         width,
        "depth":         depth,
        "chi":           chi,
        "n_candidates":  len(uniform_candidates),
        "sieve_seconds": t_elapsed,
        "xeb_mps":       xeb_mps,
        "hog_mps":       hog_mps,
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
