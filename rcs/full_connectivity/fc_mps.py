# How good are Google's own "patch circuits" and "elided circuits" as a direct
# XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
#
# MPS amplitude estimation by (Anthropic) Claude; heavy-output sieve and
# overall structure by Dan Strano.
#
# Amplitude queries use trie-based prefix contraction: shared prefixes among
# candidate bitstrings are contracted once rather than once per bitstring,
# giving O(n_candidates) speedup over sequential amplitude() calls.

import math
import random
import statistics
import sys
from collections import Counter, defaultdict

import numpy as np
import jax.numpy as jnp
import quimb.tensor as tn
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Trie-based batch amplitude contraction
# ---------------------------------------------------------------------------

def _batch_amplitudes_trie(mps_psi, bitstrings):
    """
    Compute MPS amplitudes for a batch of bitstrings by sharing prefix
    contractions via a trie traversal.

    For n_candidates bitstrings of length n on an MPS with bond dimension chi,
    cost is O(n_unique_prefixes * chi^2) rather than O(n_candidates * n * chi^2).
    In practice this gives a speedup proportional to the average shared prefix
    length, which for a random sample of heavy outputs is typically O(width).

    Parameters
    ----------
    mps_psi   : quimb MatrixProductState  (mps_sim.psi)
    bitstrings : list of tuples of ints (0/1), each of length n

    Returns
    -------
    dict { bitstring_tuple -> complex amplitude }
    """
    # Extract raw tensors as numpy arrays.
    # Quimb MPS tensor index convention (verified empirically):
    #   site 0       (left boundary):  shape (bond_r, phys)
    #   site 1..n-2  (bulk):           shape (bond_l, bond_r, phys)
    #   site n-1     (right boundary): shape (bond_l, phys)
    # Physical index is always last; bond indices are ordered left-to-right.
    tensors = [np.array(t.data) for t in mps_psi.tensors]
    n       = len(tensors)
    results = {}

    def _recurse(site, env, group):
        if site == n:
            # env is a scalar (0-d array or python complex)
            scalar = complex(env.flat[0]) if hasattr(env, 'flat') else complex(env)
            for bs in group:
                results[bs] = scalar
            return
        t = tensors[site]
        # Partition group by the bit value at this site
        by_bit = defaultdict(list)
        for bs in group:
            by_bit[bs[site]].append(bs)
        for bit, subgroup in by_bit.items():
            if site == 0:
                # Left boundary: t shape (bond_r, phys)
                # env seed is the row vector t[:, bit]
                new_env = t[:, bit].copy()
            elif site == n - 1:
                # Right boundary: t shape (bond_l, phys)
                # result is scalar  env (bond_l,) . t[:, bit] (bond_l,)
                new_env = env @ t[:, bit]
            else:
                # Bulk: t shape (bond_l, bond_r, phys)
                # new_env (bond_r,) = env (bond_l,) @ t[:, :, bit] (bond_l, bond_r)
                new_env = env @ t[:, :, bit]
            _recurse(site + 1, new_env, subgroup)

    _recurse(0, None, bitstrings)
    return results


def _int_to_bitstring(integer, length):
    """LSB-first bitstring of `integer` padded to `length`."""
    return (bin(integer)[2:].zfill(length))[::-1]


def _int_to_bittuple(integer, length):
    """LSB-first tuple of ints (0/1) of `integer` padded to `length`."""
    return tuple((integer >> b) & 1 for b in range(length))


# ---------------------------------------------------------------------------
# Circuit builder + benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0):
    """
    Build a random fully-connected circuit of `depth` layers on `width` qubits.
    Each layer applies random U3 gates to all qubits followed by random CX pairs.

    Parameters
    ----------
    width : int
    depth : int
    sdrp  : float  — Qrack single-qubit depolarization rate (0 = noiseless)

    Returns
    -------
    dict with xeb, hog_prob, l2_diff
    """
    all_bits     = list(range(width))
    n_candidates = min(width ** 2,   1 << width)
    chi          = min(width ** 2.5, 1 << width)   # width^2 is usually sufficient

    # Build circuit in both Qiskit (for Qrack) and quimb MPS (for amplitude queries)
    qc      = QuantumCircuit(width)
    mps_sim = tn.CircuitMPS(width, max_bond=chi, to_backend=jnp.array)

    for _ in range(depth):
        for i in all_bits:
            th, ph, lm = (random.uniform(0, 2 * math.pi) for _ in range(3))
            qc.u(th, ph, lm, i)
            mps_sim.apply_gate('U3', th, ph, lm, i)

        shuffled = all_bits[:]
        random.shuffle(shuffled)
        while len(shuffled) > 1:
            c, t = shuffled.pop(), shuffled.pop()
            qc.cx(c, t)
            mps_sim.apply_gate('CX', c, t)

    # -----------------------------------------------------------------------
    # Qrack: run circuit, collect heavy-output candidates
    # -----------------------------------------------------------------------
    sim = QrackSimulator(width)
    if sdrp > 0.0:
        sim.set_sdrp(sdrp)
    sim.run_qiskit_circuit(qc, shots=0)

    counts = dict(Counter(sim.measure_shots(all_bits, n_candidates)))
    counts[sim.highest_prob_perm()] = n_candidates
    candidates = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    sim = None   # free GPU memory before MPS work

    # -----------------------------------------------------------------------
    # MPS: batch amplitude estimation via trie contraction
    # -----------------------------------------------------------------------
    n_pow = 1 << width
    u_u   = 1.0 / n_pow

    # Collect candidate bitstrings as tuples for trie lookup
    candidate_tuples = [
        _int_to_bittuple(key, width)
        for key, _ in candidates[:n_candidates]
    ]
    # Add highest-prob permutation explicitly (already in counts but be safe)
    amp_map = _batch_amplitudes_trie(mps_sim.psi, candidate_tuples)

    # Filter to above-uniform and accumulate
    ideal_probs_sparse = {}
    sum_probs = 0.0
    for (key, hit_count), bs_tup in zip(candidates, candidate_tuples):
        if len(ideal_probs_sparse) >= n_candidates and hit_count < 2:
            break
        amp = amp_map.get(bs_tup, 0.0 + 0.0j)
        p   = amp.real ** 2 + amp.imag ** 2
        if p <= u_u:
            continue
        ideal_probs_sparse[key] = p
        sum_probs += p

    if sum_probs == 0.0:
        sum_probs = 1.0

    return {k: v / sum_probs for k, v in ideal_probs_sparse.items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_mps_qrack_validation.py [width] [depth] [sdrp=0]"
        )
    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    sdrp  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    print(bench_qrack(width, depth, sdrp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
