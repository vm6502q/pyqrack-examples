# How good are Google's own "patch circuits" and "elided circuits" as a direct
# XEB approximation to full Sycamore circuits?
# (Are they better than the 2019 Sycamore hardware?)
#
# MPS amplitude estimation by (Anthropic) Claude; heavy-output sieve and
# overall structure by Dan Strano.

import math
import random
import statistics
import sys
from collections import Counter

import numpy as np
import jax.numpy as jnp
import quimb.tensor as tn
from qiskit import QuantumCircuit
from pyqrack import QrackSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_to_bitstring(integer, length):
    """LSB-first bitstring of `integer` padded to `length`."""
    return (bin(integer)[2:].zfill(length))[::-1]


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
    all_bits   = list(range(width))
    n_candidates = min(width ** 2, 1 << width)   # heavy-output candidates to retain/check
    chi          = min(width ** 3, 1 << width)   # MPS bond dimension

    # Build circuit in both Qiskit (for Qrack) and quimb MPS (for amplitude queries)
    # chi = width^3 is generous; chi = width^2 is faster and usually sufficient
    qc      = QuantumCircuit(width)
    mps_sim = tn.CircuitMPS(width, max_bond=chi, to_backend=jnp.array)
 
    for _ in range(depth):
        # Random single-qubit U3 gates
        for i in all_bits:
            th, ph, lm = (random.uniform(0, 2 * math.pi) for _ in range(3))
            qc.u(th, ph, lm, i)
            mps_sim.apply_gate('U3', th, ph, lm, i)
 
        # Random fully-connected CX layer (each qubit used at most once per layer)
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
 
    # Seed candidates: highest-probability permutation + shot samples
    counts = dict(Counter(sim.measure_shots(all_bits, n_candidates)))
    counts[sim.highest_prob_perm()] = n_candidates   # guarantee at least one heavy output
    # Sort by count descending so we query MPS for the most-sampled bitstrings first
    candidates = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    sim = None   # free GPU memory before MPS amplitude queries


    # -----------------------------------------------------------------------
    # MPS: estimate amplitudes for heavy-output candidates
    # -----------------------------------------------------------------------
    n_pow = 1 << width
    u_u   = 1.0 / n_pow

    ideal_probs_sparse = {}
    sum_probs = 0.0
    for key, hit_count in candidates:
        if len(ideal_probs_sparse) >= n_candidates and hit_count < 2:
            break
        bs  = _int_to_bitstring(key, width)
        amp = complex(mps_sim.amplitude(bs))
        p   = amp.real ** 2 + amp.imag ** 2
        if p <= u_u:
            continue
        ideal_probs_sparse[key] = p
        sum_probs += p

    if sum_probs == 0.0:
        sum_probs = 1.0   # degenerate guard; shouldn't happen in practice

    # Normalize sparse ideal probabilities
    ideal_probs_sparse = {k: v / sum_probs for k, v in ideal_probs_sparse.items()}

    # -----------------------------------------------------------------------
    # Qrack control: full probability vector for XEB denominator
    # -----------------------------------------------------------------------
    control = QrackSimulator(width)
    control.run_qiskit_circuit(qc, shots=0)
    control_probs = np.asarray(control.out_probs(), dtype=np.float64)

    return calc_stats(control_probs, ideal_probs_sparse, width)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, exp_probs_sparse, width):
    """
    Compute XEB, HOG probability, and L2 difference.

    Parameters
    ----------
    ideal_probs       : np.ndarray, shape (2^width,), full probability vector
    exp_probs_sparse  : dict {int -> float}, sparse experimental probabilities
                        (normalized; missing keys have probability 0)
    width             : int
    """
    n_pow    = 1 << width
    u_u      = 1.0 / n_pow
    # QV mixture model: 50% ideal + 50% uniform (matches the QV protocol)
    model    = 0.5
    mean_u   = u_u

    # Expand sparse experiment into dense array
    exp_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in exp_probs_sparse.items():
        exp_dense[k] = v
    # Apply QV mixture
    exp_mixed = (1.0 - model) * exp_dense + model * mean_u

    threshold = float(np.median(ideal_probs))

    # Vectorised statistics
    ideal_c    = ideal_probs - u_u
    exp_c      = exp_mixed   - u_u
    numer      = float(np.dot(ideal_c, exp_c))
    denom      = float(np.dot(ideal_c, ideal_c))
    xeb        = numer / denom if denom > 0.0 else 0.0
    hog_prob   = float(exp_mixed[ideal_probs > threshold].sum())
    l2_diff    = float(np.sqrt(np.sum((ideal_probs - exp_mixed) ** 2)))

    return {
        "qubits":   width,
        "xeb":      xeb,
        "hog_prob": hog_prob,
        "l2_diff":  l2_diff,
    }


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
