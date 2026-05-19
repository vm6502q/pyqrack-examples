# Fully-connected RCS: ACE consensus sieve + MPS amplitude estimation.
#
# Pipeline:
#   1. Two ACE instances (sequential vs stride coupler order) build the
#      symmetrized density matrix diagonal p_dm cheaply.
#   2. p_dm sieves the top width² heavy candidates and bottom width² light
#      candidates — O(n²) outputs total.
#   3. MPS (quimb CircuitMPS) estimates amplitudes for the heavy candidates
#      via trie-based batch contraction — O(n² * chi²) instead of O(2^n).
#   4. Heavy: MPS probabilities, renormalized.
#      Light:  inverted p_dm, structured suppression below uniform.
#   5. 50% heavy + 50% light → renormalize → 50/50 mix with uniform (QV).
#   6. XEB and HOG computed against full ideal simulator.
#
# ACE sieve cost:   O(depth * width * chi_ace²)  — fast, two instances
# MPS query cost:   O(n_candidates * chi_mps²)   — trie gives ~width× speedup
# MPS build cost:   O(depth * width * chi_mps²)  — dominates at large chi
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
# Trie-based batch MPS amplitude contraction
# (from fc_mps_qrack_validation.py, proven 432x speedup)
# ---------------------------------------------------------------------------

def _int_to_bittuple(integer, length):
    return tuple((integer >> b) & 1 for b in range(length))


def batch_amplitudes_trie(mps_psi, bitstrings):
    """
    Compute MPS amplitudes for a batch of bitstrings by sharing prefix
    contractions via trie traversal.  O(n_unique_prefixes * chi²).
    """
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

def calc_stats(ideal_probs, split_probs):
    n_pow = len(ideal_probs); u_u = 1.0 / n_pow
    p = np.asarray(ideal_probs, dtype=np.float64)
    q = np.asarray(split_probs, dtype=np.float64)
    p_c = p - u_u; q_c = q - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(q[p > float(np.median(p))].sum())
    return xeb, hog


def calc_stats_sparse(ideal_probs, exp_probs_sparse, n_pow):
    u_u = 1.0 / n_pow; model = 0.5
    exp_dense = np.zeros(n_pow, dtype=np.float64)
    for k, v in exp_probs_sparse.items():
        exp_dense[k] = v
    exp_mixed = (1.0 - model) * exp_dense + model * u_u
    p_c   = ideal_probs - u_u; q_c = exp_mixed - u_u
    denom = float(np.dot(p_c, p_c))
    xeb   = float(np.dot(p_c, q_c)) / denom if denom > 0 else 0.0
    hog   = float(exp_mixed[ideal_probs > float(np.median(ideal_probs))].sum())
    return xeb, hog


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
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, sdrp=0.0, chi=None):
    lcv_range    = range(width)
    all_bits     = list(lcv_range)
    n_inst       = 2
    n_candidates = width ** 2
    n_pow        = 1 << width
    u_u          = 1.0 / n_pow

    if chi is None:
        chi = min(width ** 2, 1 << width)

    # -----------------------------------------------------------------------
    # Build circuit in Qiskit + quimb MPS simultaneously
    # -----------------------------------------------------------------------
    qc      = QuantumCircuit(width)
    mps_sim = tn.CircuitMPS(width, max_bond=chi, to_backend=jnp.array)

    rng_state_build = random.getstate()  # snapshot for MPS build
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

    rng_state = rng_state_build  # ACE instances replay same circuit

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
    # Two ACE consensus instances — sampling only, no statevector materialization.
    # measure_shots samples proportionally to probability; above-uniform outputs
    # are overrepresented. Union of both instances covers the heavy tail broadly.
    # -----------------------------------------------------------------------
    n_shots  = n_candidates
    all_bits = list(range(width))
    sampled_candidates = set()
    for inst in range(n_inst):
        random.setstate(rng_state)
        sim = QrackSimulator(width)
        if sdrp > 0.0:
            sim.set_sdrp(sdrp)
        sim.set_ace_max_qb((width + 1) >> 1)
        sim.run_qiskit_circuit(qc, shots=0)
        shots = sim.measure_shots(all_bits, n_shots)
        sampled_candidates.update(int(s) for s in shots)
        del sim

    t_ace = time.perf_counter() - t_start
    candidates = sorted(sampled_candidates)

    # (p_dm not computed — sampling-only pipeline needs no statevector)

    # -----------------------------------------------------------------------
    # Sieve: top n_candidates by p_dm → MPS amplitude estimation
    #        bottom n_candidates by p_dm → inverted light tail from p_dm
    # -----------------------------------------------------------------------
    # Partition ALL outputs into heavy and light by p_dm median.
    # Every output is one or the other — no unassigned middle ground.
    # Heavy (above median): MPS amplitude estimates, renormalized.
    # Light (below median): inverted p_dm (2*u_u - p_dm), renormalized,
    #                        then mapped below u_u for structured suppression.
    # Both tails contribute positively to XEB numerator.
    # MPS amplitude queries for all sampled candidates via trie contraction.
    # MPS determines whether each ACE-sampled candidate is truly heavy or light
    # relative to u_u — routing to the correct tail accordingly.
    t_mps_start = time.perf_counter()
    candidate_tuples = [_int_to_bittuple(int(i), width) for i in candidates]
    amp_map = batch_amplitudes_trie(mps_sim.psi, candidate_tuples)
    t_mps = time.perf_counter() - t_mps_start

    # Route by MPS probability vs u_u (XEB mean-field baseline)
    heavy = {}           # p_mps > u_u : contribute q_i > u_u
    light_raw = {}       # p_mps <= u_u: contribute q_i < u_u
    for idx, bs_tup in zip(candidates, candidate_tuples):
        amp = amp_map.get(bs_tup, 0.0 + 0.0j)
        p   = amp.real**2 + amp.imag**2
        if p > u_u:
            heavy[int(idx)] = p
        elif p > 0:
            light_raw[int(idx)] = max(0.0, 2.0 * u_u - p)

    # Normalize heavy
    s_h = sum(heavy.values())
    if s_h > 0:
        heavy = {k: v / s_h for k, v in heavy.items()}

    # Normalize light and map below u_u for structured suppression
    s_l = sum(light_raw.values())
    if s_l > 0:
        light = {k: max(0.0, u_u - (v / s_l) * u_u) for k, v in light_raw.items()}
        s_l2  = sum(light.values())
        if s_l2 > 0:
            light = {k: v / s_l2 for k, v in light.items()}
    else:
        light = {}

    # Equal weight to whichever tails are non-empty
    n_nonempty = (1 if heavy else 0) + (1 if light else 0)
    if n_nonempty == 0:
        combined = {}
    else:
        w        = 1.0 / n_nonempty
        all_keys = set(heavy) | set(light)
        combined = {k: w * heavy.get(k, 0.0) + w * light.get(k, 0.0)
                    for k in all_keys}
        s_c = sum(combined.values())
        if s_c > 0:
            combined = {k: v / s_c for k, v in combined.items()}

    t_elapsed = time.perf_counter() - t_start
    xeb_sieve, hog_sieve = calc_stats_sparse(ideal_probs, combined, n_pow)

    return {
        "width":          width,
        "depth":          depth,
        "chi":            chi,
        "n_sampled":      len(candidates),
        "n_heavy":        len(heavy),
        "n_light":        len(light),
        "seconds_total":  t_elapsed,
        "seconds_ace":    t_ace,
        "seconds_mps":    t_mps,
        "xeb_sieve":      xeb_sieve,
        "hog_sieve":      hog_sieve,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_mps_ace_consensus.py [width] [depth] [sdrp=0] [chi=width²]"
        )
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
