"""
Ergodic vs. non-ergodic fidelity benchmark for QrackAceBackend.

By (Anthropic) Claude

Motivation
----------
QrackAceBackend approximates a genuinely entangled multi-qubit state using
several "patch" simulators connected by an approximate ("shadow") coupling
mechanism, reconciled by probability-based corrective rotations (see
_correct() / _rotate_to_bloch() in qrack_ace_backend.py). That reconciliation
machinery is most confident exactly when a qubit's local Bloch vector is far
from maximally mixed -- i.e. when its Z-basis marginal is far from 50/50.

This motivates a physical hypothesis: since chaotic/thermalizing ("ergodic")
quantum dynamics drives local marginals toward flat (Porter-Thomas-like,
Eigenstate Thermalization Hypothesis) statistics, while non-ergodic dynamics
(integrable systems, among others) does not fully thermalize to that same
statistics, QrackAceBackend's approximation should do comparatively better
on non-ergodic circuits than on ergodic ones, at matched depth and gate
count.

Method
------
The kicked Ising model is used as the tunable ergodicity control -- a
standard benchmark in the OTOC/quantum-chaos literature (Larkin-Ovchinnikov;
widely used since, e.g. in mixed-field Ising chain chaos studies): a layer
of ZZ-coupling (implemented here as CZ on a random perfect matching of the
coupling graph each layer) alternates with a single-qubit "kick" layer
R_x(theta_x) [R_z(theta_z)]:

  - theta_z = 0   : pure transverse field. Maps to free fermions under a
                    Jordan-Wigner transform -- INTEGRABLE, does not
                    thermalize to full many-body chaos statistics.
  - theta_z != 0  : mixed field. Generically CHAOTIC / ergodic (thermalizes,
                    matches ETH).

theta_x is held fixed and identical between the two regimes, so the two
circuit families differ ONLY in ergodicity, not in overall gate count or
entangling-gate density.

For each regime, at matched depth/shots/topology, this script reports:
  - linear XEB (standard cross-entropy benchmark fidelity)
  - tail-restricted XEB (linear XEB after excluding the heaviest 25% of
    ideal-probability bitstrings -- see fidelity_battery.py for why this is
    a harder, more spoofing-resistant statistic than plain linear XEB)
  - cross-seam ZZ correlator error: mean |sampled - exact| connected <Z_i
    Z_j> correlator, for qubit pairs whose entangling gate has to cross the
    approximate patch boundary this backend introduces
  - mean marginal polarization: mean |P(qubit=1) - 0.5| over all qubits,
    from the EXACT reference -- a direct, independent check that theta_z is
    actually doing what it's supposed to (0 = fully flat/ergodic-looking
    locally, 0.5 = fully classical)

Caveat, worth knowing before reading results
---------------------------------------------
Linear XEB has high variance at SHALLOW depth: with few circuit layers, the
ideal distribution is dominated by a handful of very heavy bitstrings, and
a single lucky/unlucky sample can swing the estimate by an order of
magnitude. Prefer depth >= 6 for this reason (the default is 8). The
cross-seam ZZ correlator error does not share this instability and is the
more trustworthy statistic at any depth.

Usage
-----
    python3 ergodicity_benchmark.py [n_circuits] [depth] [shots]

Defaults: 20 circuits, depth 8, 300 shots per circuit. Runtime is
dominated by the exact reference (QrackSimulator) and ACE sampling, both
cheap at this qubit count (well under a minute total for the defaults on
an ordinary laptop).
"""

import math
import random
import sys

from pyqrack import QrackSimulator
from pyqrack.qrack_ace_backend import QrackAceBackend, Pauli

# 2 patches, near-equal size (4+4 qubits) -- small enough for an exact
# statevector reference, large enough to have real cross-patch coupling.
QC, LRC, LRR = 8, 1, 4

THETA_X = 0.9 * math.pi / 2          # generic transverse kick strength (both regimes)
THETA_Z_CHAOTIC = 0.8 * math.pi / 2  # generic longitudinal field (chaotic regime only)


def get_topology():
    ref = QrackAceBackend(QC, long_range_columns=LRC, long_range_rows=LRR, is_torus=True)
    homes = [ref._qubits[q][0][0] for q in range(QC)]
    cmap = ref.get_logical_coupling_map()
    cross_seam = sorted(set(tuple(sorted((a, c))) for (a, c) in cmap if homes[a] != homes[c]))
    return cmap, cross_seam


def gen_kicked_ising(depth, cmap, theta_z, seed):
    """theta_z == 0 -> integrable (non-ergodic); theta_z != 0 -> chaotic
    (ergodic). Coupling-layer matching is randomized per layer (seeded)
    to average over circuit instances while theta_x/theta_z stay fixed."""
    rng = random.Random(seed)
    ops = []
    for _ in range(depth):
        used = set()
        pairs = list(cmap)
        rng.shuffle(pairs)
        for (a, c) in pairs:
            if a in used or c in used:
                continue
            used.add(a)
            used.add(c)
            ops.append(("cz", a, c))
        for q in range(QC):
            ops.append(("rx", q, THETA_X))
            if theta_z != 0.0:
                ops.append(("rz", q, theta_z))
    return ops


def run_ideal(ops):
    sim = QrackSimulator(QC)
    for op in ops:
        if op[0] == "cz":
            _, a, c = op
            sim.mcz([a], c)
        elif op[0] == "rx":
            _, q, th = op
            sim.r(Pauli.PauliX, th, q)
        else:
            _, q, th = op
            sim.r(Pauli.PauliZ, th, q)
    return sim, sim.out_probs()


def exact_zz(sim, i, j):
    p00 = sim.prob_perm_rdm([i, j], [False, False])
    p01 = sim.prob_perm_rdm([i, j], [False, True])
    p10 = sim.prob_perm_rdm([i, j], [True, False])
    p11 = sim.prob_perm_rdm([i, j], [True, True])
    zz = p00 - p01 - p10 + p11
    zi = (p00 + p01) - (p10 + p11)
    zj = (p00 + p10) - (p01 + p11)
    return zz - zi * zj


def run_ace_sample(ops, config):
    b = QrackAceBackend(QC, **config)
    for op in ops:
        if op[0] == "cz":
            _, a, c = op
            b.cz(a, c)
        elif op[0] == "rx":
            _, q, th = op
            b.r(Pauli.PauliX, th, q)
        else:
            _, q, th = op
            b.r(Pauli.PauliZ, th, q)
    return b.m_all()


def linear_xeb(ideal_probs, samples):
    N = 1 << QC
    return N * sum(ideal_probs[s] for s in samples) / len(samples) - 1.0


def tail_restricted_xeb(ideal_probs, samples, exclude_frac=0.25):
    N = 1 << QC
    order = sorted(range(N), key=lambda x: -ideal_probs[x])
    heavy = set(order[: int(exclude_frac * N)])
    tail_samples = [s for s in samples if s not in heavy]
    if len(tail_samples) < 5:
        return None
    tail_size = N - len(heavy)
    tail_mass = sum(ideal_probs[x] for x in range(N) if x not in heavy)
    mean_p = sum(ideal_probs[s] for s in tail_samples) / len(tail_samples)
    return (tail_size / tail_mass) * mean_p - 1.0


def sampled_zz(samples, i, j):
    n = len(samples)
    s = zi = zj = 0.0
    for x in samples:
        bi = 1 if (x >> i) & 1 else -1
        bj = 1 if (x >> j) & 1 else -1
        s += bi * bj
        zi += bi
        zj += bj
    s /= n
    zi /= n
    zj /= n
    return s - zi * zj


def marginal_polarization(ideal_probs):
    """Mean |P(qubit=1) - 0.5| over all qubits, from the exact reference.
    0 = fully flat/ergodic-looking locally, 0.5 = fully classical. An
    independent sanity check that theta_z is actually controlling
    ergodicity in the intended direction, not a fidelity statistic itself."""
    N = 1 << QC
    total = 0.0
    for q in range(QC):
        p1 = sum(ideal_probs[x] for x in range(N) if (x >> q) & 1)
        total += abs(p1 - 0.5)
    return total / QC


def run_regime(label, theta_z, n_circuits, depth, shots):
    cmap, cross_seam = get_topology()
    config = dict(long_range_columns=LRC, long_range_rows=LRR, is_torus=True, is_error_detection=True)
    lin_fids, tail_fids, zz_errs, polarizations = [], [], [], []
    for ci in range(n_circuits):
        ops = gen_kicked_ising(depth, cmap, theta_z, seed=3000 + ci)
        sim, ideal_probs = run_ideal(ops)
        polarizations.append(marginal_polarization(ideal_probs))
        exact_corrs = {p: exact_zz(sim, *p) for p in cross_seam}

        random.seed(9000 + ci)
        samples = [run_ace_sample(ops, config) for _ in range(shots)]

        lin_fids.append(linear_xeb(ideal_probs, samples))
        tf = tail_restricted_xeb(ideal_probs, samples)
        if tf is not None:
            tail_fids.append(tf)
        sampled_corrs = {p: sampled_zz(samples, *p) for p in cross_seam}
        zz_errs.append(sum(abs(sampled_corrs[p] - exact_corrs[p]) for p in cross_seam) / len(cross_seam))

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"[{label}] theta_z={theta_z:.4f}  mean marginal polarization={avg(polarizations):.4f}  "
          f"(0=ergodic/flat, 0.5=classical)\n"
          f"    linear XEB      = {avg(lin_fids):+.4f}\n"
          f"    tail XEB        = {avg(tail_fids):+.4f}\n"
          f"    seam |ZZ error| = {avg(zz_errs):.4f}\n")
    return dict(linear=avg(lin_fids), tail=avg(tail_fids), zz_err=avg(zz_errs), polarization=avg(polarizations))


def main():
    n_circuits = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    if depth < 6:
        print(f"WARNING: depth={depth} < 6 -- linear XEB is high-variance at shallow depth "
              "(see module docstring); prefer the seam |ZZ error| statistic for depth < 6.\n")

    print("=" * 70)
    print("NON-ERGODIC regime: pure transverse field (theta_z=0), integrable/free-fermion")
    print("=" * 70)
    r_non = run_regime("non-ergodic", 0.0, n_circuits, depth, shots)

    print("=" * 70)
    print(f"ERGODIC regime: mixed field (theta_z={THETA_Z_CHAOTIC:.4f}), chaotic/thermalizing")
    print("=" * 70)
    r_erg = run_regime("ergodic", THETA_Z_CHAOTIC, n_circuits, depth, shots)

    print("=" * 70)
    print(f"SUMMARY (non-ergodic vs ergodic, {n_circuits} circuits, depth={depth}, {shots} shots/circuit)")
    print("=" * 70)
    for k in ("linear", "tail", "zz_err", "polarization"):
        print(f"  {k:12s}:  non-ergodic {r_non[k]:+.4f}   vs   ergodic {r_erg[k]:+.4f}")


if __name__ == "__main__":
    main()
