"""
Fidelity battery for QrackAceBackend, beyond plain linear XEB.

Motivation
----------
Linear XEB (F = 2^n * <P_ideal(sampled bitstring)> - 1) is known to be
"spoofable": there exist purely classical algorithms that post a high
linear-XEB score without sampling from anything close to the true output
distribution, by exploiting the fact that linear XEB only requires weak
correlation with the *heaviest* bitstrings under the ideal distribution
(Barak, Chou, Gao, Jain, Sharan 2020; Gao et al. on classical spoofing of
shallow/noisy random circuits). A high linear-XEB number is therefore
necessary-ish evidence of fidelity, but not sufficient on its own.

This script runs three additional, harder-to-game checks against the same
random circuits, comparing QrackAceBackend's sampled output to an exact
statevector reference (QrackSimulator):

  1. Hellinger fidelity (squared Bhattacharyya coefficient) between the
     ACE-sampled empirical distribution and the exact ideal distribution.
     Unlike linear XEB, this requires the *entire* distribution shape to
     match reasonably well -- not just the heaviest few outcomes -- which
     is exactly the property known spoofing constructions fail to have
     even while acing linear XEB.

  2. Tail-restricted linear XEB: linear XEB recomputed after excluding the
     heaviest bitstrings (by ideal probability) from both the sample set
     and the renormalized ideal distribution. A classical heavy-hitter
     search would show little to no correlation once you remove the peaks
     it was built to find; a genuine (even lossy) simulator should retain
     positive correlation into the tail, just at reduced magnitude.

  3. Cross-seam ZZ correlators: for pairs of qubits that live in DIFFERENT
     patches (i.e. whose entangling gate had to cross the approximate
     patch boundary this backend introduces), compare the sampled
     connected correlator <Z_i Z_j> - <Z_i><Z_j> against the exact value.
     This is a fine-grained, many-constraint, physically-interpretable
     check specific to this architecture's own approximation mechanism
     (the same coupling machinery debugged in _apply_coupling) -- it's a
     much harder target to game than a single scalar XEB number, since it
     must hold up pairwise, across many different qubit pairs and circuit
     instances, not just in aggregate.

None of this constitutes a formal hardness/anti-spoofing proof. It's a
battery of empirical checks that a pure heavy-bitstring spoofing
construction would be expected to fail, run at a scale this sandbox can
actually execute (statevector reference up to ~8-10 qubits, a few hundred
shots per circuit for the distribution-sensitive metrics). Finite-sample
noise is real at these shot counts -- treat single-circuit numbers as
noisy and look at the trend across circuits/depths, not any one value.
"""

import math
import random
import sys
from collections import Counter

from pyqrack import QrackSimulator
from pyqrack.qrack_ace_backend import QrackAceBackend

# 2 patches, near-equal size (4+4 qubits), confirmed via home-patch lookup.
QC, LRC, LRR = 8, 1, 4


def get_topology():
    ref = QrackAceBackend(QC, long_range_columns=LRC, long_range_rows=LRR, is_torus=True)
    homes = [ref._qubits[q][0][0] for q in range(QC)]
    cmap = ref.get_logical_coupling_map()
    cross_seam = sorted(set(tuple(sorted((a, c))) for (a, c) in cmap if homes[a] != homes[c]))
    intra = sorted(set(tuple(sorted((a, c))) for (a, c) in cmap if homes[a] == homes[c]))
    return cmap, cross_seam, intra


def gen_circuit(n, depth, cmap, seed):
    rng = random.Random(seed)
    ops = []
    for _ in range(depth):
        for q in range(n):
            ops.append(("u", q, rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)))
        used = set()
        pairs = list(cmap_global)
        rng.shuffle(pairs)
        for (a, c) in pairs:
            if a in used or c in used:
                continue
            used.add(a)
            used.add(c)
            ops.append(("cx", a, c))
    return ops


def run_ideal(n, ops):
    sim = QrackSimulator(n)
    for op in ops:
        if op[0] == "u":
            _, q, th, ph, lam = op
            sim.u(q, th, ph, lam)
        else:
            _, c, t = op
            sim.mcx([c], t)
    probs = sim.out_probs()
    return sim, probs


def exact_zz(sim, i, j):
    # Exact <Z_i Z_j>, <Z_i>, <Z_j> from the full statevector via prob_perm-
    # style joint/marginal reads (no sampling noise).
    p00 = sim.prob_perm_rdm([i, j], [False, False])
    p01 = sim.prob_perm_rdm([i, j], [False, True])
    p10 = sim.prob_perm_rdm([i, j], [True, False])
    p11 = sim.prob_perm_rdm([i, j], [True, True])
    zz = p00 - p01 - p10 + p11
    zi = (p00 + p01) - (p10 + p11)
    zj = (p00 + p10) - (p01 + p11)
    return zz - zi * zj


def run_ace_sample(n, ops, config):
    b = QrackAceBackend(n, **config)
    for op in ops:
        if op[0] == "u":
            _, q, th, ph, lam = op
            b.u(q, th, ph, lam)
        else:
            _, c, t = op
            b.cx(c, t)
    return b.m_all()


def linear_xeb(ideal_probs, samples, n):
    N = 1 << n
    return N * sum(ideal_probs[s] for s in samples) / len(samples) - 1.0


def hellinger_fidelity(ideal_probs, samples, n):
    N = 1 << n
    counts = Counter(samples)
    total = len(samples)
    bc = 0.0
    for x in range(N):
        qhat = counts.get(x, 0) / total
        if qhat > 0:
            bc += math.sqrt(ideal_probs[x] * qhat)
    return bc * bc  # squared Bhattacharyya coefficient ("Hellinger fidelity")


def hellinger_excess_ratio(hf, hf_base):
    # Raw excess (hf - hf_base) gets squeezed toward zero as the ideal
    # distribution scrambles: the perfect-fidelity ceiling is ALWAYS
    # exactly 1 (BC(P,P) = sum_x P(x) = 1, regardless of P's shape), but
    # the uniform-guessing floor BC(uniform,P) = sum_x sqrt(P(x)/N) rises
    # toward that SAME ceiling as P approaches flat (Cauchy-Schwarz:
    # equality iff P is uniform). So the raw difference isn't
    # depth-comparable on its own -- a shrinking headroom will squeeze any
    # genuine signal toward zero even with no change in how well ACE is
    # actually doing. Reframe in terms of Hellinger DISTANCE instead
    # (H = sqrt(1 - BC)): distance_ACE / distance_uniform is a ratio, not
    # a difference, so it isn't mechanically squeezed by a shrinking
    # headroom the way the raw excess is. ratio < 1 means ACE is closer to
    # the ideal distribution than uniform guessing; ratio > 1 means
    # farther. It's still an estimate from finite-shot plug-in fidelities,
    # so it inherits their noise -- but not the headroom-compression
    # artifact.
    dist_ace = math.sqrt(max(0.0, 1.0 - hf))
    dist_base = math.sqrt(max(0.0, 1.0 - hf_base))
    if dist_base < 1e-9:
        return float("nan")
    return dist_ace / dist_base


def uniform_baseline_hellinger(ideal_probs, n, n_shots, seed):
    # Raw Hellinger fidelity is NOT zero-centered the way linear XEB is: a
    # purely uniform-random sampler already scores a nontrivial positive
    # value against a concentrated (Porter-Thomas-like) ideal distribution
    # (by Cauchy-Schwarz, BC(uniform, P) = sum_x sqrt(P(x)/N) <= 1, with
    # equality iff P itself is uniform -- for a generic random-circuit P
    # it sits well above 0). So a raw ACE Hellinger number means little on
    # its own; report it alongside this same-shot-count uniform-sampler
    # baseline computed against the identical ideal distribution, and look
    # at the EXCESS.
    rng = random.Random(seed)
    N = 1 << n
    samples = [rng.randrange(N) for _ in range(n_shots)]
    return hellinger_fidelity(ideal_probs, samples, n)


def tail_restricted_xeb(ideal_probs, samples, n, exclude_frac=0.25):
    N = 1 << n
    order = sorted(range(N), key=lambda x: -ideal_probs[x])
    heavy = set(order[: int(exclude_frac * N)])
    tail_samples = [s for s in samples if s not in heavy]
    if len(tail_samples) < 5:
        return None, 0.0
    tail_size = N - len(heavy)
    tail_mass = sum(ideal_probs[x] for x in range(N) if x not in heavy)
    mean_p = sum(ideal_probs[s] for s in tail_samples) / len(tail_samples)
    fidelity = (tail_size / tail_mass) * mean_p - 1.0
    retained_frac = len(tail_samples) / len(samples)
    return fidelity, retained_frac


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


cmap_global = None  # set in main() before gen_circuit is called


def run_battery(n_circuits, depth, shots, config, tag, exclude_frac=0.25):
    cmap, cross_seam, _ = get_topology()
    global cmap_global
    cmap_global = cmap

    lin_fids, hell_fids, hell_baselines, hell_ratios, tail_fids, tail_retained = [], [], [], [], [], []
    zz_errs = []

    for ci in range(n_circuits):
        ops = gen_circuit(QC, depth, cmap, seed=2000 + ci)
        sim, ideal_probs = run_ideal(QC, ops)
        exact_corrs = {pair: exact_zz(sim, *pair) for pair in cross_seam}

        random.seed(9000 + ci)
        samples = [run_ace_sample(QC, ops, config) for _ in range(shots)]

        lf = linear_xeb(ideal_probs, samples, QC)
        hf = hellinger_fidelity(ideal_probs, samples, QC)
        hf_base = uniform_baseline_hellinger(ideal_probs, QC, shots, seed=70000 + ci)
        hr = hellinger_excess_ratio(hf, hf_base)
        tf, retained = tail_restricted_xeb(ideal_probs, samples, QC, exclude_frac)

        sampled_corrs = {pair: sampled_zz(samples, *pair) for pair in cross_seam}
        mean_abs_err = sum(abs(sampled_corrs[p] - exact_corrs[p]) for p in cross_seam) / len(cross_seam)

        lin_fids.append(lf)
        hell_fids.append(hf)
        hell_baselines.append(hf_base)
        hell_ratios.append(hr)
        if tf is not None:
            tail_fids.append(tf)
            tail_retained.append(retained)
        zz_errs.append(mean_abs_err)

        print(
            f"[{tag}] circuit {ci:2d}: linXEB={lf:+.3f}  Hellinger-F={hf:.3f} (uniform baseline {hf_base:.3f}, "
            f"excess {hf - hf_base:+.3f}, dist.ratio {hr:.3f})  "
            f"tailXEB={('%+.3f' % tf) if tf is not None else '  n/a':>7} "
            f"(kept {retained*100:4.1f}%)  seam|ZZ err|={mean_abs_err:.3f}"
        )

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(
        f"\n[{tag}] MEANS over {n_circuits} circuits, depth={depth}, {shots} shots/circuit, "
        f"n={QC} (2x4-qubit patches):\n"
        f"    linear XEB           = {avg(lin_fids):.4f}\n"
        f"    Hellinger F (raw)    = {avg(hell_fids):.4f}\n"
        f"    Hellinger F (uniform baseline) = {avg(hell_baselines):.4f}\n"
        f"    Hellinger F (excess over baseline)  = {avg(hell_fids) - avg(hell_baselines):.4f}\n"
        f"    Hellinger distance ratio (ACE/unif) = {avg(hell_ratios):.4f}  (< 1 = closer to ideal than uniform)\n"
        f"    tail XEB             = {avg(tail_fids):.4f}  (avg tail retained {avg(tail_retained)*100:.1f}%)\n"
        f"    seam |ZZ error|      = {avg(zz_errs):.4f}  (12 cross-seam pairs/circuit)\n"
    )
    return dict(linear=avg(lin_fids), hellinger=avg(hell_fids), hellinger_excess=avg(hell_fids) - avg(hell_baselines),
                hellinger_ratio=avg(hell_ratios), tail=avg(tail_fids), zz_err=avg(zz_errs))


def main():
    n_circuits = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    print("=" * 70)
    print("is_error_detection=True (primary configuration)")
    print("=" * 70)
    res_true = run_battery(n_circuits, depth, shots,
                            dict(long_range_columns=LRC, long_range_rows=LRR, is_torus=True,
                                 is_error_detection=True),
                            tag="ED=True")

    print("=" * 70)
    print("is_error_detection=False (comparison)")
    print("=" * 70)
    res_false = run_battery(n_circuits, depth, shots,
                             dict(long_range_columns=LRC, long_range_rows=LRR, is_torus=True,
                                  is_error_detection=False),
                             tag="ED=False")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k in ("linear", "hellinger", "hellinger_excess", "hellinger_ratio", "tail", "zz_err"):
        print(f"  {k:18s}:  ED=True {res_true[k]:+.4f}   vs   ED=False {res_false[k]:+.4f}")


if __name__ == "__main__":
    main()
