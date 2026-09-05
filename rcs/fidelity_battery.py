"""
Fidelity battery for QrackAceBackend, beyond plain linear XEB.

Revision note
-------------
An earlier version of this script used Hellinger fidelity (squared
Bhattacharyya coefficient) as a "harder to game than linear XEB" check.
That was a mistake for this purpose, and it's worth recording why, so it
doesn't get reinvented.

Linear XEB is exactly a regression slope: xeb(Q) = Cov(P,Q)/Var(P) over
the uniform (1/N-weighted) distribution of bitstrings, where the uniform
distribution U is *by construction* the zero-correlation reference point
(mean(P) = 1/N for any valid probability distribution). Mixing U into any
Q -- Q_mix = (1-p)*Q + p*U -- therefore satisfies, exactly:

    xeb(Q_mix) = (1-p) * xeb(Q)

linearly, for every p. There is no interior optimum: diluting toward
uniform can only ever shrink linear XEB toward 0, never improve it.
Hellinger fidelity has no such structure -- BC(Q_mix, P) is CONCAVE in p
(sqrt of an affine function), so it can have an interior or boundary
maximum above p=0. Checked directly on this backend's own output: the
Hellinger-optimal mixing fraction can run as high as p* ~ 0.8-1.0 at
moderate depth, and doing so drives linear XEB toward exactly 0 by the
relation above. That means Hellinger fidelity, computed as a plug-in
estimate off a sparse empirical histogram (a few thousand shots against
hundreds of states), can be *improved* by diluting away the very
correlation signal linear XEB is measuring -- rewarding partial
replacement with literal noise. That's a more direct vulnerability than
anything in the classical-XEB-spoofing literature, and it means Hellinger
fidelity was the wrong tool here, not merely one that needed a baseline
correction.

So this version reports three metrics, all built on linear XEB rather than
distributional (Bhattacharyya/Hellinger) comparison:

  1. Full-window linear XEB (the standard statistic), for continuity with
     existing benchmarks.

  2. Tail-restricted linear XEB: linear XEB recomputed after excluding the
     heaviest bitstrings (by ideal probability) from both the sample set
     and the renormalized ideal distribution. A classical heavy-hitter
     spoof (Barak, Chou, Gao, Jain, Sharan 2020) shows little to no
     correlation once the peaks it was built to find are removed; a
     genuine (even lossy) simulator should retain positive correlation
     into the tail, just at reduced magnitude. Unlike the Hellinger
     approach, this stays a linear-XEB-family statistic throughout, so it
     doesn't inherit the uniform-dilution vulnerability above -- excluding
     bitstrings changes which support the regression runs over; it isn't
     a convex blend with U.

  3. Cross-seam ZZ correlators: for pairs of qubits that live in DIFFERENT
     patches (i.e. whose entangling gate had to cross the approximate
     patch boundary this backend introduces), compare the sampled
     connected correlator <Z_i Z_j> - <Z_i><Z_j> against the exact value.
     A fine-grained, many-constraint, physically-interpretable check
     specific to this architecture's own approximation mechanism (the
     same coupling machinery debugged in _apply_coupling) -- it must hold
     up pairwise, across many different qubit pairs and circuit
     instances, not just in aggregate, which is a harder target than a
     single scalar XEB number.

None of this constitutes a formal hardness/anti-spoofing proof. It's a
battery of empirical checks run at a scale this sandbox can actually
execute (statevector reference up to ~8-10 qubits, several hundred shots
per circuit). Finite-sample noise is real at these shot counts -- treat
single-circuit numbers as noisy and look at the trend across
circuits/depths, not any one value.
"""

import math
import random
import sys

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


cmap_global = None  # set in run_battery() before gen_circuit is called


def run_battery(n_circuits, depth, shots, config, tag, exclude_frac=0.25):
    cmap, cross_seam, _ = get_topology()
    global cmap_global
    cmap_global = cmap

    lin_fids, tail_fids, tail_retained, zz_errs = [], [], [], []

    for ci in range(n_circuits):
        ops = gen_circuit(QC, depth, cmap, seed=2000 + ci)
        sim, ideal_probs = run_ideal(QC, ops)
        exact_corrs = {pair: exact_zz(sim, *pair) for pair in cross_seam}

        random.seed(9000 + ci)
        samples = [run_ace_sample(QC, ops, config) for _ in range(shots)]

        lf = linear_xeb(ideal_probs, samples, QC)
        tf, retained = tail_restricted_xeb(ideal_probs, samples, QC, exclude_frac)

        sampled_corrs = {pair: sampled_zz(samples, *pair) for pair in cross_seam}
        mean_abs_err = sum(abs(sampled_corrs[p] - exact_corrs[p]) for p in cross_seam) / len(cross_seam)

        lin_fids.append(lf)
        if tf is not None:
            tail_fids.append(tf)
            tail_retained.append(retained)
        zz_errs.append(mean_abs_err)

        print(
            f"[{tag}] circuit {ci:2d}: linXEB={lf:+.3f}  "
            f"tailXEB={('%+.3f' % tf) if tf is not None else '  n/a':>7} "
            f"(kept {retained*100:4.1f}%)  seam|ZZ err|={mean_abs_err:.3f}"
        )

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(
        f"\n[{tag}] MEANS over {n_circuits} circuits, depth={depth}, {shots} shots/circuit, "
        f"n={QC} (2x4-qubit patches):\n"
        f"    linear XEB      = {avg(lin_fids):.4f}\n"
        f"    tail XEB        = {avg(tail_fids):.4f}  (avg tail retained {avg(tail_retained)*100:.1f}%)\n"
        f"    seam |ZZ error| = {avg(zz_errs):.4f}  (12 cross-seam pairs/circuit)\n"
    )
    return dict(linear=avg(lin_fids), tail=avg(tail_fids), zz_err=avg(zz_errs))


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
    for k in ("linear", "tail", "zz_err"):
        print(f"  {k:10s}:  ED=True {res_true[k]:+.4f}   vs   ED=False {res_false[k]:+.4f}")


if __name__ == "__main__":
    main()
