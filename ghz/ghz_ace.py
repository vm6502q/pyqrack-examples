# ACE GHZ state preparation simulation
#
# By (Anthropic) Claude and Dan Strano
#
# Preparation order follows the known ACE noise model:
#   1. Start from the last qubit index, which is always a seam (or corner-seam) qubit.
#   2. Spread to every seam qubit through seam-to-seam couplings first. Seam qubits on the
#      same wall have replicas on the same simulators, so these couplings are exact.
#   3. Then spread from the seams into the interior patches, preferring controls that share
#      the target's patch simulator (bulk-to-bulk inside a patch is exact).
# Every coupling used is taken from the backend's own logical coupling map.
#
# Fidelity is reported as the standard GHZ lower bound used for hardware GHZ benchmarks:
#     F >= (P(0...0) + P(1...1) + <X^n>) / 2
# where <X^n> comes from the same prepared state, sampled again after H on every qubit.
# Computational-basis populations alone cannot distinguish GHZ from the classical mixture
# of |0...0> and |1...1>; the parity term is what certifies the coherence between them.
#
# Checks for where coherence goes when <X^n> is low:
#   * Quadrature check: <Y X^(n-1)> (Y on one qubit) measures the imaginary part of the
#     coherence. |C| = sqrt(<X^n>^2 + <Y X^(n-1)>^2) / 2 is the coherence magnitude, so a
#     low <X^n> with a high |C| means a relative phase (drift), not lost coherence. The
#     phase-corrected fidelity estimate (P(0...0) + P(1...1)) / 2 + |C| is reported too;
#     it is NOT a rigorous lower bound (|C| from two noisy quadratures is biased upward by
#     about 1/sqrt(shots)), unlike ghz_fidelity_lower_bound.
#   * Size sweep (optional, --sweep): GHZ grown along the same schedule to k qubits, with
#     population and parity of those k qubits at each size, plus how many patch simulators
#     the k qubits span. Where parity collapses shows whether coherence is lost at seams.
#     (Per-patch parity of one GHZ state is not a usable check: the parity of any proper
#     subset of a GHZ state is exactly 0 even ideally.)

import sys
import time

from collections import Counter, deque

from pyqrack import QrackAceBackend


def _replicas(sim, lq):
    qubits = sim.qubits if hasattr(sim, "qubits") else sim._qubits
    return qubits[lq]


def is_seam(sim, lq):
    return len(_replicas(sim, lq)) > 1


def ghz_schedule(sim):
    """Return a list of (control, target) CNOTs that prepares GHZ from the last qubit,
    seams first, then the patch interiors, using only couplings in the logical map."""
    n = sim.num_qubits()
    root = n - 1
    if not is_seam(sim, root):
        raise RuntimeError(f"Qubit {root} is not a seam qubit for this layout; check long_range_columns/rows.")

    adj = {q: set() for q in range(n)}
    for a, b in sim.get_logical_coupling_map():
        adj[a].add(b)
        adj[b].add(a)

    def sims_of(q):
        return {r[0] for r in _replicas(sim, q)}

    done = {root}
    order = []

    # Phase 1: seam-to-seam, breadth-first from the root, as far as seam couplings reach.
    # Phase 2: breadth-first into the bulk. Whenever a newly reached qubit is a seam, its
    # own seam-to-seam neighbours are taken before continuing into the bulk.
    seam_q = deque([root])
    bulk_q = deque()
    while seam_q or bulk_q:
        if seam_q:
            u = seam_q.popleft()
            for v in sorted(adj[u]):
                if v not in done and is_seam(sim, v):
                    done.add(v)
                    order.append((u, v))
                    seam_q.append(v)
            bulk_q.append(u)
            continue
        u = bulk_q.popleft()
        for v in sorted(adj[u]):
            if v in done:
                continue
            # Prefer a control that is a bulk qubit on v's own patch simulator (exact, intra-patch);
            # otherwise use u.
            ctrl = u
            if not is_seam(sim, v):
                vs = sims_of(v)
                for w in sorted(adj[v]):
                    if w in done and not is_seam(sim, w) and sims_of(w) & vs:
                        ctrl = w
                        break
            done.add(v)
            order.append((ctrl, v))
            (seam_q if is_seam(sim, v) else bulk_q).append(v)

    if len(done) != n:
        missing = sorted(set(range(n)) - done)
        raise RuntimeError(f"Logical coupling map does not connect every qubit; unreached: {missing}")

    return order


def prepare(n, lrc, lrr, schedule=None, k=None):
    """Prepare GHZ on the root plus the first k - 1 schedule targets (all n qubits if k is None)."""
    sim = QrackAceBackend(n, long_range_columns=lrc, long_range_rows=lrr)
    if schedule is None:
        schedule = ghz_schedule(sim)
    sim.h(n - 1)
    for c, t in schedule[: (len(schedule) if k is None else k - 1)]:
        sim.mcx([c], t)
    return sim, schedule


def _parity(counts, mask, shots):
    return sum(v * (1 - 2 * ((k & mask).bit_count() & 1)) for k, v in counts.items()) / shots


def size_sweep(n, lrc, lrr, schedule, shots, sizes):
    """Population and X-parity of GHZ grown to k qubits along the schedule, for each k in sizes."""
    members = [n - 1] + [t for _, t in schedule]
    rows = []
    for k in sizes:
        sim, _ = prepare(n, lrc, lrr, schedule, k)
        qs = members[:k]
        mask = sum(1 << q for q in qs)
        patches = len({r[0] for q in qs for r in _replicas(sim, q) if r[0] >= 0})
        seams = sum(1 for q in qs if is_seam(sim, q))
        z = Counter(sim.measure_shots(list(range(n)), shots))
        for q in qs:
            sim.h(q)
        x = Counter(sim.measure_shots(list(range(n)), shots))
        del sim
        pop = sum(v for key, v in z.items() if (key & mask) in (0, mask)) / shots
        par = _parity(x, mask, shots)
        rows.append({"k": k, "simulators_spanned": patches, "seam_qubits": seams,
                     "population": pop, "x_parity": par, "fidelity_lower_bound": 0.5 * (pop + par)})
    return rows


def bench_qrack(n, lrc, lrr, shots=16384, sweep_step=0):
    start = time.perf_counter()

    # Run 1: computational basis populations.
    sim, schedule = prepare(n, lrc, lrr)
    seams = sum(1 for q in range(n) if is_seam(sim, q))
    seam_seam = sum(1 for c, t in schedule if is_seam(sim, c) and is_seam(sim, t))
    z_counts = Counter(sim.measure_shots(list(range(n)), shots))

    # Run 2: same preparation, then H on every qubit (transversal in ACE), then parity.
    for q in range(n):
        sim.h(q)
    x_counts = Counter(sim.measure_shots(list(range(n)), shots))

    # Quadrature check, same state again: turn qubit 0 from the X basis into the Y basis
    # (undo its H, then S^dagger, then H), so the parity measures <Y_0 X^(n-1)>.
    sim.h(0)
    sim.adjs(0)
    sim.h(0)
    y_counts = Counter(sim.measure_shots(list(range(n)), shots))
    del sim

    m = (1 << n) - 1
    p_0 = z_counts.get(0, 0) / shots
    p_1 = z_counts.get(m, 0) / shots
    population = p_0 + p_1
    all_mask = (1 << n) - 1
    parity = _parity(x_counts, all_mask, shots)
    y_parity = _parity(y_counts, all_mask, shots)
    fidelity_lb = 0.5 * (population + parity)
    coherence = 0.5 * (parity * parity + y_parity * y_parity) ** 0.5
    fidelity_phase_corrected = 0.5 * population + coherence

    # Statistical error of the bound (binomial population + parity shot noise).
    se = 0.5 * ((population * (1 - population) + (1 - parity * parity)) / shots) ** 0.5

    # Diagnostic only: mean normalized Hamming distance to the nearer pole (0 = all shots
    # at a pole, 1 = all shots perfectly balanced). This is not a fidelity.
    dist = sum(v * 2 * min(k.bit_count(), n - k.bit_count()) / n for k, v in z_counts.items()) / shots

    sweep = []
    if sweep_step > 0:
        sizes = sorted(set(list(range(2, n + 1, sweep_step)) + [n]))
        sweep = size_sweep(n, lrc, lrr, schedule, shots, sizes)

    return {
        "n": n,
        "long_range_columns": lrc,
        "long_range_rows": lrr,
        "seam_qubits": seams,
        "cnots": len(schedule),
        "seam_to_seam_cnots": seam_seam,
        "seconds": time.perf_counter() - start,
        "shots": shots,
        "p_all_0": p_0,
        "p_all_1": p_1,
        "population": population,
        "x_parity": parity,
        "y_quadrature_parity": y_parity,
        "coherence_magnitude": coherence,
        "ghz_fidelity_lower_bound": fidelity_lb,
        "ghz_fidelity_lower_bound_se": se,
        "ghz_fidelity_phase_corrected_estimate": fidelity_phase_corrected,
        "mean_pole_distance": dist,
        "size_sweep": sweep,
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            "Usage: python3 ghz_ace.py [width] [long_range_columns=4] [long_range_rows=4] [shots=16384] [--sweep[=step]]"
        )

    sweep_step = 0
    args = []
    for a in sys.argv[1:]:
        if a.startswith("--sweep"):
            sweep_step = int(a.split("=", 1)[1]) if "=" in a else 1
        else:
            args.append(a)
    width = int(args[0])
    lrc = int(args[1]) if len(args) > 1 else 4
    lrr = int(args[2]) if len(args) > 2 else 4
    shots = int(args[3]) if len(args) > 3 else 16384
    result = bench_qrack(width, lrc, lrr, shots, sweep_step)
    sweep = result.pop("size_sweep")
    for k, v in result.items():
        print(f"  {k}: {v}")
    if sweep:
        print("  size sweep (k qubits along the schedule):")
        print("      k  sims  seams  population  x_parity  F_lower_bound")
        for r in sweep:
            print(f"    {r['k']:3d}  {r['simulators_spanned']:4d}  {r['seam_qubits']:5d}  "
                  f"{r['population']:10.4f}  {r['x_parity']:8.4f}  {r['fidelity_lower_bound']:13.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
