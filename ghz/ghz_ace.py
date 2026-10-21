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
# where <X^n> comes from a second run of the same preparation followed by H on every qubit.
# Computational-basis populations alone cannot distinguish GHZ from the classical mixture
# of |0...0> and |1...1>; the parity term is what certifies the coherence between them.

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


def prepare(n, lrc, lrr, schedule=None):
    sim = QrackAceBackend(n, long_range_columns=lrc, long_range_rows=lrr)
    if schedule is None:
        schedule = ghz_schedule(sim)
    sim.h(n - 1)
    for c, t in schedule:
        sim.mcx([c], t)
    return sim, schedule


def bench_qrack(n, lrc, lrr, shots=16384):
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
    del sim

    m = (1 << n) - 1
    p_0 = z_counts.get(0, 0) / shots
    p_1 = z_counts.get(m, 0) / shots
    population = p_0 + p_1
    parity = sum(v * (1 - 2 * (k.bit_count() & 1)) for k, v in x_counts.items()) / shots
    fidelity_lb = 0.5 * (population + parity)

    # Statistical error of the bound (binomial population + parity shot noise).
    se = 0.5 * ((population * (1 - population) + (1 - parity * parity)) / shots) ** 0.5

    # Diagnostic only: mean normalized Hamming distance to the nearer pole (0 = all shots
    # at a pole, 1 = all shots perfectly balanced). This is not a fidelity.
    dist = sum(v * 2 * min(k.bit_count(), n - k.bit_count()) / n for k, v in z_counts.items()) / shots

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
        "ghz_fidelity_lower_bound": fidelity_lb,
        "ghz_fidelity_lower_bound_se": se,
        "mean_pole_distance": dist,
    }


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python3 ghz_ace.py [width] [long_range_columns=4] [long_range_rows=4]")

    width = int(sys.argv[1])
    lrc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lrr = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    result = bench_qrack(width, lrc, lrr)
    for k, v in result.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
