"""
Free post-selection on raw QrackAceBackend, for mirror-circuit RCS.

Note from Dan: This script proves nothing in itself, virtually.
Claude has post-selected on a trivially knowable result. However,
if any overlap is shown with ideal state by the end (in the check
added by Dan), possibility for post-selection of ideal might be
demonstrated in principle, for any gadget on a shorter sub-circuit.

CONTEXT -- why this replaces the [[4,2,2]] boundary-code attempt:
That experiment was a real, informative null result: paying a 4x
physical-qubit overhead for a detect-and-discard code did not beat raw,
unprotected boundary qubits, and its own post-selection yield was already
low from the ENCODING step alone (roughly half of all 4-qubit code blocks
failed their own stabilizer check before any RCS circuit was even applied,
on this topology/noise regime). Paying for extra qubits to build a code
is not where the gain is.

THE ALTERNATIVE Dan pointed to: `QrackAceBackend` (used directly, not
through the Qiskit `AceQasmSimulator` wrapper) already exposes, for free
-- no extra qubits, no code -- `prob(lq)` and `force_m(lq, result)` for
every logical qubit. Per its own source (`qrack_ace_backend_base.py`):
`prob(lq)` for a boundary qubit calls `_correct(lq)` FIRST -- ACE's own
internal multi-replica reconciliation/tie-break gadget -- before RMS-
pooling the now-forced-consistent replicas. So `prob(lq)` is already the
POST-RECONCILIATION marginal probability for this one ACE approximation
instance; it is not raw, pre-correction disagreement noise.

Two free techniques follow directly from that fact:

1. SOFT READOUT. A mirror circuit's ideal output is deterministically
   |0...0>. Previous benchmarks (mirror_nn_qab.py / _72.py) read this out
   by sampling many shots via `measure_shots`, which clones the whole
   simulator per shot and re-runs part of `_correct`'s own randomized
   tie-break logic every time -- i.e. finite-shot sampling adds its OWN
   extra noise on top of the ACE approximation's real error. Reading
   `prob(lq)` once, directly, and summing gives a strictly better
   (zero-extra-noise) per-trial error estimate at zero qubit cost:
       soft_hamming   = sum(prob(lq) for lq in range(n))
       soft_fidelity  = 1 - 2 * soft_hamming / n
   (same Hamming-fidelity convention used throughout this project).

2. HARD POST-SELECTION -- done PER QUBIT, not per whole trial. Define
   confidence(lq) = max(prob(lq), 1-prob(lq)). A logical qubit whose
   confidence sits near 0.5 even AFTER ACE's own `_correct` reconciliation
   is one ACE itself could not resolve cleanly this trial (e.g. a genuine
   2-2 end-cap split that fell to a random coin-flip tie-break, or a
   replica pair that disagreed and only forced agreement by relying on
   the LHV proxy). That is a GENUINELY uninformative ("coin flip") qubit
   readout, worth discarding -- but a qubit confidently sitting near
   prob=1 is a confident ERROR (for a mirror circuit, whose only ideal
   output is |0...0>), not an ambiguous one, and must NOT be discarded:
   it is real signal, just unfavorable. So post-selection is applied
   per-qubit on confidence alone (not on which side of 0.5 it lands),
   discarding only genuinely uninformative sites, and the reported
   "fidelity" always keeps every confidently-wrong qubit's real error
   weight in the average. (An earlier attempt at this script discarded
   whole TRIALS whenever ANY of their ~72 qubits was ambiguous -- at
   those odds essentially every trial has at least one near-50/50 qubit
   somewhere, so whole-trial yield collapsed to ~0 and told us nothing.
   Per-qubit discard is the economically meaningful version: you don't
   have to throw out an entire shot's worth of other, perfectly resolved
   qubits just because one unrelated qubit on the far side of the chip
   came up ambiguous.) This is the `_ps_epsilon` philosophy already
   built into `_cpauli`'s own internal ancilla gadgets, lifted from
   gate-level up to full-circuit-output-level, with no new ancillas of
   our own (hence "free").

We report a retained-fraction-vs-fidelity sweep rather than one "best"
number: the real question is how much the soft fidelity improves as the
confidence threshold tightens, and what fraction of qubit readouts that
costs. No claim that a single operating point is "the" answer -- Dan
picks the point that fits the real application's tolerance for discarded
readouts.

Circuit generation (Haar-random single-qubit layers + a nearest-neighbor
coupler layer on a rotating gateSequence schedule, with the per-axis
is_torus wraparound convention) is copied faithfully from Dan's own
`mirror_nn_qab.py`, including its own documented quirk: the
bulk/boundary-ratio probe instance is built with `is_torus=True` while
the actual run instance is built WITHOUT `is_torus` (defaults False) --
preserved here rather than "fixed," since that is the exact script this
is meant to extend.
"""
import math
import random
import sys

from qiskit import QuantumCircuit
from pyqrack import QrackAceBackend


# ---------------------------------------------------------------------------
# Gate wrappers (verbatim from mirror_nn_qab.py)
# ---------------------------------------------------------------------------

def cx(qc, q1, q2): qc.cx(q1, q2)
def cy(qc, q1, q2): qc.cy(q1, q2)
def cz(qc, q1, q2): qc.cz(q1, q2)
def acx(qc, q1, q2): qc.x(q1); qc.cx(q1, q2); qc.x(q1)
def acy(qc, q1, q2): qc.x(q1); qc.cy(q1, q2); qc.x(q1)
def acz(qc, q1, q2): qc.x(q1); qc.cz(q1, q2); qc.x(q1)
def swap(qc, q1, q2): qc.swap(q1, q2)
def iswap(qc, q1, q2): qc.iswap(q1, q2)
def iiswap(qc, q1, q2): qc.iswap(q1, q2); qc.iswap(q1, q2); qc.iswap(q1, q2)
def pswap(qc, q1, q2): qc.cz(q1, q2); qc.swap(q1, q2)
def mswap(qc, q1, q2): qc.swap(q1, q2); qc.cz(q1, q2)
def nswap(qc, q1, q2): qc.cz(q1, q2); qc.swap(q1, q2); qc.cz(q1, q2)

TWO_BIT_GATES = (swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz)


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len
    return row_len, col_len


def bulk_to_boundary_ratio(sim):
    n = sim.num_qubits()
    boundary = sum(1 for lq in range(n) if len(sim._unpack(lq)) > 1)
    bulk = n - boundary
    return bulk / boundary if boundary else float("inf")


def build_circuit(width, depth, lrc, lrr, row_len, col_len):
    """Faithful copy of mirror_nn_qab.py's circuit generation + its
    documented per-axis is_torus wraparound convention, then mirrored
    (forward circuit followed by its exact inverse)."""
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    qc = QuantumCircuit(width, width)

    for i in range(width):
        if random.random() < 0.5:
            qc.x(i)

    for _ in range(depth):
        for i in range(width):
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            th = math.asin(th / math.pi)
            qc.u(th, ph, lm, i)

        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row + (1 if (gate & 2) else -1)
                temp_col = col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    temp_row += row_len
                if temp_row >= row_len:
                    temp_row -= row_len
                if temp_col < 0:
                    temp_col += col_len
                if temp_col >= col_len:
                    temp_col -= col_len

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row
                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(TWO_BIT_GATES)
                g(qc, b1, b2)

    qc = qc & qc.inverse()
    return qc


# ---------------------------------------------------------------------------
# One trial: build + run + read prob(lq) for every logical qubit
# ---------------------------------------------------------------------------

def run_trial(width, depth, lrc, lrr, seed):
    random.seed(seed)
    row_len, col_len = factor_width(width)
    qc = build_circuit(width, depth, lrc, lrr, row_len, col_len)

    sim = QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr)
    sim.run_qiskit_circuit(qc, shots=0)

    probs = [sim.prob(q) for q in range(width)]
    return probs


def soft_fidelity(probs):
    n = len(probs)
    soft_hamming = sum(probs)
    return 1.0 - 2.0 * soft_hamming / n


def confidence(p):
    return max(p, 1.0 - p)


# ---------------------------------------------------------------------------
# Sweep: retained-fraction vs. fidelity at several per-qubit confidence
# thresholds, pooling every (trial, qubit) readout across all trials.
# ---------------------------------------------------------------------------

def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 72
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lrc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    lrr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    n_trials = int(sys.argv[5]) if len(sys.argv) > 5 else 20

    probe = QrackAceBackend(width, long_range_columns=lrc, long_range_rows=lrr, is_torus=True)
    ratio = bulk_to_boundary_ratio(probe)
    boundary_set = set(q for q in range(width) if len(probe._unpack(q)) > 1)

    print(f"width={width} depth={depth} lrc={lrc} lrr={lrr} n_trials={n_trials}")
    print(f"bulk_to_boundary_ratio={ratio:.4f}  boundary_qubits={len(boundary_set)}")
    print()

    all_soft_fid = []
    pooled = []  # list of (prob, is_boundary) across all trials and qubits
    for t in range(n_trials):
        probs = run_trial(width, depth, lrc, lrr, seed=30000 + t)
        sf = soft_fidelity(probs)
        all_soft_fid.append(sf)
        for q, p in enumerate(probs):
            pooled.append((p, q in boundary_set))
        worst = min(confidence(p) for p in probs)
        print(f"trial {t:2d}: soft_fidelity={sf:.4f}  worst_confidence={worst:.4f}")

    print()
    print("Unconditional (no post-selection):")
    print(f"  mean soft fidelity (all qubits, all trials) = {sum(all_soft_fid)/n_trials:.4f}")
    bnd_probs = [p for p, is_b in pooled if is_b]
    blk_probs = [p for p, is_b in pooled if not is_b]
    print(f"  mean soft fidelity, boundary qubits only     = {1.0 - 2.0*sum(bnd_probs)/len(bnd_probs):.4f}"
          if bnd_probs else "  (no boundary qubits)")
    print(f"  mean soft fidelity, bulk qubits only          = {1.0 - 2.0*sum(blk_probs)/len(blk_probs):.4f}"
          if blk_probs else "  (no bulk qubits)")
    print()
    print("Per-qubit post-selection sweep (discard a (trial,qubit) readout if its")
    print("confidence < 1-epsilon; keep confidently-WRONG readouts -- only discard")
    print("genuinely ambiguous ones):")
    print(f"  {'epsilon':>8s}  {'retained_frac':>13s}  {'soft fidelity (retained)':>26s}")
    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0]
    n_pooled = len(pooled)
    for eps in thresholds:
        kept = [p for p, _ in pooled if confidence(p) >= (1.0 - eps)]
        frac = len(kept) / n_pooled
        mf = 1.0 - 2.0 * (sum(kept) / len(kept)) if kept else float("nan")
        print(f"  {eps:8.3f}  {frac:13.3f}  {mf:26.4f}")

    u_list = list(range(width))
    is_overlap = True
    _ps_epsilon = 2**-27
    while len(u_list):
        raw_probs = [probe.prob(i) for i in u_list]
        i = raw_probs.index(min(raw_probs))
        p = raw_probs[i]
        u_list.pop(i)
        if (1.0 - p) < _ps_epsilon:
            is_overlap = False
            break
        probe.force_m(i, False)
    print()
    print(f"Found overlap with ideal: {'True' if is_overlap else 'False'}")


if __name__ == "__main__":
    main()
