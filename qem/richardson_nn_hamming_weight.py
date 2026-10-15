# Zero-noise extrapolation of mirror-circuit Hamming weight.
#
# Same circuit family and transform as mitiq_nn_hamming_weight.py, but the
# Richardson extrapolation is done here, in the variable u = lambda^p rather
# than in lambda itself. Ordinary Richardson (mitiq's RichardsonFactory) is
# the special case p = 1.
#
# Why lambda^p: our depth model for ACE fidelity is
#     F(d) = F(d0) ** ((d / d0) ** p),   i.e.   log F  is linear in  d^p.
# Unitary folding by an odd scale factor lambda multiplies circuit depth by
# lambda, so the same model says the observable's log-signal is linear in
# lambda^p. With p != 1, a polynomial in lambda is the wrong basis; a
# polynomial in lambda^p is the right one.
#
# Scale factors are 1, 3, 5 by global unitary folding:
#     lambda = 2k + 1:   M -> M (M^dagger M)^k,   with M = U U^dagger (the mirror).
#
# Usage:
#   python3 richardson_nn_hamming_weight.py [width] [depth] [shots=8192] [p=fit] [transform=log]
#     p:         a number (e.g. 1.6), or "fit" to solve p from the three points
#     transform: "log"   -> extrapolate log(1 - 2h)   (matches the F^(lambda^p) model)
#                "logit" -> extrapolate logit(2h)     (as in the mitiq example)
#   where h = (Hamming weight) / (shots * width) is the mean fraction of 1s;
#   ideal h = 0, fully scrambled h = 1/2.
#
# Mitiq (and examples using on which this is based) is under the GPL 3.0.
# Hence, this example, as the entire work-in-itself, must be considered to be under GPL 3.0.
# See https://www.gnu.org/licenses/gpl-3.0.txt for details.

import math
import random
import sys
import time

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.qrack import AceQasmSimulator


SCALE_FACTORS = (1, 3, 5)


def factor_width(width):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (row_len, col_len)


def cx(sim, q1, q2):
    sim.cx(q1, q2)


def cy(sim, q1, q2):
    sim.cy(q1, q2)


def cz(sim, q1, q2):
    sim.cz(q1, q2)


def acx(sim, q1, q2):
    sim.x(q1)
    sim.cx(q1, q2)
    sim.x(q1)


def acy(sim, q1, q2):
    sim.x(q1)
    sim.cy(q1, q2)
    sim.x(q1)


def acz(sim, q1, q2):
    sim.x(q1)
    sim.cz(q1, q2)
    sim.x(q1)


def swap(sim, q1, q2):
    sim.swap(q1, q2)


def iswap(sim, q1, q2):
    sim.iswap(q1, q2)


def iiswap(sim, q1, q2):
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)
    sim.iswap(q1, q2)


def pswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)


def mswap(sim, q1, q2):
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def nswap(sim, q1, q2):
    sim.cz(q1, q2)
    sim.swap(q1, q2)
    sim.cz(q1, q2)


def random_circuit(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    lcv_range = range(width)

    # Nearest-neighbor couplers:
    gateSequence = [0, 3, 2, 1, 2, 1, 0, 3]
    two_bit_gates = swap, pswap, mswap, nswap, iswap, iiswap, cx, cy, cz, acx, acy, acz

    row_len, col_len = factor_width(width)

    qc = QuantumCircuit(width, width)
    for _ in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            th, ph, lm = (random.uniform(-math.pi, math.pi) for _ in range(3))
            # Keep it Haar-random towards the poles:
            th = math.asin(th / math.pi)
            qc.u(th, ph, lm, i)

        # Nearest-neighbor couplers:
        ############################
        gate = gateSequence.pop(0)
        gateSequence.append(gate)
        for row in range(1, row_len, 2):
            for col in range(col_len):
                temp_row = row
                temp_col = col
                temp_row = temp_row + (1 if (gate & 2) else -1)
                temp_col = temp_col + (1 if (gate & 1) else 0)

                if temp_row < 0:
                    continue
                if temp_col < 0:
                    continue
                if temp_row >= row_len:
                    continue
                if temp_col >= col_len:
                    continue

                b1 = col * row_len + row
                b2 = temp_col * row_len + temp_row

                if (b1 >= width) or (b2 >= width):
                    continue

                g = random.choice(two_bit_gates)
                g(qc, b1, b2)

    return qc


# To take a [0.0, 1.0] bounded interval to an unbounded one for OLS or Richardson extrapolation:
# Precise symmetric logit/expit limits provided by (Anthropic) Claude
# Based on a less precise version by Dan Strano

_LOGIT_CEIL = 36.7368005696771             # logit(nextafter(1.0, 0.0)), exact
_LOGIT_FLOOR = -_LOGIT_CEIL                # symmetric by choice, not the true underflow point (~-744.44)
_EXPIT_AT_FLOOR = 1.1102230246251573e-16   # = expit(_LOGIT_FLOOR), precomputed
_LOG_FLOOR = math.log(_EXPIT_AT_FLOOR)     # same floor, for log(1 - 2h)


def logit(x):
    if x >= 1.0:
        return _LOGIT_CEIL
    if x <= 0.0:
        return _LOGIT_FLOOR
    return max(_LOGIT_FLOOR, min(_LOGIT_CEIL, math.log(x / (1.0 - x))))


def expit(x):
    if x >= _LOGIT_CEIL:
        return 1.0
    if x <= _LOGIT_FLOOR:
        return _EXPIT_AT_FLOOR
    return 1.0 / (1.0 + math.exp(-x))


# Transforms of h (mean fraction of 1s) to an unbounded variable, and back.
def to_log(h):
    s = 1.0 - 2.0 * h  # signal: 1 when ideal, 0 when fully scrambled
    return _LOG_FLOOR if s <= _EXPIT_AT_FLOOR else math.log(s)


def from_log(y):
    return 0.5 * (1.0 - math.exp(min(y, 0.0)))


def to_logit(h):
    return logit(2.0 * h)


def from_logit(y):
    return 0.5 * expit(y)


TRANSFORMS = {"log": (to_log, from_log), "logit": (to_logit, from_logit)}


def fold(qc, scale_factor):
    """Global unitary folding of the mirror circuit M = U U^dagger:
    M -> M (M^dagger M)^k for scale_factor = 2k + 1. Returns the unmeasured circuit."""
    if scale_factor < 1 or scale_factor % 2 != 1:
        raise ValueError("Scale factors must be odd positive integers.")
    mirror = qc & qc.inverse()
    inv = mirror.inverse()
    out = mirror.copy()
    for _ in range((scale_factor - 1) // 2):
        out = out & inv & mirror
    return out


def mean_ones_fraction(circ, n_qubits, shot_count):
    """Run circ and return h = Hamming weight / (shots * n_qubits)."""
    qcm = circ.copy()
    # Hamming weight summed over all qubits does not depend on which wire
    # goes to which classical bit, so measure each logical wire in place.
    for q in range(n_qubits):
        qcm.measure(q, q)

    sim = AceQasmSimulator()
    counts = dict(sim.run(qcm, shots=shot_count).result().get_counts())

    hamming_weight = 0
    for k, v in counts.items():
        hamming_weight += k.count("1") * v

    return hamming_weight / (shot_count * n_qubits)


def richardson_weights(nodes):
    """Lagrange weights for extrapolating to 0 through (nodes[i], y[i]):
    y(0) = sum_i w_i * y_i,  w_i = prod_{j != i} u_j / (u_j - u_i)."""
    w = []
    for i, ui in enumerate(nodes):
        wi = 1.0
        for j, uj in enumerate(nodes):
            if j != i:
                wi *= uj / (uj - ui)
        w.append(wi)
    return w


def richardson(lambdas, ys, p):
    """Richardson extrapolation in u = lambda^p. p = 1 is ordinary Richardson."""
    u = [lam ** p for lam in lambdas]
    w = richardson_weights(u)
    return sum(wi * yi for wi, yi in zip(w, ys)), w


def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 richardson_nn_hamming_weight.py [width] [depth] [shots=8192] [p=fit] [transform=log]"
        )

    width = int(sys.argv[1])
    depth = int(sys.argv[2])
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    p = float(sys.argv[4]) if len(sys.argv) > 4 else 1.60569
    transform = sys.argv[5] if len(sys.argv) > 5 else "log"
    fwd, back = TRANSFORMS[transform]

    qc = random_circuit(width, depth)
    target = AceQasmSimulator()

    start = time.perf_counter()
    qc = transpile(qc, backend=target, optimization_level=3)
    end = time.perf_counter()
    print(f"Transpilation time: {end - start} seconds")

    # Fold AFTER transpilation, so the optimizer cannot cancel U against U^dagger.
    start = time.perf_counter()
    hs = []
    for lam in SCALE_FACTORS:
        h = mean_ones_fraction(fold(qc, lam), width, shots)
        hs.append(h)
        print(f"  scale {lam}: Hamming weight {width * h:.4f} (h = {h:.5f}, {transform} = {fwd(h):.5f})")
    ys = [fwd(h) for h in hs]

    y0, w = richardson(SCALE_FACTORS, ys, p)
    # Each point carries shot noise; Richardson amplifies it by sqrt(sum w_i^2).
    gain = math.sqrt(sum(wi * wi for wi in w))
    y0_p1, _ = richardson(SCALE_FACTORS, ys, 1.0)
    end = time.perf_counter()

    raw = width * hs[0]
    mitigated = width * back(y0)
    print(f"  Richardson in lambda^p, p = {p:.4f}: weights {[round(x, 4) for x in w]}, noise gain {gain:.2f}")
    print(f"  (ordinary Richardson, p = 1: Hamming weight {width * back(y0_p1):.4f})")
    if transform == "log" and y0 > 0.0:
        verdict = "Treat this extrapolation as failed." if y0 > 0.05 else "(Within noise of the ideal 0.)"
        print(f"  NOTE: extrapolated log(1 - 2h) = {y0:.4f} > 0 means h < 0; clipped to Hamming weight 0. {verdict}")

    print({"width": width, "depth": depth, "seconds": (end - start), "p": p, "transform": transform,
           "hamming_weight": float(raw), "mitigated_hamming_weight": float(mitigated)})
    print("(Ideal hamming_weight is 0.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
