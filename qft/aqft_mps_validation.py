#!/usr/bin/env python3
"""
aqft_mps_validation.py — Validate Quimb MPS AQFT against PyQrack exact QFT.

(C) Daniel Strano and the Qrack contributors 2026.
This file was produced almost in its entirety by (Anthropic) Claude.
Licensed under MIT: https://opensource.org/licenses/MIT

The Approximate Quantum Fourier Transform (AQFT) drops controlled-phase gates
whose rotation angle falls below a cutoff threshold (pi / 2^cutoff).  For the
QFT on a GHZ input state — the hardest case for MPS because GHZ is maximally
entangled — we compare:

    ideal  : PyQrack exact QFT (full state vector)
    approx : Quimb CircuitMPS AQFT with tunable bond dimension and cutoff

Metrics reported:
    XEB   : linear cross-entropy benchmarking fidelity
    HOG   : heavy output generation probability
    L2    : L2 norm of probability difference vector

Usage:
    python3 aqft_mps_validation.py [n_qubits] [max_bond] [cutoff]

    n_qubits : number of qubits          (default: 16)
    max_bond : MPS bond dimension chi     (default: n_qubits)
    cutoff   : AQFT phase gate cutoff    (default: n_qubits)
               drop cp(pi/2^k) for k > cutoff
"""

import math
import statistics
import sys

import jax.numpy as jnp
import numpy as np
import quimb.tensor as tn
from pyqrack import QrackSimulator
from qiskit import QuantumCircuit, transpile


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def calc_stats(ideal_probs, exp_probs):
    """XEB, HOG probability, and L2 norm between ideal and experimental."""
    n_pow = len(ideal_probs)
    mean_guess = 1.0 / n_pow
    model = 0.5
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)

    numer = denom = hog_prob = sqr_diff = 0.0

    for i in range(n_pow):
        exp = (1.0 - model) * (exp_probs[i] if i < len(exp_probs) else 0.0) + model * mean_guess
        ideal = ideal_probs[i]
        denom   += (ideal - u_u) ** 2
        numer   += (ideal - u_u) * (exp - u_u)
        sqr_diff += (ideal - exp) ** 2
        if ideal > threshold:
            hog_prob += exp

    xeb = numer / denom if denom else 0.0
    l2  = math.sqrt(sqr_diff)
    return {"xeb": float(xeb), "hog_prob": float(hog_prob), "l2_diff": float(l2)}


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------
def build_ghz(n: int, circ: QuantumCircuit) -> None:
    """Prepare GHZ state: (|0...0> + |1...1>) / sqrt(2)."""
    circ.h(0)
    for i in range(1, n):
        circ.cx(i - 1, i)


def build_qft(n: int, circ: QuantumCircuit) -> None:
    """Standard QFT (all controlled-phase gates, no cutoff)."""
    for qubit in range(n - 1, -1, -1):
        circ.h(qubit)
        for ctrl in range(qubit - 1, -1, -1):
            k = qubit - ctrl
            circ.cp(math.pi / (1 << k), ctrl, qubit)


def build_aqft(n: int, circ: QuantumCircuit, cutoff: int) -> None:
    """
    AQFT: drop controlled-phase gates with rotation pi/2^k where k > cutoff.
    Beals et al. (2003) showed cutoff = O(log n) suffices for hidden subgroup.
    """
    for qubit in range(n - 1, -1, -1):
        circ.h(qubit)
        for ctrl in range(qubit - 1, -1, -1):
            k = qubit - ctrl
            if k <= cutoff:
                circ.cp(math.pi / (1 << k), ctrl, qubit)
            # else: silently drop — this is the approximation


def bit_reverse_swap(n: int, circ: QuantumCircuit) -> None:
    """Bit-reversal permutation (swap pairs to match standard QFT output order)."""
    for i in range(n // 2):
        circ.swap(i, n - 1 - i)


# ---------------------------------------------------------------------------
# PyQrack exact reference
# ---------------------------------------------------------------------------
def pyqrack_exact_probs(n: int) -> list[float]:
    """
    Run GHZ + QFT + bit-reversal exactly in PyQrack and return output
    probability vector of length 2^n.
    """
    circ = QuantumCircuit(n)
    build_ghz(n, circ)
    build_qft(n, circ)
    bit_reverse_swap(n, circ)

    sim = QrackSimulator(n)
    sim.run_qiskit_circuit(circ, shots=0)
    return sim.out_probs()


# ---------------------------------------------------------------------------
# Quimb MPS AQFT
# ---------------------------------------------------------------------------
def quimb_aqft_probs(n: int, max_bond: int, cutoff: int) -> list[float]:
    """
    Run GHZ + AQFT in Quimb CircuitMPS and return output probability vector.

    CircuitMPS maintains state as MPS with bounded bond dimension `max_bond`.
    Gate application is O(chi^2 * n) per gate.  Dropping long-range phase gates
    (the AQFT approximation) keeps the bond dimension from growing as fast as
    the exact QFT would require.
    """
    # Build the AQFT circuit via Qiskit for gate list, then replay into Quimb
    circ = QuantumCircuit(n)
    build_ghz(n, circ)
    build_aqft(n, circ, cutoff)
    bit_reverse_swap(n, circ)

    # Transpile to a gate set Quimb's CircuitMPS understands
    basis = ["h", "cx", "cp", "swap", "x", "rz", "s", "sdg", "t", "tdg"]
    circ_t = transpile(circ, optimization_level=2, basis_gates=basis)

    mps = tn.CircuitMPS(n, max_bond=max_bond, to_backend=jnp.array)

    QUIMB_GATE_MAP = {
        "h":    "H",
        "cx":   "CX",
        "swap": "SWAP",
        "x":    "X",
        "rz":   "RZ",
        "s":    "S",
        "sdg":  "Sdg",
        "t":    "T",
        "tdg":  "Tdg",
    }

    for instr in circ_t.data:
        gate_name = instr.operation.name.lower()
        qubits    = [circ_t.find_bit(q).index for q in instr.qubits]
        params    = [float(p) for p in instr.operation.params]

        if gate_name == "cp":
            # Controlled-phase: decompose as CRZ up to global phase
            # cp(theta) = diag(1, 1, 1, e^{i*theta})
            # Quimb has no native CP but accepts arbitrary 2-qubit unitaries
            theta = params[0]
            mat = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, complex(math.cos(theta), math.sin(theta))],
            ], dtype=complex)
            mps.apply_gate_raw(mat, qubits)
            continue

        quimb_name = QUIMB_GATE_MAP.get(gate_name)
        if quimb_name is None:
            # Fallback: apply as raw unitary if available
            u = instr.operation.to_matrix()
            mps.apply_gate_raw(u, qubits)
            continue

        if params:
            mps.apply_gate(quimb_name, *params, *qubits)
        else:
            mps.apply_gate(quimb_name, *qubits)

    # Extract full probability vector via single MPS contraction.
    # mps.to_dense() contracts the full MPS into a state vector — O(n * chi^2)
    # rather than O(2^n * chi^2) from calling amplitude() in a loop.
    sv = np.array(mps.to_dense()).flatten()
    probs = np.abs(sv) ** 2

    # Quimb's qubit ordering is MSB-first; Qrack/Qiskit is LSB-first.
    # Reverse the bit ordering of each index.
    n_pow = 1 << n
    reordered = np.zeros(n_pow, dtype=float)
    for i in range(n_pow):
        j = int(format(i, f"0{n}b")[::-1], 2)
        reordered[j] = probs[i]

    # Normalise (MPS approximation may not preserve exact norm)
    total = reordered.sum()
    if total > 0:
        reordered /= total

    return reordered.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n        = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    max_bond = int(sys.argv[2]) if len(sys.argv) > 2 else n
    cutoff   = int(sys.argv[3]) if len(sys.argv) > 3 else int(math.log2(n))

    print(f"n_qubits={n}  max_bond={max_bond}  aqft_cutoff={cutoff}")
    print()

    print("Running PyQrack exact QFT on GHZ input...")
    ideal = pyqrack_exact_probs(n)
    print(f"  ideal probs: {len(ideal)} entries, sum={sum(ideal):.6f}")
    print()

    print(f"Running Quimb MPS AQFT (max_bond={max_bond}, cutoff={cutoff})...")
    approx = quimb_aqft_probs(n, max_bond, cutoff)
    print(f"  approx probs: {len(approx)} entries, sum={sum(approx):.6f}")
    print()

    stats = calc_stats(ideal, approx)
    print("Results:")
    print(f"  XEB      : {stats['xeb']:.6f}")
    print(f"  HOG prob : {stats['hog_prob']:.6f}")
    print(f"  L2 diff  : {stats['l2_diff']:.6f}")
    print()

    # Also report gate count comparison
    circ_full = QuantumCircuit(n)
    build_ghz(n, circ_full)
    build_qft(n, circ_full)
    n_gates_full = circ_full.count_ops()

    circ_aqft = QuantumCircuit(n)
    build_ghz(n, circ_aqft)
    build_aqft(n, circ_aqft, cutoff)
    n_gates_aqft = circ_aqft.count_ops()

    cp_full = n_gates_full.get("cp", 0)
    cp_aqft = n_gates_aqft.get("cp", 0)
    print(f"Gate counts (cp gates only):")
    print(f"  Full QFT : {cp_full}")
    print(f"  AQFT     : {cp_aqft}  ({100*cp_aqft/max(cp_full,1):.1f}% retained)")


if __name__ == "__main__":
    main()
