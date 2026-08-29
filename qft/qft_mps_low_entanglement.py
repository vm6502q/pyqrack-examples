#!/usr/bin/env python3
"""
Low-entanglement classical simulation of the Quantum Fourier Transform,
via matrix product states, with the final bit-reversal deferred to
classical post-processing of the sampled bit strings.

Attribution
-----------
This technique -- and the underlying result that the QFT's apparent
"maximal operator entanglement" is entirely an artifact of the final
bit-reversal permutation, while the core Hadamard-plus-controlled-phase
circuit has small, bounded entanglement regardless of qubit count -- is
the result of, and owed entirely to:

    Jielun Chen, E. Miles Stoudenmire, Steven R. White,
    "Quantum Fourier Transform Has Small Entanglement",
    PRX Quantum 4, 040318 (2023).
    arXiv:2210.08468 [quant-ph]
    https://arxiv.org/abs/2210.08468
    https://doi.org/10.1103/PRXQuantum.4.040318

This script is an independent demonstration of that result using the
`quimb` tensor-network library, offered with attribution and without
claim to the original finding, which belongs to the authors above.

What this demonstrates
-----------------------
The standard QFT circuit is usually drawn as: Hadamard + controlled-phase
gates ("the core"), followed by a swap network that reverses the qubit
order to match the conventional output-labeling convention. That swap
network is a pure permutation of computational basis labels -- it moves
no genuine entanglement, since it is not able to entangle even a
completely  unentangled ("product") state.

The paper shows this permutation is nonetheless responsible for the
"maximal operator entanglement" often attributed to the full QFT: the
core circuit alone has small, bounded operator entanglement across any
positional bipartition of the register, independent of qubit count.

Because a computational-basis permutation commutes past a final
measurement (measuring a permuted state and permuting a measured
outcome give identical distributions), the bit-reversal step can be
skipped entirely in simulation whenever the QFT is the last operation
before readout: simulate only the core circuit (cheap, bounded bond
dimension), sample from it, and classically reverse the bit order of
each sampled string. This script verifies that this reproduces the
exact QFT output distribution, then demonstrates the resulting
simulation cost staying flat, in bond dimension, out to qubit counts
where dense state-vector simulation is not remotely possible.

Requires: quimb (pip install quimb)
"""

import time
from collections import Counter

import numpy as np
import quimb.tensor as qtn


def build_qft_core_mps(n, psi0, max_bond=None):
    """Build the QFT "core" circuit -- Hadamard + controlled-phase gates
    only, deliberately WITHOUT the final bit-reversal swap network --
    as a matrix product state circuit. This is the circuit whose
    operator entanglement the referenced paper shows is small and
    bounded, regardless of n.
    """
    circ = qtn.CircuitMPS(n, psi0=psi0, max_bond=max_bond)
    for j in range(n):
        circ.h(j)
        for k in range(j + 1, n):
            angle = np.pi / (2 ** (k - j))
            circ.cu1(angle, j, k)
    return circ


def reverse_bits(bitstring):
    """Classical bit-reversal of a sampled outcome string -- the entire
    cost of "undoing" the swap network that was never applied."""
    return bitstring[::-1]


def random_product_state(n, seed=0):
    """A generic (non-trivial) product-state input. |0...0> is a
    degenerate case for the QFT specifically -- QFT|0...0> is always
    the trivial uniform superposition |+>^n regardless of whether the
    swap network is applied, since a definite |0> control makes every
    controlled-phase gate a no-op. A generic input is needed to see the
    entanglement structure this script is actually about."""
    rng = np.random.default_rng(seed)
    arrays = []
    for _ in range(n):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        arrays.append(v)
    return qtn.MPS_product_state(arrays)


def exact_qft_probs(psi0_dense, n):
    """Reference: exact, dense, standard-ordered QFT output
    distribution. Only tractable for small n -- used here purely to
    verify correctness of the low-entanglement approach, not as part
    of it."""
    N = 2 ** n
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega ** (j * k) for k in range(N)] for j in range(N)]) / np.sqrt(N)
    out = F @ psi0_dense
    return np.abs(out) ** 2


def verify_correctness(n=8, n_samples=20000, seed=42):
    """Confirm that sampling from the swap-free core circuit and
    classically bit-reversing each sample reproduces the exact,
    standard-ordered QFT output distribution."""
    print(f"--- Correctness check: n={n} qubits, {n_samples} samples ---")

    psi0 = random_product_state(n, seed=seed)
    psi0_dense = psi0.to_dense().flatten()

    circ = build_qft_core_mps(n, psi0)
    print(f"core circuit max bond dimension: {circ.psi.max_bond()} "
          f"(vs. {2 ** (n // 2)} max possible for a naive positional cut)")

    raw_samples = list(circ.sample(n_samples, seed=seed))
    reversed_samples = [reverse_bits(s) for s in raw_samples]

    counts = Counter(reversed_samples)
    empirical_probs = np.zeros(2 ** n)
    for bitstring, c in counts.items():
        empirical_probs[int(bitstring, 2)] = c / n_samples

    exact_probs = exact_qft_probs(psi0_dense, n)

    tvd = 0.5 * np.sum(np.abs(empirical_probs - exact_probs))
    print(f"total variation distance (sampled+reversed vs. exact): {tvd:.5f}")
    print("(expect a small, statistical-noise-scale value, shrinking with more samples)\n")

    print("Top-5 highest-probability outcomes, exact vs. empirical:")
    top5 = np.argsort(exact_probs)[-5:][::-1]
    for idx in top5:
        b = format(idx, f"0{n}b")
        print(f"  {b}: exact={exact_probs[idx]:.5f}  empirical={empirical_probs[idx]:.5f}")
    print()


def demonstrate_scaling(qubit_counts=(20, 40, 60, 80), n_samples=100):
    """Show simulation cost (bond dimension, wall time) staying tractable
    at qubit counts where dense state-vector simulation is not
    remotely possible."""
    print("--- Scaling demonstration ---")
    print(f"{'n qubits':>10} {'max bond dim':>14} {'build (s)':>11} "
          f"{'sample (s)':>12} {'dense would need':>20}")
    for n in qubit_counts:
        psi0 = random_product_state(n, seed=n)

        t0 = time.time()
        circ = build_qft_core_mps(n, psi0)
        build_time = time.time() - t0

        t0 = time.time()
        list(circ.sample(n_samples, seed=1))
        sample_time = time.time() - t0

        dense_gb = 2 ** n * 16 / 1e9  # complex128 state vector
        print(f"{n:>10} {circ.psi.max_bond():>14} {build_time:>11.3f} "
              f"{sample_time:>12.3f} {dense_gb:>17.2e} GB")
    print()


if __name__ == "__main__":
    verify_correctness()
    demonstrate_scaling()
