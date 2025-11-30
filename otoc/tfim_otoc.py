# Ising model Trotterization
# by Dan Strano and (OpenAI GPT) Elara

# We reduce transverse field Ising model for globally uniform J and h parameters from a 2^n-dimensional problem to an (n+1)-dimensional approximation that suffers from no Trotter error. Upon noticing most time steps for Quantinuum's parameters had roughly a quarter to a third (or thereabouts) of their marginal probability in |0> state, it became obvious that transition to and from |0> state should dominate the mechanics. Further, the first transition tends to be to or from any state with Hamming weight of 1 (in other words, 1 bit set to 1 and the rest reset 0, or n bits set for Hamming weight of n). Further, on a torus, probability of all states with Hamming weight of 1 tends to be exactly symmetric. Assuming approximate symmetry in every respective Hamming weight, the requirement for the overall probability to converge to 1.0 or 100% in the limit of an infinite-dimensional Hilbert space suggests that Hamming weight marginal probability could be distributed like a geometric series. A small correction to exact symmetry should be made to favor closeness of "like" bits to "like" bits (that is, geometric closeness on the torus of "1" bits to "1" bits and "0" bits to "0" bits), but this does not affect average global magnetization. Adding an oscillation component with angular frequency proportional to J, we find excellent agreement with Trotterization approaching the limit of infinitesimal time step, for R^2 (coefficient of determination) of normalized marginal probability distribution of ideal Trotterized simulation as described by the (n+1)-dimensional approximate model, as well as for R^2 and RMSE (root-mean-square error) of global magnetization curve values.

import math
import numpy as np
import statistics
import sys

from collections import Counter

from pyqrackising import generate_otoc_samples


def main():
    n_qubits = 16
    depth = 16
    t1 = 0
    t2 = 1
    omega = 1.5

    J, h, dt, z = -1.0, 2.0, 0.125, 4
    cycles = 3

    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        cycles = int(sys.argv[3])
    if len(sys.argv) > 4:
        butterfly_fraction = float(sys.argv[4])
    else:
        butterfly_fraction = 1 / n_qubits

    butterfly_count = int(np.round(butterfly_fraction * n_qubits))
    if butterfly_count < 1:
        raise ValueError("Butterfly fraction would select 0 qubits!")

    omega *= math.pi
    qubits = list(range(n_qubits))

    # 1/8 butterfly qubits
    ops = ['X', 'Y', 'Z']
    pauli_strings = []
    for cycle in range(cycles):
        # Add the out-of-time-order perturbation
        string = ['I'] * n_qubits
        butterfly_qubits = np.random.choice(qubits, size=butterfly_count, replace=False)
        for b in butterfly_qubits:
            string[b] = np.random.choice(ops)
        pauli_strings.append("".join(string))

    shots = 1<<(n_qubits + 2)
    experiment_probs = dict(Counter(generate_otoc_samples(n_qubits=n_qubits, J=J, h=h, z=z, theta=0, t=dt*depth, shots=shots, pauli_strings=pauli_strings)))
    experiment_probs = { k: v / shots for k, v in experiment_probs.items() }

    print({
        "qubits": n_qubits,
        "depth": depth,
        "shots": shots,
        "pauli_strings": pauli_strings,
        "marginal_prob": experiment_probs
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
