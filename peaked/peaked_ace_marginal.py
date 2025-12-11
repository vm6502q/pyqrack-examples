# Marginal probability attack with Qrack's "automatic circuit elision" ACE
# by Daniel Strano (lead author and maintainer of Qrack)

# You likely want to set the following environment variables
# (and it's not possible to set them in a script-based way
# in a simple single Python script):

# QRACK_MAX_PAGING_QB / QRACK_MAX_CPU_QB - Sets a cap on largest entangled qubit subsystem size. For large circuits, set to half the total qubit count (or maybe 1/3, if 1/2 is too large, etc.).
# QRACK_QTENSORNETWORK_THRESHOLD_QB - Controls light-cone optimizations for memory and accuracy (which are slow, but only by a factor of ~n for n qubits). For fast (less-accurate) simulation, set much higher than the count of total qubits.
# QRACK_QUNIT_SEPARABILITY_THRESHOLD - Optionally, this rounds nearly-separable single qubits to total separable, in flight. That might help or hurt.
#     (A "golden" value for large, hard circuits seems to be (1 - 1 / sqrt(2)) / 2, approximately 0.1464466.)

import operator
import random
import sys

from collections import Counter

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile


epsilon = sys.float_info.epsilon


def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]

def run_qasm(file_in):
    # Load QASM as Qiskit circuit
    qc = QuantumCircuit.from_qasm_file(file_in)
    basis_gates = QrackSimulator.get_qiskit_basis_gates()
    qc = transpile(qc, basis_gates=basis_gates)
    n_qubits = qc.num_qubits
    shots = n_qubits ** 3

    accepted_bits = {}
    while len(accepted_bits.keys()) < n_qubits:
        # Outer loop accepts "stable" fixed bits on repeated trials
        rejected_bits = n_qubits
        accepted_mask = [True] * n_qubits
        accepted_value = [False] * n_qubits
        is_first_outer = True
        force_bit = []
        while rejected_bits > 0:
            # Inner loop fixes bits with "heavy" marginal probabilities until all bits are fixed
            fixed_bits = dict(accepted_bits)
            is_first_inner = True

            # Run the original circuit
            sim = QrackSimulator(n_qubits)
            sim.run_qiskit_circuit(qc, shots=0)

            while len(fixed_bits.keys()) < n_qubits:
                # Post-select any fixed bits
                for key, value in fixed_bits.items():
                    sim.force_m(key, value)

                # Find the single highest-probability dimension
                highest_prob = sim.highest_prob_perm()
                print(f"ACE highest-probability dimension: {highest_prob}")
                print(f"Left-to-right, least-to-most significant: {int_to_bitstring(highest_prob, n_qubits)}")

                # Sample the approximate distribution
                experiment_counts = dict(Counter(sim.measure_shots(list(range(n_qubits)), shots)))
                # experiment_counts[highest_prob] = shots
                experiment_counts = sorted(experiment_counts.items(), key=operator.itemgetter(1), reverse=True)
                experiment = None

                marginal = [0.0] * n_qubits

                for b in range(n_qubits):
                    if b in fixed_bits.keys():
                        # This bit is already fixed to an eigenstate
                        marginal[b] = fixed_bits[b]
                    elif (highest_prob >> b) & 1:
                        # The single highest probability carries 50% weight
                        marginal[b] = 0.5

                # The sampled distribution carries 50% of the overall marginal weight
                for count in experiment_counts:
                    key = count[0]
                    val = count[1] / (shots << 1)
                    for b in range(n_qubits):
                        if b in fixed_bits.keys():
                            continue
                        if (key >> b) & 1:
                            marginal[b] += val

                print(f"Est. marginals: {marginal}")
                ltr = ''.join(['0' if p < 0.5 else '1' for p in marginal])
                print(f"Left-to-right, least-to-most significant: {ltr}")

                # Calculate how polarized each single-qubit RDM is
                is_new_fixed = False
                max_polar = -1
                max_bit = []
                polar = [0.0] * n_qubits
                total_polar = 0.0
                for b, p in enumerate(marginal):
                    if b in fixed_bits.keys():
                        continue

                    if p < 0.5:
                        polar[b] = 0.5 - p
                    else:
                        polar[b] = p - 0.5

                    if (polar[b] + epsilon) >= 0.5:
                        val = 0 if marginal[b] < 0.5 else 1
                        fixed_bits[b] = val
                        sim.force_m(b, val)
                        is_new_fixed = True
                        continue

                    total_polar += polar[b]

                    if polar[b] > max_polar:
                        max_polar = polar[b]
                        max_bit = [b]
                    elif (polar == max_polar):
                        max_bit.append(b)

                if is_new_fixed:
                    # Update marginal probabilities after fixing floor/ceiling bits
                    continue

                # In case no bits are stably fixed in the other loop,
                # we'll use the most-polarized bit before the inner loop fixes any more bits
                if is_first_inner:
                    max_bit = random.choice(max_bit)
                    val = marginal[max_bit]
                    if abs(val - 0.5) <= epsilon:
                        # We're at the 50/50 point, so randomize
                        val = random.random() < 0.5
                    else:
                        # We're polarized, so tend towards the favored pole
                        val = val > 0.5
                    force_bit.append((max_bit, val))
                    is_first_inner = False

                if max_polar < 0:
                    # All bits must already be fixed
                    break

                # Polarization out of total is the probability to fix the bit in this round
                f_bit = 0
                if total_polar <= epsilon:
                    opt = []
                    for b in range(n_qubits):
                        if b not in fixed_bits.keys():
                            opt.append(b)
                    f_bit = random.choice(opt)
                else:
                    sum_polar = 0.0
                    total_polar *= random.random()
                    for b in range(n_qubits):
                        if b in fixed_bits.keys():
                            continue
                        sum_polar += polar[b]
                        f_bit = b
                        if total_polar < sum_polar:
                            break

                    val = marginal[f_bit]
                    if abs(val - 0.5) <= epsilon:
                        accepted_mask[f_bit] = False
                        val = random.random()

                # Unless multiple bits are at floor/ceiling, we fix one bit per round
                fixed_bits[f_bit] = 0 if val < 0.5 else 1

            if is_first_outer:
                # If this is the first run of the outer loop,
                # we're establishing the initial combination of fixed values
                accepted_value = dict(fixed_bits)
                is_first_outer = False
                rejected_bits = n_qubits
            else:
                # After the first iteration, we reject "unstable" bits
                # on every subsequent iteration.
                rejected_bits = 0
                for b, n in fixed_bits.items():
                    if not accepted_mask[b]:
                        continue

                    # Original vs. new bit
                    o = accepted_value[b]
                    if o != n:
                        # The bit is "unstable" and returned different values across iterations
                        accepted_mask[b] = False
                        rejected_bits += 1

        # Fix the top level "stable" bits
        is_accepted = False
        for b in range(n_qubits):
            if b in accepted_bits.keys():
                continue
            if accepted_mask[b]:
                accepted_bits[b] = 1 if accepted_value[b] else 0
                is_accepted = True

        # If 0 bits were fixed, use the "backup option"
        # of the most-polarized bit in first iteration of the inner loop
        # (so we're still guaranteed to converge in the end)
        if not is_accepted:
            t = random.choice(force_bit)
            accepted_bits[t[0]] = 1 if t[1] else 0

    # Final result
    ltr = ''
    for b in range(n_qubits):
        if b not in accepted_bits.keys():
            # (This never actually occurs)
            ltr += 'x'
        else:
            ltr += '1' if accepted_bits[b] else '0'

    print("Final convergence:")
    print(f"Left-to-right, least-to-most significant: {ltr}")

def main():
    file_in = "P3_sharp_peak.qasm"
    if len(sys.argv) > 1:
        file_in = str(sys.argv[1])

    run_qasm(file_in)

    return 0


if __name__ == "__main__":
    sys.exit(main())
