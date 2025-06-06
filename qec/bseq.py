# Example of nonlocal measurement collapse with QrackAceBackend
# (Produced by Elara, a custom OpenAI GPT)

import math
import sys
from pyqrack import QrackAceBackend, Pauli

def apply_measurement_basis(sim, q, theta):
    # Rotate qubit q into the measurement basis
    sim.r(Pauli.PauliY, theta, q)

def measure_expectation(sim, shots):
    results = sim.measure_shots([0, 1], shots)
    total = 0
    for res in results:
        a = (res >> 0) & 1
        b = (res >> 1) & 1
        total += 1 if a == b else -1
    return total / shots

def run_chsh_test(shots):
    sim = QrackAceBackend(9, long_range_columns=1)
    sim.h(0)
    sim.mcx([0], 1)

    pi = math.pi

    # CHSH optimal angles (in radians)
    theta_a = 0
    theta_ap = pi / 2
    theta_b = pi / 4
    theta_bp = -pi / 4

    def expectation(theta1, theta2):
        s = QrackAceBackend(9, long_range_columns=1)
        s.h(0)
        s.mcx([0], 1)
        apply_measurement_basis(s, 0, theta1)
        apply_measurement_basis(s, 1, theta2)
        return measure_expectation(s, shots)

    E_ab = expectation(theta_a, theta_b)
    E_abp = expectation(theta_a, theta_bp)
    E_apb = expectation(theta_ap, theta_b)
    E_apbp = expectation(theta_ap, theta_bp)

    S = abs(E_ab + E_abp + E_apb - E_apbp)
    print("E(a, b)     =", E_ab)
    print("E(a, b')    =", E_abp)
    print("E(a', b)    =", E_apb)
    print("E(a', b')   =", E_apbp)
    print("CHSH S-value:", S)
    print("Bell inequality violated?", "Yes" if S > 2 else "No")

if __name__ == "__main__":
    run_chsh_test(2048)
