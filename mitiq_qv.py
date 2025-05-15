# See "Error mitigation increases the effective quantum volume of quantum computers," https://arxiv.org/abs/2203.05489
#
# Mitiq is under the GPL 3.0.
# Hence, this example, as the entire work-in-itself, must be considered to be under GPL 3.0.
# See https://www.gnu.org/licenses/gpl-3.0.txt for details.

import math
import random
import statistics
import sys
import time

import numpy as np

from collections import Counter

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector

from mitiq import zne
from mitiq.zne.scaling.folding import fold_global
from mitiq.zne.inference import RichardsonFactory


def calc_stats(ideal_probs, counts, shots):
    # For QV, we compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    n_pow = len(ideal_probs)
    n = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    diff_sqr = 0
    numer = 0
    denom = 0
    sum_hog_counts = 0
    experiment = [0] * n_pow
    for i in range(n_pow):
        count = counts[i] if i in counts else 0
        ideal = ideal_probs[i]

        experiment[i] = count

        # L2 distance
        diff_sqr += (ideal - (count / shots)) ** 2

        # XEB / EPLG
        denom += (ideal - u_u) ** 2
        numer += (ideal - u_u) * ((count / shots) - u_u)

        # QV / HOG
        if ideal > threshold:
            sum_hog_counts += count

    l2_similarity = 1 - diff_sqr ** (1/2)
    hog_prob = sum_hog_counts / shots
    xeb = numer / denom

    return {
        'l2_similarity': l2_similarity,
        'xeb': xeb,
        'hog_prob': hog_prob,
    }


def random_circuit(width, depth):
    # This is a "nearest-neighbor" coupler random circuit.
    control = AerSimulator(method="statevector")
    shots = 1 << (width + 2)

    lcv_range = range(width)
    all_bits = list(lcv_range)

    circ = QuantumCircuit(width)
    for d in range(depth):
        # Single-qubit gates
        for i in lcv_range:
            for _ in range(3):
                circ.h(i)
                circ.rz(random.uniform(0, 2 * math.pi), i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.cx(c, t)

    return circ

def logit(x):
    # Theoretically, these limit points are "infinite,"
    # but precision caps out between 36 and 37:
    if x > (1 - 5e-17):
        return 37
    # For the negative limit, the precisions caps out
    # between -37 and -38
    elif x < 1e-17:
        return -38
    return max(-38, min(37, np.log(x / (1 - x))))


def expit(x):
    # Theoretically, these limit points are "infinite,"
    # but precision caps out between 36 and 37:
    if x >= 37:
        return 1.0
    # For the negative limit, the precisions caps out
    # between -37 and -38
    elif x <= -38:
        return 0.0
    return 1 / (1 + np.exp(-x))


def execute(circ):
    """Returns the mirror circuit expectation value for unsigned integer overall bit string."""

    shots = 1 << (circ.width() + 2)
    all_bits = list(range(circ.width()))
    
    experiment = QrackSimulator(circ.width())
    experiment.run_qiskit_circuit(circ)

    control = AerSimulator(method="statevector")
    circ_aer = transpile(circ, backend=control)
    circ_aer.save_statevector()
    job = control.run(circ_aer)

    experiment_counts = dict(Counter(experiment.measure_shots(all_bits, shots)))
    control_probs = Statevector(job.result().get_statevector()).probabilities()

    stats = calc_stats(control_probs, experiment_counts, shots)

    # So as not to exceed floor at 0.0 and ceiling at 1.0, (assuming 0 < p < 1,)
    # we mitigate its logit function value (https://en.wikipedia.org/wiki/Logit)
    return logit(stats['hog_prob'])
    # return logit(stats['xeb'])
    # return logit(stats['l2_similarity'])


def main():
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python3 fc.py [width] [depth]')

    width = int(sys.argv[1])
    depth = int(sys.argv[2])

    circ = random_circuit(width, depth)

    scale_count = 10
    max_scale = 2
    factory = RichardsonFactory(scale_factors=[(1 + (max_scale - 1) * x / scale_count) for x in range(0, scale_count)])

    mitigated_fidelity = expit(zne.execute_with_zne(circ, execute, scale_noise=fold_global, factory=factory))

    print({ 'width': width, 'depth': depth, 'mitigated_fidelity': mitigated_fidelity })

    return 0


if __name__ == '__main__':
    sys.exit(main())
