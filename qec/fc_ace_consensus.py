# Fully-connected RCS: ACE majority-vote consensus via Qiskit transpilation.
#
# Three QrackAceBackend instances run the same circuit with three different
# cyclic qubit permutations, so patch boundaries fall on different logical
# qubits in each instance. The circuit is rebuilt and transpiled fresh each
# layer (top-to-bottom), with AceQasmSimulator providing the noise model for
# transpilation. Shots are pooled after inverting the qubit permutation.
#
# XEB and HOG computed against AerSimulator statevector ideal.
#
# By Dan Strano and (Anthropic) Claude.

import math
import os
import random
import statistics
import sys
import time
from collections import Counter

import numpy as np
from scipy.stats import binom

from pyqrack import QrackAceBackend
from qiskit.providers.qrack import AceQasmSimulator

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer.backends import AerSimulator
from qiskit.quantum_info import Statevector


# ---------------------------------------------------------------------------
# Qubit permutation maps
# ---------------------------------------------------------------------------

def _make_qubit_maps(width, n_inst):
    maps     = []
    inv_maps = []
    for inst in range(n_inst):
        shift = inst * width // n_inst
        fwd = [(q + shift) % width for q in range(width)]
        inv = [0] * width
        for rcs_q, ace_q in enumerate(fwd):
            inv[ace_q] = rcs_q
        maps.append(fwd)
        inv_maps.append(inv)
    return maps, inv_maps


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calc_stats(ideal_probs, counts, depth, shots):
    n_pow     = len(ideal_probs)
    n         = int(round(math.log2(n_pow)))
    threshold = statistics.median(ideal_probs)
    u_u       = statistics.mean(ideal_probs)
    numer = 0; denom = 0; sum_hog = 0
    for i in range(n_pow):
        count = counts.get(i, 0)
        ideal = ideal_probs[i]
        denom    += (ideal - u_u) ** 2
        numer    += (ideal - u_u) * ((count / shots) - u_u)
        if ideal > threshold:
            sum_hog += count
    hog_prob = sum_hog / shots
    xeb      = numer / denom if denom else 0.0
    p_val    = (1 - binom.cdf(sum_hog - 1, shots, 0.5)) if sum_hog > 0 else 1.0
    return {"qubits": n, "depth": depth,
            "xeb": xeb, "hog_prob": hog_prob, "p-value": p_val}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_qrack(width, depth, trials=1):
    n_inst    = 3
    shots     = 1 << (width + 2)
    lcv_range = range(width)
    all_bits  = list(lcv_range)

    control      = AerSimulator(method="statevector")
    noise_dummy  = AceQasmSimulator(n_qubits=width, long_range_columns=2)
    maps, inv_maps = _make_qubit_maps(width, n_inst)

    results = []

    for trial in range(trials):
        t_trial = time.perf_counter()

        # Fresh ACE instances each trial
        aces = [QrackAceBackend(width) for _ in range(n_inst)]
        if "QRACK_QUNIT_SEPARABILITY_THRESHOLD" not in os.environ:
            for ace in aces:
                for sim in ace.sim:
                    sim.set_sdrp(0.03)

        # Build full circuit incrementally, one layer at a time
        full_circ = QuantumCircuit(width)

        for d in range(depth):
            t_layer = time.perf_counter()

            # Single-qubit layer
            for i in lcv_range:
                th = random.uniform(0, 2 * math.pi)
                ph = random.uniform(0, 2 * math.pi)
                lm = random.uniform(0, 2 * math.pi)
                full_circ.u(th, ph, lm, i)

            # 2-qubit couplers (same pairs for all instances)
            pairs = []
            unused = all_bits[:]
            random.shuffle(unused)
            while len(unused) > 1:
                c = unused.pop(); t = unused.pop()
                full_circ.cx(c, t)
                pairs.append((c, t))

            # -------------------------------------------------------------------
            # Ideal ground truth via AerSimulator statevector
            # -------------------------------------------------------------------
            circ_aer = transpile(full_circ, optimization_level=3, backend=control)
            circ_aer.save_statevector()
            ideal_probs = list(
                Statevector(control.run(circ_aer).result().get_statevector()).probabilities()
            )

            # -------------------------------------------------------------------
            # ACE consensus: three instances with permuted qubit indices.
            # Rebuild and transpile the full circuit top-to-bottom for each instance,
            # using the permuted qubit map. Pool shots after inverse permutation.
            # -------------------------------------------------------------------
            pooled = Counter()
            total_shots = 0

            for inst, ace in enumerate(aces):
                # Build permuted circuit directly — no pre-transpile needed
                # since run_qiskit_circuit handles u and cx natively.
                pcirc = QuantumCircuit(width)
                for gate in full_circ.data:
                    name   = gate.operation.name
                    qidxs  = [maps[inst][q._index] for q in gate.qubits]
                    params = list(gate.operation.params)
                    if name in ('u', 'u3'):
                        pcirc.u(params[0], params[1], params[2], qidxs[0])
                    elif name == 'cx':
                        pcirc.cx(qidxs[0], qidxs[1])
                circ_ace = transpile(pcirc, optimization_level=3,
                                     basis_gates=list(noise_dummy.DEFAULT_CONFIGURATION['basis_gates']))
                circ_ace.measure_all()

                # Fresh instance for each depth (top-to-bottom replay)
                ace_inst = QrackAceBackend(width)
                if "QRACK_QUNIT_SEPARABILITY_THRESHOLD" not in os.environ:
                    for sim in ace_inst.sim:
                        sim.set_sdrp(0.03)
                ace_inst.run_qiskit_circuit(circ_ace)

                raw_shots = ace_inst.measure_shots(all_bits, shots)
                for raw in raw_shots:
                    canonical = 0
                    for ace_q in range(width):
                        if (raw >> ace_q) & 1:
                            canonical |= 1 << inv_maps[inst][ace_q]
                    pooled[canonical] += 1
                total_shots += shots
                del ace_inst

            stats = calc_stats(ideal_probs, pooled, d + 1, total_shots)
            stats["layer_seconds"] = time.perf_counter() - t_layer

            if trial == 0:
                results.append(stats)
            else:
                results[d]["xeb"]      += stats["xeb"]
                results[d]["hog_prob"] += stats["hog_prob"]
                results[d]["p-value"]  *= stats["p-value"]

            if trial == trials - 1:
                results[d]["xeb"]      /= trials
                results[d]["hog_prob"] /= trials
                results[d]["p-value"]   = results[d]["p-value"] ** (1 / trials)
                print(results[d])

        for ace in aces:
            del ace

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python3 fc_ace_consensus_qiskit.py [width] [depth] [trials=1]")
    width  = int(sys.argv[1])
    depth  = int(sys.argv[2])
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    bench_qrack(width, depth, trials)
    return 0

if __name__ == "__main__":
    sys.exit(main())
