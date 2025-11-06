# Quantum volume protocol certification

import math
import random
import statistics
import sys
import time
import tracemalloc

from pyqrack import QrackSimulator, QrackCircuit

import threading, pynvml


# See https://discuss.pytorch.org/t/measuring-peak-memory-usage-tracemalloc-for-pytorch/34067/6
# for GPU memory usage monitoring
def gpu_mem_used(id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used


def peak_monitor_start():
    global peak_monitoring
    peak_monitoring = True

    # this thread samples RAM usage as long as the current epoch of the fit loop is running
    peak_monitor_thread = threading.Thread(target=peak_monitor_func)
    peak_monitor_thread.daemon = True
    peak_monitor_thread.start()


def peak_monitor_stop():
    global peak_monitoring
    peak_monitoring = False


def peak_monitor_func():
    global nvml_peak, peak_monitoring
    nvml_peak = 0

    while True:
        nvml_peak = max(gpu_mem_used(0), nvml_peak)
        if not peak_monitoring:
            break
        time.sleep(0.1)  # 0.1sec


def bench_qrack(n, sdrp=0):
    global nvml_peak, peak_monitoring

    # This is a "quantum volume" (random) circuit.
    circ = QrackCircuit()

    lcv_range = range(n)
    all_bits = list(lcv_range)
    x_op = [0, 1, 1, 0]

    for _ in lcv_range:
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            cos0 = math.cos(th / 2)
            sin0 = math.sin(th / 2)
            u_op = [
                cos0 + 0j,
                sin0 * (-math.cos(lm) + -math.sin(lm) * 1j),
                sin0 * (math.cos(ph) + math.sin(ph) * 1j),
                cos0 * (math.cos(ph + lm) + math.sin(ph + lm) * 1j),
            ]
            circ.mtrx(u_op, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            circ.ucmtrx([unused_bits.pop()], x_op, unused_bits.pop(), 1)

    sim = QrackSimulator(n, isTensorNetwork=False)
    circ.run(sim)
    ideal_probs = [(x * (x.conjugate())).real for x in sim.out_ket()]
    del sim

    tracemalloc.start()
    traced_memory_start = tracemalloc.get_traced_memory()
    peak_monitoring = False
    nvml_peak = 0
    pynvml.nvmlInit()
    peak_monitor_start()
    start = time.perf_counter()

    sim = QrackSimulator(n, isTensorNetwork=False)
    if sdrp > 0:
        sim.set_sdrp(sdrp)
    circ.run(sim)

    interval = time.perf_counter() - start
    traced_memory_end = tracemalloc.get_traced_memory()
    nvml_after = gpu_mem_used(0)
    peak_monitor_stop()
    tracemalloc.stop()

    fidelity = sim.get_unitary_fidelity()
    approx_probs = [(x * (x.conjugate())).real for x in sim.out_ket()]

    return (
        ideal_probs,
        approx_probs,
        interval,
        fidelity,
        (traced_memory_end[1] - traced_memory_start[0]) / 1024,
        (nvml_peak - nvml_after) / (1024 * 1024),
    )


def main():
    n = 20
    sdrp = 1 / 3
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        sdrp = float(sys.argv[2])
    n_pow = 1 << n

    results = bench_qrack(n, sdrp)

    ideal_probs = results[0]
    approx_probs = results[1]
    interval = results[2]
    fidelity = results[3]
    memory_cpu = results[4]
    memory_gpu = results[5]

    # We compare probabilities of (ideal) "heavy outputs."
    # If the probability is above 2/3, the protocol certifies/passes the qubit width.
    threshold = statistics.median(ideal_probs)
    u_u = statistics.mean(ideal_probs)
    numer = 0
    denom = 0
    hog_prob = 0
    for i in range(n_pow):
        ideal = ideal_probs[i]
        experimental = approx_probs[i]

        # XEB / EPLG
        ideal_centered = ideal - u_u
        denom += ideal_centered * ideal_centered
        numer += ideal_centered * (experimental - u_u)

        # QV / HOG
        if ideal > threshold:
            hog_prob += experimental

    xeb = numer / denom

    print(
        {
            "qubits": n,
            "sdrp": sdrp,
            "seconds": interval,
            "worst_case_fidelity": fidelity,
            "xeb": xeb,
            "hog_prob": hog_prob,
            "pass": (hog_prob >= 2 / 3),
            "peak_cpu_mb": memory_cpu,
            "peak_gpu_mb": memory_gpu,
        }
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
