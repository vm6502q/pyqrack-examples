# Quantum volume protocol certification

import math
import random
import sys
import time
import tracemalloc

from pyqrack import QrackSimulator

from qiskit import QuantumCircuit

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

    lcv_range = range(n)
    all_bits = list(lcv_range)
    x_op = [0, 1, 1, 0]
    shots = 100

    circ = QuantumCircuit(n)
    for d in range(n):
        # Single-qubit gates
        for i in lcv_range:
            th = random.uniform(0, 2 * math.pi)
            ph = random.uniform(0, 2 * math.pi)
            lm = random.uniform(0, 2 * math.pi)
            circ.u(th, ph, lm, i)

        # 2-qubit couplers
        unused_bits = all_bits.copy()
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.cx(c, t)

    traced_memory_start = tracemalloc.get_traced_memory()
    peak_monitoring = False
    nvml_peak = 0
    pynvml.nvmlInit()
    peak_monitor_start()
    start = time.perf_counter()

    sim = QrackSimulator(n, isStabilizerHybrid=False)
    if sdrp > 0:
        sim.set_sdrp(sdrp)
    # Run the experiment.
    sim.run_qiskit_circuit(circ)
    sim.run_qiskit_circuit(circ.inverse())

    terminal = sim.measure_shots(all_bits, shots)
    fidelity_exp = terminal.count(0) / shots
    fidelity_est = sim.get_unitary_fidelity()

    del sim

    interval = time.perf_counter() - start
    traced_memory_end = tracemalloc.get_traced_memory()
    nvml_after = gpu_mem_used(0)
    peak_monitor_stop()
    tracemalloc.stop()

    return (
        interval,
        fidelity_est,
        fidelity_exp,
        (traced_memory_end[1] - traced_memory_start[0]) / 1024,
        (nvml_peak - nvml_after) / (1024 * 1024),
    )


def main():
    n = 20
    sdrp = 0
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        sdrp = float(sys.argv[2])
    n_pow = 1 << n

    results = bench_qrack(n, sdrp)

    interval = results[0]
    fidelity_est = results[1]
    fidelity_exp = results[2]
    memory_cpu = results[3]
    memory_gpu = results[4]

    print(
        {
            "qubits": n,
            "sdrp": sdrp,
            "seconds": interval,
            "fidelity_est": fidelity_est,
            "fidelity_exp": fidelity_exp,
            "peak_cpu_mb": memory_cpu,
            "peak_gpu_mb": memory_gpu,
        }
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
