# Quantum volume protocol certification

import math
import pynvml
import random
import statistics
import sys
import threading
import time
import tracemalloc

from pyqrack import QrackSimulator, QrackCircuit

from qiskit import QuantumCircuit

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
        if not peak_monitoring: break
        time.sleep(0.1) # 0.1sec


def bench_qrack(file, sdrp = 0):
    global nvml_peak, peak_monitoring

    circ = QuantumCircuit.from_qasm_file(file)
    n = circ.num_qubits
    circ = QrackCircuit.in_from_qiskit_circuit(circ)
    
    tracemalloc.start()
    traced_memory_start = tracemalloc.get_traced_memory()
    peak_monitoring = False
    nvml_peak = 0
    pynvml.nvmlInit()
    peak_monitor_start()
    start = time.perf_counter()

    sim = QrackSimulator(n, isTensorNetwork=False, isStabilizerHybrid=False)
    if sdrp > 0:
        sim.set_sdrp(sdrp)
    circ.run(sim)
    sim.m_all()
    fidelity = sim.get_unitary_fidelity()

    interval = time.perf_counter() - start
    traced_memory_end = tracemalloc.get_traced_memory()
    nvml_after = gpu_mem_used(0)
    peak_monitor_stop()
    tracemalloc.stop()

    return (n, interval, fidelity, (traced_memory_end[1] - traced_memory_start[0]) / 1024, (nvml_peak - nvml_after) / (1024 * 1024))


def main():
    file = "qft.qasm"
    sdrp = 0.3
    if len(sys.argv) > 1:
        file = str(sys.argv[1])
    if len(sys.argv) > 2:
        sdrp = float(sys.argv[2])

    results = bench_qrack(file, sdrp)

    n = results[0]
    interval = results[1]
    fidelity = results[2]
    memory_cpu = results[3]
    memory_gpu = results[4]

    print(n, "qubits,", sdrp, "SDRP:",
        interval, "seconds,",
        fidelity, "fidelity,",
        memory_cpu, "MB heap memory peak",
        memory_gpu, "MB GPU memory peak"
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
