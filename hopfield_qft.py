import math
import random
import sys
import time

from pyqrack import QrackSimulator, QrackNeuron


# See https://stackoverflow.com/questions/8898807/pythonic-way-to-iterate-over-bits-of-integer#answer-8898977
def bits(n):
    while n:
        b = n & (~n+1)
        yield b
        n ^= b


def main():
    start = time.perf_counter()
    width = 4
    if len(sys.argv) > 1:
        width = int(sys.argv[1])
    width_x2 = width << 1
    sim = QrackSimulator(width_x2)
    i_range = range(width)
    o_range = range(width, width_x2)
    pow_width = 1 << width
    eta = 1 / pow_width

    neurons = []
    for b in o_range:
        neurons.append(QrackNeuron(sim, i_range, b))

    for p in range(pow_width):
        # Prepare input and output.
        sim.reset_all()
        for b in range(width):
            if (p >> b) & 1:
                sim.x(b)
                sim.x(b + width)
        sim.qft(o_range)

        # Learn
        for b in i_range:
            neurons[b].learn_permutation(eta, True, False)

    sim.reset_all()
    train_time = time.perf_counter() - start

    with open(f"qft_{width}.csv", "w") as f:
        for b in i_range:
            f.write(str(neurons[b].get_angles()))

    start = time.perf_counter()
    for i in i_range:
        sim.u(i, random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))
    for b in i_range:
        neurons[b].predict()
    sim.m_all()
    predict_time = time.perf_counter() - start

    print(width, "qubits input; Train: ", train_time, "seconds, Predict:" , predict_time, "seconds.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
