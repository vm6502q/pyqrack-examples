"""
permutations_seq.py  --  Strano construction (permutations variant)
Sequential generator: yields tuples of positive integers (with repetition allowed)
in Cantor-pairing-diagonal order.
Credit: Dan Strano (construction); sequential rewrite assisted by Claude.
"""
import sys


def make_length_gen(length):
    """Generate tuples of given length (repetition allowed), same order as
    original make_tuples without striking rule."""
    t = [1] * length
    yield tuple(t)

    high_bit = 0
    i = 1

    while True:
        broke = False
        for j in range(length - 1):
            if t[j] <= t[j + 1]:
                t[j] += 1
                if (j + 1) > high_bit:
                    high_bit = j + 1
                for k in range(j):
                    t[k] = 1
                broke = True
                break

        if not broke:
            t[high_bit] += 1
            for k in range(high_bit):
                t[k] = 1

        i += 1
        yield tuple(t)


def permutations_stream():
    """Yield tuples interleaved across lengths by Cantor pairing diagonal."""
    length_gens = {}
    diagonal = 0
    while True:
        for j in range(diagonal + 1):
            length = j + 1
            if length not in length_gens:
                length_gens[length] = make_length_gen(length)
            yield next(length_gens[length])
        diagonal += 1


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python3 permutations_seq.py [n]")

    n = int(sys.argv[1])

    print("---------")
    print("N, N×N, V")
    print("---------")

    length_counts = {}
    stream = permutations_stream()
    for t in range(n):
        perm = next(stream)
        length = len(perm)
        length_counts[length] = length_counts.get(length, 0) + 1
        c = length_counts[length] + 1j * length
        print(f"{t + 1}, {c}: {perm}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
