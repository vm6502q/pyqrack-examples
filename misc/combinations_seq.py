"""
combinations_seq.py  --  Strano construction (combinations / power-set variant)
Sequential generator: yields finite subsets of N in Cantor-pairing-diagonal order.
Credit: Dan Strano (construction); sequential rewrite assisted by Claude.
"""
import sys


def make_length_gen(length):
    """Generate strictly-increasing tuples of given length, same order as
    original make_tuples with striking rule."""
    t = [1] * length
    seen = set()

    if length == 1:
        yield (1,)

    high_bit = 0
    i = 1 if length == 1 else 0

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

        if len(t) != len(set(t)):
            continue
        s = tuple(sorted(t))
        if s in seen:
            continue

        seen.add(s)
        i += 1
        yield s


def combinations_stream():
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
        raise RuntimeError("Usage: python3 combinations_seq.py [n]")

    n = int(sys.argv[1])

    print("---------")
    print("N, N×N, V")
    print("---------")

    length_counts = {}
    stream = combinations_stream()
    for t in range(n):
        combo = next(stream)
        length = len(combo)
        length_counts[length] = length_counts.get(length, 0) + 1
        c = length_counts[length] + 1j * length
        print(f"{t + 1}, {c}: {combo}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
