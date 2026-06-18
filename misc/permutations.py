import sys

def make_tuples(l, N):
    if N <= 0:
        # Empty set
        return

    # "Count by least-significant digit,"
    # where "left-most," "lowest index" is "least-significant"
    t = [1] * l

    yield tuple([j for j in t])

    high_bit = 0
    i = 1
    while i < N:
        broke = False
        for j in range(l - 1):
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

        # Striking rule (for exact power set):
        # lowest = t[0]
        # dupe = False
        # for j in range(1, len(t)):
        #     if t[j] >= lowest:
        #         dupe = True
        #         break
        #     lowest = t[j]
        # if dupe:
        #     continue

        # Else, the striking rule did not act.
        i += 1
        yield tuple([j for j in t])


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python3 permutations.py [sequence length]")

    n = int(sys.argv[1])

    # Square root of (2 * n) (equivalent, minimum side of triangle)
    r = (n << 1) ** (1 / 2)
    side_len = int(r)
    if r != side_len:
        side_len += 1

    finite_set = []
    for _i in range(side_len):
        # Iterate forward / backward
        i = side_len - (_i + 1)
        subset = [j for j in make_tuples(_i + 1, i + 1)]
        finite_set.append(subset)

    # Just Cantor's pairing function:
    output_set = []
    t = 0
    i = 0
    j = 0
    while t < n:
        # print((i, j))
        output_set.append(finite_set[j][i])
        if i == 0:
            i = j + 1
            j = 0
        else:
            i -= 1
            j += 1
        t += 1

    print(output_set)

    return 0

if __name__ == "__main__":
    sys.exit(main())
