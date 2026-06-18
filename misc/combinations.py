import sys

def make_tuples(l, N):
    if N <= 0:
        # Empty set
        return

    # "Count by least-significant digit,"
    # where "left-most," "lowest index" is "least-significant"
    t = [1] * l

    result = []

    if l == 1:
        i = 1
        result.append(tuple(t))
    else:
        i = 0

    high_bit = 0
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
        if len(t) != len(set(t)):
            continue
        s = tuple(sorted(t))
        if s in result:
            continue

        # Else, the striking rule did not act.
        i += 1
        result.append(s)

    return result


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python3 combinations.py [sequence length]")

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
        finite_set.append(make_tuples(_i + 1, i + 1))

    # Just Cantor's pairing function:
    output_set = []
    t = 0
    i, j = 0, 0
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

    print("---------")
    print("N, N×N, V")
    print("---------")

    ls = {}
    for k, v in enumerate(output_set):
        l = len(v)
        ls[l] = ls.get(l, 0) + 1
        c = ls[l] + 1j * l
        print(f"{k + 1}, {c}: {v}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
