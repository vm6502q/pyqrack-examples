import sys

def main():
    file = "marp.txt"
    if len(sys.argv) > 1:
        file = str(sys.argv[1])

    avg_fidelity = 0
    avg_time = 0
    sample_count = 0
    with open(file, 'r') as in_file:
        fidelity = 0
        time = 0
        while True:
            line = in_file.readline()
            if not line:
                break
            d = eval(line)
            fidelity = d['fidelity']
            time = d['time']
            if d['sdrp'] == 1:
                sample_count = sample_count + 1
                avg_fidelity = avg_fidelity + fidelity
                avg_time = avg_time + time

    avg_fidelity = avg_fidelity / sample_count
    avg_time = avg_time / sample_count

    print({
        avg_fidelity, "avg. fidelity,",
        avg_time, "avg. seconds"
    })

    return 0


if __name__ == '__main__':
    sys.exit(main())
