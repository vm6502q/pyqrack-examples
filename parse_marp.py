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
                # Last sample
                avg_fidelity = avg_fidelity + fidelity
                avg_time = avg_time + time
                break
            d = eval(line)
            if d['sdrp'] == 1:
                # Update count
                sample_count = sample_count + 1
                # Finalize last samples
                avg_fidelity = avg_fidelity + fidelity
                avg_time = avg_time + time
            fidelity = d['fidelity']
            time = d['time']

    avg_fidelity = avg_fidelity / sample_count
    avg_time = avg_time / sample_count

    print({
        "avg_fidelity:": avg_fidelity,
        "avg_seconds": avg_time
    })

    return 0


if __name__ == '__main__':
    sys.exit(main())
