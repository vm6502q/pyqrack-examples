import sys

def main():
    file = "marp.txt"
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python3 parse_marp.py [file_name]')

    file = str(sys.argv[1])

    avg_fidelity = {}
    avg_sdrp_time = {}
    avg_marp_time = {}
    trial_count = {}
    with open(file, 'r') as in_file:
        depth = 0
        fidelity = 0
        time = 0
        while True:
            line = in_file.readline()
            if not line:
                # Last sample
                if depth > 0:
                    if depth in avg_fidelity.keys():
                        avg_fidelity[depth] = avg_fidelity[depth] + fidelity
                        avg_sdrp_time[depth] = avg_sdrp_time[depth] + time
                    else:
                        avg_fidelity[depth] = fidelity
                        avg_sdrp_time[depth] = time
                break
            d = eval(line)
            if d['sdrp'] == 1:
                # Update count
                dpth = d['depth']
                if dpth in trial_count.keys():
                    trial_count[dpth] = trial_count[dpth] + 1
                else:
                    trial_count[dpth] = 1
                # Finalize last samples
                if depth > 0:
                    if depth in avg_fidelity.keys():
                        avg_fidelity[depth] = avg_fidelity[depth] + fidelity
                        avg_sdrp_time[depth] = avg_sdrp_time[depth] + time
                    else:
                        avg_fidelity[depth] = fidelity
                        avg_sdrp_time[depth] = time
            depth = d['depth']
            fidelity = d['fidelity']
            time = d['time']
            if depth in avg_marp_time.keys():
                avg_marp_time[depth] = avg_marp_time[depth] + time
            else:
                avg_marp_time[depth] = time

    for key in avg_fidelity.keys():
        depth = int(key)
        trials = trial_count[key]
        print({
            'depth': int(key),
            'successful_trials': trials,
            'avg_fidelity': avg_fidelity[key] / 100,
            'avg_sdrp_seconds': avg_sdrp_time[key] / trials,
            'avg_marp_seconds': avg_marp_time[key] / trials
        })

    return 0


if __name__ == '__main__':
    sys.exit(main())
