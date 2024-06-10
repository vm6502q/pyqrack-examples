import sys

def main():
    file = "marp.txt"
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python3 parse_marp.py [file_name]')

    file = str(sys.argv[1])

    avg_fidelity = []
    avg_sdrp_time = []
    avg_marp_time = []
    trial_count = []
    ideal_capacity = []
    with open(file, 'r') as in_file:
        depth = 0
        fidelity = 0
        time = 0
        while True:
            line = in_file.readline()
            if 'QRACK_MAX_PAGING_QB=' in line:
                ideal_capacity.append(int(line.split('QRACK_MAX_PAGING_QB=',1)[1]))
                avg_fidelity.append({})
                avg_sdrp_time.append({})
                avg_marp_time.append({})
                trial_count.append({})
                continue

            if not line:
                # Last sample
                if depth > 0:
                    if depth in avg_fidelity[-1].keys():
                        avg_fidelity[-1][depth] = avg_fidelity[-1][depth] + fidelity
                        avg_sdrp_time[-1][depth] = avg_sdrp_time[-1][depth] + time
                    else:
                        avg_fidelity[-1][depth] = fidelity
                        avg_sdrp_time[-1][depth] = time
                break

            d = eval(line)

            if d['sdrp'] == 1:
                # Update count
                dpth = d['depth']
                if dpth in trial_count[-1].keys():
                    trial_count[-1][dpth] = trial_count[-1][dpth] + 1
                else:
                    trial_count[dpth] = 1
                # Finalize last samples
                if depth > 0:
                    if depth in avg_fidelity[-1].keys():
                        avg_fidelity[-1][depth] = avg_fidelity[-1][depth] + fidelity
                        avg_sdrp_time[-1][depth] = avg_sdrp_time[-1][depth] + time
                    else:
                        avg_fidelity[-1][depth] = fidelity
                        avg_sdrp_time[-1][depth] = time

            depth = d['depth']
            fidelity = d['fidelity']
            time = d['time']

            if depth in avg_marp_time.keys():
                avg_marp_time[depth] = avg_marp_time[depth] + time
            else:
                avg_marp_time[depth] = time

    for i in range(len(avg_fidelity)):
        for key in avg_fidelity[i].keys():
            depth = int(key)
            trials = trial_count[i][key]
            print({
                'depth': int(key),
                'successful_trials': trials,
                'avg_fidelity': avg_fidelity[-1][key] / 100,
                'avg_sdrp_seconds': avg_sdrp_time[-1][key] / trials,
                'avg_marp_seconds': avg_marp_time[-1][key] / trials,
                'ideal_capacity_qb': ideal_capacity[-1]
            })

    return 0


if __name__ == '__main__':
    sys.exit(main())
