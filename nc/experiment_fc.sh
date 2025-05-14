# Turn off near-Clifford conversion to state vector
export QRACK_MAX_PAGING_QB=-1
export QRACK_MAX_CPU_QB=-1

for w in {2..28} ; do for r in {1..10} ; do python3 fc_2n_plus_2_qiskit_validation.py $w $(echo "$r/10" | bc -l) ; done ; done

