# Turn off near-Clifford conversion to state vector
export QRACK_MAX_PAGING_QB=-1
export QRACK_MAX_CPU_QB=-1

for w in {2..28} ; do python3 fc_qiskit_validation.py $w $w 30; done
