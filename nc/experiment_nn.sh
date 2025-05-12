# Turn off SDRP
unset QRACK_QUNIT_SEPARABILITY_THRESHOLD
# Turn off near-Clifford conversion to state vector
export QRACK_MAX_PAGING_QB=-1
export QRACK_MAX_PAGE_QB=-1
export QRACK_MAX_CPU_QB=-1
# Round to nearest Clifford circuit
export QRACK_NONCLIFFORD_ROUNDING_THRESHOLD=1

for w in 4 6 8 9 10 12 14 15 16 18 20 21 22 24 25 26 27 28 ; do python3 rcs_nn_qiskit_validation.py $w $w; done
