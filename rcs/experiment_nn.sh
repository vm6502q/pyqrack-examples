# Suggested value for SDRP:
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.024

for w in 4 6 8 9 10 12 14 15 16 18 20 21 22 24 25 26 27 28 ; do QRACK_MAX_PAGING_QB=$((($w + 1) / 2)) QRACK_MAX_CPU_QB=$((($w + 1) / 2)) python3 rcs_nn_qiskit_validation.py $w $w 30; done
