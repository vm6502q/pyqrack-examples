# Suggested value for SDRP:
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.025

for w in {2..28} ; do QRACK_MAX_PAGING_QB=$((($w + 1) / 2)) QRACK_MAX_CPU_QB=$((($w + 1) / 2)) python3 fc_qiskit_validation.py $w $w 30; done
