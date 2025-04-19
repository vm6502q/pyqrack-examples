# Suggested value for SDRP:
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.03

for w in 4 6 8 9 10 12 14 15 16 18 20 21 22 24 ; do QRACK_MAX_PAGING_QB=$((($w + 1) / 2)) QRACK_MAX_CPU_QB=$((($w + 1) / 2)) python3 ising_ace_validation.py 10 $w 2048; done
