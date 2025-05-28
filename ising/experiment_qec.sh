# Suggested value for SDRP
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.0475

for w in 4 6 8 9 10 12 14 15 16 18 20 21 22 24 25 26 27 28; do python3 ising_ace_validation.py $w 10 2048; done
