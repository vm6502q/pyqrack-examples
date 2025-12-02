for w in 4 6 8 9 10 12 14 15 16 18 20 21 22 24 25 26 27 28 ; do QRACK_MAX_PAGING_QB=$(((w + 3) / 4)) QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1 python3 tfim_ace_validation.py $w 40 0.125; done
