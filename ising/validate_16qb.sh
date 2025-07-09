for d in {1..20}; do QRACK_MAX_PAGING_QB=8 QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1 python3 ising_validation.py 16 $d 2048; done
