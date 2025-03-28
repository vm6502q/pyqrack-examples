for w in {2..28} ; do QRACK_MAX_PAGING_QB=$((($w + 1) / 2)) QRACK_MAX_CPU_QB=$((($w + 1) / 2)) python3 fc_qiskit_validation.py $w $w 20; done
