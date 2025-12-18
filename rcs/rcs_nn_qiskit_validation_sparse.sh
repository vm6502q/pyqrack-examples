for n in {4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,27,28,30}; do QRACK_SPARSE_TRUNCATION_THRESHOLD=$(python3 -c "print(1/(4*(2**$n)))") python3 rcs_nn_qiskit_validation_sparse.py $n $n; done
