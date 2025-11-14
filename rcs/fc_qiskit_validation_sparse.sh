for n in {2..30}; do QRACK_SPARSE_TRUNCATION_THRESHOLD=$(( bc -l << "2^(-$n)/$n" )) python3 fc_qiskit_validation_sparse.py $n $n; done
