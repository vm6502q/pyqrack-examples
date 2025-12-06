for n in {2..30}; do QRACK_SPARSE_TRUNCATION_THRESHOLD=$(python3 -c "print(1/(($n-1)*(2**($n/2))))") python3 fc_qrack_validation_sparse.py $n $n; done
