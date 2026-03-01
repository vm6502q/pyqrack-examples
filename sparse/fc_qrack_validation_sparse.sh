for n in {2..30}; do QRACK_SPARSE_TRUNCATION_THRESHOLD=$(python3 -c "print(max(1.7763568394002505e-15,1.0/(($n-1)*(2**$n))))") python3 fc_qrack_validation_sparse.py $n $n; done
