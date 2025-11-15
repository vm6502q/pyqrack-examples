for n in {2..30}; do QRACK_SPARSE_TRUNCATION_THRESHOLD=$( echo "scale=24; 1/(($n-1)*sqrt(2^$n))" | bc ) python3 fc_qiskit_validation_sparse.py $n $n; done
