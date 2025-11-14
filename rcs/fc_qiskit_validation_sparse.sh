for n in {2..30}; do QRACK_SPARSE_TRUNCATION_THRESHOLD=$( echo "scale=64; 1/(16*2^16)" | bc ) python3 fc_qiskit_validation_sparse.py $n $n; done
