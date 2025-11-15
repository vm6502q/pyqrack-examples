for n in {2..30}; do QRACK_SPARSE_TRUNCATION_THRESHOLD=$(python3 -c "import math; print(1/($n*math.sqrt(2**$n)))") python3 fc_qiskit_validation_sparse.py $n $n; done
