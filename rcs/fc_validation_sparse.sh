for n in {2..30}; do QRACK_SPARSE_TRUNCATION_THRESHOLD=$(python3 -c "import math; print(1/(($n-1)*math.sqrt(2**$n)))") python3 fc_validation_sparse.py $n $n; done
