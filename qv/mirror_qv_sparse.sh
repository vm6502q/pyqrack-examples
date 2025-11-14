for n in {2..50}; do QRACK_SPARSE_TRUNCATION_THRESHOLD=$(( bc -l << "2^(-$n)/$n" )) python3 mirror_qv_sparse.py $n $n; done
