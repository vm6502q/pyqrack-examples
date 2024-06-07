for depth in {1..8}; do for run in {1..100}; do python3 marp_full.py 56 ${depth} 21 | grep "{'width':"; done; done
