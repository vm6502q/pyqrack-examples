for depth in {3..12}; do for run in {1..100}; do python3 marp_2d.py 64 ${depth} 11 | grep "{'width':"; done; done
