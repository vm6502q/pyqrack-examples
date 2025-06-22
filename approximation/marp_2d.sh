for depth in {1..12}; do for run in {1..100}; do python3 marp_2d.py 54 ${depth} 21 | grep "{'width':"; done; done
