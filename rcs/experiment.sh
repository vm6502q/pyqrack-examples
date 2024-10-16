echo "sycamore_2019_elided"
for run in {1..100}; do python3 sycamore_2019_elided.py 28 10; done
echo "rcs_nn_elided"
for run in {1..100}; do python3 rcs_nn_elided.py 28 10; done
echo "fc_elided"
for run in {1..100}; do python3 fc_elided.py 28 10; done
echo "sycamore_2019_elided_time"
for run in {1..100}; do python3 sycamore_2019_elided_time.py 54 20; done
echo "rcs_nn_elided_time"
for run in {1..100}; do python3 rcs_nn_elided_time.py 54 20; done
echo "fc_elided_time"
for run in {1..100}; do python3 fc_elided_time.py 54 20; done
