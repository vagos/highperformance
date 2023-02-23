#!/bin/bash

echo "My knn Serial" > serial.txt
echo " " >> serial.txt
./myknn-serial x.txt q.txt >> serial.txt
echo " " >> serial.txt
echo "My knn Serial Alt" >> serial.txt
echo " " >> serial.txt
./myknn-serial-alt x.txt q.txt >> serial.txt

echo "My knn OpenMP" > omp.txt
echo " " >> omp.txt
for i in {1,2,4,6,8,12,16,32}
do
    echo "OMP_NUM_THREADS=$i" >> omp.txt
    echo " " >> omp.txt
    OMP_NUM_THREADS= $i ./myknn-omp x.txt q.txt >> omp.txt
    echo " " >> omp.txt
    echo " " >> omp.txt
done

# ./myknn-serial x.txt q.txt
# ./myknn-omp x.txt q.txt
