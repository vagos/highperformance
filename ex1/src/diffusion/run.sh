#!/bin/bash

# Execution examples
# export OMP_NUM_THREADS=1; ./diffusion2d_openmp 1 1 1024 1000 0.00000001
# export OMP_NUM_THREADS=4; ./diffusion2d_openmp 1 1 1024 1000 0.00000001
# mpirun -n 1 ./diffusion2d_mpi_nb 1 1 1024 1000 0.00000001
# mpirun -n 4 ./diffusion2d_mpi_nb 1 1 1024 1000 0.00000001

D=1
L=1
N=128
T=1000
DT=1e-9

# DEBUG="alacritty -e -e ./diffusion2d_mpi $D $L $N $T $DT"
# DEBUG="valgrind --leak-check=yes"
# RUN_CMD="export OMP_NUM_THREADS=4; ./diffusion2d_openmp $D $L $N $T $DT"
ARGS="--mca opal_warn_on_missing_libcuda 0"

benchmark() 
{
    N=$1
    RUN_CMD="mpirun -n 1 $ARGS ./diffusion2d_mpi $D $L $N $T $DT" 
    
    $RUN_CMD
}

SIZES=(1024 2048 4096)

for N in ${SIZES[@]}; do
  echo "Running mpi code with N=$N..."
  benchmark $N
done
