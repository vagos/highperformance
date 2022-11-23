#!/bin/bash

D=1
L=1
N=32
T=1000
DT=1e-9

# DEBUG="alacritty -e -e ./diffusion2d_mpi $D $L $N $T $DT"
# DEBUG="valgrind --leak-check=yes"

ARGS="--use-hwthread-cpus --mca opal_warn_on_missing_libcuda 0"

mpirun -n 1 ./diffusion2d_mpi_nb_io 10 -1
