#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=s659

./build/diffusion_serial 10
./build/diffusion_cuda 10
./build/diffusion_serial 11
./build/diffusion_cuda 11
