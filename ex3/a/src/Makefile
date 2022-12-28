# Makefile

SHELL := /bin/bash
CXX?=g++
CXXFLAGS?= -Wall -O3 -std=c++11 -lstdc++
NVCC?=nvcc
NVCCFLAGS?=-O3 --gpu-architecture=native

.DEFAULT: all

.PHONY: all
all: dir diffusion_serial diffusion_cuda

dir:
	@mkdir -p build

diffusion_serial: diffusion2d_serial.cpp
	$(CXX) $(CXXFLAGS) -o build/$@ $^

diffusion_cuda: diffusion2d_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o build/$@ $^


.PHONY: clean
clean:
	-rm -r build
