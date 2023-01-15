CXX = g++
GDB = -ggdb

all: vanilla fast generate

vanilla: particles.h timer.hpp utils.h
	$(CXX) vanilla.cpp -std=c++11 -g -O3 -o $@ -fopenmp

fast: particles.h timer.hpp utils.h fast.cpp
	$(CXX) -o fast fast.cpp -fopenmp -mavx -O3

generate: generate.cpp
	$(CXX) -o generate generate.cpp

.PHONY: clean
clean:
	rm -f vanilla fast
