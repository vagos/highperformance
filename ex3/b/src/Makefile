GDB = -ggdb
GXX = g++
NVC = nvcc

all: ants ants-fast

ants: ants.cpp timer.hpp utils.hpp
	$(GXX) -DLOG=\"serial\" -o ants ants.cpp $(GDB)

ants-fast: ants.cpp timer.hpp utils.hpp
	$(GXX) -DLOG=\"acc\" -o ants-fast ants.cpp -O3 -fopenacc $(GDB)
	
ants-fast-nvc: ants.cpp timer.hpp utils.hpp
	$(NVC) -DLOG=\"acc-nvc\" ants.cpp -o ants-fast-nvc -O3

clean:
	rm -f ants-*
	rm -f *.log
