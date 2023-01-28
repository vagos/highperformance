GDB = -ggdb
GXX = g++

all: ants ants-fast

ants: ants.cpp timer.hpp utils.hpp
	$(GXX) -o ants ants.cpp -std=c++17 -xc++ -lstdc++ -shared-libgcc $(GDB)

ants-fast: ants.cpp timer.hpp utils.hpp
	$(GXX) -o ants-fast ants.cpp -O2 -fopenacc $(GDB)

clean:
	rm -f ants
	rm -f *.log
