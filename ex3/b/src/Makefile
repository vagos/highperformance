GDB = -ggdb

ants: ants.cpp
	gcc -o ants ants.cpp -std=c++17 -xc++ -lstdc++ -shared-libgcc $(GDB)

clean:
	rm -f ants
	rm -f *.log
