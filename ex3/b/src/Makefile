GDB = -ggdb
GXX = g++
NVC = /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvc++ # set this as the path of your nvidia c++ compiler.

all: ants-serial ants-fast

ants-serial: ants.cpp timer.hpp utils.hpp
	$(GXX) -DLOG=\"serial\" -o ants-serial ants.cpp -O3 $(GDB)

ants-fast: ants.cpp timer.hpp utils.hpp
	$(NVC) -DLOG=\"acc\" ants.cpp -o ants-fast -O3 -acc -fast -ta=nvidia -Minfo=accel -Msafeptr

clean:
	rm -f ants-*
	rm -f *.log
