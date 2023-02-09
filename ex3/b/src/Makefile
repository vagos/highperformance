GDB =
GXX = g++
NVC = /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvc++

all: ants-fast-nvc ants-serial # ants-fast

ants-serial: ants.cpp timer.hpp utils.hpp
	$(GXX) -DLOG=\"serial\" -o ants-serial ants.cpp -O3 $(GDB)

ants-fast: ants.cpp timer.hpp utils.hpp
	$(GXX) -DLOG=\"acc\" -o ants-fast ants.cpp -O3 -lgomp -fopenacc -save-temps -I/opt/cuda/include $(GDB)
	
ants-fast-nvc: ants.cpp timer.hpp utils.hpp
	$(NVC) -DLOG=\"acc-nvc\" ants.cpp -o ants-fast-nvc -O3 -acc -fast -ta=nvidia -Minfo=accel -Msafeptr

clean:
	rm -f ants-*
	rm -f *.log
