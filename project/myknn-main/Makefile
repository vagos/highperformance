DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024

LOW ?=  0
HIGH ?= 2


CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -g -O3
CFLAGS += -DSURROGATES
LDFLAGS += -lm

NVC = /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvc++ # set this as the path of your nvidia c++ compiler.

all: gendata myknn-serial myknn-serial-alt myknn-omp myknn-mpi myknn-cuda myknn-omp-alt myknn-acc-alt

# myknn-cuda-alt: myknn-cuda-alternative.cu func.h
# 	nvcc -o myknn-cuda-alt $(CFLAGS) myknn-cuda-alternative.cu $(LDFLAGS)

myknn-acc-alt: myknn-acc-alternative.c func.h
	$(NVC) -o myknn-acc-alt $(CFLAGS) myknn-acc-alternative.c $(LDFLAGS) -acc -fast 

myknn-omp-alt: myknn-omp-alternative.c func.h
	g++ -o myknn-omp-alt $(CFLAGS) myknn-omp-alternative.c $(LDFLAGS) -fopenmp

myknn-serial-alt: myknn-serial-alternative.c func.h
	g++ -o myknn-serial-alt $(CFLAGS) myknn-serial-alternative.c $(LDFLAGS)

# myknn-acc: myknn-acc.c func.h
# 	$(NVC) -o myknn-acc $(CFLAGS) myknn-acc.c $(LDFLAGS) -acc -fast 

myknn-cuda: myknn-cuda.cu func.h
	nvcc -o myknn-cuda $(CFLAGS) myknn-cuda.cu $(LDFLAGS)

myknn-mpi: myknn-mpi.c func.h
	mpicc -o myknn-mpi $(CFLAGS) myknn-mpi.c $(LDFLAGS)

myknn-omp: myknn-omp.c func.h
	g++ -o myknn-omp $(CFLAGS) myknn-omp.c $(LDFLAGS) -fopenmp

myknn-serial: myknn-serial.c func.h
	g++ -o myknn-serial $(CFLAGS) myknn-serial.c $(LDFLAGS)

gendata: gendata.c func.h
	g++ -o gendata $(CFLAGS) gendata.c $(LDFLAGS)

clean:
	rm -f myknn-serial myknn-omp myknn-mpi myknn-cuda myknn-acc myknn-omp-alt myknn-acc-alt myknn-cuda-alt gendata myknn-serial-alt x.txt q.txt
