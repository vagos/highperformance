DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 32

LOW ?=  0
HIGH ?= 2


CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -g -O3
CFLAGS += -DSURROGATES -Wall
LDFLAGS += -lm

NVC = /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/bin/nvc++ # set this as the path of your nvidia c++ compiler.

all: gendata myknn-serial myknn-omp myknn-mpi myknn-acc

myknn-acc: myknn-acc.c func.h
	$(NVC) -o myknn-acc $(CFLAGS) myknn-acc.c $(LDFLAGS) -acc -fast -Minfo=accel

myknn-mpi: myknn-mpi.c func.h
	mpicc -o myknn-mpi $(CFLAGS) myknn-mpi.c $(LDFLAGS)

myknn-omp: myknn-omp.c func.h
	g++ -o myknn-omp $(CFLAGS) myknn-omp.c $(LDFLAGS) -fopenmp

myknn-serial: myknn-serial.c func.h
	g++ -o myknn-serial $(CFLAGS) myknn-serial.c $(LDFLAGS)

gendata: gendata.c func.h
	g++ -o gendata $(CFLAGS) gendata.c $(LDFLAGS)

clean:
	rm -f myknn-serial myknn-omp gendata
