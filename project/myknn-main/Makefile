DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 32

LOW ?=  0
HIGH ?= 2


CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -g -ggdb -O3
CFLAGS += -DSURROGATES -Wall
LDFLAGS += -lm

all: gendata myknn-serial myknn-omp

myknn-omp: myknn-omp.c func.h
	g++ -o myknn-omp $(CFLAGS) myknn-omp.c $(LDFLAGS) -fopenmp

myknn-serial: myknn-serial.c func.h
	g++ -o myknn-serial $(CFLAGS) myknn-serial.c $(LDFLAGS)

gendata: gendata.c func.h
	g++ -o gendata $(CFLAGS) gendata.c $(LDFLAGS)

clean:
	rm -f myknn-serial myknn-omp gendata
