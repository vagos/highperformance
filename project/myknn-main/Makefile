DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 32

LOW ?=  0
HIGH ?= 2


CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -g -ggdb -O3
CFLAGS += -DSURROGATES -Wall -fopenmp
LDFLAGS += -lm -fopenmp

all: gendata myknn myknnomp

myknnomp: myknnomp.o
	g++ -o myknnomp myknnomp.o $(LDFLAGS) -fopenmp

myknnomp.o: myknnomp.cpp func.c util.hpp
	g++ $(CFLAGS) -c myknnomp.cpp

gendata: gendata.o
	g++ -o gendata gendata.o $(LDFLAGS)

gendata.o: gendata.c func.c
	g++ $(CFLAGS) -c gendata.c

myknn: myknn.o
	g++ -o myknn myknn.o $(LDFLAGS)

myknn.o: myknn.c func.c
	g++ $(CFLAGS) -c myknn.c

clean:
	rm -f myknn *.o gendata
