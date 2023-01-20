# Make sure to load: gcc, mkl

CC = g++
CFLAGS = -Wall -Wextra -Wpedantic
CFLAGS += -O3 -march=native

LDFLAGS = -lm -lpthread -lz -ldl -fopenmp
LIBS = -lopenblas -llapacke

all: pca_seq pca_omp

pca_seq: pca_seq.o
	$(CC) -o $@ $@.o $(OBJECTS) $(LDFLAGS) $(LIBS)

pca_seq.o: pca.cpp
	$(CC) $(CFLAGS) -o pca_seq.o -c pca.cpp

pca_omp: pca_omp.o
	$(CC) -o $@ $@.o $(OBJECTS) $(LDFLAGS) $(LIBS)

pca_omp.o: pca_omp.cpp
	$(CC) $(CFLAGS) -fopenmp -o pca_omp.o -c pca_omp.cpp

images: pca_omp
	mkdir -p out
	bash create_images.sh

clean:
	rm -f *.o *~ pca_seq pca_omp

