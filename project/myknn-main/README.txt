# How to build and test

make

./gen_data x.txt q.txt 

./myknn-<type> x.txt q.txt

# How to add new implementation

cp myknn-serial.c myknn-<type>.c

Now, just edit the file myknn-<type>.c. All common functions are in func.h.
