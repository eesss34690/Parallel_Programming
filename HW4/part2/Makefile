all: 
	mpicxx -O3 -fno-asynchronous-unwind-tables -mfma -ffast-math -funsafe-math-optimizations -floop-parallelize-all -ftree-parallelize-loops=4 -fprefetch-loop-arrays -fno-tree-vectorize -fprofile-use -w main.cc matt.cc -o matmul
safe:
	mpicxx main.cc matmul.cc -o matmul
