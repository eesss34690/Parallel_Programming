default: conv

CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC'
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3
OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

conv: $(OBJS)
	$(CC) -o $@ $(OBJS) $(FLAGS)

cuda: $(OBJS) kernel.o
	nvcc ${CUDA_LINK_FLAGS} -lOpenCL -m64 -o $@ $(OBJS) kernel.o

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

kernel.o : kernel.cu kernel.h
	nvcc ${CUDA_COMPILE_FLAGS} -c kernel.cu -o $@

clean:
	rm -f conv *.o output.bmp ref.bmp