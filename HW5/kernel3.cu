#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define GROUP_SIZE 2
__device__ int mandel(float c_re, float c_im, int maxIteration)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < maxIteration; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int resX, int resY, int maxIterations, int *outputd) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    int thisY = (blockIdx.y * blockDim.y + threadIdx.y) * GROUP_SIZE;

    if (thisX < resX && thisY < resY)
    {
        for (int j = thisY; j < thisY + GROUP_SIZE && j < resY; j++)
        {
            for (int i = thisX; i < thisX + GROUP_SIZE && i < resX; i++)
            {
                float x = lowerX + i * stepX;
                float y = lowerY + j * stepY;

                int index = (j * resX + i);
                outputd[index] = mandel(x, y, maxIterations);                
            }
        }

    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // allocate memory first
    int * outputd, *output;
    int pixel_size = resX * resY * sizeof(int);
    size_t pitch;

    cudaMallocPitch(&outputd, &pitch, sizeof(int) * resX, resY);
    cudaHostAlloc(&output, pixel_size, cudaHostAllocDefault);

    // 1 block 256 thread, construct other portions with grids
    dim3 dimBlock(16 / GROUP_SIZE, 16 / GROUP_SIZE);
    dim3 dimGrid((int)(ceil(resX / 16.0)), (int)(ceil(resY / 16.0)));

    mandelKernel <<< dimGrid, dimBlock >>> (lowerX, lowerY, stepX, stepY, resX, resY, maxIterations, outputd);
    
    cudaDeviceSynchronize();

    // transfer to CPU
    cudaMemcpy(output, outputd, pixel_size, cudaMemcpyDeviceToHost);
    memcpy(img, output, pixel_size);
    
    cudaFree(outputd);
    cudaFreeHost(output);
}
