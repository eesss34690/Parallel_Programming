__global__ void convKernel(float* inputImage, float* filter, float* outputImage,
     const int imageWidth, const int imageHeight, const int halffilterSize)
{
    int thread_width = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int thread_height = blockIdx.y * blockDim.y + threadIdx.y;
    if (thread_width >= imageWidth && thread_height >= imageHeight)
        return;
    float4 sum = make_float4(0.0, 0.0, 0.0, 0.0);

    int i, curHeight, curWidth, j, pos, filter_idx = 0;

    for (i = -halffilterSize; i <= halffilterSize; i++)
    {
        curHeight = thread_height + i;
        if (curHeight < imageHeight && curHeight >= 0)
        {
            curHeight *= imageWidth;
            for (j = -halffilterSize; j <= halffilterSize; j++)
            {
                if (filter[filter_idx] != 0)
                {
                    curWidth = thread_width + j;
                    if (curWidth < imageWidth && curWidth >= 0)
                    {
                        pos = curWidth + curHeight;
                        sum.x += inputImage[pos] * filter[filter_idx];
                        sum.y += inputImage[pos + 1] * filter[filter_idx];
                        sum.z += inputImage[pos + 2] * filter[filter_idx];
                        sum.w += inputImage[pos + 3] * filter[filter_idx];
                    }
                }
                filter_idx++;
            }
        }
    }
    int idx = thread_width + thread_height * imageWidth;
    outputImage[idx] = sum.x;
    outputImage[idx + 1] = sum.y;
    outputImage[idx = 2] = sum.z;
    outputImage[idx + 3] = sum.w;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFECuda (int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage)
{
    // allocate memory first
    float* inputd, *filterd, *outputd;
    int filterSize = filterWidth * filterWidth;
    int ioSize = imageWidth * imageHeight;
    cudaMalloc((void**)&inputd, ioSize);
    cudaMalloc((void**)&filterd, filterSize);
    cudaMalloc((void**)&outputd, ioSize);
    cudaMemcpy(inputd, inputImage, ioSize, cudaMemcpyHostToDevice);
    cudaMemcpy(filterd, filter, filterSize, cudaMemcpyHostToDevice);

    // 1 block 256 thread, construct other portions with grids
    dim3 dimBlock(4, 16);
    dim3 dimGrid((int)(ceil(imageWidth / 16.0)), (int)(ceil(imageHeight / 16.0)));

    int halffilterSize = filterWidth / 2;

    convKernel <<< dimGrid, dimBlock >>> (inputd, filterd, outputd, imageWidth, imageHeight, halffilterSize);

    cudaMemcpy(outputImage, outputd, ioSize, cudaMemcpyDeviceToHost);

    cudaFree(outputd);
}