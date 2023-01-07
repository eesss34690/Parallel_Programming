#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth;
    int ioSize = imageWidth * imageHeight;
    cl_mem inputBuf = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, sizeof(cl_float) * ioSize, inputImage, NULL);
    cl_mem filterBuf = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, sizeof(cl_float) * filterSize, filter, NULL);
    cl_mem outputBuf = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * ioSize, NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, 0);

    cl_kernel kernel = clCreateKernel(*program, "convolution", 0);

    int halffilterSize = filterWidth / 2;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterBuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuf);
    clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    clSetKernelArg(kernel, 4, sizeof(int), &imageHeight);
    clSetKernelArg(kernel, 5, sizeof(int), &halffilterSize);

    size_t threadSize = ioSize / 4;

    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &threadSize, 0, 0, 0, 0);

    clFinish(queue);

    clEnqueueReadBuffer(queue, outputBuf, CL_TRUE, 0, sizeof(float) * ioSize, outputImage, 0, 0, 0);
}