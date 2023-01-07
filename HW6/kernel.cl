__kernel void convolution(__global const float* inputImage, __global const float* filter, __global float4* outputImage,
     const int imageWidth, const int imageHeight, const int halffilterSize)
{
    int idx = get_global_id(0) * 4, filter_idx = 0;
    int thread_width = idx % imageWidth;
    int thread_height = idx / imageWidth;
    float4 sum = 0.0, input;

    int i, curHeight, curWidth, j, pos;

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
                        input = (float4)(inputImage[pos], inputImage[pos + 1], inputImage[pos + 2],
                                 inputImage[pos + 3]);
                        sum += input * filter[filter_idx];
                    }
                }
                filter_idx++;
            }
        }
    }
    outputImage[idx / 4] = sum;
}