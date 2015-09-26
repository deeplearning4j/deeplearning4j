extern "C"
#include <cuComplex.h>

__global__ void rdiv_float(int n, float *a, float *b, float *sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        sum[i] =  b[i] / a[i];
    }

}