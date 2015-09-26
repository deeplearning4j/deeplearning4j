extern "C"
#include <cuComplex.h>

__global__ void mul_double(int n, double *a, double *b, double *sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        sum[i] = a[i] * b[i];
    }

}