extern "C"
#include <cuComplex.h>

__global__ void prod_strided_float(int n, int xOffset,float *dx,int incx,float result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= xOffset && i % incx == 0)
                                result *= dx[i];
             }

 }


