extern "C"
#include <math.h>
#include <cuComplex.h>
__global__ void norm1_strided_double(int n, int xOffset,double *dx,int incx,double result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= xOffset && i % incx == 0)
                                result += abs(dx[i]);
             }

 }


