extern "C"
#include <cuComplex.h>

__global__ void mul_strided_float(int n,int xOffset,int yOffset, float *dx, float *dy,int incx,int incy,float *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
              if(i >= xOffset && i >= yOffset &&  i % incx == 0 && i % incy == 0)
                    result[i] =  dx[i] * dy[i];
         }
}


