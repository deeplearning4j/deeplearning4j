extern "C"
#include <math.h>
__global__ void softmax_strided_float(int n,int xOffset, float *dx,int incx,float max,float sum,float *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= xOffset &&  i % incx == 0)
                                result[i] = expf(dx[i] - max) / sum;
             }

 }


