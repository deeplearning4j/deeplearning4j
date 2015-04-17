extern "C"
#include <math.h>
#include <cuComplex.h>

__global__ void abs_float(int n,int idx,float *dy,int incy,float *result) {
       for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                    if(i >= idx && i % incy == 0)
                                  result[i] =  abs(dy[i]);
       }

}