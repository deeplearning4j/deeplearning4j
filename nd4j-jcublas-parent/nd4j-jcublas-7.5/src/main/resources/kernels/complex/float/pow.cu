extern "C"
#include <math.h>
#include <cuComplex.h>

  __global__ void pow_float(int n,int idx,float *dy,int incy,float raise,float *result) {
         for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                        if(i >= idx && i % incy == 0)
                            result[i] =  powf(dy[i],raise);
          }

  }
