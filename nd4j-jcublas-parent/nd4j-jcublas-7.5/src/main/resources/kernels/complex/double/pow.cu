extern "C"
#include <math.h>
#include <cuComplex.h>

__global__ void pow_double(int n,int idx,double *dy,int incy,double raise,double *result) {
         for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                        if(i >= idx && i % incy == 0)
                            result[i] =  pow(dy[i],raise);
          }

  }
