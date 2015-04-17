extern "C"
#include <cuComplex.h>
  __global__ void sign_float(int n,int idx,float *dy,int incy,float *result) {
           for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= idx && i % incy == 0) {
                              float x = dy[i];
                              result[i] =  (x > 0) - (x < 0);
                           }
            }

    }