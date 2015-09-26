extern "C"
#include <cuComplex.h>

__global__ void sign_double(int n,int idx,double *dy,int incy,double *result) {
           for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= idx && i % incy == 0) {
                              double x = dy[i];
                              result[i] =  (x > 0) - (x < 0);
                           }
            }

    }