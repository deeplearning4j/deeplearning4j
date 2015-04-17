extern "C"
#include <cuComplex.h>

__global__ void rdiv_scalar_double(int n,int idx, float dx,float *dy,int incy,double *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                        if(i >= idx && i % incy == 0)
                            result[i] = dx / dy[i];
         }

 }


