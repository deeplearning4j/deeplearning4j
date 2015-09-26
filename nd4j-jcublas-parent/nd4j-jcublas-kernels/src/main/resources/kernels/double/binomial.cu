#include <cuda_runtime.h>
#include <curand.h>
__device__ double doBinomial(int n, double p, double *randomNumbers,curandGenerator_t s) {
  int x = 0;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  for(int i = tid; i < n; i++) {
    if(randomNumbers[i]< p )
      x++;
  }
  return x;
}


extern "C"
__global__ void binomial_double(int len,int n,double *ps,double *randomNumbers,double *result, curandGenerator_t s) {
          int tid = threadIdx.x + blockIdx.x * blockDim.x;
          for(int i = tid; i < len; i += blockDim.x * gridDim.x) {
              result[i] = doBinomial(n,ps[i],randomNumbers,s);
          }
}

