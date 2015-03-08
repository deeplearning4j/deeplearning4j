#include <curand_kernel.h>

extern "C"
__global__ void uniform_float(int n,float lower,float upper,float *result) {
          int tid = threadIdx.x + blockIdx.x * blockDim.x;
          for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
              float u = result[i];
              result[i] = u * upper + (1 - u) * lower;
          }
}

