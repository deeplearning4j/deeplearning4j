#include <curand_kernel.h>

extern "C"
__global__ void uniform_double(int n,double lower,double upper,double *result) {
          int tid = threadIdx.x + blockIdx.x * blockDim.x;
          for(int i = tid; i < n; i += blockDim.x * gridDim.x) {
              double u = result[i];
              result[i] = u * upper + (1 - u) * lower;
          }
}