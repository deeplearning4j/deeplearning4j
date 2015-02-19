extern "C"
__global__ void div_strided_float(int n,const float *d_X, const float *d_Y, float *d_Z)
 {
      const int  tid = blockDim.x * blockIdx.x + threadIdx.x;
      const int  increment = blockDim.x * gridDim.x;

      for (int i = tid; i < n ; i += increment) {
            d_Z[i] = d_X[i] / d_Y[i];
      }
 }


