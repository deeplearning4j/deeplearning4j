extern "C"
 __global__ void sub_strided_double(int n,const double *d_X, const double *d_Y, double *d_Z)
{
      const int  tid = blockDim.x * blockIdx.x + threadIdx.x;
      const int  increment = blockDim.x * gridDim.x;

      for (int i = tid; i < n ; i += increment) {
            d_Z[i]= d_X[i] - d_Y[i];
       }
 }


