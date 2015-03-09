 extern "C"

//scalar and current element
__device__ double op(double d1,double d2,double *params);

__device__ void transform(int n, int idx,double dx,double *dy,int incy,double *params,double *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                           result[i] = op(dx,dy[i],params);
         }

 }


