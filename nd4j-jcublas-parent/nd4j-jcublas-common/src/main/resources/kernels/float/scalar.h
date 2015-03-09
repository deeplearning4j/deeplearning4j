 extern "C"

//scalar and current element
__device__ float op(float d1,float d2,float *params);

__device__ void transform(int n, int idx,float dx,float *dy,int incy,float *params,float *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                           result[i] = op(dx,dy[i],params);
         }

 }


