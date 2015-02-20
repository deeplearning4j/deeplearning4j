extern "C"
__global__ void add_strided_float(int n, float *dx, float *dy,int incx,int incy) {
       int  dxIdx = blockDim.x * blockIdx.x + threadIdx.x;
        int dyIdx = blockDim.y * blockIdx.y + threadIdx.y;

        if(dxIdx < n && dxIdx % incx == 0 && dyIdx < n && dyIdx % incy == 0)
                 dy[dyIdx] += dx[dxIdx];

 }


