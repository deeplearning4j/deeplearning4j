#include <math.h>


__global__ void neg_float(int n,int idx,float *dy,int incy,float *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                       if(i >= idx && i % incy == 0)
                           result[i] =  -dy[i];
         }

 }

__global__ void tanh_float(int n,int idx,float *dy,int incy,float *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                       if(i >= idx && i % incy == 0)
                           result[i] =  tanh(-dy[i]);
         }

 }

__global__ void exp_float(int n,int idx,float *dy,int incy,float *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                       if(i >= idx && i % incy == 0)
                           result[i] =  exp(-dy[i]);
         }

 }

__global__ void sigmoid_float(int n,int idx,float *dy,int incy,float *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                       if(i >= idx && i % incy == 0)
                           result[i] =  1 / (1 + exp(-dy[i]));
         }

 }

 __global__ void log_float(int n,int idx,float *dy,int incy,float *result) {
         for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                        if(i >= idx && i % incy == 0)
                            result[i] =  log(dy[i]);
          }

  }

   __global__ void floor_float(int n,int idx,float *dy,int incy,float *result) {
           for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= idx && i % incy == 0)
                              result[i] =  floor(dy[i]);
            }

    }

     __global__ void ceil_float(int n,int idx,float *dy,int incy,float *result) {
             for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                            if(i >= idx && i % incy == 0)
                                result[i] =  ceil(dy[i]);
              }

      }

       __global__ void abs_float(int n,int idx,float *dy,int incy,float *result) {
               for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                              if(i >= idx && i % incy == 0)
                                  result[i] =  abs(dy[i]);
                }

        }

       __global__ void pow_float(int n,int idx,float *dy,int incy,float *result,float raise) {
         for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                        if(i >= idx && i % incy == 0)
                            result[i] =  pow(dy[i],raise);
          }

  }

   __global__ void log_float(int n,int idx,float *dy,int incy,float *result) {
           for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= idx && i % incy == 0)
                              result[i] =  sqrt(dy[i]);
            }

    }


   __global__ void sign_float(int n,int idx,float *dy,int incy,float *result) {
           for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= idx && i % incy == 0) {
                              float x = dy[i];
                              result[i] =  (x > 0) - (x < 0);
                           }
            }

    }