#include <math.h>


__global__ void neg_double(int n,int idx,double *dy,int incy,double *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                       if(i >= idx && i % incy == 0)
                           result[i] =  -dy[i];
         }

 }

__global__ void tanh_double(int n,int idx,double *dy,int incy,double *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                       if(i >= idx && i % incy == 0)
                           result[i] =  tanh(-dy[i]);
         }

 }

__global__ void exp_double(int n,int idx,double *dy,int incy,double *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                       if(i >= idx && i % incy == 0)
                           result[i] =  exp(-dy[i]);
         }

 }

__global__ void sigmoid_double(int n,int idx,double *dy,int incy,double *result) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                       if(i >= idx && i % incy == 0)
                           result[i] =  1 / (1 + exp(-dy[i]));
         }

 }

 __global__ void log_double(int n,int idx,double *dy,int incy,double *result) {
         for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                        if(i >= idx && i % incy == 0)
                            result[i] =  log(dy[i]);
          }

  }

   __global__ void floor_double(int n,int idx,double *dy,int incy,double *result) {
           for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= idx && i % incy == 0)
                              result[i] =  floor(dy[i]);
            }

    }

     __global__ void ceil_double(int n,int idx,double *dy,int incy,double *result) {
             for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                            if(i >= idx && i % incy == 0)
                                result[i] =  ceil(dy[i]);
              }

      }

       __global__ void abs_double(int n,int idx,double *dy,int incy,double *result) {
               for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                              if(i >= idx && i % incy == 0)
                                  result[i] =  abs(dy[i]);
                }

        }

       __global__ void pow_double(int n,int idx,double *dy,int incy,double *result,double raise) {
         for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                        if(i >= idx && i % incy == 0)
                            result[i] =  pow(dy[i],raise);
          }

  }

   __global__ void log_double(int n,int idx,double *dy,int incy,double *result) {
           for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= idx && i % incy == 0)
                              result[i] =  sqrt(dy[i]);
            }

    }


   __global__ void sign_double(int n,int idx,double *dy,int incy,double *result) {
           for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                          if(i >= idx && i % incy == 0) {
                              double x = dy[i];
                              result[i] =  (x > 0) - (x < 0);
                           }
            }

    }