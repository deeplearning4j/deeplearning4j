extern "C"
__global__ void mean_strided_double(int n, int xOffset,double *dx,int incx,double *result) {
                  extern __shared__ double sdata[];

                 // perform first level of reduction,
                 // reading from global memory, writing to shared memory
                 unsigned int tid = threadIdx.x;
                 unsigned int i = blockIdx.x*blockDim.x * 2 + threadIdx.x;
                 unsigned int gridSize = blockDim.x * 2 * gridDim.x;

                 double temp = 0;

                 // we reduce multiple elements per thread.  The number is determined by the
                 // number of active thread blocks (via gridDim).  More blocks will result
                 // in a larger gridSize and therefore fewer elements per thread
                 while (i < n) {
                   if(i >= xOffset && i % incx == 0) {
                    temp += dx[i];
                    // ensure we don't read out of bounds
                    if (i + blockDim.x < n) {
                            temp += dx[i + blockDim.x];
                      }

                   }
                    i += gridSize;
                 }

                 // each thread puts its local mean into shared memory
                 sdata[tid] = temp;
                 __syncthreads();


                 // do reduction in shared mem
                 if (blockDim.x >= 512) {
                 if (tid < 256) {
                      sdata[tid] = temp = temp + sdata[tid + 256];
                     }
                   __syncthreads();
                 }

                 if (blockDim.x >= 256) {
                      if (tid < 128) {
                        sdata[tid] = temp = temp + sdata[tid + 128];
                      }
                     __syncthreads();
                 }

                 if (blockDim.x >= 128) {
                      if (tid <  64)  {
                           sdata[tid] = temp = temp + sdata[tid +  64];
                       }
                       __syncthreads();
                  }

                 if (tid < 32) {
                     // now that we are using warp-synchronous programming (below)
                     // we need to declare our shared memory volatile so that the compiler
                     // doesn't reorder stores to it and induce incorrect behavior.
                     volatile double* smem = sdata;
                     if (blockDim.x >=  64) {
                         smem[tid] = temp = temp + smem[tid + 32];
                      }
                     if (blockDim.x >=  32) {
                         smem[tid] = temp = temp + smem[tid + 16];
                     }
                     if (blockDim.x >=  16) {
                         smem[tid] = temp = temp + smem[tid +  8];
                      }
                     if (blockDim.x >=   8) {
                          smem[tid] = temp = temp + smem[tid +  4];
                      }
                     if (blockDim.x >=   4) {
                         smem[tid] = temp = temp + smem[tid +  2];
                      }
                     if (blockDim.x >=   2) {
                         smem[tid] = temp = temp + smem[tid +  1];
                      }
                 }

                 // write result for this block to global mem
                 if (tid == 0)
                     result[blockIdx.x] = sdata[0] / (double) n;

 }


