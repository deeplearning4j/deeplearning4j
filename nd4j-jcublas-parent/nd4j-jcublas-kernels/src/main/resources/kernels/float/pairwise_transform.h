#include <math.h>

//x[i] and y[i]
__device__ float op(float d1,float d2,float *params);
__device__ float op(float d1,float *params);

__device__ void transform(int n,int xOffset,int yOffset, float *dx, float *dy,int incx,int incy,float *params,float *result,int incz) {

    int totalThreads = gridDim.x * blockDim.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (incy == 0) {
        if ((blockIdx.x == 0) && (tid == 0)) {
            for (; i < n; i++) {
                result[i * incz] = op(dx[i * incx],params);
            }

        }
    } else if ((incx == incy) && (incx > 0)) {
        /* equal, positive, increments */
        if (incx == 1) {
            /* both increments equal to 1 */
            for (; i < n; i += totalThreads) {
                  result[i * incz] = op(dx[i],dy[i],params);
            }
        } else {
            /* equal, positive, non-unit increments. */
            for (; i < n; i += totalThreads) {
                result[i * incz] = op(dx[i * incx],dy[i * incy],params);
            }
        }
    } else {
        /* unequal or nonpositive increments */
        for (; i < n; i += totalThreads) {
            result[i * incz] = op(dx[i * incx],dy[i * incy],params);
        }
    }
}


