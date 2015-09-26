#include <math.h>

//x[i] and y[i]
__device__ double op(double d1,double d2,double *params);
__device__ double op(double d1,double *params);

__device__ void transform(int n,int xOffset,int yOffset, double *dx, double *dy,int incx,int incy,double *params,double *result,int incz) {

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


