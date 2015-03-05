#include <math.h>

//x[i] and y[i]
__device__ float op(float d1,float d2,float *params);
__device__ float op(float d1,float *params);

__device__ void transform(int n,int xOffset,int yOffset, float *dx, float *dy,int incx,int incy,float *params,float *result) {

    int totalThreads = gridDim.x * blockDim.x;
    int ctaStart = blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (incy == 0) {
        if ((blockIdx.x == 0) && (tid == 0)) {
            /* FIXME: This code is functionally correct, but inefficient */
            int ix = (incx < 0) ? ((1 - n) * incx) : 0;
            for (; i < n; i++) {
                result[i] = op(dx[i * incx],params);
                ix += incx;
            }

        }
    } else if ((incx == incy) && (incx > 0)) {
        /* equal, positive, increments */
        if (incx == 1) {
            /* both increments equal to 1 */
            for (; i < n; i += totalThreads) {
                  result[i] = op(dx[i],dy[i],params);
              }
        } else {
            /* equal, positive, non-unit increments. */
            for (; i < n; i += totalThreads) {
                result[i * incx] = op(dx[i * incx],dy[i * incx],params);
            }
        }
    } else {
        /* unequal or nonpositive increments */
        int ix = ((incx < 0) ? ((1 - n) * incx) : 0);
        int iy = ((incy < 0) ? ((1 - n) * incy) : 0);
        for (; i < n; i += totalThreads) {
            result[iy + i * incy] = op(dx[ix + i * incx],dy[iy + i * incy],params);
        }
    }
}


