/*
 * postprocess_impl.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#include <postprocess.h>

namespace nd4j {
    namespace functions {
        namespace reduce {
/*
*
 *
 * @param n
 * @param xOffset
 * @param dx
 * @param incx
 * @param extraParams
 * @param result

template <typename T>
__device__ void postProcessLoop(int n,int xOffset,T *dx,int incx,T *extraParams,T *result) {
	int tid = threadIdx.x;
	int i = xOffset + (blockIdx.x * blockDim.x + tid);

	for(; i < n; i += gridDim.x * blockDim.x * incx) {
		result[i] = postProcess(result[i],n,xOffset,dx,incx,extraParams,result);
	}
}

}



extern "C"
__global__ void postProcessLoop_double(int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
	nd4j::functions::reduce::postProcessLoop<double>(n,xOffset,dx,incx,extraParams,result);
}

extern "C"
__global__ void postProcessLoop_float(int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
	nd4j::functions::reduce::postProcessLoop<float>(n,xOffset,dx,incx,extraParams,result);
}

*/

        }
    }
}
