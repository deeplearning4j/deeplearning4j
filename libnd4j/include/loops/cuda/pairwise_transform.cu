
#ifndef PAIRWISE_TRANSFORM_CU
#define PAIRWISE_TRANSFORM_CU

#ifdef __CUDACC__

#include "../pairwise_transform.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <op_boilerplate.h>
#include <helpers/TAD.h>
#include <types/float16.h>
#include <helpers/DebugHelper.h>

#include "../legacy_ops.h"


/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */

template <typename T, typename opType>
__device__ void pairWiseTransformGeneric(
		T *dx,
		T *dy,
		T *params,
		T *result,
		Nd4jLong *xShapeInfo, int xRank,
		Nd4jLong *yShapeInfo, int yRank,
		Nd4jLong *resultShapeInfo, int zRank, 
		int *allocationPointer, 
		Nd4jLong *tadOnlyShapeInfo) {

	functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<opType>(
	    dx,
	    xShapeInfo,
	    dy,
	    yShapeInfo,
	    result,
	    resultShapeInfo,
	    params,
	    allocationPointer,
	    nullptr, tadOnlyShapeInfo);
}

/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */

/*
extern "C" __global__ void pairWiseTransformDouble(
		int opNum,
		double *dx,
		double *dy,
		double *params,
		double *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<double>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank, allocationPointer, tadOnlyShapeInfo);

}

extern "C" __global__ void pairWiseTransformFloat(
		int opNum,
		float *dx,
		float *dy,
		float *params,
		float *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<float>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank, allocationPointer, tadOnlyShapeInfo);

}

extern "C" __global__ void pairWiseTransformHalf(
		int opNum,
		float16 *dx,
		float16 *dy,
		float16 *params,
		float16 *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<float16>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank, allocationPointer, tadOnlyShapeInfo);
}

*/

/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */

/*
template <typename T>
__device__ void pairWiseTransformGeneric(
		int opNum,
		T *dx,
		T *dy,
		T *params,
		T *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer, int *tadOnlyShapeInfo) {

	__shared__ UnifiedSharedMemory *manager;

     if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::pairwise_transforms::PairWiseTransform<T>), sizeof(shape::TAD), xRank);
    }
    __syncthreads();

	functions::pairwise_transforms::PairWiseTransform<T>::transformCuda(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			result,
			resultShapeInfo,
			params,
			xIndexes,
			yIndexes,
			resultIndexes,
			allocationPointer,
			manager,
			tadOnlyShapeInfo);

}
*/

/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
/*
extern "C" __global__ void pairWiseTransformDoubleIndex(
		int opNum,
		double *dx,
		double *dy,
		double *params,
		double *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<double>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank,
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer, tadOnlyShapeInfo);

}

extern "C" __global__ void pairWiseTransformFloatIndex(
		int opNum,
		float *dx,
		float *dy,
		float *params,
		float *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<float>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank,
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer, tadOnlyShapeInfo);
}

extern "C" __global__ void pairWiseTransformHalfIndex(
		int opNum,
		float16 *dx,
		float16 *dy,
		float16 *params,
		float16 *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<float16>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank,
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer, tadOnlyShapeInfo);
}

*/

/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
template<typename T, typename opType>
__device__ void pairWiseTransformStridedGeneric(
		Nd4jLong n,
		T *dx,
		T *dy,
		Nd4jLong incx,
		Nd4jLong incy,
		T *params,
		T *result,
		Nd4jLong incz, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo) {

	functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<opType>(
		n,
		dx,
		dy,
		incx,
		incy,
		params,
		result,
		incz,
		allocationPointer,
		nullptr,
		tadOnlyShapeInfo);
}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
/*
extern "C" __global__ void pairWiseTransformStridedDouble(
		int opNum,
		Nd4jLong n,
		double *dx,
		double *dy,
		int incx,
		int incy,
		double *params,
		double *result,
		int incz, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformStridedGeneric<double>(
			opNum,
			n,
			dx,
			dy,
			incx,
			incy,
			params,
			result,
			incz, allocationPointer, tadOnlyShapeInfo);
}

extern "C" __global__ void pairWiseTransformStridedFloat(
		int opNum,
		Nd4jLong n,
		float *dx,
		float *dy,
		int incx,
		int incy,
		float *params,
		float *result,
		int incz, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformStridedGeneric<float>(
			opNum,
			n,
			dx,
			dy,
			incx,
			incy,
			params,
			result,
			incz, allocationPointer, tadOnlyShapeInfo);
}


extern "C" __global__ void pairWiseTransformStridedHalf(
		int opNum,
		Nd4jLong n,
		float16 *dx,
		float16 *dy,
		int incx,
		int incy,
		float16 *params,
		float16 *result,
		int incz, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformStridedGeneric<float16>(
			opNum,
			n,
			dx,
			dy,
			incx,
			incy,
			params,
			result,
			incz, allocationPointer, tadOnlyShapeInfo);
}
*/


// pwt shape
DISPATCH_KERNEL_SIMPLE(pwtSimpleShaped_, pairWiseTransformGeneric, float, INPUT(float *dx, float *dy, float *params, float *result, Nd4jLong *xShapeInfo, int xRank, Nd4jLong *yShapeInfo, int yRank, Nd4jLong *resultShapeInfo, int zRank, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo), PARAMS(dx, dy, params, result, xShapeInfo, xRank, yShapeInfo, yRank, resultShapeInfo, zRank, allocationPointer, tadOnlyShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(pwtSimpleShaped_, pairWiseTransformGeneric, double, INPUT(double *dx, double *dy, double *params, double *result, Nd4jLong *xShapeInfo, int xRank, Nd4jLong *yShapeInfo, int yRank, Nd4jLong *resultShapeInfo, int zRank, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo), PARAMS(dx, dy, params, result, xShapeInfo, xRank, yShapeInfo, yRank, resultShapeInfo, zRank, allocationPointer, tadOnlyShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(pwtSimpleShaped_, pairWiseTransformGeneric, float16, INPUT(float16 *dx, float16 *dy, float16 *params, float16 *result, Nd4jLong *xShapeInfo, int xRank, Nd4jLong *yShapeInfo, int yRank, Nd4jLong *resultShapeInfo, int zRank, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo), PARAMS(dx, dy, params, result, xShapeInfo, xRank, yShapeInfo, yRank, resultShapeInfo, zRank, allocationPointer, tadOnlyShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))

// pwt strided
DISPATCH_KERNEL_SIMPLE(pwtSimpleStrided_, pairWiseTransformStridedGeneric, float, INPUT(Nd4jLong n, float *dx, float *dy, Nd4jLong incx, Nd4jLong incy, float *params, float *result, Nd4jLong incz, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo), PARAMS(n, dx, dy, incx, incy, params, result, incz, allocationPointer, tadOnlyShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(pwtSimpleStrided_, pairWiseTransformStridedGeneric, double, INPUT(Nd4jLong n, double *dx, double *dy, Nd4jLong incx, Nd4jLong incy, double *params, double *result, Nd4jLong incz, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo), PARAMS(n, dx, dy, incx, incy, params, result, incz, allocationPointer, tadOnlyShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(pwtSimpleStrided_, pairWiseTransformStridedGeneric, float16, INPUT(Nd4jLong n, float16 *dx, float16 *dy, Nd4jLong incx, Nd4jLong incy, float16 *params, float16 *result, Nd4jLong incz, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo), PARAMS(n, dx, dy, incx, incy, params, result, incz, allocationPointer, tadOnlyShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))




namespace functions {
    namespace pairwise_transforms {

            template<>
            __host__ void PairWiseTransform<float>::execudaCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float *dx, Nd4jLong xStride, float *y, Nd4jLong yStride, float *result, Nd4jLong resultStride, float *extraParams, Nd4jLong n) {
                cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	            if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
		            printf("F4 opNum:[%i]; <<<X: [%i]; Y: [%i]; Z: [%i]>>>\n", opNum, launchDims.x,launchDims.y, launchDims.z);
	            }

	            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

            	//pairWiseTransformStridedFloat<<<launchDims.x, launchDims.y, launchDims.z, *stream>>> ( opNum, n, dx, y, xStride, yStride, extraParams, result, resultStride, allocationPointer, deviceTADShapeInfo);
            	DISPATCH_SIMPLE(pwtSimpleStrided, float, PARAMS(n, dx, y, xStride, yStride, extraParams, result, resultStride, allocationPointer, deviceTADShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))

	            DEBUG_KERNEL(stream, opNum);
            }

            template<>
            __host__ void PairWiseTransform<float16>::execudaCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float16 *dx, Nd4jLong xStride, float16 *y, Nd4jLong yStride, float16 *result, Nd4jLong resultStride, float16 *extraParams, Nd4jLong n) {
                cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

	            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		            printf("H4 opNum:[%i]\n", opNum);

	            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

            	//pairWiseTransformStridedHalf<<<launchDims.x,launchDims.y, launchDims.z, *stream>>> ( opNum, n, dx, y, xStride, yStride, extraParams, result, resultStride, allocationPointer, deviceTADShapeInfo);
            	DISPATCH_SIMPLE(pwtSimpleStrided, float16, PARAMS(n, dx, y, xStride, yStride, extraParams, result, resultStride, allocationPointer, deviceTADShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))

	            DEBUG_KERNEL(stream, opNum);
            }


           template<>
            __host__ void PairWiseTransform<double>::execudaCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *dx, Nd4jLong xStride, double *y, Nd4jLong yStride, double *result, Nd4jLong resultStride, double *extraParams, Nd4jLong n) {
                cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		            printf("D4 opNum:[%i]\n", opNum);

	            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

            	//pairWiseTransformStridedDouble<<<launchDims.x,launchDims.y, launchDims.z, *stream>>> ( opNum, n, dx, y, xStride, yStride, extraParams, result, resultStride, allocationPointer, deviceTADShapeInfo);
            	DISPATCH_SIMPLE(pwtSimpleStrided, double, PARAMS(n, dx, y, xStride, yStride, extraParams, result, resultStride, allocationPointer, deviceTADShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))


	            DEBUG_KERNEL(stream, opNum);
            }

            template<>
            __host__ void PairWiseTransform<float>::execudaCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float *dx, Nd4jLong *xShapeInfo, float *y, Nd4jLong *yShapeInfo, float *result, Nd4jLong *resultShapeInfo, float *extraParams) {
                cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		            printf("D6 opNum:[%i]\n", opNum);

            	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	            auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);
	            auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);
            	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	            //pairWiseTransformFloat<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>( opNum, dx, y, extraParams, result, xShapeInfo,  shape::rank(hostXShapeInfo), yShapeInfo,  shape::rank(hostYShapeInfo), resultShapeInfo,  shape::rank(hostZShapeInfo), allocationPointer, deviceTADShapeInfo);
                DISPATCH_SIMPLE(pwtSimpleShaped, float, PARAMS(dx, y, extraParams, result, xShapeInfo,  shape::rank(hostXShapeInfo), yShapeInfo,  shape::rank(hostYShapeInfo), resultShapeInfo,  shape::rank(hostZShapeInfo), allocationPointer, deviceTADShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))

	            DEBUG_KERNEL(stream, opNum);
            }


            template<>
            __host__ void PairWiseTransform<float16>::execudaCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float16 *dx, Nd4jLong *xShapeInfo, float16 *y, Nd4jLong *yShapeInfo, float16 *result, Nd4jLong *resultShapeInfo, float16 *extraParams) {
                cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		            printf("H6 opNum:[%i]\n", opNum);

            	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	            auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);
	            auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);
            	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	            //pairWiseTransformHalf<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>( opNum, dx, y, extraParams, result, xShapeInfo,  shape::rank(hostXShapeInfo), yShapeInfo,  shape::rank(hostYShapeInfo), resultShapeInfo,  shape::rank(hostZShapeInfo), allocationPointer, deviceTADShapeInfo);
                DISPATCH_SIMPLE(pwtSimpleShaped, float16, PARAMS(dx, y, extraParams, result, xShapeInfo,  shape::rank(hostXShapeInfo), yShapeInfo,  shape::rank(hostYShapeInfo), resultShapeInfo,  shape::rank(hostZShapeInfo), allocationPointer, deviceTADShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))

	            DEBUG_KERNEL(stream, opNum);
            }


            template<>
            __host__ void PairWiseTransform<double>::execudaCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *dx, Nd4jLong *xShapeInfo, double *y, Nd4jLong *yShapeInfo, double *result, Nd4jLong *resultShapeInfo, double *extraParams) {
                cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		            printf("H6 opNum:[%i]\n", opNum);

            	auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
	            auto hostYShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[7]);
	            auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);
            	auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

	            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);

	            //pairWiseTransformDouble<<<launchDims.x,launchDims.y, launchDims.z, *stream>>>( opNum, dx, y, extraParams, result, xShapeInfo,  shape::rank(hostXShapeInfo), yShapeInfo,  shape::rank(hostYShapeInfo), resultShapeInfo,  shape::rank(hostZShapeInfo), allocationPointer, deviceTADShapeInfo);
                DISPATCH_SIMPLE(pwtSimpleShaped, double, PARAMS(dx, y, extraParams, result, xShapeInfo,  shape::rank(hostXShapeInfo), yShapeInfo,  shape::rank(hostYShapeInfo), resultShapeInfo,  shape::rank(hostZShapeInfo), allocationPointer, deviceTADShapeInfo), OPS_A(PAIRWISE_TRANSFORM_OPS))

                DEBUG_KERNEL(stream, opNum);
            }

            /*
            template<typename T>
            __device__ void PairWiseTransform<T>::transformCuda(const int opNum, Nd4jLong n, T *dx, T *y, int incx, int incy, T *extraParams, T *result, int incz, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo) {
                    DISPATCH_BY_OPNUM(transformCuda, PARAMS(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo), PAIRWISE_TRANSFORM_OPS);
			}


            template<typename T>
			__device__ void PairWiseTransform<T>::transformCuda(const int opNum, T *dx, int *xShapeBuffer, T *y, int *yShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager,int *tadOnlyShapeInfo) {
                    DISPATCH_BY_OPNUM(transformCuda, PARAMS(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo), PAIRWISE_TRANSFORM_OPS);
			}
*/

			/*
            template<typename T>
			__device__ void PairWiseTransform<T>::transformCuda(const int opNum, T *dx, int *xShapeBuffer, T *y, int *yShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams, int *indexes, int *yIndexes, int *resultIndexes, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo) {
                    DISPATCH_BY_OPNUM(transform, PARAMS(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo), PAIRWISE_TRANSFORM_OPS);
			}
			*/


			/*
            template<typename T>
			template<typename OpType>
	        __device__ void PairWiseTransform<T>::transform(T *dx, int *xShapeBuffer, T *y, int *yShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams, int *indexes, int *yIndexes, int *resultIndexes,  int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo) {
	        	int tid = blockIdx.x * blockDim.x + threadIdx.x;
		        Nd4jLong n = shape::length(xShapeBuffer);

		        for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
			        result[resultIndexes[i]] = OpType::op(dx[indexes[i]],y[yIndexes[i]], extraParams);
		        }
	        }
	        */

	        /**
	 *
	 */
	    template<typename T>
        template<typename OpType>
	    __device__ void PairWiseTransform<T>::transformCuda(T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo) {
		    int tid = blockIdx.x * blockDim.x + threadIdx.x;

		    __shared__ int xRank;
		    __shared__ int yRank;
		    __shared__ int resultRank;

		    __shared__ int xEWS;
		    __shared__ int yEWS;
		    __shared__ int zEWS;

		    __shared__ char xOrder;
		    __shared__ char yOrder;
		    __shared__ char zOrder;

		    __shared__ bool xRow;
		    __shared__ bool yRow;
		    __shared__ bool zRow;

		    if (threadIdx.x == 0) {
		        xRank = shape::rank(xShapeBuffer);
    		    yRank = shape::rank(yShapeBuffer);
	    	    resultRank = shape::rank(resultShapeBuffer);

		        xEWS = shape::elementWiseStride(xShapeBuffer);
		        yEWS = shape::elementWiseStride(yShapeBuffer);
    		    zEWS = shape::elementWiseStride(resultShapeBuffer);

	    	    xOrder = shape::order(xShapeBuffer);
		        yOrder = shape::order(yShapeBuffer);
		        zOrder = shape::order(resultShapeBuffer);

		        xRow = shape::isRowVector(xShapeBuffer);
		        yRow = shape::isRowVector(yShapeBuffer);
		        zRow = shape::isRowVector(resultShapeBuffer);

		    }
		    __syncthreads();

		    Nd4jLong n = shape::length(xShapeBuffer);
		    if((xEWS >= 1 && yEWS == xEWS && zEWS == xEWS &&  xOrder == yOrder && zOrder == xOrder) || (xEWS >= 1 && yEWS == xEWS && zEWS == xEWS && xRow && yRow && zRow)) {
			    // TODO: this is wrong, and should be moved to host side
			    transformCuda<OpType>(
					n,
					dx,
					y,
					xEWS,
					yEWS,
					extraParams,
					result,
					zEWS, allocationPointer, manager, tadOnlyShapeInfo);

    		} else {

    			if (dx == result) {
					Nd4jLong xCoord[MAX_RANK];
					Nd4jLong yCoord[MAX_RANK];

	    			for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
		    			shape::ind2subC(xRank,shape::shapeOf(xShapeBuffer), i, xCoord);
			    		shape::ind2subC(yRank,shape::shapeOf(yShapeBuffer), i, yCoord);

				    	auto xOffset = shape::getOffset(0, shape::shapeOf(xShapeBuffer), shape::stride(xShapeBuffer), xCoord, xRank);
					    auto yOffset = shape::getOffset(0, shape::shapeOf(yShapeBuffer), shape::stride(yShapeBuffer), yCoord, yRank);
    					result[xOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);
	    			}
		    	} else {
					Nd4jLong xCoord[MAX_RANK];
			    	Nd4jLong yCoord[MAX_RANK];
    		    	Nd4jLong resultCoord[MAX_RANK];

    				for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
	    				shape::ind2subC(xRank,shape::shapeOf(xShapeBuffer), i, xCoord);
		    			shape::ind2subC(yRank,shape::shapeOf(yShapeBuffer), i, yCoord);
			    		shape::ind2subC(resultRank,shape::shapeOf(resultShapeBuffer), i, resultCoord);

    					auto xOffset = shape::getOffset(0, shape::shapeOf(xShapeBuffer), shape::stride(xShapeBuffer), xCoord, xRank);
	    				auto yOffset = shape::getOffset(0, shape::shapeOf(yShapeBuffer), shape::stride(yShapeBuffer), yCoord, yRank);
		    			auto resultOffset = shape::getOffset(0, shape::shapeOf(resultShapeBuffer), shape::stride(resultShapeBuffer), resultCoord, resultRank);
			    		result[resultOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);
    				}
    			}
    		}
    	}


     /*
	 template<typename T>
	 __device__ void transform(
			T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			Nd4jLong n,
			int *indexes,
			int *allocationPointer,
			UnifiedSharedMemory *manager,
			int *tadOnlyShapeInfo) {
		transform(dx,
				xShapeBuffer,
				y,
				yShapeBuffer,
				result,
				resultShapeBuffer,
				extraParams,
				indexes,
				indexes,
				indexes, allocationPointer, manager, tadOnlyShapeInfo);
	    }


	 template<typename T>
	 __device__ void transform(
			T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *indexes,
			int *yIndexes,
			int *allocationPointer,
			UnifiedSharedMemory *manager,
			int *tadOnlyShapeInfo) {
		transform(dx,
				xShapeBuffer,
				y,
				yShapeBuffer,
				result,
				resultShapeBuffer,
				extraParams,
				indexes,
				yIndexes,
				indexes, allocationPointer, manager, tadOnlyShapeInfo);
	    }
*/

	    /**
	 *
	 * @param n
	 * @param xOffset
	 * @param yOffset
	 * @param resultOffset
	 * @param dx
	 * @param dy
	 * @param incx
	 * @param incy
	 * @param params
	 * @param result
	 * @param incz
	 * @param blockSize
	 */
    template<typename T>
    template<typename OpType>
	__device__ void PairWiseTransform<T>::transformCuda(
			Nd4jLong n,
			T *dx,
			T *dy,
			Nd4jLong incx,
			Nd4jLong incy,
			T *params,
			T *result,
			Nd4jLong incz,int *allocationPointer,
			UnifiedSharedMemory *manager,
			Nd4jLong *tadOnlyShapeInfo) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (incx == incy && incy == incz && incx == 1) {
			for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
				result[i] = OpType::op(dx[i], dy[i], params);
			}
		} else {
			for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
				result[i * incz] = OpType::op(dx[i * incx], dy[i * incy], params);
			}
		}
	}
    }
}




#endif // CUDA_CC

#endif // PAIRWISE_TRANSFORM_CU