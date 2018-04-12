/*
 * transform.h
 *
 *  Created on: Dec 28, 2015
 *  @author: agibsonccc
 *  @author: raver119@gmail.com
 */

#ifndef TRANSFORM_H_
#define TRANSFORM_H_
#include <vector>
#include <templatemath.h>
#include <ops/ops.h>
#include <ops/special_ops.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pairwise_util.h>
#include <dll.h>

//#include <loops/reduce.h>
//#include <loops/scalar.h>
//#include <loops/indexreduce.h>
//#include <loops/broadcasting.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include "legacy_ops.h"


namespace functions {
    namespace transform {

        template<typename T>
        class Transform {
        public:

#ifdef __CUDACC__

	template<typename OpType>
	static  __device__ void transformCuda(
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *resultShapeInfo,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, int *tadShapeInfo, Nd4jIndex *tadOffsets);

	template<typename OpType>
	static  __device__ void transformCuda(
			Nd4jIndex n,
			T *dy,
			int incy,
			T *params,
			T *result,
			int resultStride,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager);

	static  __device__ void transformCuda(
			const int opNum,
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *resultShapeInfo,
			int *allocationPointer,
			T *reductionPointer,
			UnifiedSharedMemory *manager, int *tadShapeInfo, Nd4jIndex *tadOffsets);


	static  __device__ void transformCuda(
			const int opNum,
			Nd4jIndex n,
			T *dy,
			int incy,
			T *params,
			T *result,
			int resultStride,
			int *allocationPointer,
			T *reductionPointer,
			UnifiedSharedMemory *manager);

	static _CUDA_H void executeTransformStrided(dim3 launchDims, cudaStream_t *stream, int opNum, Nd4jIndex n, T *x, int xStride, T *extraParams, T *z, int zStride, int *allocationPointer, T *reductionPointer);
	
	static _CUDA_H void executeTransformShaped(dim3 launchDims, cudaStream_t *stream, int opNum, T *x, int *xShape, int xRank, T *extraParams, T *z, int *zShape, int zRank, int *allocationPointer, T *reductionPointer,  int *tadShapeInfo, Nd4jIndex *tadOffsets);

#endif


			static void exec(int opNum, T *dx, int xStride, T *result, int resultStride, T *extraParams, const int n);

			static void exec(int opNum, T *dx, int *xShapeInfo, T *result, int *resultShapeInfo, T *extraParams, int *tadShapeInfo, Nd4jIndex *tadOffsets);


			template<typename OpType>
			static ND4J_EXPORT void exec(T *dx, int *xShapeInfo, T *result, int *resultShapeInfo, T *extraParams, int *tadShapeInfo, Nd4jIndex *tadOffsets);

			template <typename OpType>
			static ND4J_EXPORT void exec(T *dx, int xStride, T *result, int resultStride, T *extraParams, const int n);
        };
    }
}


#endif /* TRANSFORM_H_ */
