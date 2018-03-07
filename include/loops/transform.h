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
#include <loops/reduce.h>
#include <loops/scalar.h>
#include <loops/indexreduce.h>
#include <loops/broadcasting.h>

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
            /**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
	virtual __inline__ __device__ void transform(
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *indexes) {
		Nd4jIndex n = shape::length(shapeInfo);
		int totalThreads = gridDim.x * blockDim.x;
		Nd4jIndex i = blockIdx.x * blockDim.x + threadIdx.x;

		/* equal, positive, non-unit increments. */
#pragma unroll
		for (; i < n; i+= totalThreads) {
			result[indexes[i]] = op(dy[indexes[i]], params);
		}

	}


	/**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
template<typename OpType>
static __inline__ __device__ void transformCuda(
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *resultShapeInfo,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, int *tadShapeInfo, Nd4jIndex *tadOffsets) {

		if(OpType::requiresSpecial) {
			OpType::execSpecialCuda(dy,shapeInfo,result,resultShapeInfo,params, allocationPointer, reductionPointer, manager, tadShapeInfo, tadOffsets);
			return;
		} else {

		int *xShape = shape::shapeOf(shapeInfo);
		int *xStride = shape::stride(shapeInfo);
		char xOrder = shape::order(shapeInfo);
		char resultOrder = shape::order(resultShapeInfo);
		int xRank = shape::rank(shapeInfo);
		int xOffset = shape::offset(shapeInfo);

		int xElementWiseStride = shape::elementWiseStride(shapeInfo);
		int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
		int tid = blockIdx.x * blockDim.x + threadIdx.x;


		__shared__ int length;
		if(threadIdx.x == 0)
			length = shape::length(shapeInfo);
		__syncthreads();

		if(xElementWiseStride >= 1 && resultElementWiseStride >= 1 && xOrder == resultOrder) {
			transformCuda<OpType>(
					length,
					dy,
					xElementWiseStride,
					params,
					result,
					resultElementWiseStride, allocationPointer, reductionPointer, manager);
		}
		else {
			/* equal, positive, non-unit increments. */
			//long allocSize = sizeof(int) * xRank;
			//int *xIdx = shape::cuMalloc(manager->getT1ShapeBuffer(), allocSize);
			int xCoord[MAX_RANK];

#pragma unroll
			for (Nd4jIndex i = tid; i < length; i+= gridDim.x * blockDim.x) {
				//int *xIdx = shape::ind2sub(xRank, xShape, i, xIdx);
				shape::ind2sub(xRank,shape::shapeOf(shapeInfo),i, xCoord);
				Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xCoord, xRank);
				Nd4jIndex resultOffset2 = shape::getOffset(0,xShape,shape::stride(resultShapeInfo),xCoord,xRank);
				result[resultOffset2] = OpType::op(dy[xOffset2], params);
			}
		}
	  }
	}

	/**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
template<typename OpType>
	static  __inline__ __device__ void transformCuda(
			Nd4jIndex n,
			T *dy,
			int incy,
			T *params,
			T *result,
			int resultStride,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {
		int totalThreads = gridDim.x * blockDim.x;
		Nd4jIndex i = blockIdx.x * blockDim.x + threadIdx.x;

		if(incy == 1 && resultStride == 1) {
			/* equal, positive, non-unit increments. */
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i] = OpType::op(dy[i], params);
			}
		}
		else {
			/* equal, positive, non-unit increments. */
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i * resultStride] = OpType::op(dy[i * incy], params);
			}
		}


	}

	static  __inline__ __device__ void transformCuda(
			const int opNum,
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *resultShapeInfo,
			int *allocationPointer,
			T *reductionPointer,
			UnifiedSharedMemory *manager, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
                                DISPATCH_BY_OPNUM(transformCuda, PARAMS(dy, shapeInfo, params, result, resultShapeInfo, allocationPointer, reductionPointer, manager, tadShapeInfo, tadOffsets), TRANSFORM_OPS);
	}


	static  __inline__ __device__ void transformCuda(
			const int opNum,
			Nd4jIndex n,
			T *dy,
			int incy,
			T *params,
			T *result,
			int resultStride,
			int *allocationPointer,
			T *reductionPointer,
			UnifiedSharedMemory *manager) {
                                DISPATCH_BY_OPNUM(transformCuda, PARAMS(n, dy, incy, params, result, resultStride, allocationPointer, reductionPointer, manager), TRANSFORM_OPS);
	}
#endif


			static void exec(int opNum, T *dx, int xStride, T *result, int resultStride, T *extraParams, const int n) {
                                DISPATCH_BY_OPNUM(exec, PARAMS(dx, xStride, result, resultStride, extraParams, n), TRANSFORM_OPS);
			}

			static void exec(
				int opNum,
				T *dx,
				int *xShapeInfo,
				T *result,
				int *resultShapeInfo,
				T *extraParams,
				int *indexes,
				int *resultIndexes, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
                            DISPATCH_BY_OPNUM(exec, PARAMS(dx, xShapeInfo, result, resultShapeInfo, extraParams, indexes, resultIndexes, tadShapeInfo, tadOffsets), TRANSFORM_OPS);
			}


			static void exec(
				int opNum,
				T *dx,
				int *xShapeInfo,
				T *result,
				int *resultShapeInfo,
				T *extraParams, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
                                DISPATCH_BY_OPNUM(exec, PARAMS(dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), TRANSFORM_OPS);
			}


			template<typename OpType>
			static void _CUDA_H exec(
                    T *dx,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T *extraParams, int *tadShapeInfo, Nd4jIndex *tadOffsets) {

                if(OpType::requiresSpecial) {
                    OpType::execSpecial(dx,xShapeInfo,result,resultShapeInfo,extraParams, tadShapeInfo, tadOffsets);
                    return;
                }

                int n = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);

                if(xElementWiseStride >= 1 && resultElementWiseStride >= 1 && shape::order(xShapeInfo) == shape::order(resultShapeInfo)) {
                    exec<OpType>(dx,xElementWiseStride,result,resultElementWiseStride,extraParams,n);
                }
                else {
                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int xStridesIter[MAX_RANK];
                    int resultStridesIter[MAX_RANK];
                    int *xShape = shape::shapeOf(xShapeInfo);
                    int *xStride = shape::stride(xShapeInfo);
                    int *resultStride = shape::stride(resultShapeInfo);
                    int rank = shape::rank(xShapeInfo);
                    if(PrepareTwoRawArrayIter<T>(rank,
                                                 xShape,
                                                 dx,
                                                 xStride,
                                                 result,
                                                 resultStride,
                                                 &rank,
                                                 shapeIter,
                                                 &dx,
                                                 xStridesIter,
                                                 &result,
                                                 resultStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter);
                        {
                            // Process the innermost dimension
                            T *xIter = dx;
                            T *resultIter = result;
                            resultIter[0] = OpType::op(xIter[0], extraParams);
                        }
                        ND4J_RAW_ITER_TWO_NEXT(dim,
                                               rank,
                                               coord,
                                               shapeIter,
                                               dx,
                                               xStridesIter,
                                               result,
                                               resultStridesIter);

                    }

                }
            }


			template<typename OpType>
			static void exec(
				T *dx,
				int *xShapeInfo,
				T *result,
				int *resultShapeInfo,
				T *extraParams,
				int *indexes,
				int *resultIndexes, int *tadShapeInfo, Nd4jIndex *tadOffsets) {

				int n = shape::length(xShapeInfo);
#pragma omp parallel for simd schedule(guided) proc_bind(AFFINITY) default(shared)
				for (Nd4jIndex i = 0; i < n; i++) {
					result[resultIndexes[i]] = OpType::op(dx[indexes[i]], extraParams);
				}
			}

            template<typename OpType>
            static void exec(T *dx,
                             int xStride,
                             T *result,
                             int resultStride,
                             T *extraParams,
                             const int n) {

                int elementsPerThread = n / ELEMENT_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                int span = (n / num_threads) + 8;

                if (xStride == 1 && resultStride == 1) {

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        int tid = omp_get_thread_num();
                        int start = span * tid;
                        int end = span * (tid + 1);
                        if (end > n) end = n;

#pragma omp simd
                        for (Nd4jIndex i = start; i < end; i++) {
                            result[i] = OpType::op(dx[i], extraParams);
                        }
                    }
                } else {

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        int tid = omp_get_thread_num();
                        int start = span * tid;
                        int end = span * (tid + 1);
                        if (end > n) end = n;

#pragma omp simd
                        for (Nd4jIndex i = start; i < end; i++) {
                            result[i*resultStride] = OpType::op(dx[i * xStride], extraParams);
                        }
                    }
                }
            }
        };
    }
}




#ifdef __CUDACC__
/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
template <typename T>
__device__ void transformGeneric(
		int opNum,
		Nd4jIndex n,
		T *dy,
		int incy,
		T *params,
		T *result,
		int resultStride, int *allocationPointer, T *reductionPointer) {

	__shared__ UnifiedSharedMemory *manager;

	if(threadIdx.x == 0) {
	    extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::transform::Transform<T>), sizeof(shape::TAD), 0);
	}
	__syncthreads();

	functions::transform::Transform<T>::transformCuda(
		opNum,
		n,
		dy,
		incy,
		params,
		result,
		resultStride,
		allocationPointer,
		reductionPointer,
		manager);
}

template <typename T, typename OpClass>
__device__ void transformSimpleGeneric(
		Nd4jIndex n,
		T *dy,
		int incy,
		T *params,
		T *result,
		int resultStride, int *allocationPointer, T *reductionPointer) {

	__shared__ UnifiedSharedMemory *manager;

	if(threadIdx.x == 0) {
	    extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::transform::Transform<T>), sizeof(shape::TAD), 0);
	}
	__syncthreads();

	functions::transform::Transform<T>::template transformCuda<OpClass>(
		n,
		dy,
		incy,
		params,
		result,
		resultStride,
		allocationPointer,
		reductionPointer,
		manager);
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
/*
template <typename T>
__device__ void transformGeneric(
		int opNum,
		T *dy,
		int *xShapeInfo, int xRank,
		T *params,
		T *result,int *resultShapeInfo, int zRank, int *allocationPointer, T *reductionPointer) {

	__shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::transform::Transform<T>), sizeof(shape::TAD), xRank);
	}
	__syncthreads();


	functions::transform::Transform<T>::transformCuda(
	    opNum,
	    dy,
	    xShapeInfo,
	    params,
	    result,
	    resultShapeInfo,
	    allocationPointer,
	    reductionPointer,
	    manager);
}
*/


template <typename T, typename OpClass>
__device__ void transformSimpleGeneric(
		T *dy,
		int *xShapeInfo, int xRank,
		T *params,
		T *result,int *resultShapeInfo, int zRank, int *allocationPointer, T *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::transform::Transform<T>), sizeof(shape::TAD), xRank);
	}
	__syncthreads();


	functions::transform::Transform<T>::template transformCuda<OpClass>(
	    dy,
	    xShapeInfo,
	    params,
	    result,
	    resultShapeInfo,
	    allocationPointer,
	    reductionPointer,
	    manager, tadShapeInfo, tadOffsets);
}



/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
template <typename T>
__device__ void transformGenericIndexes(
		int opNum,
		T *dy,
		int *xShapeInfo, int xRank,
		T *params,
		T *result,int *indexes, int *allocationPointer, T *reductionPointer) {

	__shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::transform::Transform<T>), sizeof(shape::TAD), xRank);
	}
	__syncthreads();

/*
	functions::transform::Transform<T>::transformCuda(
	        opNum,
	        dy,
	        xShapeInfo,
	        params,
	        result,
	        indexes,
	        allocationPointer,
	        reductionPointer,
	        manager);
	        */
}



/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
extern "C" __global__ void transformDoubleIndexes(
		int opNum,
		double *dy,
		int *shapeInfo, int xRank,
		double *params,
		double *result,int *indexes, int *allocationPointer, double *reductionPointer) {

	transformGenericIndexes<double>(
			opNum,
			dy,
			shapeInfo, xRank,
			params,
			result,indexes, allocationPointer, reductionPointer);
}

/**
 * The c and driver interface
 *  for th kernels
 * @param opNum the op number
 * @param n the length of the problem
 * @param idx
 * the start index
 * @param dy the vector to transform
 * @param incy the stride for the vector
 * @param params the extra parameters for the problem
 * @param result the result storage
 * @param blockernelHeight the block size for the problem
 */
extern "C" __global__ void transformFloatIndexes(
		int opNum,
		float *dy,
		int *shapeInfo, int xRank,
		float *params,
		float *result,int *indexes, int *allocationPointer, float *reductionPointer) {

	transformGenericIndexes<float>(
			opNum,
			dy,
			shapeInfo, xRank,
			params,
			result,indexes, allocationPointer, reductionPointer);

}

extern "C" __global__ void transformHalfIndexes(
		int opNum,
		float16 *dy,
		int *shapeInfo, int xRank,
		float16 *params,
		float16 *result,int *indexes, int *allocationPointer, float16 *reductionPointer) {

	transformGenericIndexes<float16>(
			opNum,
			dy,
			shapeInfo, xRank,
			params,
			result,indexes, allocationPointer, reductionPointer);

}



/**
* This is utility kernel, that updates given special buffer with proper values in device memory
*/
extern "C" __global__ void prepareShapeBuffer(int *dimension, int *maxDimension, int *specialPointer, int rows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    dimension[0] = 0;
    maxDimension[0] = 1;

    specialPointer[0] = 2;
    specialPointer[1] = rows;
    specialPointer[2] = 1;
    specialPointer[3] = 1;
    specialPointer[4] = 1;
    specialPointer[5] = 0;
    specialPointer[6] = 1;
    specialPointer[7] = 99;
}

extern "C" __global__ void prepareDimensionalShapeBuffer(int *xShapeInfoBuffer, float *extraParams, int *zShapeInfo) {
    // extraParams[0] - number of dimensions
    // extraParams[1] - dimension
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    int targetDimension = (int) extraParams[1];
    printf("Target dimension: [%i]\n", targetDimension);

    int targetWidth = shape::shapeOf(xShapeInfoBuffer)[targetDimension];
    printf("Target rank: [%i]\n", targetWidth);
}

template <typename T>
__device__ void fillIsMaxGeneric(T *dx, long length, long idx) {

   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   for (long i = tid; i < length; i+= blockDim.x * gridDim.x) {
        dx[i] = (i == idx? 1.0 : 0.0);
   }
}

extern "C" __global__ void fillIsMaxFloat(float *dx, long length, long idx) {
    fillIsMaxGeneric<float>(dx, length, idx);
}

extern "C" __global__ void fillIsMaxDouble(double *dx, long length, long idx) {
    fillIsMaxGeneric<double>(dx, length, idx);
}

extern "C" __global__ void fillIsMaxHalf(float16 *dx, long length, long idx) {
    fillIsMaxGeneric<float16>(dx, length, idx);
}

template <typename T>
__device__ void fillDimensionalIsMaxGeneric(T *dX, int *xShapeInfo, T *dZ, int *zShapeInfo, int *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jIndex *tadOffsets) {

    __shared__ int tadLength;
    __shared__ int tadEWS;
    __shared__ int numTads;

    __shared__ int *tadShape;
    __shared__ int *tadStride;
    __shared__ int tadRank;
    __shared__ char tadOrder;

    if (threadIdx.x == 0) {
        tadLength = shape::tadLength(zShapeInfo, dimension, dimensionLength);
        tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
        numTads = shape::length(zShapeInfo) / tadLength;

        tadShape = shape::shapeOf(tadOnlyShapeInfo);
        tadStride = shape::stride(tadOnlyShapeInfo);
        tadRank = shape::rank(tadOnlyShapeInfo);
        tadOrder = shape::order(tadOnlyShapeInfo);
    }
    __syncthreads();



    for (int r = blockIdx.x; r < numTads; r+= gridDim.x) {
        int tadOffsetForBlock = tadOffsets[r];

        int highestElement = (int) dX[r];

        if (dimensionLength > 1 || tadEWS < 1) {
            int xCoord[MAX_RANK];

            for (int e = threadIdx.x; e < tadLength; e += blockDim.x) {
                shape::ind2subC(tadRank,tadShape, e, xCoord);

                Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

                dZ[xOffset] = (e == highestElement? (T) 1.0f : (T) 0.0f);
            }
        } else {
            for (int e = threadIdx.x; e < tadLength; e += blockDim.x) {
                // so, we just set dZ[e] for each TAD. Sure, e should be replaced with
                int idx = tadOffsetForBlock + (e * tadEWS);
                dZ[idx] = (e == highestElement? (T) 1.0f : (T) 0.0f);
            }
        }

    }
}

extern "C" __global__ void fillDimensionalIsMaxFloat(float *dx, int *xShapeInfo, float *dz, int *zShapeInfo, int *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jIndex *tadOffsets) {
    fillDimensionalIsMaxGeneric<float>(dx, xShapeInfo, dz, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
}

extern "C" __global__ void fillDimensionalIsMaxDouble(double *dx, int *xShapeInfo, double *dz, int *zShapeInfo, int *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jIndex *tadOffsets) {
    fillDimensionalIsMaxGeneric<double>(dx, xShapeInfo, dz, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
}

extern "C" __global__ void fillDimensionalIsMaxHalf(float16 *dx, int *xShapeInfo, float16 *dz, int *zShapeInfo, int *tadOnlyShapeInfo, int *dimension, int dimensionLength, Nd4jIndex *tadOffsets) {
    fillDimensionalIsMaxGeneric<float16>(dx, xShapeInfo, dz, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
}

template <typename T>
__device__ void concatKernelGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									T *result,
									int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, int *zTadShape, Nd4jIndex *zOffsets) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int zRank = shape::rank(resultShapeInfo);

	T **dataT = (T **) data;
	int **shapeInfoPointers = (int **) inputShapeInfos;
	int **tadShapes = (int **) tadPointers;
	Nd4jIndex **tadOffsets = (Nd4jIndex **) offsetPointers;


    //__shared__ int tDim[1];
        __shared__ int baseIdx;

		__shared__ int yLength;
		__shared__ char yOrder;
		__shared__ int yEWS;

		char zOrder = shape::order(resultShapeInfo);

		int zEWS = shape::elementWiseStride(resultShapeInfo);
		int tadEWS = shape::elementWiseStride(zTadShape);
		int zLength = shape::length(resultShapeInfo);

        __shared__ int arrOffset;
		__shared__ int numTads;


        if (shape::isVector(resultShapeInfo)) {
			//if (threadIdx.x == 0)
			//	printf("Vector here\n");
				
			if (zEWS >= 1) {
				for (int r = blockIdx.x; r < numArrays; r += gridDim.x) {
					if(shape::isVector(shapeInfoPointers[r]) || shape::order(shapeInfoPointers[r]) == shape::order(resultShapeInfo)) {
						yLength = shape::length(shapeInfoPointers[r]);
						yEWS = shape::elementWiseStride(shapeInfoPointers[r]);
						// FIXME: this is bad
						__shared__ int baseIdx;
						if (threadIdx.x == 0) {
							baseIdx = 0;
							for (int f = 0; f < r; f++) {
								baseIdx += shape::length(shapeInfoPointers[f]);
							}
						}
						__syncthreads();
						for (int i = threadIdx.x; i < yLength && baseIdx + i < zLength; i += blockDim.x) {
							result[baseIdx + i * zEWS] = dataT[r][i * yEWS];
						}
						__syncthreads();
					} else {
						if (tid == 0)
							printf("Non-matched order for vector\n");
					}
				}
			} else {
				if (tid == 0)
					printf("Vector Non-1 zEWS\n");
			}
			return;
		}


		bool _vec = shape::isVector(resultShapeInfo);


		// TODO: to be pulled into separate kernel. matrix concatenation
		for (int r = 0; r < numArrays; r ++) {

			int *currentShape = shapeInfoPointers[r];
			T *currentData = dataT[r];
			int *currentTad = tadShapes[r];
			Nd4jIndex *currentOffsets = tadOffsets[r];


			if (threadIdx.x == 0) {
				yLength = shape::length(currentTad);
				yOrder = shape::order(currentTad);
				yEWS = shape::elementWiseStride(currentTad);
                numTads = shape::length(currentShape) / yLength;

                arrOffset = 0;
				for (int f = 0; f < r; f++) {
					arrOffset +=  shape::length(tadShapes[f]);
				}

			}
			__syncthreads();

            if (yLength == 1 && _vec) {
				//if (threadIdx.x == 0)
				//	printf("Branch 0\n");

                // edge case, each thread will handle it's own tad then
                for (int j = tid; j < numTads; j += blockDim.x * gridDim.x) {
                    Nd4jIndex inputOffset = currentOffsets[j];
				    Nd4jIndex resultOffset = zOffsets[j];

				    T *dataTAD = currentData + inputOffset;
				    T *resultTAD = result + resultOffset;

                    int sub[MAX_RANK];

                    if (shape::order(zTadShape) == 'f') {
				        shape::ind2sub(shape::rank(zTadShape),shape::shapeOf(zTadShape),arrOffset, sub);
				    } else {
				        shape::ind2subC(shape::rank(zTadShape),shape::shapeOf(zTadShape),arrOffset, sub);
				    }
				    Nd4jIndex baseOffset = shape::getOffset(0,shape::shapeOf(zTadShape),shape::stride(zTadShape), sub, shape::rank(zTadShape));

				    resultTAD += baseOffset;

					int yRank = shape::rank(currentTad);
					int tadRank = shape::rank(zTadShape);

					shape::ind2subC(yRank, shape::shapeOf(currentTad), 0,sub);

					Nd4jIndex yOffset = shape::getOffset(0, shape::shapeOf(currentTad), shape::stride(currentTad), sub, yRank);
					resultOffset = shape::getOffset(0, shape::shapeOf(zTadShape), shape::stride(zTadShape), sub, tadRank);

					resultTAD[resultOffset] =  dataTAD[yOffset];
                }
            } else {
				//if (threadIdx.x == 0)
				//	printf("Branch 1\n");

			    for (int j = blockIdx.x; j < numTads; j += gridDim.x) {
				    Nd4jIndex inputOffset = currentOffsets[j];
				    Nd4jIndex resultOffset = zOffsets[j];

				    T *dataTAD = currentData + inputOffset;
				    T *resultTAD = result + resultOffset;

                    int sub[MAX_RANK];

				    shape::ind2subC(shape::rank(zTadShape),shape::shapeOf(zTadShape),arrOffset, sub);
				    Nd4jIndex baseOffset = shape::getOffset(0,shape::shapeOf(zTadShape),shape::stride(zTadShape), sub, shape::rank(zTadShape));

				    resultTAD += baseOffset;

				    if (zOrder == yOrder && yEWS > 0  && tadEWS > 0) {
				        //if (threadIdx.x == 0)
				        //    printf("Branch A\n");

					    for (int i = threadIdx.x; i < yLength; i += blockDim.x) {
						    resultTAD[i * tadEWS] = dataTAD[i * yEWS];
					    }
				    } else {
					    if(tadEWS > 0 && shape::order(resultShapeInfo) == shape::order(currentTad)) {
					        //if (threadIdx.x == 0)
				            //    printf("Branch B\n");

						    if (threadIdx.x == 0) {
							    baseIdx = 0;
							    for (int f = 0; f < r; f++) {
							    	baseIdx += shape::length(shapeInfoPointers[f]);
						    	}
					    		//printf("R: %i; baseIdx: %i;\n", baseIdx);
				    		}
			    			__syncthreads();

		    				if (numTads == 1) {
	    						for(int k = threadIdx.x; k < yLength; k+= blockDim.x) {
    								resultTAD[baseIdx + k * tadEWS] = dataTAD[k];
							    }
						    } else {
							    int yIdx[MAX_RANK];
							    int yRank = shape::rank(currentTad);

							    for (int i = threadIdx.x; i < yLength; i+= blockDim.x) {
								    shape::ind2subC(yRank, shape::shapeOf(currentTad), i, yIdx);
								    int yOffset = shape::getOffset(0, shape::shapeOf(currentTad), shape::stride(currentTad), yIdx, yRank);

								    resultTAD[baseIdx + i * tadEWS] =  dataTAD[yOffset];
							    }
						    }
						    __syncthreads();
					    } else {
                            //if (threadIdx.x == 0)
				            //    printf("Branch C; yLength: %i;\n", yLength);

						    int yIdx[MAX_RANK];
						    int yRank = shape::rank(currentTad);
						    int tadRank = shape::rank(zTadShape);

						    for (int i = threadIdx.x; i < yLength; i+= blockDim.x) {
							    shape::ind2subC(yRank, shape::shapeOf(currentTad), i,yIdx);

							    int yOffset = shape::getOffset(0, shape::shapeOf(currentTad), shape::stride(currentTad), yIdx, yRank);
							    int resultOffset = shape::getOffset(0, shape::shapeOf(zTadShape), shape::stride(zTadShape), yIdx, tadRank);

							    resultTAD[resultOffset] =  dataTAD[yOffset];
						    }
					    }
				    }
				    __syncthreads();
			    }
			}
			__syncthreads();
		}
}

template <typename T>
__device__ void concatKernelScalarGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									T *result,
									int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    T **input = (T **) data;

    for (int i = tid; i < numArrays; i += blockDim.x * gridDim.x) {
			result[i] = input[i][0];
	}
}

extern "C" __global__ void concatKernelScalarFloat(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float *result,
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelScalarGeneric<float>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelScalarHalf(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float16 *result,
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelScalarGeneric<float16>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelScalarDouble(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  double *result,
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelScalarGeneric<double>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}


template <typename T>
__device__ void concatKernelHStackGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									T *result,
									int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    // we expect all data coming in as vectors, and result as 2D matrix
    // the only significant difference here is the fact that input lengths might be different
    int **inputShapes = (int**) inputShapeInfos;
     T **input = (T **) data;

     __shared__ int inputEWS;
     __shared__ int resultEWS;
     __shared__ int inputLength;

     if (threadIdx.x == 0) {
        resultEWS = shape::elementWiseStride(resultShapeInfo);
     }
     __syncthreads();

     for (int r = blockIdx.x; r < numArrays; r+= gridDim.x) {

        __shared__ int baseIdx;
		if (threadIdx.x == 0) {
			baseIdx = 0;
			for (int f = 0; f < r; f++) {
			    baseIdx += shape::length(inputShapes[f]);
		    }
		}
		__syncthreads();


        T *inputData = (T *) input[r];

        if (threadIdx.x == 0) {
         inputEWS = shape::elementWiseStride(inputShapes[r]);
         inputLength = shape::length(inputShapes[r]);
        }
        __syncthreads();

        for(int i = threadIdx.x; i < inputLength; i += blockDim.x) {
            result[baseIdx + i * resultEWS] = inputData[i * inputEWS];
        }
        __syncthreads();
     }
}

extern "C" __global__ void concatKernelHStackFloat(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float *result,
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelHStackGeneric<float>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelHStackDouble(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  double *result,
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelHStackGeneric<double>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}


extern "C" __global__ void concatKernelHStackHalf(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float16 *result,
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelHStackGeneric<float16>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

template <typename T>
__device__ void concatKernelVStackGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									T *result,
									int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    /*
     this is special case for concat: we group bunch of vectors into 2D matrix
     also: we expect each inputShapeInfo to have EWS, be a vector, and have equal size
     */

     int **inputShapes = (int**) inputShapeInfos;
     T **input = (T **) data;

     __shared__ int inputEWS;
     __shared__ int resultEWS;
     __shared__ int inputLength;

     if (threadIdx.x == 0) {
        inputLength = shape::length(inputShapes[0]);
        inputEWS = shape::elementWiseStride(inputShapes[0]);
        resultEWS = shape::elementWiseStride(resultShapeInfo);
     }
     __syncthreads();

     for (int r = blockIdx.x; r < numArrays; r+= gridDim.x) {

        int resultOffset = r * inputLength * resultEWS;
        T *inputData = (T *) input[r];

        for(int i = threadIdx.x; i < inputLength; i += blockDim.x) {
            result[resultOffset + i * resultEWS] = inputData[i * inputEWS];
        }
     }
}

extern "C" __global__ void concatKernelVStackFloat(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float *result,
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelVStackGeneric<float>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelVStackDouble(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  double *result,
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelVStackGeneric<double>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelVStackHalf(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  float16 *result,
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {

    concatKernelVStackGeneric<float16>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}


extern "C" __global__ void concatKernelDouble(int dimension,
											  int numArrays,
											  Nd4jPointer *data,
											  Nd4jPointer *inputShapeInfo,
											  double *result,
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, int *zTadShape, Nd4jIndex *zOffsets) {
	concatKernelGeneric<double>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers, zTadShape, zOffsets);
}

extern "C" __global__ void concatKernelFloat(int dimension,
											 int numArrays,
											 Nd4jPointer *data,
											 Nd4jPointer *inputShapeInfo,
											 float *result,
											 int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, int *zTadShape, Nd4jIndex *zOffsets) {
	concatKernelGeneric<float>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers, zTadShape, zOffsets);
}

extern "C" __global__ void concatKernelHalf(int dimension,
											 int numArrays,
											 Nd4jPointer *data,
											 Nd4jPointer *inputShapeInfo,
											 float16 *result,
											 int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers, int *zTadShape, Nd4jIndex *zOffsets) {
	concatKernelGeneric<float16>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers, zTadShape, zOffsets);
}


template <typename T>
__device__ void pullRowsKernelGeneric(T *x,
                                     int *xShapeInfo,
                                     T *z,
                                     int *zShapeInfo,
                                     int n,
                                     int *indexes,
                                     int *tadShapeInfo,
                                     Nd4jIndex *tadOffsets,
                                     int *zTadShapeInfo,
                                     Nd4jIndex *zTadOffsets) {


    int xEWS = shape::elementWiseStride(tadShapeInfo);
    int zEWS = shape::elementWiseStride(zTadShapeInfo);
    int tadLength = shape::length(tadShapeInfo);


    if (xEWS >= 1 && zEWS >= 1) {
        for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
            T *rX = x + tadOffsets[indexes[idx]];
            T *rZ = z + zTadOffsets[idx];

            for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
                rZ[i * zEWS] = rX[i * xEWS];
            }
        }
    } else {

        int xCoord[MAX_RANK];
		int zCoord[MAX_RANK];

        for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
            T *rX = x + tadOffsets[indexes[idx]];
            T *rZ = z + zTadOffsets[idx];

            for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
                shape::ind2subC(shape::rank(tadShapeInfo),shape::shapeOf(tadShapeInfo), i, xCoord);
		    	shape::ind2subC(shape::rank(zTadShapeInfo),shape::shapeOf(zTadShapeInfo), i, zCoord);

		    	Nd4jIndex xOffset = shape::getOffset(0, shape::shapeOf(tadShapeInfo), shape::stride(tadShapeInfo), xCoord, shape::rank(tadShapeInfo));
	    		Nd4jIndex zOffset = shape::getOffset(0, shape::shapeOf(zTadShapeInfo), shape::stride(zTadShapeInfo), zCoord, shape::rank(zTadShapeInfo));

                rZ[zOffset] = rX[xOffset];
            }
        }
    }
}

extern "C" __global__ void pullRowsKernelHalf(
                                     float16 *x,
                                     int *xShapeInfo,
                                     float16 *z,
                                     int *zShapeInfo,
                                     int n,
                                     int *indexes,
                                     int *tadShapeInfo,
                                     Nd4jIndex *tadOffsets,
                                     int *zTadShapeInfo,
                                     Nd4jIndex *zTadOffsets) {
    pullRowsKernelGeneric<float16>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

extern "C" __global__ void pullRowsKernelFloat(float *x,
                                     int *xShapeInfo,
                                     float *z,
                                     int *zShapeInfo,
                                     int n,
                                     int *indexes,
                                     int *tadShapeInfo,
                                     Nd4jIndex *tadOffsets,
                                     int *zTadShapeInfo,
                                     Nd4jIndex *zTadOffsets) {
    pullRowsKernelGeneric<float>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

extern "C" __global__ void pullRowsKernelDouble(double *x,
                                     int *xShapeInfo,
                                     double *z,
                                     int *zShapeInfo,
                                     int n,
                                     int *indexes,
                                     int *tadShapeInfo,
                                     Nd4jIndex *tadOffsets,
                                     int *zTadShapeInfo,
                                     Nd4jIndex *zTadOffsets) {
    pullRowsKernelGeneric<double>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

template <typename T>
__device__ void convertToHalfGeneric(T *dx, int n, half *dz) {
    int tid = threadIdx.x + blockIdx.x * gridDim.x;

    for (Nd4jIndex i = tid; i < n; i += blockDim.x * gridDim.x ) {
        dz[i] = __float2half((float) dx[i]);
    }
}

extern "C" __global__ void kernelFloatsToHalfs(float *dx, int n, half *dz) {
    convertToHalfGeneric<float>(dx, n, dz);
}

extern "C" __global__ void kernelDoublesToHalfs(double *dx, int n, half *dz) {
    convertToHalfGeneric<double>(dx, n, dz);
}

template <typename T>
__device__ void convertHalfsToGeneric(half *dx, int n, T *dz) {
    int tid = threadIdx.x + blockIdx.x * gridDim.x;

    for (Nd4jIndex i = tid; i < n; i += blockDim.x * gridDim.x ) {
        dz[i] = (T) __half2float(dx[i]);
    }
}

extern "C" __global__ void kernelHalfsToDoubles(half *dx, int n, double *dz) {
    convertHalfsToGeneric<double>(dx, n, dz);
}

extern "C" __global__ void kernelHalfsToFloats(half *dx, int n, float *dz) {
    convertHalfsToGeneric<float>(dx, n, dz);
}

/**
 * This kernel accumulates X arrays, and stores result into Z
 *
 * @tparam T
 * @param x
 * @param z
 * @param n
 * @param length
 */
template<typename T>
__device__ void accumulateKernelGeneric(T **x, T *z, int n, const Nd4jIndex length) {
    __shared__ T *shmem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char sharedmem[];
        shmem = (T *) sharedmem;
    }
    __syncthreads();

    for (int r = blockDim.x * blockIdx.x; r < length; r += blockDim.x * gridDim.x) {
        shmem[threadIdx.x] = 0.0f;

        Nd4jIndex baseIdx = r;

        // aggregation step, we roll over all arrays
        for (int ar = 0; ar < n; ar++) {
            T *cdata = (T *) x[ar];
            cdata += baseIdx;

            if (baseIdx + threadIdx.x < length)
                shmem[threadIdx.x] += cdata[threadIdx.x];
        }

        T *wdata = z + baseIdx;

        // saving accumulated values
        if (baseIdx + threadIdx.x < length) {
            wdata[threadIdx.x] = shmem[threadIdx.x];
       }
    }
}


extern "C" __global__ void accumulateKernelHalf(float16 **dx, float16 *dz, int n, Nd4jIndex length) {
    accumulateKernelGeneric<float16>(dx, dz, n, length);
}

extern "C" __global__ void accumulateKernelFloat(float **dx, float *dz, int n, Nd4jIndex length) {
    accumulateKernelGeneric<float>(dx, dz, n, length);
}

extern "C" __global__ void accumulateKernelDouble(double **dx, double *dz, int n, Nd4jIndex length) {
    accumulateKernelGeneric<double>(dx, dz, n, length);
}


template <typename T>
__device__ void averagingKernelGeneric(T **dx, T *dz, int n, Nd4jIndex length, bool propagate) {

    __shared__ T *shmem;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char sharedmem[];
        shmem = (T *) sharedmem;
    }
    __syncthreads();


    // each block cycles over it's own part of arrays
    for (int r = blockDim.x * blockIdx.x; r < length; r += blockDim.x * gridDim.x) {
        shmem[threadIdx.x] = (T) 0.0f;

        Nd4jIndex baseIdx = r;

        // aggregation step, we roll over all arrays
        for (int ar = 0; ar < n; ar++) {
            T *cdata = (T *) dx[ar];
            cdata += baseIdx;

            if (baseIdx + threadIdx.x < length)
                shmem[threadIdx.x] += cdata[threadIdx.x];
        }


        // average data in shared memory
        if (baseIdx + threadIdx.x < length)
            shmem[threadIdx.x] /= n;

        // div step & write out step
        if (dz != nullptr) {
            T *wdata = dz + baseIdx;

            if (baseIdx + threadIdx.x < length) {
                wdata[threadIdx.x] = shmem[threadIdx.x];
            }
        }

        // propagate averaged data to all arrays
        if (propagate)
            for (int ar = 0; ar < n; ar++) {
                T *cdata = (T *) dx[ar];
                cdata += baseIdx;

                if (baseIdx + threadIdx.x < length)
                    cdata[threadIdx.x] = shmem[threadIdx.x];
            }
    }
}


extern "C" __global__ void averagingKernelHalf(float16 **dx, float16 *dz, int n, Nd4jIndex length, bool propagate) {
    averagingKernelGeneric<float16>(dx, dz, n, length, propagate);
}

extern "C" __global__ void averagingKernelFloat(float **dx, float *dz, int n, Nd4jIndex length, bool propagate) {
    averagingKernelGeneric<float>(dx, dz, n, length, propagate);
}

extern "C" __global__ void averagingKernelDouble(double **dx, double *dz, int n, Nd4jIndex length, bool propagate) {
    averagingKernelGeneric<double>(dx, dz, n, length, propagate);
}

template<typename T>
__device__ void tearKernelGeneric(T *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {

    __shared__ Nd4jIndex tadLength;
    __shared__ int tadEWS;
    __shared__ int zEWS;
    __shared__ int tadRank;
    __shared__ Nd4jIndex numTads;
    __shared__ int zRank;
    __shared__ int *tadShape;
    __shared__ int *tadStride;
    __shared__ int *zShape;
    __shared__ int *zStride;

    if (threadIdx.x == 0) {
        tadLength = shape::length(tadShapeInfo);
        tadEWS = shape::elementWiseStride(tadShapeInfo);
        zEWS = shape::elementWiseStride(zShapeInfo);
        tadRank = shape::rank(tadShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;
        zRank = shape::rank(zShapeInfo);
        tadShape = shape::shapeOf(tadShapeInfo);
        tadStride = shape::stride(tadShapeInfo);
        zShape = shape::shapeOf(zShapeInfo);
        zStride = shape::stride(zShapeInfo);
    }
    __syncthreads();

    for (Nd4jIndex r = blockIdx.x; r < numTads; r += gridDim.x) {
        T *z = (T *) targets[r];
        T *s = x + tadOffsets[r];

        if (zEWS > 0 && tadEWS > 0) {
        for (Nd4jIndex i = threadIdx.x; i < tadLength; i += blockDim.x) {
            z[i * zEWS] = s[i * tadEWS];
        }
        } else {
            int xCoord[MAX_RANK];
            int zCoord[MAX_RANK];

            for (Nd4jIndex j = 0; j < tadLength; j++) {
                shape::ind2sub(tadRank,tadShape, j, xCoord);
                shape::ind2sub(zRank, zShape, j, zCoord);

                Nd4jIndex xOffset = shape::getOffset(0, tadShape, tadStride, xCoord, tadRank);
                Nd4jIndex zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                z[zOffset] = s[xOffset];
            }
        }
    }
}

extern "C" __global__ void tearKernelDouble(double *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
    tearKernelGeneric<double>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

extern "C" __global__ void tearKernelFloat(float *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
    tearKernelGeneric<float>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

extern "C" __global__ void tearKernelHalf(float16 *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
    tearKernelGeneric<float16>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}


template<typename T>
__device__ void shuffleKernelGeneric(T **dX, int **xShapeInfo, T **dZ, int **zShapeInfo, int N, int *shuffleMap, int **tadOnlyShapeInfo, Nd4jIndex **tadOffsets) {

            // we assume that shuffle map for each X contains pair TAD Y

            __shared__ int tadLength;
            __shared__ int tadEWS;
            __shared__ int tadRank;
            __shared__ int numTads;
            __shared__ int *tadShape;
            __shared__ int *tadStride;
            __shared__ int yStride;


        for (int f = 0; f < N; f++) {
            T *x = (T *) dX[f];
            T *z = (T *) dZ[f];



            __syncthreads();

            if (threadIdx.x == 0) {
                tadLength = shape::length(tadOnlyShapeInfo[f]);
                tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
                tadRank = shape::rank(tadOnlyShapeInfo[f]);
                numTads = shape::length(xShapeInfo[f]) / tadLength;

                tadShape = shape::shapeOf(tadOnlyShapeInfo[f]);
                tadStride = shape::stride(tadOnlyShapeInfo[f]);
            }
            __syncthreads();


            // we roll over the pairs of TADs, thus limit is numTads / 2
            for (Nd4jIndex r = blockIdx.x; r < numTads; r += blockDim.x) {
                if (shuffleMap[r] < 0)
                    continue;

                Nd4jIndex oldOffset = tadOffsets[f][r];
                Nd4jIndex newOffset = tadOffsets[f][shuffleMap[r]];



                T *rX = x + oldOffset;
                T *rY = x + newOffset;

                T *zX = z + oldOffset;
                T *zY = z + newOffset;

                // so we're going to change TAD[oldOffset] with TAD[newOffset]
                if (tadEWS == 1) {
                    for (Nd4jIndex i = threadIdx.x; i < tadLength; i += blockDim.x) {
                        T oldX = rX[i];

                        rX[i] = rY[i];
                        zY[i] = oldX;
                    }

                } else {
                    // well have to iterate using ind2sub
                        int xCoord[MAX_RANK];
                        int yCoord[MAX_RANK];
                        for (Nd4jIndex i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                            shape::ind2subC(tadRank,tadShape, i, xCoord);
                            shape::ind2subC(tadRank,tadShape, i, yCoord);

                            Nd4jIndex xOffset = shape::getOffset(oldOffset, tadShape, tadStride, xCoord, tadRank);
                            Nd4jIndex yOffset = shape::getOffset(newOffset, tadShape, tadStride, yCoord, tadRank);

                            T oldX = x[xOffset];
                            z[xOffset] = x[yOffset];
                            z[yOffset] = oldX;
                        }
                    }
            }
        }
}

extern "C" __global__ void shuffleKernelDouble(double **x, int **xShapeInfo, double **z, int **zShapeInfo, int N, int *shuffleMap, int **tadOnlyShapeInfo, Nd4jIndex **tadOffsets) {
    shuffleKernelGeneric<double>(x, xShapeInfo, z, zShapeInfo, N, shuffleMap, tadOnlyShapeInfo, tadOffsets);
}

extern "C" __global__ void shuffleKernelFloat(float **x, int **xShapeInfo, float **z, int **zShapeInfo, int N, int *shuffleMap, int **tadOnlyShapeInfo, Nd4jIndex **tadOffsets) {
    shuffleKernelGeneric<float>(x, xShapeInfo, z, zShapeInfo, N, shuffleMap, tadOnlyShapeInfo, tadOffsets);
}

extern "C" __global__ void shuffleKernelHalf(float16 **x, int **xShapeInfo, float16 **z, int **zShapeInfo, int N, int *shuffleMap, int **tadOnlyShapeInfo, Nd4jIndex **tadOffsets) {
    shuffleKernelGeneric<float16>(x, xShapeInfo, z, zShapeInfo, N, shuffleMap, tadOnlyShapeInfo, tadOffsets);
}

// transform strided
DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, float, INPUT(Nd4jIndex n, float *x, int xStride, float *extraParams, float *z, int zStride, int *allocationPointer, float *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, double, INPUT(Nd4jIndex n, double *x, int xStride, double *extraParams, double *z, int zStride, int *allocationPointer, double *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, float16, INPUT(Nd4jIndex n, float16 *x, int xStride, float16 *extraParams, float16 *z, int zStride, int *allocationPointer, float16 *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

// transform shaped
DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, float, INPUT(float *x, int *xShape, int xRank, float *extraParams, float *z, int *zShape, int zRank, int *allocationPointer, float *reductionPointer,  int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, double, INPUT(double *x, int *xShape, int xRank, double *extraParams, double *z, int *zShape, int zRank, int *allocationPointer, double *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, float16, INPUT(float16 *x, int *xShape, int xRank, float16 *extraParams, float16 *z, int *zShape, int zRank, int *allocationPointer, float16 *reductionPointer,  int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(TRANSFORM_OPS))

#endif

#endif /* TRANSFORM_H_ */
