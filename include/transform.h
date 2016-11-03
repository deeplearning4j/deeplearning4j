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
#include <ops.h>
#ifndef __CUDACC__
#include <omp.h>
#endif
#include <pairwise_util.h>
#include <dll.h>
#include "reduce.h"
#include "scalar.h"
#include "indexreduce.h"
#include "broadcasting.h"
#include <shape.h>
#include <ops.h>
#include <special_ops.h>
#include <op_boilerplate.h>
#include "types/float8.h"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif


#define TRANSFORM_OPS \
	(0, simdOps::Abs), \
        (1, simdOps::Ceiling), \
        (2, simdOps::Cosine), \
        (3, simdOps::Exp), \
        (4, simdOps::Floor), \
        (5, simdOps::Log), \
        (6, simdOps::Neg), \
        (7, simdOps::Pow), \
        (8, simdOps::Round), \
        (9, simdOps::SetRange), \
        (10,simdOps::Sigmoid), \
        (11,simdOps::Sign), \
        (12,simdOps::Sin), \
        (13,simdOps::SoftPlus), \
        (14,simdOps::Sqrt), \
        (15,simdOps::Tanh), \
        (16,simdOps::ACos), \
        (17,simdOps::ASin), \
        (18,simdOps::ATan), \
        (19,simdOps::HardTanh), \
        (20,simdOps::SoftSign), \
        (21,simdOps::ELU), \
        (22,simdOps::ELUDerivative), \
        (23,simdOps::TanhDerivative), \
        (24,simdOps::TimesOneMinus), \
        (25,simdOps::HardTanhDerivative), \
        (26,simdOps::Ones), \
        (27,simdOps::Identity), \
        (28,simdOps::Stabilize), \
        (29,simdOps::SigmoidDerivative), \
        (30,simdOps::SoftSignDerivative), \
        (31,simdOps::LeakyRELU), \
        (32,simdOps::LeakyRELUDerivative), \
        (33,simdOps::RELU), \
        (34,simdOps::Step), \
        (35,simdOps::OneMinus), \
        (36,simdOps::Col2Im), \
        (37,simdOps::Im2col), \
        (38,simdOps::SoftMax), \
        (39,simdOps::SoftMaxDerivative), \
        (40,simdOps::LogSoftMax), \
        (41,simdOps::IsMax), \
        (42,simdOps::SpecialDerivative), \
        (43,simdOps::DropOut), \
        (44,simdOps::DropOutInverted), \
        (45,simdOps::CompareAndSet), \
        (46,simdOps::ReplaceNans) ,\
        (47,simdOps::StabilizeFP16) ,\
        (48,simdOps::Histogram), \
        (49,simdOps::Cube), \
        (50,simdOps::CubeDerivative)


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
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager) {

		if(OpType::requiresSpecial) {
			OpType::execSpecialCuda(dy,shapeInfo,result,resultShapeInfo,params, allocationPointer, reductionPointer, manager);
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
			UnifiedSharedMemory *manager) {
                                DISPATCH_BY_OPNUM(transformCuda, PARAMS(dy, shapeInfo, params, result, resultShapeInfo, allocationPointer, reductionPointer, manager), TRANSFORM_OPS);
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
				int *resultIndexes, int *tadShapeInfo, int *tadOffsets) {
                            DISPATCH_BY_OPNUM(exec, PARAMS(dx, xShapeInfo, result, resultShapeInfo, extraParams, indexes, resultIndexes, tadShapeInfo, tadOffsets), TRANSFORM_OPS);
			}


			static void exec(
				int opNum,
				T *dx,
				int *xShapeInfo,
				T *result,
				int *resultShapeInfo,
				T *extraParams, int *tadShapeInfo, int *tadOffsets) {
                                DISPATCH_BY_OPNUM(exec, PARAMS(dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), TRANSFORM_OPS);
			}


			template<typename OpType>
			static void exec(
                    T *dx,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T *extraParams, int *tadShapeInfo, int *tadOffsets) {

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
                            /* Process the innermost dimension */
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
				int *resultIndexes, int *tadShapeInfo, int *tadOffsets) {

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
                            result[i*resultStride] = OpType::op(dx[i * resultStride], extraParams);
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
		T *result,int *resultShapeInfo, int zRank, int *allocationPointer, T *reductionPointer) {

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
__device__ void fillDimensionalIsMaxGeneric(T *dX, int *xShapeInfo, T *dZ, int *zShapeInfo, int *tadOnlyShapeInfo, int *dimension, int dimensionLength, int *tadOffsets) {

    __shared__ int tadLength;
    __shared__ int tadEWS;
    __shared__ int numTads;

    if (threadIdx.x == 0) {

        tadLength = shape::tadLength(zShapeInfo, dimension, dimensionLength);
        tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
        numTads = shape::length(zShapeInfo) / tadLength;
/*
        if (blockIdx.x == 0) {
            printf("original X shape: \n");
            shape::printShapeInfoLinear(xShapeInfo);

            printf("original Z shape: \n");
            shape::printShapeInfoLinear(zShapeInfo);

            printf("Target dimension: [%i], dimensionLength: [%i], numTads: [%i], rnumTads: [%i]\n", dimension[0], dimensionLength, numTads, tad->numTads);

            printf("TAD shape: \n");
            shape::printShapeInfoLinear(tadOnlyShapeInfo);

            printf("TAD shape2: \n");
            shape::printShapeInfoLinear(tad->tadOnlyShapeInfo);
        }
        */
    }
    __syncthreads();


    for (int r = blockIdx.x; r < numTads; r+= gridDim.x) {
        int tadOffsetForBlock = tadOffsets[r];

        int highestElement = (int) dX[r];

//        if (threadIdx.x == 0)
//            printf("TAD: [%i], highestElement: [%i], numTads: [%i], tadLength: [%i], tadOffset: [%i], tadEWS: [%i]\n", r, highestElement, numTads, tadLength, tadOffsetForBlock, tadEWS);


        for (int e = threadIdx.x; e < tadLength; e += blockDim.x) {
            // so, we just set dZ[e] for each TAD. Sure, e should be replaced with
            int idx = tadOffsetForBlock + (e * tadEWS);
            dZ[idx] = (e == highestElement? (T) 1.0 : (T) 0.0);
        }

    }
}

extern "C" __global__ void fillDimensionalIsMaxFloat(float *dx, int *xShapeInfo, float *dz, int *zShapeInfo, int *tadOnlyShapeInfo, int *dimension, int dimensionLength, int *tadOffsets) {
    fillDimensionalIsMaxGeneric<float>(dx, xShapeInfo, dz, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
}

extern "C" __global__ void fillDimensionalIsMaxDouble(double *dx, int *xShapeInfo, double *dz, int *zShapeInfo, int *tadOnlyShapeInfo, int *dimension, int dimensionLength, int *tadOffsets) {
    fillDimensionalIsMaxGeneric<double>(dx, xShapeInfo, dz, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
}

extern "C" __global__ void fillDimensionalIsMaxHalf(float16 *dx, int *xShapeInfo, float16 *dz, int *zShapeInfo, int *tadOnlyShapeInfo, int *dimension, int dimensionLength, int *tadOffsets) {
    fillDimensionalIsMaxGeneric<float16>(dx, xShapeInfo, dz, zShapeInfo, tadOnlyShapeInfo, dimension, dimensionLength, tadOffsets);
}

template <typename T>
__device__ void concatKernelGeneric(int dimension,
									int numArrays,
									Nd4jPointer *data,
									Nd4jPointer *inputShapeInfos,
									T *result,
									int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ UnifiedSharedMemory *manager;

	int zRank = shape::rank(resultShapeInfo);

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
		manager->init(sizeof(UnifiedSharedMemory), 0, 0, sizeof(shape::TAD), zRank + 2);
	}
	__syncthreads();

	T **dataT = (T **) data;
	int **shapeInfoPointers = (int **) inputShapeInfos;
	int **tadShapes = (int **) tadPointers;
	int **tadOffsets = (int **) offsetPointers;


    __shared__ int tDim[1];
    __shared__ int baseIdx;

		__shared__ shape::TAD *tad;
		__shared__ int yLength;
		__shared__ char yOrder;
		__shared__ int yEWS;
		if (threadIdx.x == 0) {
			tDim[0] = dimension;
			tad = new shape::TAD(); //(xShapeInfo,dimension,dimensionLength) // (manager->getTADSpace())
			tad->setExternalBuffers((void *) manager);
			//    tad->initWithExternalTAD(manager->getT1ShapeBuffer(), manager->getXShapeBuffer(), dimension, dimensionLength);
			tad->init(resultShapeInfo, tDim, 1);
			tad->createTadOnlyShapeInfo();
		}
		__syncthreads();

		char zOrder = shape::order(resultShapeInfo);

		int zEWS = shape::elementWiseStride(resultShapeInfo);
		int tadEWS = shape::elementWiseStride(tad->tadOnlyShapeInfo);
		int zLength = shape::length(resultShapeInfo);

        __shared__ int arrOffset;
		__shared__ int numTads;


        if (shape::isVector(resultShapeInfo)) {
			//if (threadIdx.x == 0)
				//printf("Vector here\n");
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


		// TODO: to be pulled into separate kernel. matrix concatenation
		for (int r = blockIdx.x; r < numArrays; r += gridDim.x) {

			int *currentShape = shapeInfoPointers[r];
			T *currentData = dataT[r];
			int *currentTad = tadShapes[r];
			int *currentOffsets = tadOffsets[r];


			if (threadIdx.x == 0) {
				//inputTAD = new((unsigned char *)managerInput->getTADSpace()) shape::TAD(); //(xShapeInfo,dimension,dimensionLength)
				//inputTAD->setExternalBuffers((void *) managerInput);
				//inputTAD->initWithExternalTAD(currentTad, currentShape, tDim, 1);
				//inputTAD->init(shapeInfoPointers[r], &dimension, 1);
				//inputTAD->createTadOnlyShapeInfo();

				yLength = shape::length(currentTad);
				yOrder = shape::order(currentTad);
				yEWS = shape::elementWiseStride(currentTad);
                numTads = shape::length(currentShape) / yLength;
			}
			__syncthreads();


			if (threadIdx.x == 0) {
				arrOffset = 0;
				for (int f = 0; f < r; f++) {
					arrOffset +=  shape::length(tadShapes[f]);
				}
			}
			__syncthreads();

			for (int j = 0; j < numTads; j++) {
				int inputOffset = currentOffsets[j];
				int resultOffset = tad->tadOffset(j);

				T *dataTAD = currentData + inputOffset;
				T *resultTAD = result + resultOffset;

                int sub[MAX_RANK];

				shape::ind2subC(shape::rank(tad->tadOnlyShapeInfo),shape::shapeOf(tad->tadOnlyShapeInfo),arrOffset, sub);
				Nd4jIndex baseOffset = shape::getOffset(0,shape::shapeOf(tad->tadOnlyShapeInfo),shape::stride(tad->tadOnlyShapeInfo), sub, shape::rank(tad->tadOnlyShapeInfo));

				resultTAD += baseOffset;

				if (zOrder == yOrder && yEWS > 0  && tadEWS > 0) {
					for (int i = threadIdx.x; i < yLength; i += blockDim.x) {
						resultTAD[i * tadEWS] = dataTAD[i * yEWS];
					}
				} else {
					if(tadEWS > 0 && shape::order(resultShapeInfo) == shape::order(currentTad)) {

						// FIXME: this is bad

						if (threadIdx.x == 0) {
							baseIdx = 0;
							for (int f = 0; f < r; f++) {
								baseIdx += shape::length(shapeInfoPointers[f]);
							}
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
								shape::ind2sub(yRank, shape::shapeOf(currentTad), i, yIdx);
								int yOffset = shape::getOffset(0, shape::shapeOf(currentTad), shape::stride(currentTad), yIdx, yRank);

								resultTAD[baseIdx + i * tadEWS] =  dataTAD[yOffset];
							}
						}
						__syncthreads();
					} else {
						int yIdx[MAX_RANK];
						int yRank = shape::rank(currentTad);
						int tadRank = shape::rank(tad->tadOnlyShapeInfo);

						for (int i = threadIdx.x; i < yLength; i+= blockDim.x) {
							shape::ind2sub(yRank, shape::shapeOf(currentTad), i,yIdx);

							int yOffset = shape::getOffset(0, shape::shapeOf(currentTad), shape::stride(currentTad), yIdx, yRank);
							int resultOffset = shape::getOffset(0, shape::shapeOf(tad->tadOnlyShapeInfo), shape::stride(tad->tadOnlyShapeInfo), yIdx, tadRank);

							resultTAD[resultOffset] =  dataTAD[yOffset];
						}
					}
				}
				__syncthreads();
			}
			__syncthreads();

//			if (threadIdx.x == 0)
//				delete inputTAD;
		}

		if (threadIdx.x == 0 && tad != NULL)
			delete tad;
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
        resultEWS = shape::elementWiseStride(resultShapeInfo);
     }
     __syncthreads();

     for (int r = blockIdx.x; r < numArrays; r+= gridDim.x) {

        int resultOffset = r * inputLength * resultEWS;
        T *inputData = (T *) input[r];

        if (threadIdx.x == 0) {
         inputEWS = shape::elementWiseStride(inputShapes[r]);
        }
        __syncthreads();

        for(int i = threadIdx.x; i < inputLength; i += blockDim.x) {
            result[resultOffset + i * resultEWS] = inputData[i * inputEWS];
        }
        __syncthreads();
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
											  int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
	concatKernelGeneric<double>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelFloat(int dimension,
											 int numArrays,
											 Nd4jPointer *data,
											 Nd4jPointer *inputShapeInfo,
											 float *result,
											 int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
	concatKernelGeneric<float>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}

extern "C" __global__ void concatKernelHalf(int dimension,
											 int numArrays,
											 Nd4jPointer *data,
											 Nd4jPointer *inputShapeInfo,
											 float16 *result,
											 int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
	concatKernelGeneric<float16>(dimension, numArrays, data, inputShapeInfo, result, resultShapeInfo, tadPointers, offsetPointers);
}


template <typename T>
__device__ void pullRowsKernelGeneric(T *x,
                                     int *xShapeInfo,
                                     T *z,
                                     int *zShapeInfo,
                                     int n,
                                     int *indexes,
                                     int *tadShapeInfo,
                                     int *tadOffsets,
                                     int *zTadShapeInfo,
                                     int *zTadOffsets) {


    int xEWS = shape::elementWiseStride(tadShapeInfo);
    int zEWS = shape::elementWiseStride(zTadShapeInfo);
    int tadLength = shape::length(tadShapeInfo);

    for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
        int tadOffsetForBlock = tadOffsets[indexes[idx]];

        T *rX = x + tadOffsetForBlock;
        T *rZ = z + zTadOffsets[idx];

        for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
            rZ[i * zEWS] = rX[i * xEWS];
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
                                     int *tadOffsets,
                                     int *zTadShapeInfo,
                                     int *zTadOffsets) {
    pullRowsKernelGeneric<float16>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

extern "C" __global__ void pullRowsKernelFloat(float *x,
                                     int *xShapeInfo,
                                     float *z,
                                     int *zShapeInfo,
                                     int n,
                                     int *indexes,
                                     int *tadShapeInfo,
                                     int *tadOffsets,
                                     int *zTadShapeInfo,
                                     int *zTadOffsets) {
    pullRowsKernelGeneric<float>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

extern "C" __global__ void pullRowsKernelDouble(double *x,
                                     int *xShapeInfo,
                                     double *z,
                                     int *zShapeInfo,
                                     int n,
                                     int *indexes,
                                     int *tadShapeInfo,
                                     int *tadOffsets,
                                     int *zTadShapeInfo,
                                     int *zTadOffsets) {
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
        shmem[threadIdx.x] = 0.0f;

        Nd4jIndex baseIdx = r;

        // aggregation step, we roll over all arrays
        for (int ar = 0; ar < n; ar++) {
            T *cdata = (T *) dx[ar];
            cdata += baseIdx;

            if (baseIdx + threadIdx.x < length)
                shmem[threadIdx.x] += cdata[threadIdx.x];
        }

        // div step & write out step
        T *wdata = dz + baseIdx;

        if (baseIdx + threadIdx.x < length) {
            shmem[threadIdx.x] /= n;
            wdata[threadIdx.x] = shmem[threadIdx.x];
        }

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
__device__ void shuffleKernelGeneric(T **dX, int **xShapeInfo, T **dZ, int **zShapeInfo, int N, int *shuffleMap, int **tadOnlyShapeInfo, int **tadOffsets) {

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

                int oldOffset = tadOffsets[f][r];
                int newOffset = tadOffsets[f][shuffleMap[r]];



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

extern "C" __global__ void shuffleKernelDouble(double **x, int **xShapeInfo, double **z, int **zShapeInfo, int N, int *shuffleMap, int **tadOnlyShapeInfo, int **tadOffsets) {
    shuffleKernelGeneric<double>(x, xShapeInfo, z, zShapeInfo, N, shuffleMap, tadOnlyShapeInfo, tadOffsets);
}

extern "C" __global__ void shuffleKernelFloat(float **x, int **xShapeInfo, float **z, int **zShapeInfo, int N, int *shuffleMap, int **tadOnlyShapeInfo, int **tadOffsets) {
    shuffleKernelGeneric<float>(x, xShapeInfo, z, zShapeInfo, N, shuffleMap, tadOnlyShapeInfo, tadOffsets);
}

extern "C" __global__ void shuffleKernelHalf(float16 **x, int **xShapeInfo, float16 **z, int **zShapeInfo, int N, int *shuffleMap, int **tadOnlyShapeInfo, int **tadOffsets) {
    shuffleKernelGeneric<float16>(x, xShapeInfo, z, zShapeInfo, N, shuffleMap, tadOnlyShapeInfo, tadOffsets);
}

// transform strided
DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, float, INPUT(Nd4jIndex n, float *x, int xStride, float *extraParams, float *z, int zStride, int *allocationPointer, float *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, double, INPUT(Nd4jIndex n, double *x, int xStride, double *extraParams, double *z, int zStride, int *allocationPointer, double *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformStrided_, transformSimpleGeneric, float16, INPUT(Nd4jIndex n, float16 *x, int xStride, float16 *extraParams, float16 *z, int zStride, int *allocationPointer, float16 *reductionPointer), PARAMS(n, x, xStride, extraParams, z, zStride, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

// transform shaped
DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, float, INPUT(float *x, int *xShape, int xRank, float *extraParams, float *z, int *zShape, int zRank, int *allocationPointer, float *reductionPointer), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, double, INPUT(double *x, int *xShape, int xRank, double *extraParams, double *z, int *zShape, int zRank, int *allocationPointer, double *reductionPointer), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))
DISPATCH_KERNEL_SIMPLE(transformShaped_, transformSimpleGeneric, float16, INPUT(float16 *x, int *xShape, int xRank, float16 *extraParams, float16 *z, int *zShape, int zRank, int *allocationPointer, float16 *reductionPointer), PARAMS(x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer), OPS_A(TRANSFORM_OPS))

#endif

#endif /* TRANSFORM_H_ */
