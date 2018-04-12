
#ifndef REDUCE_H
#define REDUCE_H
#include <dll.h>
//#include <string>
#include <helpers/sharedmem.h>
#include <stdio.h>
#include <helpers/shape.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <templatemath.h>
#include <helper_cuda.h>
#include <nd4jmalloc.h>
#include <pairwise_util.h>
#include <ops/ops.h>
#include <ops/special_accumulation_ops.h>
#include <op_boilerplate.h>

#pragma once
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include "legacy_ops.h"

//an op for the kernel
namespace functions {
    namespace reduce {

/**
 * A reduce function
 * reduces a vector down to
 * a subset of itself
 * via aggregating member
 * elements.
 */
        template<typename T>
        class ReduceFunction {
        public:
#ifdef __CUDACC__
            template<typename OpType>
			static __device__ void transformCuda1D(T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				T *reductionBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets);

            template<typename OpType>
			static __device__ void execScalarCuda(
				T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				T *reductionBuffer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo);

            template<typename OpType>
			static __device__ void transformCuda3D(
				T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				T *reductionBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets);

            template<typename OpType>
			static __device__ void transformCudaXD(
				T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				T *reductionBuffer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo,
				Nd4jIndex *tadOffsets);

			/**
			 *
			 * @param sPartialsRef
			 * @param tid
			 * @param extraParams
			 */
            template<typename OpType>
			static __device__ void aggregatePartials(T *sPartials, int tid, int numItems, T *extraParams);

            static __host__ void execReduceScalar(dim3 launchDims, cudaStream_t *stream, int opNum, T *x, int *xShapeInfo, T *extraParams, T *z, int *zShapeInfo, int *dimension, int dimensionLength, T *reductionBuffer, int *tadOnlyShapeInfo);

            static __host__ void execReduceXD(dim3 launchDims, cudaStream_t *stream, int opNum, int rank, T *x, int *xShape, T *extraParams, T *z, int *zShape, int *dimension, int dimensionLength, T *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets);
#endif

            /**
             * Reduce down to 1 number
             * @param x the input
             * @param xShapeInfo the shape information
             * for the input
             * @param extraParams the extra params
             * @return
             */
            template<typename OpType>
            static _CUDA_H T execScalar(T *x, int *xShapeInfo, T *extraParams);


            static T execScalar(const int opNum, T *x, int *xShapeInfo, T *extraParams);

            static void exec(const int opNum,
                             T *x,
                             int *xShapeInfo,
                             T *extraParams,
                             T *result,
                             int *resultShapeInfoBuffer,
                             int *dimension,
                             int dimensionLength,
                             int *tadShapeInfo,
                             Nd4jIndex *tadOffset);

            /**
             * Execute on the cpu
             * @param x the input data
             * @param xShapeInfo the shape information for x
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfoBuffer the shape information
             * @param dimension the dimension to perform
             * the reduce along long
             * @param dimensionLength the length of the dimension buffer
             */


            template<typename OpType>
            static void _CUDA_H exec(T *x,
                             int *xShapeInfo,
                             T *extraParams,
                             T *result,
                             int *resultShapeInfoBuffer,
                             int *dimension,
                             int dimensionLength,
                             int *tadShapeInfo,
                             Nd4jIndex *tadOffset);

            /**
            * CPU implementation
            * @param x the input data
            * @param xShapeInfo the shape information for
            * the input data
            * @param extraParams the extra parameters for the problem
            * @param result the result buffer
            * @param resultShapeInfo the shape information
            */
            template<typename OpType>
            static void _CUDA_H exec(T *x,
                             int *xShapeInfo,
                             T *extraParams,
                             T *result,
                             int *resultShapeInfo);



            /**
            * Reduce down to 1 number
            * @param x the input
            * @param xShapeInfo the shape information
            * for the input
            * @param extraParams the extra params
            * @return
            */
            template<typename OpType>
            static T _CUDA_H execScalar(const T *x, int xElementWiseStride, Nd4jIndex length, T *extraParams);
        };

#ifdef __CUDACC__
        /**
    *
    * @param extraParams
    * @param sPartials
    * @param sMemSize
    */
        template<typename T>
        __device__ void initializeShared(T *extraParams, T **sPartials, int sMemSize);

#endif

    }

}

#endif

