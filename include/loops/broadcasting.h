/*
 * broadcasting.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef BROADCASTING_H_
#define BROADCASTING_H_
#include <dll.h>
#include <helpers/sharedmem.h>
#include <helpers/shape.h>
#include <templatemath.h>
#include <helper_cuda.h>
#include <pairwise_util.h>
#include <ops/ops.h>
#include <op_boilerplate.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef __JNI__
#include <jni.h>
#endif

#include <helpers/TAD.h>

#include "legacy_ops.h"

namespace functions {
    namespace broadcast {

/**
 * Broadcast operation
 * for broadcasting a smaller tensor
 * along long a bigger one.
 */
        template<typename T>
        class Broadcast {
        public:

#ifdef __CUDACC__

            template<typename OpType>
			static __device__ void transformCuda(
			T *x,
			int *xShapeInfo,
			T *y,
			int *yShapeInfo,
			T *result,
			int *resultShapeInfo,
			int *dimension,
			int dimensionLength, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ);


            static __host__ void executeBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, T *x, int *xShapeInfo, T *y, int *yShapeInfo, T *result, int *resultShapeInfo, int *dimension, int dimensionLength, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets, int *tadOnlyShapeInfoZ, Nd4jIndex *tadOffsetsZ);

#endif

            static void exec(const int opNum,
                             T *x,
                             int *xShapeInfo,
                             T *y,
                             int *yShapeInfo,
                             T *result,
                             int *resultShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             int *tadShapeInfo,
                             Nd4jIndex *tadOffset,
                             int *tadShapeInfoZ,
                             Nd4jIndex *tadOffsetZ);

            /**
             * CPU execution
             * @param x the input
             * @param xShapeInfo the x shape information
             * @param y the y data
             * @param yShapeInfo the y shape information
             * @param result the result
             * @param resultShapeInfo the result shape information
             * @param dimension the dimension to broadcast along long
             * @param dimensionLength the length of the dimension buffer
             */
            template<typename OpType>
            static void exec(T *x,
                             int *xShapeInfo,
                             T *y,
                             int *yShapeInfo,
                             T *result,
                             int *resultShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             int *tadShapeInfo,
                             Nd4jIndex *tadOffset,
                             int *tadShapeInfoZ,
                             Nd4jIndex *tadOffsetZ);
        };
    }
}

#endif /* BROADCASTING_H_ */
