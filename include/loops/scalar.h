/*
 * scalar.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SCALAR_H_
#define SCALAR_H_
#include <dll.h>

#ifdef __JNI__
#include <jni.h>
#endif
#include <templatemath.h>
#include <ops/ops.h>
#include <op_boilerplate.h>
#include "helpers/logger.h"
#include <helper_cuda.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <types/float16.h>
#endif

#include "legacy_ops.h"

namespace functions {
    namespace scalar {
/**
 * Apply a scalar
 *  operation to an array
 */
        template<typename T>
        class ScalarTransform {

        public:

#ifdef __CUDACC__

            __host__
            static inline  void executeCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *x, int xStride, T *result, int resultStride, T scalar, T *extraParams, Nd4jIndex n);

            __host__
            static inline void executeCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *x, int *xShapeInfo, T *result, int *resultShapeInfo, T scalar, T *extraParams);

            __host__
            static inline void executeCudaAlongDimension(dim3& launchDims, Nd4jPointer *extraPointers,int opNum, T *x, int *xShapeInfo, T *z, int *zShapeInfo, T *scalars, T *extraParams, int *dimension, int dimensionLength);

/*
            template<typename OpType>
            __device__
            static inline void transformCuda(T *x, int *xShapeInfo, T *extraParams, T *z, int *zShapeInfo, T *scalars, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *tadShapeInfoZ, Nd4jIndex *tadOffsetsZ);

            template<typename OpType>
            __device__
            static inline void transformCuda(T scalar, T *dy, int *shapeInfo, T *params, T *result, int *resultShapeInfo, int *allocationBuffer, UnifiedSharedMemory *manager);


            template<typename OpType>
            __device__
            static inline void transform(Nd4jIndex n, T scalar, T *dy, T *params, T *result, int *indexes, int *allocationBuffer, UnifiedSharedMemory *manager);


            template<typename OpType>
            __device__
	        static inline void transformCuda(Nd4jIndex n, T dx, T *dy, int incy, T *params, T *result, int resultStride, int *allocationBuffer, UnifiedSharedMemory *manager);
*/

#include "cuda/scalar_temp.cu"
#endif
            template <typename OpType>
            static void transform(T *x, int *xShapeInfo, T *extraParams, T *z, int *zShapeInfo, T *scalars, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *tadShapeInfoZ, Nd4jIndex *tadOffsetsZ);

            static void transform(int opNum, T *x, int *xShapeInfo, T *extraParams, T *z, int *zShapeInfo, T *scalars, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *tadShapeInfoZ, Nd4jIndex *tadOffsetsZ);

            static void transform(const int opNum, T *x, int *xShapeInfo, T *result, int *resultShapeInfo, T scalar, T *extraParams, int *indexes, int *resultIndexes);

            static void transform(const int opNum, T *x, int xStride, T *result, int resultStride, T scalar, T *extraParams, const Nd4jIndex n);

            static void transform(const int opNum, T *x, int *xShapeInfo, T *result, int *resultShapeInfo, T scalar, T *extraParams);




            /**
         * CPU implementation of scalar operation
         * @param x the input
         * @param xStride the stride for the input
         * @param result the result buffer
         * @param resultStride the stride for the result
         * @param scalar the scalar to apply
         * @param extraParams the extra parameters where
         * neccssary
         * @param n the number of elements to loop over
         */

            template <typename OpType>
            static void transform(T *x, int *xShapeInfo, T *result, int *resultShapeInfo, T scalar, T *extraParams, int *indexes, int *resultIndexes);



            /*
             * ScalarOp along dimension
             */


            /**
         * CPU implementation of scalar operation
         * @param x the input
         * @param xStride the stride for the input
         * @param result the result buffer
         * @param resultStride the stride for the result
         * @param scalar the scalar to apply
         * @param extraParams the extra parameters where
         * neccssary
         * @param n the number of elements to loop over
         */

            template<typename OpType>
            static  void transform(T *x, int *xShapeInfo, T *result, int *resultShapeInfo, T scalar, T *extraParams);


            /**
             * CPU implementation of scalar operation
             * @param x the input
             * @param xStride the stride for the input
             * @param result the result buffer
             * @param resultStride the stride for the result
             * @param scalar the scalar to apply
             * @param extraParams the extra parameters where
             * neccssary
             * @param n the number of elements to loop over
             */

            template<typename OpType>
            static void transform(T *x, int xStride, T *result, int resultStride, T scalar, T *extraParams, const Nd4jIndex n);
        };
    }
}


#ifdef __CUDACC__
#include "cuda/scalar.kernels"
#endif

#endif /* SCALAR_H_ */
