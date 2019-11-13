/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/*
 * scalar.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SCALAR_H_
#define SCALAR_H_
#include <OmpLaunchHelper.h>
#include <dll.h>
#include <helpers/DebugHelper.h>

#ifdef __JNI__
#include <jni.h>
#endif
#include <templatemath.h>
#include <ops/ops.h>
#include <op_boilerplate.h>
#include "helpers/logger.h"

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
        template<typename X, typename Y, typename Z>
        class ScalarTransform {

        public:

#ifdef __CUDACC__

            template <typename OpType>
            __host__
            static void intermediateShaped(dim3& launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, void* vscalar, void *vextraParams, int *allocPointer);

            template <typename OpType>
            __host__
            static void intermediateAlongDimension(dim3& launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, void *z, Nd4jLong *zShapeInfo, void *scalars, void *extraParams, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ);

            __host__
            static void executeCudaShaped(dim3& launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *result, Nd4jLong *resultShapeInfo, Nd4jLong *hzShapeInfo, void* scalar, void *extraParams);

            __host__
            static void executeCudaAlongDimension(dim3& launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, void *z, Nd4jLong *zShapeInfo, void *scalars, void *extraParams, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ);

#else
            template <typename OpType>
            static void transform(void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, void *scalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ, const uint64_t start, const uint64_t stop);

            static void transform(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, void *scalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ, const uint64_t start, const uint64_t stop);

            static void transform(const int opNum, void *x, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo,  void *scalar,  void *extraParams, const uint64_t start, const uint64_t stop);

            static void transform(const int opNum, void *x, Nd4jLong xStride, void *result, Nd4jLong resultStride, void *scalar, void *extraParams, const uint64_t len, const uint64_t start, const uint64_t stop);




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
         * @param len the number of elements to loop over
         */

            template<typename OpType>
            static  void transform(void *x, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *scalar, void *extraParams, const uint64_t start, const uint64_t stop);


            /**
             * CPU implementation of scalar operation
             * @param x the input
             * @param xStride the stride for the input
             * @param result the result buffer
             * @param resultStride the stride for the result
             * @param scalar the scalar to apply
             * @param extraParams the extra parameters where
             * neccssary
             * @param len the number of elements to loop over
             */

            template<typename OpType>
            static void transform(void *x, Nd4jLong xStride, void *result, Nd4jLong resultStride, void *scalar, void *extraParams, const uint64_t len, const uint64_t start, const uint64_t stop);
#endif
        };
    }
}


#endif /* SCALAR_H_ */
