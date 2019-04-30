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

#ifndef SCALAR_BOOL_H_
#define SCALAR_BOOL_H_
#include <dll.h>

#ifdef __JNI__
#include <jni.h>
#endif
#include <templatemath.h>
#include <ops/ops.h>
#include <op_boilerplate.h>
#include "helpers/logger.h"
#include <helper_cuda.h>
#include <OmpLaunchHelper.h>
#include <helpers/DebugHelper.h>

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
        template<typename X, typename Z>
        class ScalarBoolTransform {

        public:

#ifdef __CUDACC__
                        
            template<typename OpType>
            __device__
            static void transformCuda(void* scalar, void *vy, Nd4jLong *shapeInfo, void *vparams, void *vresult, Nd4jLong *resultShapeInfo, int *allocationBuffer);
            
            template<typename OpType>
            __device__
            static void transformCuda(Nd4jLong n, void* vx, void *vy, Nd4jLong yEWS, void *vparams, void *vz, Nd4jLong zEWS, int *allocationBuffer);

            template<typename OpType>
            __device__
            static void transformCuda(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vz, Nd4jLong *zShapeInfo, void *vscalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ);

            template <typename OpType>
            __host__
            static void intermediateAlongDimension(dim3& launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, void *z, Nd4jLong *zShapeInfo, void *scalars, void *extraParams, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ);

            template <typename OpType>
            __host__
            static void intermediateShaped(dim3& launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, void* vscalar, void *vextraParams, int *allocPointer);            
            
            __host__
            static void executeCudaShaped(dim3& launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void* scalar, void *extraParams);

            __host__
            static void executeCudaAlongDimension(dim3& launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, void *z, Nd4jLong *zShapeInfo, void *scalars, void *extraParams, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ);


/*
#include "cuda/scalar_temp.cu"
*/
#endif
            template <typename OpType>
            static void transform(void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, void *scalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ);
 
           static void transform(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, void *scalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ);

            static void transform(const int opNum, void *x, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo,  void *scalar,  void *extraParams);

            static void transform(const int opNum, void *x, Nd4jLong xStride, void *result, Nd4jLong resultStride, void *scalar, void *extraParams, const Nd4jLong n);




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
            static  void transform(void *x, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *scalar, void *extraParams);


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
            static void transform(void *x, Nd4jLong xStride, void *result, Nd4jLong resultStride, void *scalar, void *extraParams, const Nd4jLong n);
        };
    }
}


#endif /* SCALAR_H_ */
