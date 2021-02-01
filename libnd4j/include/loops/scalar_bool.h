/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <system/dll.h>

#ifdef __JNI__
#include <jni.h>
#endif
#include <math/templatemath.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>
#include "helpers/logger.h"
#include <helpers/OmpLaunchHelper.h>
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
            static void transformCuda(const void* scalar,
                                      const void *vy, const Nd4jLong *shapeInfo,
                                      void *vparams,
                                      void *vresult, const Nd4jLong *resultShapeInfo,
                                      int *allocationBuffer);
            
            template<typename OpType>
            __device__
            static void transformCuda(Nd4jLong n,
                                      const void* vx, const void *vy, Nd4jLong yEWS,
                                      void *vparams,
                                      void *vz, Nd4jLong zEWS,
                                      int *allocationBuffer);

            template<typename OpType>
            __device__
            static void transformCuda(const void *vx, const Nd4jLong *xShapeInfo,
                                      void *vextraParams,
                                      void *vz, const Nd4jLong *zShapeInfo,
                                      const void *vscalars,
                                      int *dimension, int dimensionLength,
                                      const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                                      const Nd4jLong *tadShapeInfoZ, const Nd4jLong *tadOffsetsZ);

            template <typename OpType>
            __host__
            static void intermediateAlongDimension(dim3& launchDims, cudaStream_t *stream,
                                                   const void *x, const Nd4jLong *xShapeInfo,
                                                   void *z, const Nd4jLong *zShapeInfo,
                                                   const void *scalars,
                                                   void *extraParams,
                                                   int *dimension, int dimensionLength,
                                                   const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                                                   const Nd4jLong *tadShapeInfoZ, const Nd4jLong *tadOffsetsZ);

            template <typename OpType>
            __host__
            static void intermediateShaped(dim3& launchDims, cudaStream_t *stream,
                                           const void *vx, const Nd4jLong *xShapeInfo,
                                           void *vz, const Nd4jLong *zShapeInfo,
                                           const void* vscalar,
                                           void *vextraParams,
                                           int *allocPointer);
            
            __host__
            static void executeCudaShaped(dim3& launchDims, cudaStream_t *stream,
                                          int opNum,
                                          const void *x, const Nd4jLong *xShapeInfo,
                                          void *result, const Nd4jLong *resultShapeInfo,
                                          const void* scalar,
                                          const void *extraParams);

            __host__
            static void executeCudaAlongDimension(dim3& launchDims, cudaStream_t *stream,
                                                  int opNum,
                                                  const void *x, const Nd4jLong *xShapeInfo,
                                                  void *z, const Nd4jLong *zShapeInfo,
                                                  const void *scalars,
                                                  void *extraParams,
                                                  int *dimension, int dimensionLength,
                                                  const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                                                  const Nd4jLong *tadShapeInfoZ, const Nd4jLong *tadOffsetsZ);

/*
#include "cuda/scalar_temp.cu"
*/
#else
            template <typename OpType>
            static void transform(const void *x, const Nd4jLong *xShapeInfo,
                                  void *extraParams,
                                  void *z, const Nd4jLong *zShapeInfo,
                                  const void *scalars,
                                  int *dimension, int dimensionLength,
                                  const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                                  const Nd4jLong *tadShapeInfoZ, const Nd4jLong *tadOffsetsZ,
                                  uint64_t start, uint64_t stop);
 
           static void transform(int opNum,
                                 const void *x, const Nd4jLong *xShapeInfo,
                                 void *extraParams,
                                 void *z, const Nd4jLong *zShapeInfo,
                                 const void *scalars,
                                 int *dimension, int dimensionLength,
                                 const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                                 const Nd4jLong *tadShapeInfoZ, const Nd4jLong *tadOffsetsZ,
                                 uint64_t start, uint64_t stop);

            static void transform(int opNum,
                                  const void *x, const Nd4jLong *xShapeInfo,
                                  void *result, const Nd4jLong *resultShapeInfo,
                                  const void *scalar,
                                  void *extraParams,
                                  uint64_t start, uint64_t stop);

            static void transform(int opNum,
                                  const void *x, Nd4jLong xStride,
                                  void *result, Nd4jLong resultStride,
                                  const void *scalar,
                                  void *extraParams,
                                  uint64_t n, uint64_t start, uint64_t stop);




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
            static  void transform(const void *x, const Nd4jLong *xShapeInfo,
                                   void *result, const Nd4jLong *resultShapeInfo,
                                   const void *scalar,
                                   void *extraParams,
                                   uint64_t start, uint64_t stop);


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
            static void transform(const void *x, Nd4jLong xStride,
                                  void *result, Nd4jLong resultStride,
                                  const void *scalar, void *extraParams,
                                  uint64_t n, uint64_t start, uint64_t stop);
#endif
        };
    }
}


#endif /* SCALAR_H_ */
