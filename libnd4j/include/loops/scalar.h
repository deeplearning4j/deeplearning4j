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

#ifndef SCALAR_H_
#define SCALAR_H_
#include <helpers/DebugHelper.h>
#include <helpers/OmpLaunchHelper.h>

#ifdef __JNI__
#include <jni.h>
#endif
#include <math/templatemath.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>

#include "helpers/logger.h"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <types/float16.h>
#endif
#include <loops/legacy_ops.h>

namespace functions {
namespace scalar {
/**
 * Apply a scalar
 *  operation to an array
 */
template <typename X, typename Y, typename Z>
class ScalarTransform {
 public:
#ifdef __CUDACC__

  template <typename OpType>
  SD_HOST static void intermediateShaped(dim3 &launchDims, cudaStream_t *stream, const void *vx,
                                         const sd::LongType *xShapeInfo, const sd::LongType *hxShapeInfo, void *vz,
                                         const sd::LongType *zShapeInfo, const sd::LongType *hzShapeInfo,
                                         const void *vscalar, void *vextraParams, int *allocPointer);

  template <typename OpType>
  SD_HOST static void intermediateAlongDimension(dim3 &launchDims, cudaStream_t *stream, const void *x,
                                                 const sd::LongType *xShapeInfo, void *z,
                                                 const sd::LongType *zShapeInfo, const void *scalars, void *extraParams,
                                                 int *dimension, int dimensionLength, const sd::LongType *tadShapeInfo,
                                                 const sd::LongType *tadOffsets, const sd::LongType *tadShapeInfoZ,
                                                 const sd::LongType *tadOffsetsZ);

  SD_HOST
  static void executeCudaShaped(dim3 &launchDims, cudaStream_t *stream, int opNum, const void *x,
                                const sd::LongType *xShapeInfo, const sd::LongType *hxShapeInfo, void *result,
                                const sd::LongType *resultShapeInfo, const sd::LongType *hzShapeInfo,
                                const void *scalar, void *extraParams);

  SD_HOST
  static void executeCudaAlongDimension(dim3 &launchDims, cudaStream_t *stream, int opNum, const void *x,
                                        const sd::LongType *xShapeInfo, void *z, const sd::LongType *zShapeInfo,
                                        const void *scalars, void *extraParams, int *dimension, int dimensionLength,
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                                        const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ);

#else
  template <typename OpType>
  static void transform(const void *x, const sd::LongType *xShapeInfo, void *extraParams, void *z,
                        const sd::LongType *zShapeInfo, const void *scalars, int *dimension, int dimensionLength,
                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                        const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ, uint64_t start,
                        uint64_t stop);

  static void transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams, void *z,
                        const sd::LongType *zShapeInfo, const void *scalars, int *dimension, int dimensionLength,
                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                        const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ, uint64_t start,
                        uint64_t stop);

  static void transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *result,
                        const sd::LongType *resultShapeInfo, const void *scalar, void *extraParams, uint64_t start,
                        uint64_t stop);

  static void transform(int opNum, const void *x, sd::LongType xStride, void *result, sd::LongType resultStride,
                        const void *scalar, void *extraParams, uint64_t len, uint64_t start, uint64_t stop);

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

  template <typename OpType>
  static void transform(const void *x, const sd::LongType *xShapeInfo, void *result,
                        const sd::LongType *resultShapeInfo, const void *scalar, void *extraParams, uint64_t start,
                        uint64_t stop);

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

  template <typename OpType>
  static void transform(const void *x, sd::LongType xStride, void *result, sd::LongType resultStride,
                        const void *scalar, void *extraParams, uint64_t len, uint64_t start, uint64_t stop);
#endif
};
}  // namespace scalar
}  // namespace functions

#endif /* SCALAR_H_ */
