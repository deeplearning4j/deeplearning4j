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

#ifndef REDUCE_LONG_H
#define REDUCE_LONG_H

//#include <string>
#include <helpers/shape.h>
#include <math/templatemath.h>
#include <memory/Workspace.h>
#include <ops/ops.h>
#include <stdio.h>
#include <system/op_boilerplate.h>
#include <system/pairwise_util.h>

#pragma once

#include <loops/legacy_ops.h>

// an op for the kernel
namespace functions {
namespace reduce {

/**
 * A reduce function
 * reduces a vector down to
 * a subset of itself
 * via aggregating member
 * elements.
 */
template <typename X, typename Z>
class SD_LIB_HIDDEN ReduceLongFunction {
 public:
#ifdef __CUDACC__
  template <typename OpType>
  static SD_DEVICE void aggregatePartials(void *sPartials, sd::LongType tid, sd::LongType numItems, void *extraParams);

  template <typename OpType>
  static SD_DEVICE void execScalarCuda(const void *vx, const sd::LongType *xShapeInfo, void *extraParams, void *vz,
                                       const sd::LongType *zShapeInfo, void *reductionBuffer,
                                       const sd::LongType *tadOnlyShapeInfo);

  template <typename OpType>
  static SD_DEVICE void transformCudaXD(const void *vx, const sd::LongType *outerXTadShapeInfo,
                                        const sd::LongType *innerXTadShapeInfo, void *extraParams,
                                        void *vreductionBuffer, void *vz, const sd::LongType *zShapeInfo);

  template <typename OpType>
  static SD_HOST void intermediateScalar(dim3 launchDims, cudaStream_t *stream, const void *vx,
                                         const sd::LongType *xShapeInfo, const sd::LongType *hXShapeInfo,
                                         void *extraParams, void *vz, const sd::LongType *zShapeInfo,
                                         const sd::LongType *hZShapeInfo, int *dimension, int dimensionLength,
                                         void *reductionBuffer, const sd::LongType *tadOnlyShapeInfo);

  template <typename OpType>
  static SD_HOST void intermediateXD(dim3 launchDims, cudaStream_t *stream, const void *vx,
                                     const sd::LongType *dXShapeInfo, const sd::LongType *hXShapeInfo,
                                     void *extraParams, void *vreductionBuffer, void *vz,
                                     const sd::LongType *dZShapeInfo, const sd::LongType *hZShapeInfo, const int *dims);

  static SD_HOST void execReduceScalar(dim3 launchDims, cudaStream_t *stream, int opNum, const void *vx,
                                       const sd::LongType *xShapeInfo, const sd::LongType *hXShapeInfo,
                                       void *extraParams, void *vz, const sd::LongType *zShapeInfo,
                                       const sd::LongType *hZShapeInfo, int *dimension, int dimensionLength,
                                       void *reductionBuffer, const sd::LongType *tadOnlyShapeInfo);

  static SD_HOST void execReduceXD(dim3 launchDims, cudaStream_t *stream, int opNum, const void *vx,
                                   const sd::LongType *dXShapeInfo, const sd::LongType *hXShapeInfo, void *extraParams,
                                   void *vreductionBuffer, void *vz, const sd::LongType *dZShapeInfo,
                                   const sd::LongType *hZShapeInfo, const int *dims);

#else

  /**
   * Reduce down to 1 number
   * @param x the input
   * @param xShapeInfo the shape information
   * for the input
   * @param extraParams the extra params
   * @return
   */
  template <typename OpType>
  static SD_HOST Z execScalar(const void *x, const sd::LongType *xShapeInfo, void *extraParams);

  template <typename OpType>
  static SD_HOST void execScalar(const void *x, const sd::LongType *xShapeInfo, void *extraParams, void *z,
                                 const sd::LongType *zShapeInfo);

  static Z execScalar(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams);

  static void execScalar(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams, void *z,
                         const sd::LongType *zShapeInfo);

  static void exec(int opNum, sd::memory::Workspace *workspace, const void *vx, const sd::LongType *xShapeInfo,
                   void *vextraParams, void *vz, const sd::LongType *zShapeInfo, const int *dims);

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

  template <typename OpType>
  static void SD_HOST exec(sd::memory::Workspace *workspace, const void *vx, const sd::LongType *xShapeInfo,
                           void *vextraParams, void *vz, const sd::LongType *zShapeInfo, const int *dims);

  /**
   * CPU implementation
   * @param x the input data
   * @param xShapeInfo the shape information for
   * the input data
   * @param extraParams the extra parameters for the problem
   * @param result the result buffer
   * @param resultShapeInfo the shape information
   */
  template <typename OpType>
  static void SD_HOST exec(const void *x, const sd::LongType *xShapeInfo, void *extraParams, void *result,
                           const sd::LongType *resultShapeInfo);

  /**
   * Reduce down to 1 number
   * @param x the input
   * @param xShapeInfo the shape information
   * for the input
   * @param extraParams the extra params
   * @return
   */
  template <typename OpType>
  static Z SD_HOST execScalar(const void *x, sd::LongType xElementWiseStride, sd::LongType length, void *extraParams);
#endif
};

#ifdef __CUDACC__
/**
 *
 * @param extraParams
 * @param sPartials
 * @param sMemSize
 */
template <typename T>
SD_DEVICE void initializeShared(T *extraParams, T **sPartials, int sMemSize);

#endif

}  // namespace reduce

}  // namespace functions

#endif
