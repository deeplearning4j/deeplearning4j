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
 * reduce3.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCE3_H_
#define REDUCE3_H_

#include <helpers/DebugHelper.h>
#include <helpers/OmpLaunchHelper.h>

#include <loops/legacy_ops.h>
#include <math/templatemath.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>


using namespace simdOps;

namespace functions {
namespace reduce3 {

/**
 * Reduce involving
 * 2 arrays
 */
template <typename X, typename Y>
class  Reduce3 {
 public:
#ifdef __CUDACC__
  virtual SD_DEVICE inline Y opAtomic(X d1, X d2, Y *extraParamsRef) = 0;

  /**
   * Aggregate shared memory
   * @param sPartialsRef
   * @param tid
   * @param extraParams
   */
  template <typename OpType>
  static SD_DEVICE void aggregatePartials(void *sPartials, sd::LongType tid, sd::LongType numItems, void *extraParams);

  template <typename OpType>
  static SD_DEVICE void execScalarCuda(const void *x, const sd::LongType *xShapeInfo, const void *y,
                                       const sd::LongType *yShapeInfo, void *extraParams, void *z,
                                       const sd::LongType *zShapeInfo, sd::LongType *allocationPointer, void *reductionBuffer,
                                       const sd::LongType *tadOnlyShapeInfo);

  template <typename OpType>
  static SD_DEVICE void transformAll(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                     const sd::LongType *yShapeInfo, void *extraParams, void *vz,
                                     const sd::LongType *zShapeInfo, sd::LongType*dimension,
                                     sd::LongType dimensionLength,
                                     int postProcessOrNot,
                                     sd::LongType *allocationPointer, const sd::LongType *xTadShapeInfo,
                                     const sd::LongType *xOffsets, const sd::LongType *yTadShapeInfo,
                                     const sd::LongType *yOffsets);

  /**
   Perform a reduction
   @param n the number of elements
   @param xOffset the starting offset
   @param dx the data to perform the reduction on
   @param incx the increment on which to perform the reduction
   @param extraParams extra parameters used for calculations
   @param result where to store the result of the reduction
  */
  template <typename OpType>
  static SD_DEVICE void transform(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                  const sd::LongType *yShapeInfo, void *extraParams, void *vz,
                                  const sd::LongType *zShapeInfo, sd::LongType *dimension,
                                  sd::LongType dimensionLength,
                                  int postProcessOrNot, sd::LongType *allocationPointer, const sd::LongType *tadOnlyShapeInfo,
                                  const sd::LongType *tadOffsets, const sd::LongType *yTadOnlyShapeInfo,
                                  const sd::LongType *yTadOffsets);

  static SD_DEVICE void execCuda(int opNum, const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                 const sd::LongType *yShapeInfo, void *extraParams, void *vz,
                                 const sd::LongType *zShapeInfo, sd::LongType *dimension,
                                 sd::LongType dimensionLength,
                                 int postProcessOrNot, sd::LongType *allocationPointer, const sd::LongType *tadOnlyShapeInfo,
                                 const sd::LongType *tadOffsets, const sd::LongType *yTadOnlyShapeInfo,
                                 const sd::LongType *yTadOffsets);

  static SD_DEVICE void execAllCuda(int opNum, const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                    const sd::LongType *yShapeInfo, void *extraParams, void *vz,
                                    const sd::LongType *zShapeInfo, sd::LongType *dimension,
                                    sd::LongType dimensionLength,
                                    int postProcessOrNot,
                                    sd::LongType *allocationPointer, const sd::LongType *tadOnlyShapeInfo,
                                    const sd::LongType *tadOffsets, const sd::LongType *yTadOnlyShapeInfo,
                                    const sd::LongType *yTadOffsets);

  static SD_DEVICE void execScalarCuda(int opNum, const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                       const sd::LongType *yShapeInfo, void *extraParams, void *vz,
                                       const sd::LongType *zShapeInfo, sd::LongType *allocationPointer, void *reductionBuffer,
                                       const sd::LongType *tadOnlyShapeInfo);

  static SD_HOST void exec(dim3 launchDims, cudaStream_t *stream, int opNum, const void *vx,
                           const sd::LongType *xShapeInfo, const void *vy, const sd::LongType *yShapeInfo,
                           void *extraParams, void *vz, const sd::LongType *zShapeInfo, sd::LongType *dimension,
                           sd::LongType dimensionLength, int postProcessOrNot, sd::LongType *allocationPointer,
                           const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                           const sd::LongType *yTadOnlyShapeInfo, const sd::LongType *yTadOffsets);

  static SD_HOST void execAll(dim3 launchDims, cudaStream_t *stream, int opNum, const void *vx,
                              const sd::LongType *xShapeInfo, const void *vy, const sd::LongType *yShapeInfo,
                              void *extraParams, void *vz, const sd::LongType *zShapeInfo, sd::LongType *dimension,
                              sd::LongType dimensionLength, int postProcessOrNot, sd::LongType *allocationPointer,
                              const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                              const sd::LongType *yTadOnlyShapeInfo, const sd::LongType *yTadOffsets);

  static SD_HOST void execScalar(dim3 launchDims, cudaStream_t *stream, int opNum, const void *vx,
                                 const sd::LongType *xShapeInfo, const void *vy, const sd::LongType *yShapeInfo,
                                 void *extraParams, void *vz, const sd::LongType *zShapeInfo,
                                 sd::LongType *allocationPointer,
                                 void *reductionBuffer, const sd::LongType *tadOnlyShapeInfo);

#else

  template <typename OpType>
  static void execScalar(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, const void *vy,
                         const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo);

  static void execScalar(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParamsVals, const void *y,
                         const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo);

  template <typename OpType>
  static void exec(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, const void *vy,
                   const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo, sd::LongType *dimension,
                   sd::LongType dimensionLength, sd::LongType start, sd::LongType stop);

  template <typename OpType>
  static void exec(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, const void *vy,
                   const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo, sd::LongType *dimension,
                   sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets, sd::LongType start,
                   sd::LongType stop);

  template <typename OpType>
  static void execAll(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, const void *vy,
                      const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                      sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo, const sd::LongType *xOffsets,
                      const sd::LongType *yTadShapeInfo, const sd::LongType *yOffsets, sd::LongType start, sd::LongType stop);

  static void exec(int opNum, const void *vx, const sd::LongType *xShapeInfo, void *extraParamsVals, const void *vy,
                   const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo, sd::LongType *dimension,
                   long long int dimensionLength, sd::LongType start, sd::LongType stop);

  static void exec(int opNum, const void *vx, const sd::LongType *xShapeInfo, void *extraParamsVals, const void *vy,
                   const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo, sd::LongType *dimension,
                   sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets, sd::LongType start,
                   sd::LongType stop);

  static void execAll(int opNum, const void *vx, const sd::LongType *xShapeInfo, void *extraParamsVals, const void *vy,
                      const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                      sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo, const sd::LongType *xOffsets,
                      const sd::LongType *yTadShapeInfo, const sd::LongType *yOffsets, sd::LongType start, sd::LongType stop);
#endif
};

}  // namespace reduce3
}  // namespace functions



#endif /* REDUCE3_H_ */
