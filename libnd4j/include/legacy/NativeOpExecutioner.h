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

//
// Created by agibsonccc on 1/28/16.
//

#ifndef NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
#define NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H

#include <array/ArrayOptions.h>
#include <execution/LaunchContext.h>
#include <ops/specials.h>
#include <ops/specials_sparse.h>
#include <types/types.h>
#include <helpers/shape.h>

/**
 * Native op executioner:
 *
 */

class SD_LIB_EXPORT NativeOpExecutioner {
 public:
  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfo
   */
  static void execIndexReduceScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                    const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                    const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParamsVals
   * @param y
   * @param yShapeInfo
   * @param result
   * @param resultShapeInfoBuffer
   * @param dimension
   * @param dimensionLength
   */
  static void execReduce3Scalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                                const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                void *hZ, const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParamsVals
   * @param y
   * @param yShapeInfo
   * @param result
   * @param resultShapeInfo
   */
  static void execReduce3(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                          const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                          const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                          const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParamsVals
   * @param y
   * @param yShapeInfo
   * @param result
   * @param resultShapeInfoBuffer
   * @param dimension
   * @param dimensionLength
   */
  static void execReduce3(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                          const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                          const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                          const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                          long long int *dimension,
                          sd::LongType dimensionLength, const sd::LongType *xTadOnlyShapeInfo, const sd::LongType *xTadOffsets,
                          const sd::LongType *yTadOnlyShapeInfo, const sd::LongType *yTadOffsets);

  static void execReduce3All(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                             const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             long long int *dimension,
                             sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo, const sd::LongType *xOffsets,
                             const sd::LongType *yTadShapeInfo, const sd::LongType *yOffsets);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfoBuffer
   * @param dimension
   * @param dimensionLength
   */
  static void execIndexReduce(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                              const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                              const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                              long long int *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo,
                              const sd::LongType *tadOffsets);

  /**
   *
   * @param opNum
   * @param x
   * @param xStride
   * @param result
   * @param resultStride
   * @param scalar
   * @param extraParams
   * @param n
   */
  static void execScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                         const void *dX, const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                         void *dZ, const sd::LongType *dZShapeInfo, const void *hScalar,
                         const sd::LongType *hSscalarShapeInfo, const void *dScalar,
                         const sd::LongType *dSscalarShapeInfo, void *extraParams, bool allowParallelism = true);

  static void execScalarBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                             void *dZ, const sd::LongType *dZShapeInfo, const void *hScalar,
                             const sd::LongType *hSscalarShapeInfo, const void *dScalar,
                             const sd::LongType *dSscalarShapeInfo, void *extraParams, bool allowParallelism = true);

  static void execScalarInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                            const void *dX, const sd::LongType *dXShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                            void *dZ, const sd::LongType *dZShapeInfo, const void *hScalar,
                            const sd::LongType *hSscalarShapeInfo, const void *dScalar,
                            const sd::LongType *dSscalarShapeInfo, void *extraParams, bool allowParallelism = true);

  static void execScalar(sd::LaunchContext *lc, int opNum, void const *hX, sd::LongType const *hXShapeInfo,
                         void const *dX, sd::LongType const *dXShapeInfo, void *extraParams, void *hZ,
                         sd::LongType const *hZShapeInfo, void *dZ, sd::LongType const *dZShapeInfo,
                         void const *hScalars, sd::LongType const *hScalarShapeInfo, void const *dScalars,
                         sd::LongType const *dScalarShapeInfo, long long int *dimension, sd::LongType dimensionLength,
                         sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets,
                         sd::LongType const *tadShapeInfoZ, sd::LongType const *tadOffsetsZ);

  static void execScalarBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             const void *hScalars, const sd::LongType *hScalarShapeInfo, const void *dScalars,
                             const sd::LongType *dScalarShapeInfo, long long int *dimension, sd::LongType dimensionLength,
                             const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                             const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ);

  static void execScalarInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                            const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                            const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                            const void *hScalars, const sd::LongType *hScalarShapeInfo, const void *dScalars,
                            const sd::LongType *dScalarShapeInfo, long long int *dimension, sd::LongType dimensionLength,
                            const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                            const sd::LongType *tadShapeInfoZ, const sd::LongType *tadOffsetsZ);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param y
   * @param yShapeInfo
   * @param result
   * @param resultShapeInfo
   * @param dimension
   * @param dimensionLength
   */
  static void execBroadcast(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                            const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                            const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                            const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                            long long int *dimension,
                            sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                            const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);

  static void execBroadcast(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                            const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                            const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                            const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execInverseBroadcast(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                   const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                   const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                   void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
                                   const sd::LongType *dZShapeInfo,
                                   long long int *dimension, sd::LongType dimensionLength,
                                   const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                                   const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);

  static void execBroadcastBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                void *hZ, const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                void *extraParams, sd::LongType *dimension, sd::LongType dimensionLength,
                                const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                                const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);

  static void execBroadcastBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                void *hZ, const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                void *extraParams);

  static void execInverseBroadcastBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                       const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                       const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                       void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
                                       const sd::LongType *dZShapeInfo, void *extraParams,
                                       long long int *dimension,
                                       sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
                                       const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ,
                                       const sd::LongType *tadOffsetsZ);

  static void execBroadcastInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                               const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                               const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                               void *hZ, const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                               long long int *dimension, sd::LongType dimensionLength, const sd::LongType *tadOnlyShapeInfo,
                               const sd::LongType *tadOffsets, const sd::LongType *tadOnlyShapeInfoZ,
                               const sd::LongType *tadOffsetsZ);

  static void execBroadcastInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                               const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                               const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                               void *hZ, const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execInverseBroadcastInt(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                      const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                      const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                      void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
                                      const sd::LongType *dZShapeInfo, long long int *dimension, sd::LongType dimensionLength,
                                      const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                                      const sd::LongType *tadOnlyShapeInfoZ, const sd::LongType *tadOffsetsZ);

  /**
   *
   * @param opNum
   * @param dx
   * @param xStride
   * @param y
   * @param yStride
   * @param result
   * @param resultStride
   * @param extraParams
   * @param n
   */
  static void execPairwiseTransform(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                    const void *dX, const sd::LongType *dXShapeInfo, const void *hY,
                                    const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo,
                                    void *hZ, const sd::LongType *hZShapeInfo, void *dZ,
                                    const sd::LongType *dZShapeInfo, void *extraParams);

  static void execPairwiseBoolTransform(sd::LaunchContext *lc, int opNum, const void *hX,
                                        const sd::LongType *hXShapeInfo, const void *dX,
                                        const sd::LongType *dXShapeInfo, const void *hY,
                                        const sd::LongType *hYShapeInfo, const void *dY,
                                        const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                        void *dZ, const sd::LongType *dZShapeInfo, void *extraParams);

  static void execPairwiseIntTransform(sd::LaunchContext *lc, int opNum, const void *hX,
                                       const sd::LongType *hXShapeInfo, const void *dX, const sd::LongType *dXShapeInfo,
                                       const void *hY, const sd::LongType *hYShapeInfo, const void *dY,
                                       const sd::LongType *dYShapeInfo, void *hZ, const sd::LongType *hZShapeInfo,
                                       void *dZ, const sd::LongType *dZShapeInfo, void *extraParams);

  /**
   *
   * @param opNum
   * @param dx
   * @param xStride
   * @param result
   * @param resultStride
   * @param extraParams
   * @param n
   */
  static void execTransformFloat(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                 const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                 const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                 void *extraParams, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets);

  static void execTransformAny(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                               const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                               const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                               void *extraParams, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                               bool allowParallelism = true);

  static void execTransformStrict(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                  const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                  const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                  void *extraParams, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets);

  static void execTransformSame(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                void *extraParams, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets);

  static void execTransformBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                const void *dX, const sd::LongType *dXShapeInfo, void *hZ,
                                const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                void *extraParams, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets);
  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfo
   */
  static void execReduceFloat(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                              const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                              const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                              long long int *dimension, sd::LongType dimensionLength);

  static void execReduceSame(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             sd::LongType *dimension,
                             sd::LongType dimensionLength);

  static void execReduceBool(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             long long int *dimension,
                             sd::LongType dimensionLength);

  static void execReduceLong(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             sd::LongType *dimension,
                             sd::LongType dimensionLength);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @return
   */
  static void execReduceFloatScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                    const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                    const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execReduceBoolScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                   const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                   const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execReduceSameScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                   const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                   const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execReduceLongScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                   const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                   const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo);

  static void execReduce3TAD(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                             const void *dX, const sd::LongType *dXShapeInfo, void *extraParamsVals, const void *hY,
                             const sd::LongType *hYShapeInfo, const void *dY, const sd::LongType *dYShapeInfo, void *hZ,
                             const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                             long long int *dimension,
                             sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                             const sd::LongType *yTadShapeInfo, const sd::LongType *yTadOffsets);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfoBuffer
   * @param dimension
   * @param dimensionLength
   */
  static void execSummaryStats(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                               const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                               const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                               long long int *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo,
                               const sd::LongType *tadOffsets, bool biasCorrected);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfo
   */
  static void execSummaryStats(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                               const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                               const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                               bool biasCorrected);

  /**
   *
   * @param opNum
   * @param x
   * @param xShapeInfo
   * @param extraParams
   * @param result
   * @param resultShapeInfo
   */
  static void execSummaryStatsScalar(sd::LaunchContext *lc, int opNum, const void *hX, const sd::LongType *hXShapeInfo,
                                     const void *dX, const sd::LongType *dXShapeInfo, void *extraParams, void *hZ,
                                     const sd::LongType *hZShapeInfo, void *dZ, const sd::LongType *dZShapeInfo,
                                     bool biasCorrected);

  static void execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, void *hZ,
                         const sd::LongType *hZShapeBuffer, void *dZ, const sd::LongType *dZShapeBuffer,
                         void *extraArguments);

  static void execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, const void *hX,
                         const sd::LongType *hXShapeBuffer, const void *dX, const sd::LongType *dXShapeBuffer, void *hZ,
                         const sd::LongType *hZShapeBuffer, void *dZ, const sd::LongType *dZShapeBuffer,
                         void *extraArguments);

  static void execRandom(sd::LaunchContext *lc, int opNum, sd::Pointer state, const void *hX,
                         const sd::LongType *hXShapeBuffer, const void *dX, const sd::LongType *dXShapeBuffer,
                         const void *hY, const sd::LongType *hYShapeBuffer, const void *dY,
                         const sd::LongType *dYShapeBuffer, void *hZ, const sd::LongType *hZShapeBuffer, void *dZ,
                         const sd::LongType *dZShapeBuffer, void *extraArguments);

  inline static void execSort(sd::NDArray *x, bool descending) {
    auto xType = x->dataType();

    BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::sortGeneric(x, descending), SD_COMMON_TYPES);
  }

  static void execSort(sd::NDArray *x, sd::LongType *dimension,  sd::LongType dimensionLength,
                       bool descending) {
    auto xType = x->dataType();

    BUILD_SINGLE_SELECTOR(
        xType, sd::SpecialMethods,
        ::sortTadGeneric(x, dimension, dimensionLength, descending),
        SD_COMMON_TYPES);
  }

  inline static void execSortCooIndices(sd::LongType *indices, void *x, sd::LongType length,
                                        const sd::LongType *xShapeInfo) {
    auto xType = sd::ArrayOptions::dataType(xShapeInfo);
    int rank = shape::rank(xShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, sd::sparse::SparseUtils, ::sortCooIndicesGeneric(indices, x, length, rank),
                          SD_COMMON_TYPES);
  }

  inline static void execRavelMultiIndex(sd::LongType *indices, sd::LongType *flatIndices, sd::LongType length,
                                         sd::LongType *shapeInfo, int mode) {
    sd::sparse::IndexUtils::ravelMultiIndex(indices, flatIndices, length, shapeInfo, mode);
  }

  inline static void execUnravelIndex(sd::LongType *indices, sd::LongType *flatIndices, sd::LongType length,
                                      sd::LongType *shapeInfo) {
    sd::sparse::IndexUtils::unravelIndex(indices, flatIndices, length, shapeInfo);
  }

  inline static sd::LongType encodeBitmap(sd::NDArray *x, sd::LongType N, long long int *dz,
                                          float threshold) {
    auto xType = x->dataType();

    BUILD_SINGLE_SELECTOR(xType, return sd::SpecialMethods, ::encodeBitmapGeneric(x, N, dz, threshold),
                          SD_FLOAT_TYPES);
  }

  inline static void decodeBitmap(sd::NDArray *dx, sd::LongType N, sd::NDArray *z) {
    auto zType = z->dataType();

    BUILD_SINGLE_SELECTOR(zType, sd::SpecialMethods, ::decodeBitmapGeneric(dx,z,N), SD_FLOAT_TYPES);
  }
};

#endif  // NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
