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

//
// Created by agibsonccc on 1/28/16.
//

#ifndef NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
#define NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H


#include <types/types.h>
#include <system/dll.h>
#include <ops/specials.h>
#include <ops/specials_sparse.h>
#include <execution/LaunchContext.h>
#include <array/ArrayOptions.h>
#include <helpers/shape.h>

/**
 * Native op executioner:
 *
 */

class ND4J_EXPORT NativeOpExecutioner {
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
    static void execIndexReduceScalar(sd::LaunchContext  *lc,
                                    int opNum,
                                    const void *hX, const Nd4jLong *hXShapeInfo,
                                    const void *dX, const Nd4jLong *dXShapeInfo,
                                    void *extraParams,
                                    void *hZ, const Nd4jLong *hZShapeInfo,
                                    void *dZ, const Nd4jLong *dZShapeInfo);

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
    static void execReduce3Scalar(sd::LaunchContext  *lc,
                            int opNum,
                            const void *hX, const Nd4jLong *hXShapeInfo,
                            const void *dX, const Nd4jLong *dXShapeInfo,
                            void *extraParamsVals,
                            const void *hY, const Nd4jLong *hYShapeInfo,
                            const void *dY, const Nd4jLong *dYShapeInfo,
                            void *hZ, const Nd4jLong *hZShapeInfo,
                            void *dZ, const Nd4jLong *dZShapeInfo);


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
    static void execReduce3(sd::LaunchContext  *lc,
                            int opNum,
                            const void *hX, const Nd4jLong *hXShapeInfo,
                            const void *dX, const Nd4jLong *dXShapeInfo,
                            void *extraParamsVals,
                            const void *hY, const Nd4jLong *hYShapeInfo,
                            const void *dY, const Nd4jLong *dYShapeInfo,
                            void *hZ, const Nd4jLong *hZShapeInfo,
                            void *dZ, const Nd4jLong *dZShapeInfo);

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
    static void execReduce3(sd::LaunchContext  *lc,
                            int opNum,
                            const void *hX, const Nd4jLong *hXShapeInfo,
                            const void *dX, const Nd4jLong *dXShapeInfo,
                            void *extraParamsVals,
                            const void *hY, const Nd4jLong *hYShapeInfo,
                            const void *dY, const Nd4jLong *dYShapeInfo,
                            void *hZ, const Nd4jLong *hZShapeInfo,
                            void *dZ, const Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            const Nd4jLong *xTadOnlyShapeInfo, const Nd4jLong *xTadOffsets,
                            const Nd4jLong *yTadOnlyShapeInfo, const Nd4jLong *yTadOffsets);

    static void execReduce3All(sd::LaunchContext  *lc,
                            int opNum,
                            const void *hX, const Nd4jLong *hXShapeInfo,
                            const void *dX, const Nd4jLong *dXShapeInfo,
                            void *extraParamsVals,
                            const void *hY, const Nd4jLong *hYShapeInfo,
                            const void *dY, const Nd4jLong *dYShapeInfo,
                            void *hZ, const Nd4jLong *hZShapeInfo,
                            void *dZ, const Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                            const Nd4jLong *yTadShapeInfo, const Nd4jLong *yOffsets);

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
    static void execIndexReduce(sd::LaunchContext  *lc,
                                int opNum,
                                const void *hX, const Nd4jLong *hXShapeInfo,
                                const void *dX, const Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, const Nd4jLong *hZShapeInfo,
                                void *dZ, const Nd4jLong *dZShapeInfo,
                                int *dimension, int dimensionLength,
                                const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets);

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
    static void execScalar(sd::LaunchContext  *lc,
                           int opNum,
                           const void *hX, const Nd4jLong *hXShapeInfo,
                           const void *dX, const Nd4jLong *dXShapeInfo,
                           void *hZ, const Nd4jLong *hZShapeInfo,
                           void *dZ, const Nd4jLong *dZShapeInfo,
                           const void *hScalar, const Nd4jLong *hSscalarShapeInfo,
                           const void *dScalar, const Nd4jLong *dSscalarShapeInfo,
                           void *extraParams,
                           bool allowParallelism = true);

static void execScalarBool(sd::LaunchContext  *lc,
                            int opNum,
                            const void *hX, const Nd4jLong *hXShapeInfo,
                            const void *dX, const Nd4jLong *dXShapeInfo,
                            void *hZ, const Nd4jLong *hZShapeInfo,
                            void *dZ, const Nd4jLong *dZShapeInfo,
                            const void *hScalar, const Nd4jLong *hSscalarShapeInfo,
                            const void *dScalar, const Nd4jLong *dSscalarShapeInfo,
                            void *extraParams,
                            bool allowParallelism = true);

static void execScalarInt(sd::LaunchContext  *lc,
                               int opNum,
                               const void *hX, const Nd4jLong *hXShapeInfo,
                               const void *dX, const Nd4jLong *dXShapeInfo,
                               void *hZ, const Nd4jLong *hZShapeInfo,
                               void *dZ, const Nd4jLong *dZShapeInfo,
                               const void *hScalar, const Nd4jLong *hSscalarShapeInfo,
                               const void *dScalar, const Nd4jLong *dSscalarShapeInfo,
                               void *extraParams,
                               bool allowParallelism = true);

 static void execScalar(sd::LaunchContext  *lc,
                            int opNum,
                            void const* hX, Nd4jLong const* hXShapeInfo,
                            void const* dX, Nd4jLong const* dXShapeInfo,
                            void *extraParams,
                            void *hZ, Nd4jLong const* hZShapeInfo,
                            void *dZ, Nd4jLong const* dZShapeInfo,
                            void const* hScalars, Nd4jLong const* hScalarShapeInfo,
                            void const* dScalars, Nd4jLong const* dScalarShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets,
                            Nd4jLong const* tadShapeInfoZ, Nd4jLong const* tadOffsetsZ);

 static void execScalarBool(sd::LaunchContext  *lc,
                            int opNum,
                            const void *hX, const Nd4jLong *hXShapeInfo,
                            const void *dX, const Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hZ, const Nd4jLong *hZShapeInfo,
                            void *dZ, const Nd4jLong *dZShapeInfo,
                            const void *hScalars, const Nd4jLong *hScalarShapeInfo,
                            const void *dScalars, const Nd4jLong *dScalarShapeInfo,
                            int *dimension, int dimensionLength,
                            const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                            const Nd4jLong *tadShapeInfoZ, const Nd4jLong *tadOffsetsZ);

 static void execScalarInt(sd::LaunchContext  *lc,
                           int opNum,
                           const void *hX, const Nd4jLong *hXShapeInfo,
                           const void *dX, const Nd4jLong *dXShapeInfo,
                           void *extraParams,
                           void *hZ, const Nd4jLong *hZShapeInfo,
                           void *dZ, const Nd4jLong *dZShapeInfo,
                           const void *hScalars, const Nd4jLong *hScalarShapeInfo,
                           const void *dScalars, const Nd4jLong *dScalarShapeInfo,
                           int *dimension, int dimensionLength,
                           const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                           const Nd4jLong *tadShapeInfoZ, const Nd4jLong *tadOffsetsZ);


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
    static void execBroadcast(sd::LaunchContext  *lc,
                              int opNum,
                              const void *hX, const Nd4jLong *hXShapeInfo,
                              const void *dX, const Nd4jLong *dXShapeInfo,
                              const void *hY, const Nd4jLong *hYShapeInfo,
                              const void *dY, const Nd4jLong *dYShapeInfo,
                              void *hZ, const Nd4jLong *hZShapeInfo,
                              void *dZ, const Nd4jLong *dZShapeInfo,
                              int *dimension, int dimensionLength,
                              const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                              const Nd4jLong *tadOnlyShapeInfoZ,const Nd4jLong *tadOffsetsZ);

    static void execBroadcast(sd::LaunchContext* lc,
                              int opNum,
                              const void *hX, const Nd4jLong *hXShapeInfo,
                              const void *dX, const Nd4jLong *dXShapeInfo,
                              const void *hY, const Nd4jLong *hYShapeInfo,
                              const void *dY, const Nd4jLong *dYShapeInfo,
                              void *hZ, const Nd4jLong *hZShapeInfo,
                              void *dZ, const Nd4jLong *dZShapeInfo);

    static void execInverseBroadcast(sd::LaunchContext  *lc,
                                     int opNum,
                                     const void *x, const Nd4jLong *xShapeInfo,
                                     const void *dX, const Nd4jLong *dXShapeInfo,
                                     const void *y, const Nd4jLong *yShapeInfo,
                                     const void *dY, const Nd4jLong *dYShapeInfo,
                                     void *result, const Nd4jLong *resultShapeInfo,
                                     void *dZ, const Nd4jLong *dZShapeInfo,
                                     int *dimension, int dimensionLength,
                                     const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                     const Nd4jLong *tadOnlyShapeInfoZ, const Nd4jLong *tadOffsetsZ);


    static void execBroadcastBool(sd::LaunchContext  *lc,
                                  int opNum,
                                  const void *hX, const Nd4jLong *hXShapeInfo,
                                  const void *dX, const Nd4jLong *dXShapeInfo,
                                  const void *hY, const Nd4jLong *hYShapeInfo,
                                  const void *dY, const Nd4jLong *dYShapeInfo,
                                  void *hZ, const Nd4jLong *hZShapeInfo,
                                  void *dZ, const Nd4jLong *dZShapeInfo,
                                  void *extraParams,
                                  int *dimension, int dimensionLength,
                                  const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                  const Nd4jLong *tadOnlyShapeInfoZ,const Nd4jLong *tadOffsetsZ);

    static void execBroadcastBool(sd::LaunchContext* lc,
                                  int opNum,
                                  const void *hX, const Nd4jLong *hXShapeInfo,
                                  const void *dX, const Nd4jLong *dXShapeInfo,
                                  const void *hY, const Nd4jLong *hYShapeInfo,
                                  const void *dY, const Nd4jLong *dYShapeInfo,
                                  void *hZ, const Nd4jLong *hZShapeInfo,
                                  void *dZ, const Nd4jLong *dZShapeInfo,
                                  void *extraParams);

    static void execInverseBroadcastBool(sd::LaunchContext  *lc,
                                         int opNum,
                                         const void *x, const Nd4jLong *xShapeInfo,
                                         const void *dX, const Nd4jLong *dXShapeInfo,
                                         const void *y, const Nd4jLong *yShapeInfo,
                                         const void *dY, const Nd4jLong *dYShapeInfo,
                                         void *result, const Nd4jLong *resultShapeInfo,
                                         void *dZ, const Nd4jLong *dZShapeInfo,
                                         void *extraParams,
                                         int *dimension, int dimensionLength,
                                         const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                         const Nd4jLong *tadOnlyShapeInfoZ, const Nd4jLong *tadOffsetsZ);

    static void execBroadcastInt(sd::LaunchContext  *lc,
                                 int opNum,
                                 const void *hX, const Nd4jLong *hXShapeInfo,
                                 const void *dX, const Nd4jLong *dXShapeInfo,
                                 const void *hY, const Nd4jLong *hYShapeInfo,
                                 const void *dY, const Nd4jLong *dYShapeInfo,
                                 void *hZ, const Nd4jLong *hZShapeInfo,
                                 void *dZ, const Nd4jLong *dZShapeInfo,
                                 int *dimension, int dimensionLength,
                                 const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                 const Nd4jLong *tadOnlyShapeInfoZ,const Nd4jLong *tadOffsetsZ);

    static void execBroadcastInt(sd::LaunchContext* lc,
                                 int opNum,
                                 const void *hX, const Nd4jLong *hXShapeInfo,
                                 const void *dX, const Nd4jLong *dXShapeInfo,
                                 const void *hY, const Nd4jLong *hYShapeInfo,
                                 const void *dY, const Nd4jLong *dYShapeInfo,
                                 void *hZ, const Nd4jLong *hZShapeInfo,
                                 void *dZ, const Nd4jLong *dZShapeInfo);

    static void execInverseBroadcastInt(sd::LaunchContext  *lc,
                                        int opNum,
                                        const void *x, const Nd4jLong *xShapeInfo,
                                        const void *dX, const Nd4jLong *dXShapeInfo,
                                        const void *y, const Nd4jLong *yShapeInfo,
                                        const void *dY, const Nd4jLong *dYShapeInfo,
                                        void *result, const Nd4jLong *resultShapeInfo,
                                        void *dZ, const Nd4jLong *dZShapeInfo,
                                        int *dimension, int dimensionLength,
                                        const Nd4jLong *tadOnlyShapeInfo, const Nd4jLong *tadOffsets,
                                        const Nd4jLong *tadOnlyShapeInfoZ, const Nd4jLong *tadOffsetsZ);

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
    static void execPairwiseTransform(sd::LaunchContext  *lc,
                                      int opNum,
                                      const void *hX, const Nd4jLong *hXShapeInfo,
                                      const void *dX, const Nd4jLong *dXShapeInfo,
                                      const void *hY, const Nd4jLong *hYShapeInfo,
                                      const void *dY, const Nd4jLong *dYShapeInfo,
                                      void *hZ, const Nd4jLong *hZShapeInfo,
                                      void *dZ, const Nd4jLong *dZShapeInfo,
                                      void *extraParams);

    static void execPairwiseBoolTransform(sd::LaunchContext  *lc,
                                          int opNum,
                                          const void *hX, const Nd4jLong *hXShapeInfo,
                                          const void *dX, const Nd4jLong *dXShapeInfo,
                                          const void *hY, const Nd4jLong *hYShapeInfo,
                                          const void *dY, const Nd4jLong *dYShapeInfo,
                                          void *hZ, const Nd4jLong *hZShapeInfo,
                                          void *dZ, const Nd4jLong *dZShapeInfo,
                                          void *extraParams);

    static void execPairwiseIntTransform(sd::LaunchContext  *lc,
                                         int opNum,
                                         const void *hX, const Nd4jLong *hXShapeInfo,
                                         const void *dX, const Nd4jLong *dXShapeInfo,
                                         const void *hY, const Nd4jLong *hYShapeInfo,
                                         const void *dY, const Nd4jLong *dYShapeInfo,
                                         void *hZ, const Nd4jLong *hZShapeInfo,
                                         void *dZ, const Nd4jLong *dZShapeInfo,
                                         void *extraParams);

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
    static void execTransformFloat(sd::LaunchContext  *lc,
                                   int opNum,
                                   const void *hX, const Nd4jLong *hXShapeInfo,
                                   const void *dX, const Nd4jLong *dXShapeInfo,
                                   void *hZ, const Nd4jLong *hZShapeInfo,
                                   void *dZ, const Nd4jLong *dZShapeInfo,
                                   void *extraParams,
                                   const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets);

static void execTransformAny(sd::LaunchContext  *lc,
                             int opNum,
                             const void *hX, const Nd4jLong *hXShapeInfo,
                             const void *dX, const Nd4jLong *dXShapeInfo,
                             void *hZ, const Nd4jLong *hZShapeInfo,
                             void *dZ, const Nd4jLong *dZShapeInfo,
                             void *extraParams,
                             const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                             bool allowParallelism = true);

static void execTransformStrict(sd::LaunchContext  *lc,
                                int opNum,
                                const void *hX, const Nd4jLong *hXShapeInfo,
                                const void *dX, const Nd4jLong *dXShapeInfo,
                                void *hZ, const Nd4jLong *hZShapeInfo,
                                void *dZ, const Nd4jLong *dZShapeInfo,
                                void *extraParams,
                                const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets);

static void execTransformSame(sd::LaunchContext  *lc,
                              int opNum,
                              const void *hX, const Nd4jLong *hXShapeInfo,
                              const void *dX, const Nd4jLong *dXShapeInfo,
                              void *hZ, const Nd4jLong *hZShapeInfo,
                              void *dZ, const Nd4jLong *dZShapeInfo,
                              void *extraParams,
                              const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets);

static void execTransformBool(sd::LaunchContext  *lc,
                              int opNum,
                              const void *hX, const Nd4jLong *hXShapeInfo,
                              const void *dX, const Nd4jLong *dXShapeInfo,
                              void *hZ, const Nd4jLong *hZShapeInfo,
                              void *dZ, const Nd4jLong *dZShapeInfo,
                              void *extraParams,
                              const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    static void execReduceFloat(sd::LaunchContext  *lc,
                                int opNum,
                                const void *hX, const Nd4jLong *hXShapeInfo,
                                const void *dX, const Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, const Nd4jLong *hZShapeInfo,
                                void *dZ, const Nd4jLong *dZShapeInfo,
                                int *dimension, int dimensionLength);

    static void execReduceSame(sd::LaunchContext  *lc,
                               int opNum,
                               const void *hX, const Nd4jLong *hXShapeInfo,
                               const void *dX, const Nd4jLong *dXShapeInfo,
                               void *extraParams,
                               void *hZ, const Nd4jLong *hZShapeInfo,
                               void *dZ, const Nd4jLong *dZShapeInfo,
                               int *dimension, int dimensionLength);

    static void execReduceBool(sd::LaunchContext  *lc,
                               int opNum,
                               const void *hX, const Nd4jLong *hXShapeInfo,
                               const void *dX, const Nd4jLong *dXShapeInfo,
                               void *extraParams,
                               void *hZ, const Nd4jLong *hZShapeInfo,
                               void *dZ, const Nd4jLong *dZShapeInfo,
                               int *dimension, int dimensionLength);

    static void execReduceLong(sd::LaunchContext  *lc,
                               int opNum,
                               const void *hX, const Nd4jLong *hXShapeInfo,
                               const void *dX, const Nd4jLong *dXShapeInfo,
                               void *extraParams,
                               void *hZ, const Nd4jLong *hZShapeInfo,
                               void *dZ, const Nd4jLong *dZShapeInfo,
                               int *dimension, int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    static void execReduceFloatScalar(sd::LaunchContext  *lc,
                                      int opNum,
                                      const void *hX, const Nd4jLong *hXShapeInfo,
                                      const void *dX, const Nd4jLong *dXShapeInfo,
                                      void *extraParams,
                                      void *hZ, const Nd4jLong *hZShapeInfo,
                                      void *dZ, const Nd4jLong *dZShapeInfo);

    static void execReduceBoolScalar(sd::LaunchContext  *lc,
                                     int opNum,
                                     const void *hX, const Nd4jLong *hXShapeInfo,
                                     const void *dX, const Nd4jLong *dXShapeInfo,
                                     void *extraParams,
                                     void *hZ, const Nd4jLong *hZShapeInfo,
                                     void *dZ, const Nd4jLong *dZShapeInfo);

    static void execReduceSameScalar(sd::LaunchContext  *lc,
                                     int opNum,
                                     const void *hX, const Nd4jLong *hXShapeInfo,
                                     const void *dX, const Nd4jLong *dXShapeInfo,
                                     void *extraParams,
                                     void *hZ, const Nd4jLong *hZShapeInfo,
                                     void *dZ, const Nd4jLong *dZShapeInfo);

    static void execReduceLongScalar(sd::LaunchContext  *lc,
                                     int opNum,
                                     const void *hX, const Nd4jLong *hXShapeInfo,
                                     const void *dX, const Nd4jLong *dXShapeInfo,
                                     void *extraParams,
                                     void *hZ, const Nd4jLong *hZShapeInfo,
                                     void *dZ, const Nd4jLong *dZShapeInfo);

    static void execReduce3TAD(sd::LaunchContext  *lc,
                               int opNum,
                               const void *hX, const Nd4jLong *hXShapeInfo,
                               const void *dX, const Nd4jLong *dXShapeInfo,
                               void *extraParamsVals,
                               const void *hY, const Nd4jLong *hYShapeInfo,
                               const void *dY, const Nd4jLong *dYShapeInfo,
                               void *hZ, const Nd4jLong *hZShapeInfo,
                               void *dZ, const Nd4jLong *dZShapeInfo,
                               int *dimension, int dimensionLength,
                               const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                               const Nd4jLong *yTadShapeInfo, const Nd4jLong *yTadOffsets);

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
    static void execSummaryStats(sd::LaunchContext  *lc,
                                 int opNum,
                                 const void *hX, const Nd4jLong *hXShapeInfo,
                                 const void *dX, const Nd4jLong *dXShapeInfo,
                                 void *extraParams,
                                 void *hZ, const Nd4jLong *hZShapeInfo,
                                 void *dZ, const Nd4jLong *dZShapeInfo,
                                 int *dimension, int dimensionLength,
                                 const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
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
    static void execSummaryStats(sd::LaunchContext  *lc,
                                 int opNum,
                                 const void *hX, const Nd4jLong *hXShapeInfo,
                                 const void *dX, const Nd4jLong *dXShapeInfo,
                                 void *extraParams,
                                 void *hZ, const Nd4jLong *hZShapeInfo,
                                 void *dZ, const Nd4jLong *dZShapeInfo,
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
    static void execSummaryStatsScalar(sd::LaunchContext  *lc,
                                       int opNum,
                                       const void *hX, const Nd4jLong *hXShapeInfo,
                                       const void *dX, const Nd4jLong *dXShapeInfo,
                                       void *extraParams,
                                       void *hZ, const Nd4jLong *hZShapeInfo,
                                       void *dZ, const Nd4jLong *dZShapeInfo,
                                       bool biasCorrected);


    static void execRandom(sd::LaunchContext  *lc,
                           int opNum,
                           Nd4jPointer state,
                           void *hZ, const Nd4jLong *hZShapeBuffer,
                           void *dZ, const Nd4jLong *dZShapeBuffer,
                           void *extraArguments);

    static void execRandom(sd::LaunchContext  *lc,
                           int opNum,
                           Nd4jPointer state,
                           const void *hX, const Nd4jLong *hXShapeBuffer,
                           const void *dX, const Nd4jLong *dXShapeBuffer,
                           void *hZ, const Nd4jLong *hZShapeBuffer,
                           void *dZ, const Nd4jLong *dZShapeBuffer,
                           void *extraArguments);

    static void execRandom(sd::LaunchContext  *lc,
                           int opNum,
                           Nd4jPointer state,
                           const void *hX, const Nd4jLong *hXShapeBuffer,
                           const void *dX, const Nd4jLong *dXShapeBuffer,
                           const void *hY, const Nd4jLong *hYShapeBuffer,
                           const void *dY, const Nd4jLong *dYShapeBuffer,
                           void *hZ, const Nd4jLong *hZShapeBuffer,
                           void *dZ, const Nd4jLong *dZShapeBuffer,
                           void *extraArguments);



    inline static void execSort(void *x, const Nd4jLong *xShapeInfo, bool descending) {
        auto xType = sd::ArrayOptions::dataType(xShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::sortGeneric(x, xShapeInfo, descending), LIBND4J_TYPES);
    }

    static void execSort(void *x, const Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets, bool descending) {
        auto xType = sd::ArrayOptions::dataType(xShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::sortTadGeneric(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending), LIBND4J_TYPES);
    }

    inline static void execSortCooIndices(Nd4jLong *indices, void *x, Nd4jLong length, const Nd4jLong *xShapeInfo) {
        auto xType = sd::ArrayOptions::dataType(xShapeInfo);
        int rank = shape::rank(xShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, sd::sparse::SparseUtils, ::sortCooIndicesGeneric(indices, x, length, rank), LIBND4J_TYPES);
    }


    inline static Nd4jLong encodeBitmap(void *dx, const Nd4jLong *xShapeInfo, Nd4jLong N, int *dz, float threshold) {
        auto xType = sd::ArrayOptions::dataType(xShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, return sd::SpecialMethods, ::encodeBitmapGeneric(dx, xShapeInfo, N, dz, threshold), FLOAT_TYPES);
    }

    inline static void decodeBitmap(const void *dx, Nd4jLong N, void *dz, const Nd4jLong *zShapeInfo) {
        auto zType = sd::ArrayOptions::dataType(zShapeInfo);

        BUILD_SINGLE_SELECTOR(zType, sd::SpecialMethods, ::decodeBitmapGeneric(dx, N, dz, zShapeInfo), FLOAT_TYPES);
    }

};


#endif //NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
