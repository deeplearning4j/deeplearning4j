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
#include <dll.h>
#include <loops/aggregates.h>
#include <ops/specials.h>
#include <ops/specials_sparse.h>
#include <execution/LaunchContext.h>

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
    static void execIndexReduceScalar(nd4j::LaunchContext  *lc,
                                    int opNum, 
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *extraParams,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo);

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
    static void execReduce3Scalar(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParamsVals,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo);
    

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
    static void execReduce3(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParamsVals,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo);

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
    static void execReduce3(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParamsVals,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *xTadOnlyShapeInfo, Nd4jLong *xTadOffsets,
                            Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets);

    static void execReduce3All(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParamsVals,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets,
                            Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets);

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
    static void execIndexReduce(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                int *dimension, int dimensionLength,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

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
    static void execScalar(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            void *hScalar, Nd4jLong *hSscalarShapeInfo,
                            void *dScalar, Nd4jLong *dSscalarShapeInfo,
                            void *extraParams);

static void execScalarBool(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            void *hScalar, Nd4jLong *hSscalarShapeInfo,
                            void *dScalar, Nd4jLong *dSscalarShapeInfo,
                            void *extraParams);

 static void execScalar(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            void *hScalars, Nd4jLong *hScalarShapeInfo,
                            void *dScalars, Nd4jLong *dScalarShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                            Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ);

 static void execScalarBool(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            void *hScalars, Nd4jLong *hScalarShapeInfo,
                            void *dScalars, Nd4jLong *dScalarShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                            Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ);

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
    static void execBroadcast(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
                            Nd4jLong *tadOnlyShapeInfoZ,Nd4jLong *tadOffsetsZ);

    static void execInverseBroadcast(nd4j::LaunchContext  *lc,
                                     int opNum,
                                     void *x, Nd4jLong *xShapeInfo,
                                     void *dX, Nd4jLong *dXShapeInfo,
                                     void *y, Nd4jLong *yShapeInfo,
                                     void *dY, Nd4jLong *dYShapeInfo,
                                     void *result, Nd4jLong *resultShapeInfo,
                                     void *dZ, Nd4jLong *dZShapeInfo,
                                     int *dimension, int dimensionLength,
                                     Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
                                     Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ);


    static void execBroadcastBool(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
                            Nd4jLong *tadOnlyShapeInfoZ,Nd4jLong *tadOffsetsZ);

    static void execInverseBroadcastBool(nd4j::LaunchContext  *lc,
                                     int opNum,
                                     void *x, Nd4jLong *xShapeInfo,
                                     void *dX, Nd4jLong *dXShapeInfo,
                                     void *y, Nd4jLong *yShapeInfo,
                                     void *dY, Nd4jLong *dYShapeInfo,
                                     void *result, Nd4jLong *resultShapeInfo,
                                     void *dZ, Nd4jLong *dZShapeInfo,
                                     int *dimension, int dimensionLength,
                                     Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets,
                                     Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ);
    
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
    static void execPairwiseTransform(nd4j::LaunchContext  *lc,
                                    int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *hY, Nd4jLong *hYShapeInfo,
                                    void *dY, Nd4jLong *dYShapeInfo,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo,
                                    void *extraParams);

    static void execPairwiseBoolTransform(nd4j::LaunchContext  *lc,
                                    int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *hY, Nd4jLong *hYShapeInfo,
                                    void *dY, Nd4jLong *dYShapeInfo,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo,
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
    static void execTransformFloat(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                void *extraParams,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

static void execTransformAny(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                void *extraParams,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

static void execTransformStrict(nd4j::LaunchContext  *lc,
                                    int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo,
                                    void *extraParams,
                                    Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

static void execTransformSame(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                void *extraParams,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

static void execTransformBool(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                void *extraParams,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);    
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    static void execReduceFloat(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParams,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

    static void execReduceSame(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                int *dimension, int dimensionLength,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

    static void execReduceBool(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                int *dimension, int dimensionLength,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

    static void execReduceLong(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                int *dimension, int dimensionLength,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    static void execReduceFloatScalar(nd4j::LaunchContext  *lc,
                                    int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *extraParams,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo);

    static void execReduceBoolScalar(nd4j::LaunchContext  *lc,
                                        int opNum,
                                        void *hX, Nd4jLong *hXShapeInfo,
                                        void *dX, Nd4jLong *dXShapeInfo,
                                        void *extraParams,
                                        void *hZ, Nd4jLong *hZShapeInfo,
                                        void *dZ, Nd4jLong *dZShapeInfo);

    static void execReduceSameScalar(nd4j::LaunchContext  *lc,
                                        int opNum,
                                        void *hX, Nd4jLong *hXShapeInfo,
                                        void *dX, Nd4jLong *dXShapeInfo,
                                        void *extraParams,
                                        void *hZ, Nd4jLong *hZShapeInfo,
                                        void *dZ, Nd4jLong *dZShapeInfo);

    static void execReduceLongScalar(nd4j::LaunchContext  *lc,
                                    int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *extraParams,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo);    

    static void execReduce3TAD(nd4j::LaunchContext  *lc,
                            int opNum,
                            void *hX, Nd4jLong *hXShapeInfo,
                            void *dX, Nd4jLong *dXShapeInfo,
                            void *extraParamsVals,
                            void *hY, Nd4jLong *hYShapeInfo,
                            void *dY, Nd4jLong *dYShapeInfo,
                            void *hZ, Nd4jLong *hZShapeInfo,
                            void *dZ, Nd4jLong *dZShapeInfo,
                            int *dimension, int dimensionLength, 
                            Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                            Nd4jLong *yTadShapeInfo, Nd4jLong *yTadOffsets);

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
    static void execSummaryStats(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
                                int *dimension, int dimensionLength,
                                Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
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
    static void execSummaryStats(nd4j::LaunchContext  *lc,
                                int opNum,
                                void *hX, Nd4jLong *hXShapeInfo,
                                void *dX, Nd4jLong *dXShapeInfo,
                                void *extraParams,
                                void *hZ, Nd4jLong *hZShapeInfo,
                                void *dZ, Nd4jLong *dZShapeInfo,
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
    static void execSummaryStatsScalar(nd4j::LaunchContext  *lc,
                                    int opNum,
                                    void *hX, Nd4jLong *hXShapeInfo,
                                    void *dX, Nd4jLong *dXShapeInfo,
                                    void *extraParams,
                                    void *hZ, Nd4jLong *hZShapeInfo,
                                    void *dZ, Nd4jLong *dZShapeInfo,
                                    bool biasCorrected);   


    static void execRandom(nd4j::LaunchContext  *lc,
                            int opNum,
                            Nd4jPointer state,
                            void *hZ, Nd4jLong *hZShapeBuffer,
                            void *dZ, Nd4jLong *dZShapeBuffer,
                            void *extraArguments);

    static void execRandom(nd4j::LaunchContext  *lc,
                            int opNum,
                            Nd4jPointer state,
                            void *hX, Nd4jLong *hXShapeBuffer,
                            void *dX, Nd4jLong *dXShapeBuffer,
                            void *hZ, Nd4jLong *hZShapeBuffer,
                            void *dZ, Nd4jLong *dZShapeBuffer,
                            void *extraArguments);

    static void execRandom(nd4j::LaunchContext  *lc,
                          int opNum,
                          Nd4jPointer state,
                          void *hX, Nd4jLong *hXShapeBuffer,
                          void *dX, Nd4jLong *dXShapeBuffer,
                          void *hY, Nd4jLong *hYShapeBuffer,
                          void *dY, Nd4jLong *dYShapeBuffer,
                          void *hZ, Nd4jLong *hZShapeBuffer,
                          void *dZ, Nd4jLong *dZShapeBuffer,
                          void *extraArguments);



    template <typename X>
    static FORCEINLINE void execAggregate(nd4j::LaunchContext  *lc,
                              int opNum,
                              void **varguments,
                              int numArguments,
                              Nd4jLong **shapeArguments,
                              int numShapeArguments,
                              int *indexArguments,
                              int numIndexArguments,
                              int **intArrays,
                              int numIntArrays,
                              void *vrealArguments,
                              int numRealArguments) {

        auto arguments = reinterpret_cast<X **>(varguments);
        auto realArguments = reinterpret_cast<X *>(vrealArguments);

        functions::aggregate::AggregatedFunction<X>::exec(opNum, arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
    }
    

    inline static void execSort(void *x, Nd4jLong *xShapeInfo, bool descending) {
        auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, nd4j::SpecialMethods, ::sortGeneric(x, xShapeInfo, descending), LIBND4J_TYPES);
    }

    static void execSort(void *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
        auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, nd4j::SpecialMethods, ::sortTadGeneric(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending), LIBND4J_TYPES);
    }

    inline static void execSortCooIndices(Nd4jLong *indices, void *values, Nd4jLong length, int rank) {
        nd4j::sparse::SparseUtils<Nd4jLong>::sortCooIndicesGeneric(indices, reinterpret_cast<Nd4jLong *>(values), length, rank);
    }


    inline static Nd4jLong encodeBitmap(void *dx, Nd4jLong *xShapeInfo, Nd4jLong N, int *dz, float threshold) {
        auto xType = nd4j::ArrayOptions::dataType(xShapeInfo);

        BUILD_SINGLE_SELECTOR(xType, return nd4j::SpecialMethods, ::encodeBitmapGeneric(dx, xShapeInfo, N, dz, threshold), FLOAT_TYPES);
    }

    inline static void decodeBitmap(void *dx, Nd4jLong N, void *dz, Nd4jLong *zShapeInfo) {
        auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

        BUILD_SINGLE_SELECTOR(zType, nd4j::SpecialMethods, ::decodeBitmapGeneric(dx, N, dz, zShapeInfo), FLOAT_TYPES);
    }

};


#endif //NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
