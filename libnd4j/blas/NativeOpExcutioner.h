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

#include <loops/broadcasting.h>
#include <loops/indexreduce.h>
#include <loops/pairwise_transform.h>
#include <loops/reduce_float.h>
#include <loops/reduce3.h>
#include <loops/summarystatsreduce.h>
#include <loops/transform_same.h>
#include <loops/scalar.h>
#include <loops/aggregates.h>
#include <loops/random.h>
#include <pointercast.h>
#include <ops/specials.h>
#include <ops/specials_sparse.h>
#include <types/types.h>
#include <dll.h>

/**
 * Native op executioner:
 *
 */

class ND4J_EXPORT NativeOpExcutioner {
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
    static Nd4jLong execIndexReduceScalar(int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo);

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
    static void execIndexReduce(int opNum,
                                void *x,
                                Nd4jLong *xShapeInfo,
                                void *extraParams,
                                Nd4jLong *result,
                                Nd4jLong *resultShapeInfoBuffer,
                                int *dimension,
                                int dimensionLength,
                                Nd4jLong *tadShapeInfo,
                                Nd4jLong *tadOffsets);

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
    static void execBroadcast(int opNum,
                              void *x,
                              Nd4jLong *xShapeInfo,
                              void *y,
                              Nd4jLong *yShapeInfo,
                              void *result,
                              Nd4jLong *resultShapeInfo,
                              int *dimension,
                              int dimensionLength,
                              Nd4jLong *tadOnlyShapeInfo,
                              Nd4jLong *tadOffsets,
                              Nd4jLong *tadOnlyShapeInfoZ,
                              Nd4jLong *tadOffsetsZ);


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
    static void execPairwiseTransform(int opNum,
                                      void *dx,
                                      Nd4jLong *xShapeInfo,
                                      void *y,
                                      Nd4jLong *yShapeInfo,
                                      void *result,
                                      Nd4jLong *resultShapeInfo,
                                      void *extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    static void execReduceFloat(int opNum,
                           void *x,
                           Nd4jLong *xShapeInfo,
                           void *extraParams,
                           void *result,
                           Nd4jLong *resultShapeInfo,
                           int *dimension,
                           int dimensionLength,
                           Nd4jLong *tadShapeInfo,
                           Nd4jLong *tadOffsets);

    static void execReduceSame(int opNum,
                                void *x,
                                Nd4jLong *xShapeInfo,
                                void *extraParams,
                                void *result,
                                Nd4jLong *resultShapeInfo,
                                int *dimension,
                                int dimensionLength,
                                Nd4jLong *tadShapeInfo,
                                Nd4jLong *tadOffsets);

    static void execReduceBool(int opNum,
                                void *x,
                                Nd4jLong *xShapeInfo,
                                void *extraParams,
                                void *result,
                                Nd4jLong *resultShapeInfo,
                                int *dimension,
                                int dimensionLength,
                                Nd4jLong *tadShapeInfo,
                                Nd4jLong *tadOffsets);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    static void execReduceFloatScalar(int opNum,
                              void *x,
                              Nd4jLong *xShapeInfo,
                              void *extraParams,
                              void *z,
                              Nd4jLong *zShapeInfo);

    static void execReduceBoolScalar(int opNum,
                                      void *x,
                                      Nd4jLong *xShapeInfo,
                                      void *extraParams,
                                      void *z,
                                      Nd4jLong *zShapeInfo);

    static void execReduceSameScalar(int opNum,
                                      void *x,
                                      Nd4jLong *xShapeInfo,
                                      void *extraParams,
                                      void *z,
                                      Nd4jLong *zShapeInfo);

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
    static void execReduce3(int opNum,
                            void *x,
                            Nd4jLong *xShapeInfo,
                            void *extraParamsVals,
                            void *y,
                            Nd4jLong *yShapeInfo,
                            void *result,
                            Nd4jLong *resultShapeInfo);    


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
    static void execReduce3Scalar(int opNum,
                               void *x,
                               Nd4jLong *xShapeInfo,
                               void *extraParamsVals,
                               void *y,
                               Nd4jLong *yShapeInfo,
                               void *z,
                               Nd4jLong *zShapeInfo);

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
    static void execReduce3(int opNum,
                            void *x,
                            Nd4jLong *xShapeInfo,
                            void *extraParamsVals,
                            void *y,
                            Nd4jLong *yShapeInfo,
                            void *result,
                            Nd4jLong *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength);

    static void execReduce3All(int opNum,
                            void *x,
                            Nd4jLong *xShapeInfo,
                            void *extraParamsVals,
                            void *y,
                            Nd4jLong *yShapeInfo,
                            void *result,
                            Nd4jLong *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength,
                            Nd4jLong *xTadShapeInfo,
                            Nd4jLong *xOffsets,
                            Nd4jLong *yTadShapeInfo,
                            Nd4jLong *yOffsets);

    static void execReduce3TAD(int opNum,
                            void *x,
                            Nd4jLong *xShapeInfo,
                            void *extraParamsVals,
                            void *y,
                            Nd4jLong *yShapeInfo,
                            void *result,
                            Nd4jLong *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength, 
                            Nd4jLong *tadShapeInfo, 
                            Nd4jLong *tadOffsets);


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
    static void execScalar(int opNum,
                           void *x,
                           Nd4jLong *xShapeInfo,
                           void *result,
                           Nd4jLong *resultShapeInfo,
                           void *scalar,
                           Nd4jLong *scalarShapeInfo,
                           void *extraParams);


    static void execScalar(int opNum,
                           void *x,
                           Nd4jLong *xShapeInfo,
                           void *extraParams,
                           void *z,
                           Nd4jLong *zShapeInfo,
                           void *scalars,
                           Nd4jLong *scalarShapeInfo,
                           int *dimension,
                           int dimensionLength,
                           Nd4jLong *tadShapeInfo,
                           Nd4jLong *tadOffsets,
                           Nd4jLong *tadShapeInfoZ,
                           Nd4jLong *tadOffsetsZ);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    static void execSummaryStats(int opNum,
                                 void *x,
                                 Nd4jLong *xShapeInfo,
                                 void *extraParams,
                                 void *result,
                                 Nd4jLong *resultShapeInfo,
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
    static double execSummaryStatsScalar(int opNum,
                                    void *x,
                                    Nd4jLong *xShapeInfo,
                                    void *extraParams,
                                    bool biasCorrected);

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
    static void execSummaryStats(int opNum,
                                 void *x,
                                 Nd4jLong *xShapeInfo,
                                 void *extraParams,
                                 void *result,
                                 Nd4jLong *resultShapeInfoBuffer,
                                 int *dimension,
                                 int dimensionLength,
                                 bool biasCorrected);

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
    static void execTransform(int opNum,
                              void *dx,
                              Nd4jLong *xShapeInfo,
                              void *result,
                              Nd4jLong *resultShapeInfo,
                              void *extraParams,
                              Nd4jLong *tadShapeInfo,
                              Nd4jLong *tadOffsets);
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
    static void execTransform(int opNum,
                              void *dx,
                              Nd4jLong *xShapeInfo,
                              void *result,
                              Nd4jLong *resultShapeInfo,
                              void *extraParams,
                              Nd4jLong *xIndexes,
                              Nd4jLong *resultIndexes,
                              Nd4jLong *tadShapeInfo,
                              Nd4jLong *tadOffsets);

    template <typename X>
    static FORCEINLINE void execAggregate(int opNum,
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

    static void execRandom(int opNum,
                           Nd4jPointer state,
                           void *z,
                           Nd4jLong *zShapeBuffer, 
                           void *extraArguments);

    static void execRandom(int opNum,
                           Nd4jPointer state,
                           void *x,
                           Nd4jLong *xShapeBuffer,
                           void *z,
                           Nd4jLong *zShapeBuffer, 
                           void *extraArguments);

    static void execRandom(int opNum,
                           Nd4jPointer state,
                           void *x,
                           Nd4jLong *xShapeBuffer,
                           void *y,
                           Nd4jLong *yShapeBuffer,
                           void *z,
                           Nd4jLong *zShapeBuffer,
                           void *extraArguments);

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

        BUILD_SINGLE_SELECTOR(xType, nd4j::SpecialMethods, ::encodeBitmapGeneric(dx, xShapeInfo, N, dz, threshold), FLOAT_TYPES);
    }

    inline static void decodeBitmap(void *dx, Nd4jLong N, void *dz, Nd4jLong *zShapeInfo) {
        auto zType = nd4j::ArrayOptions::dataType(zShapeInfo);

        BUILD_SINGLE_SELECTOR(zType, nd4j::SpecialMethods, ::decodeBitmapGeneric(dx, N, dz, zShapeInfo), FLOAT_TYPES);
    }

};


#endif //NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
