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
// Created by agibsonccc on 2/21/16.
//

#ifndef NATIVEOPERATIONS_NATIVEOPS_H
#define NATIVEOPERATIONS_NATIVEOPS_H


#ifndef thread_local
# if __STDC_VERSION__ >= 201112 && !defined __STDC_NO_THREADS__
#  define thread_local _Thread_local
# elif defined _WIN32 && ( \
       defined _MSC_VER || \
       defined __ICL || \
       defined __DMC__ || \
       defined __BORLANDC__ )
#  define thread_local __declspec(thread)
/* note that ICC (linux) and Clang are covered by __GNUC__ */
# elif defined __GNUC__ || \
       defined __SUNPRO_C || \
       defined __xlC__
#  define thread_local __thread
# else
#  error "Cannot define thread_local"
# endif
#endif

#include <pointercast.h>
#include <types/float16.h>
#include <cnpy.h>

//DO NOT REMOVE: THIS IS AN EDITOR SEMANTICS THING FOR CLION
//IT DEFINES THE EXPORT MACRO FOR THE EDITOR AND THEN
//RE ADDS THE DEFINITION VIA dll.h
#ifdef  _WIN32
#define ND4J_EXPORT __declspec(dllexport)
#else
#define ND4J_EXPORT
#endif
#include <dll.h>
#include <helpers/BlasHelper.h>

/*
int tad_threshold = 1;
int element_threshold = 32;

bool debug = false;
bool verbose = false;
*/

#include <array/ShapeList.h>
#include <graph/VariablesSet.h>
#include <graph/GraphState.h>
#include <graph/execution/LogicExecutor.h>
#include <graph/ResultWrapper.h>

class ND4J_EXPORT NativeOps {

public:


    /**
     *
     * @param num
     */
    void setElementThreshold(int num);

    /**
     *
     * @param num
     */
    void setTADThreshold(int num);

    /**
       *
       * @param opNum
       * @param x
       * @param xShapeInfo
       * @param extraParams
       */
    double   execIndexReduceScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                         double *x,
                                         Nd4jLong *xInfo,
                                         double *extraParams);

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
    void   execIndexReduceDouble(Nd4jPointer *extraPointers,int opNum,
                                 double *x,
                                 Nd4jLong *xInfo,
                                 double *extraParams,
                                 double *result,
                                 Nd4jLong *resultShapeInfoBuffer,
                                 int *dimension, int dimensionLength);
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
    void   execBroadcastDouble(
            Nd4jPointer *extraPointers,
            int opNum,
            double *x,
            Nd4jLong *xInfo,
            double *y,
            Nd4jLong *yInfo,
            double *result,
            Nd4jLong *resultShapeInfo,
            int *dimension,
            int dimensionLength);



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
    void   execPairwiseTransformDouble(Nd4jPointer *extraPointers,
                                       int opNum,
                                       double *dx,
                                       Nd4jLong xStride,
                                       double *y,
                                       Nd4jLong yStride,
                                       double *result,
                                       Nd4jLong resultStride,
                                       double *extraParams,
                                       Nd4jLong n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    void execPairwiseTransformDouble(Nd4jPointer *extraPointers,
                                     int opNum,
                                     double *dx,
                                     Nd4jLong *xInfo,
                                     double *y,
                                     Nd4jLong *yInfo,
                                     double *result,
                                     Nd4jLong *resultShapeInfo,
                                     double *extraParams,
                                     Nd4jLong *xIndexes,
                                     Nd4jLong *yIndexes,
                                     Nd4jLong *resultIndexes);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void execPairwiseTransformDouble(
            Nd4jPointer *extraPointers,
            int opNum,
            double *dx,
            Nd4jLong *xShapeInfo,
            double *y,
            Nd4jLong *yShapeInfo,
            double *result,
            Nd4jLong *resultShapeInfo,
            double *extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceDouble(Nd4jPointer *extraPointers,int opNum,
                            double *x,
                            Nd4jLong *xInfo,
                            double *extraParams,
                            double *result,
                            Nd4jLong *resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceDouble(Nd4jPointer *extraPointers,
                            int opNum,
                            double *x,
                            Nd4jLong *xInfo,
                            double *extraParams,
                            double *result,
                            Nd4jLong *resultShapeInfo,
                            int *dimension,
                            int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    double execReduceScalarDouble(Nd4jPointer *extraPointers,
                                  int opNum,
                                  double *x,
                                  Nd4jLong *xInfo,
                                  double *extraParams);

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
    void   execReduce3Double(Nd4jPointer *extraPointers,
                             int opNum,
                             double *x,
                             Nd4jLong *xInfo,
                             double *extraParamsVals,
                             double *y,
                             Nd4jLong *yInfo,
                             double *result,
                             Nd4jLong *resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    double   execReduce3ScalarDouble(Nd4jPointer *extraPointers,
                                     int opNum,
                                     double *x,
                                     Nd4jLong *xInfo,
                                     double *extraParamsVals,
                                     double *y,
                                     Nd4jLong *yInfo);
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
    void   execReduce3Double(Nd4jPointer *extraPointers,
                             int opNum,
                             double *x,
                             Nd4jLong *xInfo,
                             double *extraParamsVals,
                             double *y,
                             Nd4jLong *yInfo,
                             double *result,
                             Nd4jLong *resultShapeInfoBuffer,
                             int *dimension,
                             int dimensionLength);

    void execReduce3AllDouble(Nd4jPointer *extraPointers,
                             int opNum,
                             double *x,
                             Nd4jLong *xInfo,
                             double *extraParamsVals,
                             double *y,
                             Nd4jLong *yInfo,
                             double *result,
                             Nd4jLong *resultShapeInfoBuffer,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *xTadShapeInfo,
                             Nd4jLong *xOffsets,
                             Nd4jLong *yTadShapeInfo,
                             Nd4jLong *yOffsets);

    void execReduce3AllFloat(Nd4jPointer *extraPointers,
                              int opNum,
                              float *x,
                              Nd4jLong *xInfo,
                              float *extraParamsVals,
                              float *y,
                              Nd4jLong *yInfo,
                              float *result,
                              Nd4jLong *resultShapeInfoBuffer,
                              int *dimension,
                              int dimensionLength,
                              Nd4jLong *xTadShapeInfo,
                              Nd4jLong *xOffsets,
                              Nd4jLong *yTadShapeInfo,
                              Nd4jLong *yOffsets);

    void execReduce3AllHalf(Nd4jPointer *extraPointers,
                              int opNum,
                              float16 *x,
                              Nd4jLong *xInfo,
                              float16 *extraParamsVals,
                              float16 *y,
                              Nd4jLong *yInfo,
                              float16 *result,
                              Nd4jLong *resultShapeInfoBuffer,
                              int *dimension,
                              int dimensionLength,
                              Nd4jLong *xTadShapeInfo,
                              Nd4jLong *xOffsets,
                              Nd4jLong *yTadShapeInfo,
                              Nd4jLong *yOffsets);





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
    void   execScalarDouble(Nd4jPointer *extraPointers,
                            int opNum,
                            double *x,
                            Nd4jLong xStride,
                            double *result,
                            Nd4jLong resultStride,
                            double scalar,
                            double *extraParams,
                            Nd4jLong n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param n
     */
    void execScalarDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          double *x,
                          Nd4jLong *xInfo,
                          double *result,
                          Nd4jLong *resultShapeInfo,
                          double scalar,
                          double *extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param n
     * @param xIndexes
     * @param resultIndexes
     */
    void execScalarDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          double *x,
                          Nd4jLong *xInfo,
                          double *result,
                          Nd4jLong *resultShapeInfo,
                          double scalar,
                          double *extraParams,
                          Nd4jLong n,
                          Nd4jLong *xIndexes,
                          Nd4jLong *resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    double   execSummaryStatsScalarDouble(Nd4jPointer *extraPointers,
                                          int opNum,
                                          double *x,
                                          Nd4jLong *xInfo,
                                          double *extraParams,
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
    void   execSummaryStatsDouble(Nd4jPointer *extraPointers,
                                  int opNum,
                                  double *x,
                                  Nd4jLong *xInfo,
                                  double *extraParams,
                                  double *result,
                                  Nd4jLong *resultShapeInfo,
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
    void   execSummaryStatsDouble(Nd4jPointer *extraPointers,
                                  int opNum,
                                  double *x,
                                  Nd4jLong *xInfo,
                                  double *extraParams,
                                  double *result,
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
    void   execTransformDouble(Nd4jPointer *extraPointers,int opNum,
                               double *dx,
                               Nd4jLong xStride,
                               double *result,
                               Nd4jLong resultStride,
                               double *extraParams,
                               Nd4jLong n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void   execTransformDouble(Nd4jPointer *extraPointers,int opNum,
                               double *dx,
                               Nd4jLong *xInfo,
                               double *result,
                               Nd4jLong *resultShapeInfo,
                               double *extraParams);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void   execTransformDouble(Nd4jPointer *extraPointers,
                               int opNum,
                               double* dx,
                               Nd4jLong *xShapeInfo,
                               double* result,
                               Nd4jLong *resultShapeInfo,
                               double* extraParams,
                               Nd4jLong *xIndexes,
                               Nd4jLong *resultIndexes);

    /**
    *
    * @param opNum
    * @param x
    * @param xShapeInfo
    * @param extraParams
    */
    float   execIndexReduceScalarFloat(Nd4jPointer *extraPointers,
                                       int opNum,
                                       float *x,
                                       Nd4jLong *xShapeInfo,
                                       float *extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    float execIndexReduceScalarHalf(Nd4jPointer *extraPointers,
                                    int opNum,
                                    float16 *x,
                                    Nd4jLong *xShapeInfo,
                                    float16 *extraParams);

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
    void   execIndexReduceFloat(Nd4jPointer *extraPointers,int opNum,
                                float *x,
                                Nd4jLong *xShapeInfo,
                                float *extraParams,
                                float *result,
                                Nd4jLong *resultShapeInfoBuffer,
                                int *dimension, int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    void   execIndexReduceHalf(Nd4jPointer *extraPointers,int opNum,
                               float16 *x,
                               Nd4jLong *xShapeInfo,
                               float16 *extraParams,
                               float16 *result,
                               Nd4jLong *resultShapeInfoBuffer,
                               int *dimension, int dimensionLength);
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
    void   execBroadcastFloat(
            Nd4jPointer *extraPointers,
            int opNum,
            float *x,
            Nd4jLong *xShapeInfo,
            float *y,
            Nd4jLong *yShapeInfo,
            float *result,
            Nd4jLong *resultShapeInfo,
            int *dimension,
            int dimensionLength);

    /**
     *
     * @param extraPointers
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
    void   execBroadcastHalf(
            Nd4jPointer *extraPointers,
            int opNum,
            float16 *x,
            Nd4jLong *xShapeInfo,
            float16 *y,
            Nd4jLong *yShapeInfo,
            float16 *result,
            Nd4jLong *resultShapeInfo,
            int *dimension, int dimensionLength);



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
    void   execPairwiseTransformFloat(Nd4jPointer *extraPointers,
                                      int opNum,
                                      float *dx,
                                      Nd4jLong xStride,
                                      float *y,
                                      Nd4jLong yStride,
                                      float *result,
                                      Nd4jLong resultStride,
                                      float *extraParams,
                                      Nd4jLong n);

    /**
     *
     * @param extraPointers
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
    void   execPairwiseTransformHalf(Nd4jPointer *extraPointers,
                                     int opNum,
                                     float16 *dx,
                                     Nd4jLong xStride,
                                     float16 *y,
                                     Nd4jLong yStride,
                                     float16 *result,
                                     Nd4jLong resultStride,
                                     float16 *extraParams,
                                     Nd4jLong n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    void execPairwiseTransformFloat(Nd4jPointer *extraPointers,
                                    int opNum,
                                    float *dx,
                                    Nd4jLong *xShapeInfo,
                                    float *y,
                                    Nd4jLong *yShapeInfo,
                                    float *result,
                                    Nd4jLong *resultShapeInfo,
                                    float *extraParams,
                                    Nd4jLong *xIndexes,
                                    Nd4jLong *yIndexes,
                                    Nd4jLong *resultIndexes);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    void execPairwiseTransformHalf(Nd4jPointer *extraPointers,
                                   int opNum,
                                   float16 *dx,
                                   Nd4jLong *xShapeInfo,
                                   float16 *y,
                                   Nd4jLong *yShapeInfo,
                                   float16 *result,
                                   Nd4jLong *resultShapeInfo,
                                   float16 *extraParams,
                                   Nd4jLong *xIndexes,
                                   Nd4jLong *yIndexes,
                                   Nd4jLong *resultIndexes);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void execPairwiseTransformFloat(Nd4jPointer *extraPointers,
                                    int opNum,
                                    float *dx,
                                    Nd4jLong *xShapeInfo,
                                    float *y,
                                    Nd4jLong *yShapeInfo,
                                    float *result,
                                    Nd4jLong *resultShapeInfo,
                                    float *extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    void execPairwiseTransformHalf(Nd4jPointer *extraPointers,
                                   int opNum,
                                   float16 *dx,
                                   Nd4jLong *xShapeInfo,
                                   float16 *y,
                                   Nd4jLong *yShapeInfo,
                                   float16 *result,
                                   Nd4jLong *resultShapeInfo,
                                   float16 *extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(Nd4jPointer *extraPointers,
                           int opNum,
                           float *x,
                           Nd4jLong *xShapeInfo,
                           float *extraParams,
                           float *result,
                           Nd4jLong *resultShapeInfo);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceHalf(Nd4jPointer *extraPointers,
                          int opNum,
                          float16 *x,
                          Nd4jLong *xShapeInfo,
                          float16 *extraParams,
                          float16 *result,
                          Nd4jLong *resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(Nd4jPointer *extraPointers,
                           int opNum,
                           float *x,
                           Nd4jLong *xShapeInfo,
                           float *extraParams,
                           float *result,
                           Nd4jLong *resultShapeInfo,
                           int *dimension,
                           int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     * @param dimension
     * @param dimensionLength
     */

    void   execReduceHalf(Nd4jPointer *extraPointers,
                          int opNum,
                          float16 *x,
                          Nd4jLong *xShapeInfo,
                          float16 *extraParams,
                          float16 *result,
                          Nd4jLong *resultShapeInfo,
                          int *dimension,
                          int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    float execReduceScalarFloat(Nd4jPointer *extraPointers,
                                int opNum,
                                float *x,
                                Nd4jLong *xShapeInfo,
                                float *extraParams);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    float execReduceScalarHalf(Nd4jPointer *extraPointers,
                               int opNum,
                               float16 *x,
                               Nd4jLong *xShapeInfo,
                               float16 *extraParams);

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
    void   execReduce3Float(Nd4jPointer *extraPointers,int opNum,
                            float *x,
                            Nd4jLong *xShapeInfo,
                            float *extraParamsVals,
                            float *y,
                            Nd4jLong *yShapeInfo,
                            float *result,
                            Nd4jLong *resultShapeInfo);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param result
     * @param resultShapeInfo
     */
    void   execReduce3Half(Nd4jPointer *extraPointers,int opNum,
                           float16 *x,
                           Nd4jLong *xShapeInfo,
                           float16 *extraParamsVals,
                           float16 *y,
                           Nd4jLong *yShapeInfo,
                           float16 *result,
                           Nd4jLong *resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    float   execReduce3ScalarFloat(Nd4jPointer *extraPointers,
                                   int opNum,
                                   float *x,
                                   Nd4jLong *xShapeInfo,
                                   float *extraParamsVals,
                                   float *y,
                                   Nd4jLong *yShapeInfo);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @return
     */
    float   execReduce3ScalarHalf(Nd4jPointer *extraPointers,
                                  int opNum,
                                  float16 *x,
                                  Nd4jLong *xShapeInfo,
                                  float16 *extraParamsVals,
                                  float16 *y,
                                  Nd4jLong *yShapeInfo);

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
    void   execReduce3Float(Nd4jPointer *extraPointers,
                            int opNum,
                            float *x,
                            Nd4jLong *xShapeInfo,
                            float *extraParamsVals,
                            float *y,
                            Nd4jLong *yShapeInfo,
                            float *result,
                            Nd4jLong *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength);

    /**
     *
     * @param extraPointers
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
    void   execReduce3Half(Nd4jPointer *extraPointers,
                           int opNum,
                           float16 *x,
                           Nd4jLong *xShapeInfo,
                           float16 *extraParamsVals,
                           float16 *y,
                           Nd4jLong *yShapeInfo,
                           float16 *result,
                           Nd4jLong *resultShapeInfoBuffer,
                           int *dimension,
                           int dimensionLength);
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
    void   execScalarFloat(Nd4jPointer *extraPointers,
                           int opNum,
                           float *x,
                           Nd4jLong xStride,
                           float *result,
                           Nd4jLong resultStride,
                           float scalar,
                           float *extraParams,
                           Nd4jLong n);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xStride
     * @param result
     * @param resultStride
     * @param scalar
     * @param extraParams
     * @param n
     */
    void   execScalarHalf(Nd4jPointer *extraPointers,
                          int opNum,
                          float16 *x,
                          Nd4jLong xStride,
                          float16 *result,
                          Nd4jLong resultStride,
                          float scalar,
                          float16 *extraParams,
                          Nd4jLong n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param n
     */
    void execScalarFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         float *x,
                         Nd4jLong *xShapeInfo,
                         float *result,
                         Nd4jLong *resultShapeInfo,
                         float scalar,
                         float *extraParams);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     */
    void execScalarHalf(Nd4jPointer *extraPointers,
                        int opNum,
                        float16 *x,
                        Nd4jLong *xShapeInfo,
                        float16 *result,
                        Nd4jLong *resultShapeInfo,
                        float scalar,
                        float16 *extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param n
     * @param xIndexes
     * @param resultIndexes
     */
    void execScalarFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         float *x,
                         Nd4jLong *xShapeInfo,
                         float *result,
                         Nd4jLong *resultShapeInfo,
                         float scalar,
                         float *extraParams,
                         Nd4jLong *xIndexes,
                         Nd4jLong *resultIndexes);


    /*
     * Special case: scalarOp alang dimension
     */
    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param scalars
     * @param extraParams
     * @param dimension
     * @param dimensionLength
     */
    void execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                         float *x,
                         Nd4jLong *xShapeInfo,
                         float *z,
                         Nd4jLong *zShapeInfo,
                         float *scalars,
                         float *extraParams,
                         int *dimension,
                         int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param scalars
     * @param extraParams
     * @param dimension
     * @param dimensionLength
     */
    void execScalarDouble(Nd4jPointer *extraPointers,int opNum,
                          double *x,
                          Nd4jLong *xShapeInfo,
                          double *z,
                          Nd4jLong *zShapeInfo,
                          double *scalars,
                          double *extraParams,
                          int *dimension,
                          int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param scalars
     * @param extraParams
     * @param dimension
     * @param dimensionLength
     */
    void execScalarHalf(Nd4jPointer *extraPointers,int opNum,
                        float16 *x,
                        Nd4jLong *xShapeInfo,
                        float16 *z,
                        Nd4jLong *zShapeInfo,
                        float16 *scalars,
                        float16 *extraParams,
                        int *dimension,
                        int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    float   execSummaryStatsScalarFloat(Nd4jPointer *extraPointers,int opNum,float *x,
                                        Nd4jLong *xShapeInfo,
                                        float *extraParams, bool biasCorrected);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param biasCorrected
     * @return
     */
    float   execSummaryStatsScalarHalf(Nd4jPointer *extraPointers,
                                       int opNum,
                                       float16 *x,
                                       Nd4jLong *xShapeInfo,
                                       float16 *extraParams,
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
    void   execSummaryStatsFloat(Nd4jPointer *extraPointers,int opNum,
                                 float *x,
                                 Nd4jLong *xShapeInfo,
                                 float *extraParams,
                                 float *result,
                                 Nd4jLong *resultShapeInfo,bool biasCorrected);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     * @param biasCorrected
     */
    void   execSummaryStatsHalf(Nd4jPointer *extraPointers,
                                int opNum,
                                float16 *x,
                                Nd4jLong *xShapeInfo,
                                float16 *extraParams,
                                float16 *result,
                                Nd4jLong *resultShapeInfo,bool biasCorrected);

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
    void   execSummaryStatsFloat(Nd4jPointer *extraPointers,
                                 int opNum,
                                 float *x,
                                 Nd4jLong *xShapeInfo,
                                 float *extraParams,
                                 float *result,
                                 Nd4jLong *resultShapeInfoBuffer,
                                 int *dimension,
                                 int dimensionLength,
                                 bool biasCorrected);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     * @param biasCorrected
     */
    void   execSummaryStatsHalf(Nd4jPointer *extraPointers,
                                int opNum,
                                float16 *x,
                                Nd4jLong *xShapeInfo,
                                float16 *extraParams,
                                float16 *result,
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
    void   execTransformFloat(Nd4jPointer *extraPointers,
                              int opNum,
                              float *dx,
                              Nd4jLong xStride,
                              float *result,
                              Nd4jLong resultStride,
                              float *extraParams,
                              Nd4jLong n);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xStride
     * @param result
     * @param resultStride
     * @param extraParams
     * @param n
     */
    void   execTransformHalf(Nd4jPointer *extraPointers,
                             int opNum,
                             float16 *dx,
                             Nd4jLong xStride,
                             float16 *result,
                             Nd4jLong resultStride,
                             float16 *extraParams,
                             Nd4jLong n);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void   execTransformFloat(Nd4jPointer *extraPointers,
                              int opNum,
                              float *dx,
                              Nd4jLong *xShapeInfo,
                              float *result,
                              Nd4jLong *resultShapeInfo,
                              float *extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    void   execTransformHalf(Nd4jPointer *extraPointers,
                             int opNum,
                             float16 *dx,
                             Nd4jLong *xShapeInfo,
                             float16 *result,
                             Nd4jLong *resultShapeInfo,
                             float16 *extraParams);

    /**
     *
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param n
     */
    void   execTransformFloat(Nd4jPointer *extraPointers,
                              int opNum,
                              float *dx,
                              Nd4jLong *xShapeInfo,
                              float *result,
                              Nd4jLong *resultShapeInfo,
                              float *extraParams,
                              Nd4jLong *xIndexes,
                              Nd4jLong *resultIndexes);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param resultIndexes
     */
    void   execTransformHalf(Nd4jPointer *extraPointers,
                             int opNum,
                             float16 *dx,
                             Nd4jLong *xShapeInfo,
                             float16 *result,
                             Nd4jLong *resultShapeInfo,
                             float16 *extraParams,
                             Nd4jLong *xIndexes,
                             Nd4jLong *resultIndexes);


    /**
* Append an input array
* to the end of a flat array
* in a particular order
* @param offset the offset of the array to start at
* @param order the order
* @param result the result array
* @param resultShapeInfo the shape info for te array
* @param input the input for the array
* @param inputShapeInfo the shape information for that array
*/
    void flattenFloat(
            Nd4jPointer *extraPointers,
            int offset,
            char order,
            float *result,
            Nd4jLong *resultShapeInfo,
            float *input,
            Nd4jLong *inputShapeInfo);


    /**
     *
     * @param extraPointers
     * @param offset
     * @param order
     * @param result
     * @param resultShapeInfo
     * @param input
     * @param inputShapeInfo
     */
    void flattenHalf(
            Nd4jPointer *extraPointers,
            int offset,
            char order,
            float16 *result,
            Nd4jLong *resultShapeInfo,
            float16 *input,
            Nd4jLong *inputShapeInfo);

    /**
* Append an input array
* to the end of a flat array
* in a particular order
* @param offset the offset of the array to start at
* @param order the order
* @param result the result array
* @param resultShapeInfo the shape info for te array
* @param input the input for the array
* @param inputShapeInfo the shape information for that array
*/
    void flattenDouble(
            Nd4jPointer *extraPointers,
            int offset,
            char order,
            double *result,
            Nd4jLong *resultShapeInfo,
            double *input,
            Nd4jLong *inputShapeInfo);

    /**
     * Concatenate multi array of the same shape together
     * along a particular dimension
     */
    void concatFloat(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            float *result,
            Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers);
    /**
    * Concatenate multi array of the same shape together
    * along a particular dimension
    */
    void concatDouble(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            double *result,
            Nd4jLong *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);

    /**
    * Concatenate multi array of the same shape together
    * along a particular dimension
    */
    void concatInt(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            int *result,
            Nd4jLong *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);

    /**
    * Concatenate multi array of the same shape together
    * along a particular dimension
    */
    void concatLong(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            Nd4jLong *result,
            Nd4jLong *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);

    /**
     *
     * @param extraPointers
     * @param dimension
     * @param numArrays
     * @param data
     * @param inputShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param tadPointers
     * @param offsetPointers
     */
    void concatHalf(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            float16 *result,
            Nd4jLong *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);


    void specialConcatFloat(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            float *result,
            Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers);
    /**
    * Concatenate multi array of the same shape together
    * along a particular dimension
    */
    void specialConcatDouble(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            double *result,
            Nd4jLong *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);

    /**
    * Concatenate multi array of the same shape together
    * along a particular dimension
    */
    void specialConcatInt(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            int *result,
            Nd4jLong *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);

    /**
    * Concatenate multi array of the same shape together
    * along a particular dimension
    */
    void specialConcatLong(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            Nd4jLong *result,
            Nd4jLong *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);

    /**
     *
     * @param extraPointers
     * @param dimension
     * @param numArrays
     * @param data
     * @param inputShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param tadPointers
     * @param offsetPointers
     */
    void specialConcatHalf(
            Nd4jPointer *extraPointers,
            int dimension,
            int numArrays,
            Nd4jPointer *data,
            Nd4jPointer *inputShapeInfo,
            float16 *result,
            Nd4jLong *resultShapeInfo,
            Nd4jPointer *tadPointers,
            Nd4jPointer *offsetPointers);

    /**
     * This method implementation exists only for cuda.
     * The other backends should have dummy method for JNI compatibility reasons.
     */
    void initializeDevicesAndFunctions();

    void initializeFunctions(Nd4jPointer *functions);

    /**
     * This method acquires memory chunk of requested size on host side
     *
     * @param pointer pointer that'll be used for allocation
     * @param memorySize memory size, in bytes
     * @param flags optional parameter
     */
    Nd4jPointer mallocHost(Nd4jLong memorySize, int flags);

    /**
     * This method acquires memory chunk of requested size on specified device
     *
     * @param pointer pointer that'll be used for allocation
     * @param memorySize memory size, in bytes
     * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
     * @param flags optional parameter
     */
    Nd4jPointer mallocDevice(Nd4jLong memorySize, Nd4jPointer ptrToDeviceId, int flags);

    /**
     * This method releases previously allocated host memory space
     *
     * @param pointer pointer that'll be freed
     */
    int freeHost(Nd4jPointer pointer);

    /**
     * This method releases previously allocated memory space on device
     *
     * @param pointer pointer that'll be freed
     * @param ptrToDeviceId pointer to deviceId.
     */
    int freeDevice(Nd4jPointer pointer, Nd4jPointer ptrToDeviceId);

    /**
     *
     * @return
     */
    int ompGetMaxThreads();

    /**
     *
     * @return
     */
    int ompGetNumThreads();

    /**
     *
     * @param threads
     */
    void setOmpNumThreads(int threads);

    /**
     *
     * @param threads
     */
    void setOmpMinThreads(int threads);




    /**
     *
     * @return
     */
    Nd4jPointer createContext();

    /**
     *
     * @return
     */
    Nd4jPointer createStream();

    /**
     *
     * @return
     */
    Nd4jPointer createEvent();

    /**
     *
     * @param event
     * @param stream
     * @return
     */
    int registerEvent(Nd4jPointer event, Nd4jPointer stream);

    /**
     *
     * @param event
     * @return
     */
    int destroyEvent(Nd4jPointer event);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    int setDevice(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @return
     */
    int getDevice();

    /**
     *
     * @param stream
     * @return
     */
    int streamSynchronize(Nd4jPointer stream);

    /**
     *
     * @param event
     * @return
     */
    int eventSynchronize(Nd4jPointer event);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    Nd4jLong getDeviceFreeMemory(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    Nd4jLong getDeviceTotalMemory(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    int getDeviceMajor(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    int getDeviceMinor(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @param ptrToDeviceId
     * @return
     */
    const char * getDeviceName(Nd4jPointer ptrToDeviceId);

    /**
     *
     * @param dst
     * @param src
     * @param size
     * @param flags
     * @param reserved
     * @return
     */
    int memcpy(Nd4jPointer dst,
               Nd4jPointer src,
               Nd4jLong size,
               int flags,
               Nd4jPointer reserved);

    /**
     *
     * @param dst
     * @param src
     * @param size
     * @param flags
     * @param reserved
     * @return
     */
    int memcpyAsync(Nd4jPointer dst,
                    Nd4jPointer src,
                    Nd4jLong size,
                    int flags,
                    Nd4jPointer reserved);

    /**
     *
     * @param dst
     * @param value
     * @param size
     * @param flags
     * @param reserved
     * @return
     */
    int memset(Nd4jPointer dst,
               int value,
               Nd4jLong size,
               int flags,
               Nd4jPointer reserved);

    /**
     *
     * @param dst
     * @param value
     * @param size
     * @param flags
     * @param reserved
     * @return
     */
    int memsetAsync(Nd4jPointer dst,
                    int value,
                    Nd4jLong size,
                    int flags,
                    Nd4jPointer reserved);

    /**
     *
     * @param dst
     * @param src
     * @param size
     * @param flags
     * @param reserved
     * @return
     */
    int memcpyConstantAsync(Nd4jLong dst,
                            Nd4jPointer src,
                            Nd4jLong size,
                            int flags,
                            Nd4jPointer reserved);

    /**
     *
     * @return
     */
    Nd4jPointer getConstantSpace();

    /**
     *
     * @return
     */
    int getAvailableDevices();

    /**
     *
     * @param reallyEnable
     */
    void enableDebugMode(bool reallyEnable);

    /**
     *
     * @param reallyEnable
     */
    void enableVerboseMode(bool reallyEnable);

    /**
     *
     * @param gridSize
     */
    void setGridLimit(int gridSize);

    /**
     *
     * @param xShapeInfo
     * @param dimension
     * @param dimensionLength
     * @param targetBuffer
     * @param offsetsBuffer
     */
    void tadOnlyShapeInfo(Nd4jLong *xShapeInfo,
                          int *dimension,
                          int dimensionLength,
                          Nd4jLong *targetBuffer,
                          Nd4jLong *offsetsBuffer);

    /*
     * PullRow special op
     */

    /**
     *
     * @param extraPointers
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param n
     * @param indexes
     * @param tadShapeInfo
     * @param tadOffsets
     * @param zTadShapeInfo
     * @param zTadOffsets
     */
    void pullRowsHalf(Nd4jPointer *extraPointers,
                      float16 *x,
                      Nd4jLong *xShapeInfo,
                      float16 *z,
                      Nd4jLong *zShapeInfo,
                      Nd4jLong n,
                      Nd4jLong *indexes,
                      Nd4jLong *tadShapeInfo,
                      Nd4jLong *tadOffsets,
                      Nd4jLong *zTadShapeInfo,
                      Nd4jLong *zTadOffsets);

    /**
     *
     * @param extraPointers
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param n
     * @param indexes
     * @param tadShapeInfo
     * @param tadOffsets
     * @param zTadShapeInfo
     * @param zTadOffsets
     */
    void pullRowsFloat(Nd4jPointer *extraPointers,
                       float *x,
                       Nd4jLong *xShapeInfo,
                       float* z,
                       Nd4jLong *zShapeInfo,
                       Nd4jLong n,
                       Nd4jLong *indexes,
                       Nd4jLong *tadShapeInfo,
                       Nd4jLong *tadOffsets,
                       Nd4jLong *zTadShapeInfo,
                       Nd4jLong *zTadOffsets);

    /**
     *
     * @param extraPointers
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param n
     * @param indexes
     * @param tadShapeInfo
     * @param tadOffsets
     * @param zTadShapeInfo
     * @param zTadOffsets
     */
    void pullRowsDouble(Nd4jPointer *extraPointers,
                        double *x,
                        Nd4jLong *xShapeInfo,
                        double *z,
                        Nd4jLong *zShapeInfo,
                        Nd4jLong n,
                        Nd4jLong *indexes,
                        Nd4jLong *tadShapeInfo,
                        Nd4jLong *tadOffsets,
                        Nd4jLong *zTadShapeInfo,
                        Nd4jLong *zTadOffsets);

    /**
     * Array averaging op
     */
    /**
     *
     * @param extras
     * @param dx
     * @param dz
     * @param n
     * @param length
     * @param propagate
     */
    void averageHalf(Nd4jPointer *extras,
                     Nd4jPointer *dx,
                     float16 *dz,
                     int n,
                     Nd4jLong length,
                     bool propagate);

    /**
     *
     * @param extras
     * @param dx
     * @param dz
     * @param n
     * @param length
     * @param propagate
     */
    void averageFloat(Nd4jPointer *extras,
                      Nd4jPointer *dx,
                      float *dz,
                      int n,
                      Nd4jLong length,
                      bool propagate);

    /**
     *
     * @param extras
     * @param dx
     * @param dz
     * @param n
     * @param length
     * @param propagate
     */
    void averageDouble(Nd4jPointer *extras,
                       Nd4jPointer *dx,
                       double *dz,
                       int n,
                       Nd4jLong length,
                       bool propagate);


    void accumulateHalf(Nd4jPointer *extras,
                          Nd4jPointer *dx,
                          float16 *dz,
                          int n,
                          Nd4jLong length);


    void accumulateFloat(Nd4jPointer *extras,
                          Nd4jPointer *dx,
                          float *dz,
                          int n,
                          Nd4jLong length);

    void accumulateDouble(Nd4jPointer *extras,
                       Nd4jPointer *dx,
                       double *dz,
                       int n,
                       Nd4jLong length);


    /**
     * P2P enabler
     */
    /**
     *
     * @param enable
     */
    void enableP2P(bool enable);

    /**
     *
     */
    void checkP2P();

    /**
     *
     * @return
     */
    bool isP2PAvailable();

    /**
     * Shuffle methods
     */

    /**
     *
     * @param extras
     * @param dx
     * @param xShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param N
     * @param shuffleMap
     * @param tadShapeInfo
     * @param tadOffsets
     */
    void shuffleDouble(Nd4jPointer *extras,
                       Nd4jPointer *dx,
                       Nd4jPointer *xShapeInfo,
                       Nd4jPointer *dz,
                       Nd4jPointer *zShapeInfo,
                       int N,
                       int *shuffleMap,
                       Nd4jPointer *tadShapeInfo,
                       Nd4jPointer *tadOffsets);

    /**
     *
     * @param extras
     * @param dx
     * @param xShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param N
     * @param shuffleMap
     * @param tadShapeInfo
     * @param tadOffsets
     */
    void shuffleFloat(Nd4jPointer *extras,
                      Nd4jPointer *dx,
                      Nd4jPointer *xShapeInfo,
                      Nd4jPointer *dz,
                      Nd4jPointer *zShapeInfo,
                      int N,
                      int *shuffleMap,
                      Nd4jPointer *tadShapeInfo,
                      Nd4jPointer *tadOffsets);


    /**
     *
     * @param extras
     * @param dx
     * @param xShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param N
     * @param shuffleMap
     * @param tadShapeInfo
     * @param tadOffsets
     */
    void shuffleHalf(Nd4jPointer *extras,
                     Nd4jPointer *dx,
                     Nd4jPointer *xShapeInfo,
                     Nd4jPointer *dz,
                     Nd4jPointer *zShapeInfo,
                     int N,
                     int *shuffleMap,
                     Nd4jPointer *tadShapeInfo,
                     Nd4jPointer *tadOffsets);

    /**
     * Type Conversions
     */

    /**
     *
     * @param extras
     * @param srcType
     * @param x
     * @param N
     * @param dstType
     * @param z
     */
    void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, Nd4jLong N, int dstType, Nd4jPointer z);


    /**
     *
     * @return
     */
    bool isExperimentalEnabled();

    /**
     * Aggregate
     */

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param arguments
     * @param numArguments
     * @param shapeArguments
     * @param numShapeArguments
     * @param indexArguments
     * @param numIndexArguments
     * @param intArrays
     * @param numIntArrays
     * @param realArguments
     * @param numRealArguments
     */
    void execAggregateFloat(Nd4jPointer *extraPointers,
                            int opNum,
                            float **arguments,
                            int numArguments,
                            Nd4jLong **shapeArguments,
                            int numShapeArguments,
                            int *indexArguments,
                            int numIndexArguments,
                            int **intArrays,
                            int numIntArrays,
                            float *realArguments,
                            int numRealArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param arguments
     * @param numArguments
     * @param shapeArguments
     * @param numShapeArguments
     * @param indexArguments
     * @param numIndexArguments
     * @param intArrays
     * @param numIntArrays
     * @param realArguments
     * @param numRealArguments
     */
    void execAggregateDouble(Nd4jPointer *extraPointers,
                             int opNum,
                             double **arguments,
                             int numArguments,
                             Nd4jLong **shapeArguments,
                             int numShapeArguments,
                             int *indexArguments,
                             int numIndexArguments,
                             int **intArrays,
                             int numIntArrays,
                             double *realArguments,
                             int numRealArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param arguments
     * @param numArguments
     * @param shapeArguments
     * @param numShapeArguments
     * @param indexArguments
     * @param numIndexArguments
     * @param intArrays
     * @param numIntArrays
     * @param realArguments
     * @param numRealArguments
     */
    void execAggregateHalf(Nd4jPointer *extraPointers,
                           int opNum,
                           float16 **arguments,
                           int numArguments,
                           Nd4jLong **shapeArguments,
                           int numShapeArguments,
                           int *indexArguments,
                           int numIndexArguments,
                           int **intArrays,
                           int numIntArrays,
                           float16 *realArguments,
                           int numRealArguments);


    /**
     *
     * @param extraPointers
     * @param numAggregates
     * @param opNum
     * @param maxArgs
     * @param maxShapes
     * @param maxIntArrays
     * @param maxIntArraySize
     * @param maxIdx
     * @param maxReals
     * @param ptrToArguments
     */
    void execAggregateBatchFloat(Nd4jPointer *extraPointers,
                                 int numAggregates,
                                 int opNum,
                                 int maxArgs,
                                 int maxShapes,
                                 int maxIntArrays,
                                 int maxIntArraySize,
                                 int maxIdx,
                                 int maxReals,
                                 void *ptrToArguments);

    /**
     *
     * @param extraPointers
     * @param numAggregates
     * @param opNum
     * @param maxArgs
     * @param maxShapes
     * @param maxIntArrays
     * @param maxIntArraySize
     * @param maxIdx
     * @param maxReals
     * @param ptrToArguments
     */
    void execAggregateBatchDouble(Nd4jPointer *extraPointers,
                                  int numAggregates,
                                  int opNum,
                                  int maxArgs,
                                  int maxShapes,
                                  int maxIntArrays,
                                  int maxIntArraySize,
                                  int maxIdx,
                                  int maxReals,
                                  void *ptrToArguments);

    /**
     *
     * @param extraPointers
     * @param numAggregates
     * @param opNum
     * @param maxArgs
     * @param maxShapes
     * @param maxIntArrays
     * @param maxIntArraySize
     * @param maxIdx
     * @param maxReals
     * @param ptrToArguments
     */
    void execAggregateBatchHalf(Nd4jPointer *extraPointers,
                                int numAggregates,
                                int opNum,
                                int maxArgs,
                                int maxShapes,
                                int maxIntArrays,
                                int maxIntArraySize,
                                int maxIdx,
                                int maxReals,
                                void *ptrToArguments);

    /**
     * Random operations
     */

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         Nd4jPointer state,
                         float *z,
                         Nd4jLong *zShapeBuffer,
                         float *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param y
     * @param yShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         Nd4jPointer state,
                         float *x,
                         Nd4jLong *xShapeBuffer,
                         float *y,
                         Nd4jLong *yShapeBuffer,
                         float *z,
                         Nd4jLong *zShapeBuffer,
                         float *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         Nd4jPointer state,
                         float *x,
                         Nd4jLong *xShapeBuffer,
                         float *z,
                         Nd4jLong *zShapeBuffer,
                         float *extraArguments);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          Nd4jPointer state,
                          double *z,
                          Nd4jLong *zShapeBuffer,
                          double *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param y
     * @param yShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          Nd4jPointer state,
                          double *x,
                          Nd4jLong *xShapeBuffer,
                          double *y,
                          Nd4jLong *yShapeBuffer,
                          double *z,
                          Nd4jLong *zShapeBuffer,
                          double *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          Nd4jPointer state,
                          double *x,
                          Nd4jLong *xShapeBuffer,
                          double *z,
                          Nd4jLong *zShapeBuffer,
                          double *extraArguments);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomHalf(Nd4jPointer *extraPointers,
                        int opNum,
                        Nd4jPointer state,
                        float16 *z,
                        Nd4jLong *zShapeBuffer,
                        float16 *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param y
     * @param yShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomHalf(Nd4jPointer *extraPointers,
                        int opNum,
                        Nd4jPointer state,
                        float16 *x,
                        Nd4jLong *xShapeBuffer,
                        float16 *y,
                        Nd4jLong *yShapeBuffer,
                        float16 *z,
                        Nd4jLong *zShapeBuffer,
                        float16 *extraArguments);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param state
     * @param x
     * @param xShapeBuffer
     * @param z
     * @param zShapeBuffer
     * @param extraArguments
     */
    void execRandomHalf(Nd4jPointer *extraPointers,
                        int opNum,
                        Nd4jPointer state,
                        float16 *x,
                        Nd4jLong *xShapeBuffer,
                        float16 *z,
                        Nd4jLong *zShapeBuffer,
                        float16 *extraArguments);



    /**
     *
     * @param extraPointers
     * @param seed
     * @param bufferSize
     * @param ptrToBuffer
     * @return
     */
    Nd4jPointer initRandom(Nd4jPointer *extraPointers,
                           long seed,
                           long bufferSize,
                           Nd4jPointer ptrToBuffer);

    /**
     *
     * @param extraPointers
     * @param seed
     * @param ptrRandom
     */
    void refreshBuffer(Nd4jPointer *extraPointers,
                       long seed,
                       Nd4jPointer ptrRandom);

    /**
     *
     * @param extraPointers
     * @param seed
     * @param ptrRandom
     */
    void reSeedBuffer(Nd4jPointer *extraPointers,
                      long seed,
                      Nd4jPointer ptrRandom);

    /**
     *
     * @param ptrRandom
     */
    void destroyRandom(Nd4jPointer ptrRandom);

    /**
     * Grid operations
     */

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xStride
     * @param dy
     * @param yStride
     * @param dz
     * @param zStride
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateStridedFloat(Nd4jPointer *extras,
                                       const int opTypeA,
                                       const int opNumA,
                                       const int opTypeB,
                                       const int opNumB,
                                       Nd4jLong N,
                                       float *dx,
                                       Nd4jLong xStride,
                                       float *dy,
                                       Nd4jLong yStride,
                                       float *dz,
                                       Nd4jLong zStride,
                                       float *extraA,
                                       float *extraB,
                                       float scalarA,
                                       float scalarB);

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xShapeInfo
     * @param dy
     * @param yShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateShapeFloat(Nd4jPointer *extras,
                                     const int opTypeA,
                                     const int opNumA,
                                     const int opTypeB,
                                     const int opNumB,
                                     Nd4jLong N,
                                     float *dx,
                                     Nd4jLong *xShapeInfo,
                                     float *dy,
                                     Nd4jLong *yShapeInfo,
                                     float *dz,
                                     Nd4jLong *zShapeInfo,
                                     float *extraA,
                                     float *extraB,
                                     float scalarA,
                                     float scalarB);

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xStride
     * @param dy
     * @param yStride
     * @param dz
     * @param zStride
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateStridedDouble(Nd4jPointer *extras,
                                        const int opTypeA,
                                        const int opNumA,
                                        const int opTypeB,
                                        const int opNumB,
                                        Nd4jLong N,
                                        double *dx,
                                        Nd4jLong xStride,
                                        double *dy,
                                        Nd4jLong yStride,
                                        double *dz,
                                        Nd4jLong zStride,
                                        double *extraA,
                                        double *extraB,
                                        double scalarA,
                                        double scalarB);

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xShapeInfo
     * @param dy
     * @param yShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateShapeDouble(Nd4jPointer *extras,
                                      const int opTypeA,
                                      const int opNumA,
                                      const int opTypeB,
                                      const int opNumB,
                                      Nd4jLong N,
                                      double *dx,
                                      Nd4jLong *xShapeInfo,
                                      double *dy,
                                      Nd4jLong *yShapeInfo,
                                      double *dz,
                                      Nd4jLong *zShapeInfo,
                                      double *extraA,
                                      double *extraB,
                                      double scalarA,
                                      double scalarB);

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xStride
     * @param dy
     * @param yStride
     * @param dz
     * @param zStride
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateStridedHalf(Nd4jPointer *extras,
                                      const int opTypeA,
                                      const int opNumA,
                                      const int opTypeB,
                                      const int opNumB,
                                      Nd4jLong N,
                                      float16 *dx,
                                      Nd4jLong xStride,
                                      float16 *dy,
                                      Nd4jLong yStride,
                                      float16 *dz,
                                      Nd4jLong zStride,
                                      float16 *extraA,
                                      float16 *extraB,
                                      float scalarA,
                                      float scalarB);

    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param N
     * @param dx
     * @param xShapeInfo
     * @param dy
     * @param yShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     */
    void execMetaPredicateShapeHalf(Nd4jPointer *extras,
                                    const int opTypeA,
                                    const int opNumA,
                                    const int opTypeB,
                                    const int opNumB,
                                    Nd4jLong N,
                                    float16 *dx,
                                    Nd4jLong *xShapeInfo,
                                    float16 *dy,
                                    Nd4jLong *yShapeInfo,
                                    float16 *dz,
                                    Nd4jLong *zShapeInfo,
                                    float16 *extraA,
                                    float16 *extraB,
                                    float scalarA,
                                    float scalarB);


    /**
     *
     * @param extras
     * @param opTypeA
     * @param opNumA
     * @param opTypeB
     * @param opNumB
     * @param dx
     * @param xShapeInfo
     * @param dy
     * @param yShapeInfo
     * @param dz
     * @param zShapeInfo
     * @param dimension
     * @param dimensionLength
     * @param tadShapeInfo
     * @param tadOffsets
     * @param extraA
     * @param extraB
     * @param scalarA
     * @param scalarB
     * @param scalarReturned
     */
    void execMetaPredicateReduceFloat(Nd4jPointer *extras,
                                      const int opTypeA,
                                      const int opNumA,
                                      const int opTypeB,
                                      const int opNumB,
                                      float *dx,
                                      Nd4jLong *xShapeInfo,
                                      float *dy,
                                      Nd4jLong *yShapeInfo,
                                      float *dz,
                                      Nd4jLong *zShapeInfo,
                                      int *dimension,
                                      int dimensionLength,
                                      Nd4jLong *tadShapeInfo,
                                      Nd4jLong *tadOffsets,
                                      float *extraA,
                                      float *extraB,
                                      float scalarA,
                                      float scalarB,
                                      bool scalarReturned);


/**
 *
 * @param data
 * @param shapeBuffer
 * @param wordSize
 * @param headerSize
 * @return
 */
    Nd4jPointer numpyHeaderForNd4j(Nd4jPointer data,Nd4jPointer shapeBuffer,Nd4jLong wordSize,Nd4jLong *headerSize) {
        Nd4jLong *shapeBufferCast = reinterpret_cast<Nd4jLong *>(shapeBuffer);
        int  rank = shape::rank(shapeBufferCast);
        Nd4jLong *shape = shape::shapeOf(shapeBufferCast);
        unsigned int *npShape = new unsigned int[rank];
        for(int i = 0; i < rank; i++) {
            npShape[i] = shape[i];
        }

        Nd4jLong length = shape::prodLong(shape,rank);
        auto npHeader = cnpy::createNpyHeader(data,npShape,rank,wordSize);
        char *ret = new char[npHeader.size() + 1];
        int count = 0;
        for(int i = 0; i < npHeader.size(); i++) {
            if (npHeader[i] != '\0') {
                ret[count] = npHeader[i];
                count++;
            }
            else {
                nd4j_debug("Found null terminated at %d. Skipping\n",i);
            }
        }

        ret[count] = '\0';
        count++;
        *headerSize = count;
        return reinterpret_cast<Nd4jPointer>(ret);

    }

/**
   * Load numpy from a header
    * based on the cnpy parse from header method.
   * @param data the header data to parse
   * @return a pointer to a numpy cnpy:NpyArray struct
   */
    Nd4jPointer loadNpyFromHeader(Nd4jPointer data) {
        char *header = reinterpret_cast<char *>(data);

        cnpy::NpyArray arr = cnpy::loadNpyFromHeader(header);
        cnpy::NpyArray *ret = new cnpy::NpyArray();
        int totalLengthOfShape = 1;
        for(int i = 0; i < arr.shape.size(); i++) {
            totalLengthOfShape *= arr.shape[i];
        }

        ret->data = arr.data;
        ret->wordSize = arr.wordSize;
        ret->shape = arr.shape;
        return reinterpret_cast<Nd4jPointer>(ret);
    }


/**
   * Create a numpy array from an nd4j
   * array
   * @param data a pointer to the data
   * @param shapeBuffer  the shapebuffer for the nd4j array
   * @param wordSize  the word size (4 for float, 8 for doubles)
   * @return a pointer to a numpy array
   */
    Nd4jPointer numpyFromNd4j(Nd4jPointer data,Nd4jPointer shapeBuffer,Nd4jLong wordSize) {
        Nd4jLong *shapeBufferCast = reinterpret_cast<Nd4jLong *>(shapeBuffer);
        int  rank = shape::rank(shapeBufferCast);
        Nd4jLong *shape = shape::shapeOf(shapeBufferCast);
        unsigned int *npShape = new unsigned int[rank];
        for(int i = 0; i < rank; i++) {
            npShape[i] = shape[i];
        }

        Nd4jLong length = shape::prodLong(shape,rank);
        auto npHeader = cnpy::createNpyHeader(data,npShape,rank,wordSize);
        char *dataChar = reinterpret_cast<char *>(data);
        char *npHeaderData = npHeader.data();
        char *ret = new char[(wordSize * length) +  npHeader.size()];
        char *cursorStart = ret;
        std::memcpy(reinterpret_cast<void *>(ret), reinterpret_cast<void *>(npHeaderData), npHeader.size() * sizeof(Nd4jLong));
        //move to next
        cursorStart += npHeader.size();
        std::memcpy(reinterpret_cast<void *>(ret), reinterpret_cast<void *>(dataChar), length * wordSize * sizeof(Nd4jLong));
        Nd4jPointer  rettPointer = reinterpret_cast<Nd4jPointer>(ret);
        return rettPointer;
    }


/**
 *
 * @param npyArray
 * @return
 */
    Nd4jPointer shapeBufferForNumpy(Nd4jPointer npyArray) {
        cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
        auto shape = new unsigned int[arr.shape.size()];
        for(unsigned int i = 0; i < arr.shape.size(); i++) {
            shape[i] = arr.shape[i];
        }

        auto shapeBuffer = shape::shapeBufferOfNpy(arr.shape.size(), shape, arr.fortranOrder);
        delete[] shape;
        return reinterpret_cast<Nd4jPointer>(shapeBuffer);
    }


/**
* Get the shape buffer from a
* numpy array.
* **Warning** this allocates memory
* @param npyArray
* @return
*/
    Nd4jPointer shapeBufferForNumpyHeader(Nd4jPointer npyArray) {
        cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char *>(npyArray));
        auto shape = new unsigned int[arr.shape.size()];
        for(unsigned int i = 0; i < arr.shape.size(); i++) {
            shape[i] = arr.shape[i];
        }

        auto shapeBuffer = shape::shapeBufferOfNpy(arr.shape.size(), shape, arr.fortranOrder);
        delete[] shape;
        return reinterpret_cast<Nd4jPointer>(shapeBuffer);
    }



/**
 *
 * @param npyArray
 * @return
 */
    Nd4jPointer dataPointForNumpyHeader(Nd4jPointer npyArray) {
        cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char *>(npyArray));
        unsigned  char *dataToPrint = reinterpret_cast<unsigned  char *>(arr.data);
        return dataToPrint;
    }

/**
 *
 * @param npyArray
 * @return
 */
    Nd4jPointer dataPointForNumpyStruct(Nd4jPointer npyArrayStruct) {
        cnpy::NpyArray *arrPointer = reinterpret_cast<cnpy::NpyArray *>(npyArrayStruct);
        unsigned  char *dataToPrint = reinterpret_cast<unsigned  char *>(arrPointer->data);
        return reinterpret_cast<Nd4jPointer>(dataToPrint);
    }

/**
 *
 * @param npyArray
 * @param fromFile
 * @return
 */
    Nd4jPointer dataPointForNumpy(Nd4jPointer npyArray) {
        char *npyArrayBuffer = reinterpret_cast<  char *>(npyArray);
        cnpy::NpyArray arr = cnpy::loadNpyFromPointer(npyArrayBuffer);
        return dataPointForNumpyStruct(reinterpret_cast<Nd4jPointer>(&arr));
    }

/**
 * Load a numpy array from a file
 * and return it as an Nd4jPointer
 * @param path
 * @return
 */
    Nd4jPointer numpyFromFile(std::string path) {
        char *numpyBuffer = cnpy::loadFile(path.data());
        return reinterpret_cast<Nd4jPointer >(numpyBuffer);
    }


/**
  * Get the element size for a numpy array
  * @param npyArray  the numpy array's address
  * to get the length for
  * @return
  */
    int elementSizeForNpyArray(Nd4jPointer npyArray) {
        cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
        cnpy::NpyArray *arrPointer = &arr;
        int size = arrPointer->wordSize;
        // arrPointer->destruct();
        return size;
    }


/**
* Get the element size for a numpy array
* @param npyArray  the numpy array's address
* to get the length for
* @return
*/
    int elementSizeForNpyArrayHeader(Nd4jPointer npyArray) {
        cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char *>(npyArray));
        cnpy::NpyArray *arrPointer = &arr;
        int size = arrPointer->wordSize;
        return size;
    }


    void releaseNumpy(Nd4jPointer npyArray) {
        free(reinterpret_cast<void *>(npyArray));
    }


    /**
     * Return the length of a shape buffer
     * based on the pointer
     * @param buffer  the buffer pointer to check
     * @return
     */
    int lengthForShapeBufferPointer(Nd4jPointer buffer);


      /**
   * The pointer to get the address for
   *
   * @param address the address to get the pointer
   * @return the pointer for the given address
   */

    Nd4jPointer pointerForAddress(Nd4jLong address);

    /**
     * This method takes single N-dimensional tensor, and copies its TADs to target arrays
     *
     * @param x
     * @param xShapeInfo
     * @param targets
     * @param zShapeInfo
     * @return
     */
    void tearDouble(Nd4jPointer *extraPointers, double *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong
     *tadShapeInfo, Nd4jLong *tadOffsets);

    /**
     * This method takes single N-dimensional tensor, and copies its TADs to target arrays
     *
     * @param x
     * @param xShapeInfo
     * @param targets
     * @param zShapeInfo
     * @return
     */
    void tearFloat(Nd4jPointer *extraPointers, float *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

    /**
     * This method takes single N-dimensional tensor, and copies its TADs to target arrays
     *
     * @param x
     * @param xShapeInfo
     * @param targets
     * @param zShapeInfo
     * @return
     */
    void tearHalf(Nd4jPointer *extraPointers, float16 *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);


    Nd4jLong encodeBitmapFloat(Nd4jPointer *extraPointers, float *dx, Nd4jLong N, int *dz, float threshold);

    Nd4jLong encodeBitmapDouble(Nd4jPointer *extraPointers, double *dx, Nd4jLong N, int *dz, float threshold);

    Nd4jLong encodeBitmapHalf(Nd4jPointer *extraPointers, float16 *dx, Nd4jLong N, int *dz, float threshold);

    void decodeBitmapFloat(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float *dz);

    void decodeBitmapDouble(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, double *dz);

    void decodeBitmapHalf(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float16 *dz);


    void encodeThresholdP1Double(Nd4jPointer *extraPointers, double *dx, Nd4jLong N, int *dz, float threshold);

    void encodeThresholdP1Half(Nd4jPointer *extraPointers, float16 *dx, Nd4jLong N, int *dz, float threshold);

    void encodeThresholdP1Float(Nd4jPointer *extraPointers, float *dx, Nd4jLong N, int *dz, float threshold);


    void encodeThresholdP2Int(Nd4jPointer *extraPointers, int *dx, Nd4jLong N, int *dz);


    void encodeThresholdP3Float(Nd4jPointer *extraPointers, float *dx, int *offsets, Nd4jLong N, int *dz);

    void encodeThresholdP3Double(Nd4jPointer *extraPointers, double *dx, int *offsets, Nd4jLong N, int *dz);

    void encodeThresholdP3Half(Nd4jPointer *extraPointers, float16 *dx, int *offsets, Nd4jLong N, int *dz);


    void decodeThresholdFloat(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float *dz);

    void decodeThresholdDouble(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, double *dz);

    void decodeThresholdHalf(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float16 *dz);



    void sortFloat(Nd4jPointer *extraPointers, float *x, Nd4jLong *xShapeInfo, bool descending);

    void sortDouble(Nd4jPointer *extraPointers, double *x, Nd4jLong *xShapeInfo, bool descending);

    void sortHalf(Nd4jPointer *extraPointers, float16 *x, Nd4jLong *xShapeInfo, bool descending);



    void sortTadFloat(Nd4jPointer *extraPointers, float *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending);

    void sortTadDouble(Nd4jPointer *extraPointers, double *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending);

    void sortTadHalf(Nd4jPointer *extraPointers, float16 *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending);


    // special sort impl for sorting out COO indices and values
    void sortCooIndicesFloat(Nd4jPointer *extraPointers, Nd4jLong *indices, float *values, Nd4jLong length, int rank);

    void sortCooIndicesDouble(Nd4jPointer *extraPointers, Nd4jLong *indices, double *values, Nd4jLong length, int rank);

    void sortCooIndicesHalf(Nd4jPointer *extraPointers, Nd4jLong *indices, float16 *values, Nd4jLong length, int rank);


    Nd4jLong* mmapFile(Nd4jPointer *extraPointers, const char *fileName, Nd4jLong length);

    void munmapFile(Nd4jPointer *extraPointers, Nd4jLong* ptrMap, Nd4jLong length);


    // flatbuffers execution
    nd4j::graph::ResultWrapper* executeFlatGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer);
    nd4j::graph::ResultWrapper* executeFlatGraphDouble(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer);
    nd4j::graph::ResultWrapper* executeFlatGraphHalf(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer);

    // protobuf execution
    Nd4jPointer executeProtoGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer protoBufferPointer);
    Nd4jPointer executeProtoGraphFloat(Nd4jPointer *extraPointers, const char *fileName);

    const char* getAllCustomOps();

    const char* getAllOperations();

    // customOp executioner
    int execCustomOpFloat(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace);
    int execCustomOpDouble(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace);
    int execCustomOpHalf(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float16* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace);

    nd4j::ShapeList* calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs);
    nd4j::ShapeList* calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs);
    nd4j::ShapeList* calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs);

    nd4j::ShapeList* calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs);
    nd4j::ShapeList* calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs);
    nd4j::ShapeList* calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs);

    void deleteShapeList(Nd4jPointer shapeList);

    int registerGraphFloat(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer);
    int registerGraphDouble(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer);
    int registerGraphHalf(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer);

    nd4j::graph::VariablesSet<float>* executeStoredGraphFloat(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs);
    nd4j::graph::VariablesSet<double>* executeStoredGraphDouble(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs);
    nd4j::graph::VariablesSet<float16>* executeStoredGraphHalf(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs);

    int unregisterGraph(Nd4jPointer *extraPointers, Nd4jLong graphId);

    void deleteIntArray(Nd4jPointer pointer);
    void deleteLongArray(Nd4jPointer pointer);
    void deletePointerArray(Nd4jPointer pointer);

    void deleteVariablesSetFloat(Nd4jPointer pointer);
    void deleteVariablesSetDouble(Nd4jPointer pointer);
    void deleteVariablesSetHalf(Nd4jPointer pointer);

    // GraphState creation
    Nd4jPointer getGraphStateHalf(Nd4jLong id);
    Nd4jPointer getGraphStateFloat(Nd4jLong id);
    Nd4jPointer getGraphStateDouble(Nd4jLong id);

    void deleteGraphStateHalf(Nd4jPointer state);
    void deleteGraphStateFloat(Nd4jPointer state);
    void deleteGraphStateDouble(Nd4jPointer state);

    void deleteResultWrapper(Nd4jPointer ptr);

    int estimateThresholdFloat(Nd4jPointer *extraPointers, Nd4jPointer x, int N, float threshold);
    int estimateThresholdDouble(Nd4jPointer *extraPointers, Nd4jPointer x, int N, float threshold);
    int estimateThresholdHalf(Nd4jPointer *extraPointers, Nd4jPointer x, int N, float threshold);

    // this method executes op that requires scope to be present: if/while/cond/whatever
    Nd4jStatus execCustomOpWithScopeHalf(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs);
    Nd4jStatus execCustomOpWithScopeFloat(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs);
    Nd4jStatus execCustomOpWithScopeDouble(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs);




};






#endif //NATIVEOPERATIONS_NATIVEOPS_H
