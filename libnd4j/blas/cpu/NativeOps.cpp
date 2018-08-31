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

#define __STDC_CONSTANT_MACROS

#include "../NativeOps.h"
#include "../NativeOpExcutioner.h"
#include "../NDArray.h"
#include "../GraphExecutioner.h"
#include <graph/GraphHolder.h>
#include <templatemath.h>
#include <types/float8.h>
#include <loops/type_conversions.h>
#include <loops/aggregates.h>
#include <helpers/helper_ptrmap.h>
#include <helpers/logger.h>
#include <pointercast.h>
#include <pairwise_util.h>


#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#else
#include <io.h>
#include <helpers/mman.h>
#endif
#include <sys/types.h>

#include <ops/declarable/CustomOperations.h>
#include <errno.h>


char *name;
bool nameSet = false;


#ifdef __EXPERIMENTAL__
bool experimentalSupport = true;
#else
bool experimentalSupport = false;
#endif

#include <ops/specials.h>
#include "../Environment.h"
#include <TAD.h>
#include <ops/declarable/OpRegistrator.h>
#include <graph/Context.h>
#include <graph/ResultWrapper.h>

using namespace nd4j;

void NativeOps::setElementThreshold(int num) {
    if (num > 0)
        nd4j::Environment::getInstance()->setElementwiseThreshold(num);
}

void NativeOps::setTADThreshold(int num) {
    if (num > 0)
        nd4j::Environment::getInstance()->setTadThreshold(num);
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
double   NativeOps::execIndexReduceScalarDouble(Nd4jPointer *extraPointers,
                                                int opNum,
                                                double *x,
                                                Nd4jLong *xShapeInfo,
                                                double *extraParams) {
    return NativeOpExcutioner<double>::execIndexReduceScalar(opNum,
                                                             x,
                                                             xShapeInfo,
                                                             extraParams);

}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execIndexReduceDouble(Nd4jPointer *extraPointers,int opNum,
                                        double *x,
                                        Nd4jLong *xShapeInfo,
                                        double *extraParams,
                                        double *result,
                                        Nd4jLong *resultShapeInfo,
                                        int *dimension,
                                        int dimensionLength) {
    Nd4jLong *tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    Nd4jLong *tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    NativeOpExcutioner<double>::execIndexReduce(opNum,
                                                x,
                                                xShapeInfo,
                                                extraParams,
                                                result,
                                                resultShapeInfo,
                                                dimension,
                                                dimensionLength,
                                                tadShapeInfo,
                                                tadOffsets);
}


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
void   NativeOps::execBroadcastDouble(Nd4jPointer *extraPointers,int opNum,
                                      double *x,
                                      Nd4jLong *xShapeInfo,
                                      double *y,
                                      Nd4jLong *yShapeInfo,
                                      double *result,
                                      Nd4jLong *resultShape,
                                      int *dimension, int dimensionLength) {
    Nd4jLong *tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    Nd4jLong *tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    Nd4jLong *tadShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[2]);
    Nd4jLong *tadOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[3]);
    NativeOpExcutioner<double>::execBroadcast(
            opNum,
            x,
            xShapeInfo,
            y,
            yShapeInfo,
            result, resultShape,
            dimension,
            dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}



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
void   NativeOps::execPairwiseTransformDouble(Nd4jPointer *extraPointers,
                                              int opNum,
                                              double *dx,
                                              Nd4jLong xStride,
                                              double *y,
                                              Nd4jLong yStride,
                                              double *result,
                                              Nd4jLong resultStride,
                                              double *extraParams,
                                              Nd4jLong n) {
    NativeOpExcutioner<double>::execPairwiseTransform(opNum,
                                                      dx,
                                                      xStride,
                                                      y,
                                                      yStride,
                                                      result,
                                                      resultStride,
                                                      extraParams,
                                                      n);
}

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
void NativeOps::execPairwiseTransformDouble(
        Nd4jPointer *extraPointers,
        int opNum,
        double *dx,
        Nd4jLong *xShapeInfo,
        double *y,
        Nd4jLong *yShapeInfo,
        double *result,
        Nd4jLong *resultShapeInfo,
        double *extraParams,
        Nd4jLong *xIndexes,
        Nd4jLong *yIndexes,
        Nd4jLong *resultIndexes) {
    NativeOpExcutioner<double>::execPairwiseTransform(
            opNum,
            dx,
            xShapeInfo,
            y,
            yShapeInfo,
            result,
            resultShapeInfo,
            extraParams,
            xIndexes,
            yIndexes,
            resultIndexes);
}

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
void NativeOps::execPairwiseTransformDouble(
        Nd4jPointer *extraPointers,
        int opNum,
        double *dx,
        Nd4jLong *xShapeInfo,
        double *y,
        Nd4jLong *yShapeInfo,
        double *result,
        Nd4jLong *resultShapeInfo,
        double *extraParams) {
    NativeOpExcutioner<double>::execPairwiseTransform(
            opNum,
            dx,
            xShapeInfo,
            y,
            yShapeInfo,
            result,
            resultShapeInfo,
            extraParams);
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execReduceDouble(
        Nd4jPointer *extraPointers,
        int opNum,
        double *x,
        Nd4jLong *xShapeInfo,
        double *extraParams,
        double *result,
        Nd4jLong *resultShapeInfo) {
    result[0] = NativeOpExcutioner<double>::execReduceScalar(
            opNum,
            x,
            xShapeInfo,
            extraParams);

}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execReduceDouble(Nd4jPointer *extraPointers,int opNum,
                                   double *x,
                                   Nd4jLong *xShapeInfo,
                                   double *extraParams,
                                   double *result,
                                   Nd4jLong *resultShapeInfo,
                                   int *dimension,int dimensionLength) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    NativeOpExcutioner<double>::execReduce(opNum,
                                           x,
                                           xShapeInfo,
                                           extraParams,
                                           result,
                                           resultShapeInfo,
                                           dimension,
                                           dimensionLength,
                                           tadShapeInfo,
                                           tadOffsets);
}

void   NativeOps::execReduceHalf(Nd4jPointer *extraPointers,int opNum,
                                 float16 *x,
                                 Nd4jLong *xShapeInfo,
                                 float16 *extraParams,
                                 float16 *result,
                                 Nd4jLong *resultShapeInfo,
                                 int *dimension,
                                 int dimensionLength) {
    // no-op
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @return
 */
double NativeOps::execReduceScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                         double *x,
                                         Nd4jLong *xShapeInfo,
                                         double *extraParams) {
    return NativeOpExcutioner<double>::execReduceScalar(opNum,x,xShapeInfo,extraParams);
}

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
void   NativeOps::execReduce3Double(Nd4jPointer *extraPointers,int opNum,
                                    double *x,
                                    Nd4jLong *xShapeInfo,
                                    double *extraParams,
                                    double *y,
                                    Nd4jLong *yShapeInfo,
                                    double *result,
                                    Nd4jLong *resultShapeInfo) {
    NativeOpExcutioner<double>::execReduce3(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo);
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParamsVals
 * @param y
 * @param yShapeInfo
 */
double   NativeOps::execReduce3ScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                            double *x,
                                            Nd4jLong *xShapeInfo,
                                            double *extraParams,
                                            double *y,
                                            Nd4jLong *yShapeInfo) {
    return NativeOpExcutioner<double>::execReduce3Scalar(opNum,x,xShapeInfo,extraParams,y,yShapeInfo);
}
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
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execReduce3Double(Nd4jPointer *extraPointers,int opNum,
                                    double *x,
                                    Nd4jLong *xShapeInfo,
                                    double *extraParams,
                                    double *y,
                                    Nd4jLong *yShapeInfo,
                                    double *result,
                                    Nd4jLong *resultShapeInfo,
                                    int *dimension,
                                    int dimensionLength) {

    if (extraPointers == nullptr || extraPointers[2] == 0) {
        NativeOpExcutioner<double>::execReduce3(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength);
    } else {
        // going tad-way
        auto tadShapeInfo = reinterpret_cast<Nd4jLong *> (extraPointers[0]);
        auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

        NativeOpExcutioner<double>::execReduce3TAD(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets);
    }

}
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
void   NativeOps::execScalarDouble(
        Nd4jPointer *extraPointers,
        int opNum,
        double *x,
        Nd4jLong xStride,
        double *result,
        Nd4jLong resultStride,
        double scalar,
        double *extraParams,
        Nd4jLong n) {
    NativeOpExcutioner<double>::execScalar(
            opNum,
            x,
            xStride,
            result,
            resultStride,
            scalar,
            extraParams,
            n);

}

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
void NativeOps::execScalarDouble(
        Nd4jPointer *extraPointers,
        int opNum,
        double *x,
        Nd4jLong *xShapeInfo,
        double *result,
        Nd4jLong *resultShapeInfo,
        double scalar,
        double *extraParams) {
    NativeOpExcutioner<double>::execScalar(
            opNum,
            x,
            xShapeInfo,
            result,
            resultShapeInfo,
            scalar,
            extraParams);
}

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
void NativeOps::execScalarDouble(
        Nd4jPointer *extraPointers,
        int opNum,
        double *x,
        Nd4jLong *xShapeInfo,
        double *result,
        Nd4jLong *resultShapeInfo,
        double scalar,
        double *extraParams,
        Nd4jLong n,
        Nd4jLong *xIndexes,
        Nd4jLong *resultIndexes) {
    NativeOpExcutioner<double>::execScalar(
            opNum,
            x,
            xShapeInfo,
            result,
            resultShapeInfo,
            scalar,
            extraParams,
            xIndexes,
            resultIndexes);

}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
double   NativeOps::execSummaryStatsScalarDouble(Nd4jPointer *extraPointers, int opNum,double *x,
                                                 Nd4jLong *xShapeInfo,
                                                 double *extraParams,bool biasCorrected) {
    return NativeOpExcutioner<double>::execSummaryStatsScalar(
            opNum,
            x,
            xShapeInfo,
            extraParams,biasCorrected);
}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execSummaryStatsDouble(Nd4jPointer *extraPointers, int opNum,
                                         double *x,
                                         Nd4jLong *xShapeInfo,
                                         double *extraParams,
                                         double *result,
                                         Nd4jLong *resultShapeInfo,bool biasCorrected) {
    NativeOpExcutioner<double>::execSummaryStats(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            biasCorrected);
}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execSummaryStatsDouble(Nd4jPointer *extraPointers, int opNum,double *x,
                                         Nd4jLong *xShapeInfo,
                                         double *extraParams,
                                         double *result,
                                         Nd4jLong *resultShapeInfo,
                                         int *dimension, int dimensionLength,bool biasCorrected) {
    NativeOpExcutioner<double>::execSummaryStats(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,biasCorrected);

}
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
void   NativeOps::execTransformDouble(Nd4jPointer *extraPointers, int opNum,
                                      double *dx,
                                      Nd4jLong xStride,
                                      double *result,
                                      Nd4jLong resultStride,
                                      double *extraParams, Nd4jLong n) {
    NativeOpExcutioner<double>::execTransform(opNum,dx,xStride,result,resultStride,extraParams,n);
}

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
void   NativeOps::execTransformDouble(
        Nd4jPointer *extraPointers, int opNum,
        double *dx,
        Nd4jLong *xShapeInfo,
        double *result,
        Nd4jLong *resultShapeInfo,
        double *extraParams) {

    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

    NativeOpExcutioner<double>::execTransform(
            opNum,
            dx,
            xShapeInfo,
            result,
            resultShapeInfo,
            extraParams,
            tadShapeInfo,
            tadOffsets);
}

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
void   NativeOps::execTransformDouble(
        Nd4jPointer *extraPointers,
        int opNum,
        double *dx,
        Nd4jLong *xShapeInfo,
        double *result,
        Nd4jLong *resultShapeInfo,
        double *extraParams,
        Nd4jLong *xIndexes,
        Nd4jLong *resultIndexes) {
    NativeOpExcutioner<double>::execTransform(
            opNum,
            dx,
            xShapeInfo,
            result,
            resultShapeInfo,
            extraParams,
            xIndexes,
            resultIndexes, nullptr, nullptr);

}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
float   NativeOps::execIndexReduceScalarFloat(Nd4jPointer *extraPointers, int opNum,
                                              float *x,
                                              Nd4jLong *xShapeInfo,
                                              float *extraParams) {
    return NativeOpExcutioner<float>::execIndexReduceScalar(opNum,x,xShapeInfo,extraParams);
}

float   NativeOps::execIndexReduceScalarHalf(Nd4jPointer *extraPointers, int opNum,
                                             float16 *x,
                                             Nd4jLong *xShapeInfo,
                                             float16 *extraParams) {
    // no-op
    return 0.0;
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execIndexReduceFloat(Nd4jPointer *extraPointers, int opNum,
                                       float *x,
                                       Nd4jLong *xShapeInfo,
                                       float *extraParams,
                                       float *result,
                                       Nd4jLong *resultShapeInfo,
                                       int *dimension, int dimensionLength) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    NativeOpExcutioner<float>::execIndexReduce(opNum,x,xShapeInfo,extraParams,result,resultShapeInfo,dimension,dimensionLength,tadShapeInfo, tadOffsets);
}

void   NativeOps::execIndexReduceHalf(Nd4jPointer *extraPointers, int opNum,
                                      float16 *x,
                                      Nd4jLong *xShapeInfo,
                                      float16 *extraParams,
                                      float16 *result,
                                      Nd4jLong *resultShapeInfo,
                                      int *dimension, int dimensionLength) {
    // no-op
}

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
void   NativeOps::execBroadcastFloat(Nd4jPointer *extraPointers,int opNum,
                                     float *x,
                                     Nd4jLong *xShapeInfo,
                                     float *y,
                                     Nd4jLong *yShapeInfo,
                                     float *result,Nd4jLong *resultShapeInfo,
                                     int *dimension, int dimensionLength) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    auto tadShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[2]);
    auto tadOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[3]);
    NativeOpExcutioner<float>::execBroadcast(opNum,x,xShapeInfo,y,yShapeInfo,result, resultShapeInfo, dimension,dimensionLength,
                                             tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

void   NativeOps::execBroadcastHalf(Nd4jPointer *extraPointers,int opNum,
                                    float16 *x,
                                    Nd4jLong *xShapeInfo,
                                    float16 *y,
                                    Nd4jLong *yShapeInfo,
                                    float16 *result,Nd4jLong *resultShapeInfo,
                                    int *dimension, int dimensionLength) {
    // no-op
}

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
void   NativeOps::execPairwiseTransformFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *dx,
        Nd4jLong xStride,
        float *y,
        Nd4jLong yStride,
        float *result,
        Nd4jLong resultStride,
        float *extraParams, Nd4jLong n) {
    NativeOpExcutioner<float>::execPairwiseTransform(
            opNum,
            dx,
            xStride,
            y,
            yStride,
            result,
            resultStride,
            extraParams,
            n);
}

void   NativeOps::execPairwiseTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *dx,
        Nd4jLong xStride,
        float16 *y,
        Nd4jLong yStride,
        float16 *result,
        Nd4jLong resultStride,
        float16 *extraParams, Nd4jLong n) {
    // no-op
}

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
void NativeOps::execPairwiseTransformFloat(
        Nd4jPointer *extraPointers,
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
        Nd4jLong *resultIndexes) {
    NativeOpExcutioner<float>::execPairwiseTransform(
            opNum,
            dx,
            xShapeInfo,
            y,
            yShapeInfo,
            result,
            resultShapeInfo,
            extraParams,
            xIndexes,
            yIndexes,
            resultIndexes);

}

void NativeOps::execPairwiseTransformHalf(
        Nd4jPointer *extraPointers,
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
        Nd4jLong *resultIndexes) {
    // no-op
}

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
void NativeOps::execPairwiseTransformFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *dx,
        Nd4jLong *xShapeInfo,
        float *y,
        Nd4jLong *yShapeInfo,
        float *result,
        Nd4jLong * resultShapeInfo,
        float *extraParams) {
    NativeOpExcutioner<float>::execPairwiseTransform(opNum,dx,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,extraParams);
}

void NativeOps::execPairwiseTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *dx,
        Nd4jLong *xShapeInfo,
        float16 *y,
        Nd4jLong *yShapeInfo,
        float16 *result,
        Nd4jLong *resultShapeInfo,
        float16 *extraParams) {
    // no-op
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execReduceFloat(Nd4jPointer *extraPointers,int opNum,
                                  float *x,
                                  Nd4jLong *xShapeInfo,
                                  float *extraParams,
                                  float *result,
                                  Nd4jLong *resultShapeInfo) {
    int dimension[1] = {MAX_DIMENSION};
    NativeOpExcutioner<float>::execReduce(opNum,
                                          x,
                                          xShapeInfo,
                                          extraParams,
                                          result,
                                          resultShapeInfo,
                                          dimension,
                                          1,
                                          nullptr,
                                          nullptr);
}

void   NativeOps::execReduceHalf(Nd4jPointer *extraPointers,int opNum,
                                 float16 *x,
                                 Nd4jLong *xShapeInfo,
                                 float16 *extraParams,
                                 float16 *result,
                                 Nd4jLong *resultShapeInfo) {
    // no-op
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execReduceFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *x,
        Nd4jLong *xShapeInfo,
        float *extraParams,
        float *result,
        Nd4jLong *resultShapeInfo,
        int *dimension,
        int dimensionLength) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    NativeOpExcutioner<float>::execReduce(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength, tadShapeInfo, tadOffsets);
}


/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @return
 */
float NativeOps::execReduceScalarFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *x,
        Nd4jLong *xShapeInfo,
        float *extraParams) {
    return NativeOpExcutioner<float>::execReduceScalar(opNum,x,xShapeInfo,extraParams);
}

float NativeOps::execReduceScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        Nd4jLong *xShapeInfo,
        float16 *extraParams) {
    // no-op

    return 0.0;
}

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
void   NativeOps::execReduce3Float(
        Nd4jPointer *extraPointers,
        int opNum,
        float *x,
        Nd4jLong *xShapeInfo,
        float *extraParams,
        float *y,
        Nd4jLong *yShapeInfo,
        float *result,
        Nd4jLong *resultShapeInfo) {
    NativeOpExcutioner<float>::execReduce3(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            y,
            yShapeInfo,
            result,
            resultShapeInfo);
}

void   NativeOps::execReduce3Half(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        Nd4jLong *xShapeInfo,
        float16 *extraParamsVals,
        float16 *y,
        Nd4jLong *yShapeInfo,
        float16 *result,
        Nd4jLong *resultShapeInfo) {
    // no-op
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParamsVals
 * @param y
 * @param yShapeInfo
 */
float   NativeOps::execReduce3ScalarFloat(Nd4jPointer *extraPointers,int opNum,
                                          float *x,
                                          Nd4jLong *xShapeInfo,
                                          float *extraParams,
                                          float *y,
                                          Nd4jLong *yShapeInfo) {
    return NativeOpExcutioner<float>::execReduce3Scalar(opNum,x,xShapeInfo,extraParams,y,yShapeInfo);
}

float   NativeOps::execReduce3ScalarHalf(Nd4jPointer *extraPointers,int opNum,
                                         float16 *x,
                                         Nd4jLong *xShapeInfo,
                                         float16 *extraParams,
                                         float16 *y,
                                         Nd4jLong *yShapeInfo) {
    // no-op
    return 0.0;
}

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
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execReduce3Float(Nd4jPointer *extraPointers,int opNum,
                                   float *x,
                                   Nd4jLong *xShapeInfo,
                                   float *extraParams,
                                   float *y,
                                   Nd4jLong *yShapeInfo,
                                   float *result,
                                   Nd4jLong *resultShapeInfo,
                                   int *dimension,
                                   int dimensionLength) {
    if (extraPointers == nullptr || extraPointers[2] == nullptr) {
        NativeOpExcutioner<float>::execReduce3(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength);
    } else {
        // going tad-way
        auto tadShapeInfo = reinterpret_cast<Nd4jLong *> (extraPointers[0]);
        auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

        NativeOpExcutioner<float>::execReduce3TAD(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets);
    }

}

void   NativeOps::execReduce3Half(Nd4jPointer *extraPointers,int opNum,
                                  float16 *x,
                                  Nd4jLong *xShapeInfo,
                                  float16 *extraParams,
                                  float16 *y,
                                  Nd4jLong *yShapeInfo,
                                  float16 *result,
                                  Nd4jLong *resultShapeInfo,
                                  int *dimension,
                                  int dimensionLength) {
    // no-op
}

void NativeOps::execReduce3AllDouble(Nd4jPointer *extraPointers,
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
                                     Nd4jLong *yOffsets) {

    NativeOpExcutioner<double>::execReduce3All(opNum, x, xInfo, extraParamsVals, y, yInfo, result, resultShapeInfoBuffer, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);
}

void NativeOps::execReduce3AllFloat(Nd4jPointer *extraPointers,
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
                                    Nd4jLong *yOffsets) {

    NativeOpExcutioner<float>::execReduce3All(opNum, x, xInfo, extraParamsVals, y, yInfo, result, resultShapeInfoBuffer, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);
}

void NativeOps::execReduce3AllHalf(Nd4jPointer *extraPointers,
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
                                   Nd4jLong *yOffsets) {

#ifndef __ANDROID__
    // TODO: make this work with android-x86 as well
    NativeOpExcutioner<float16>::execReduce3All(opNum, x, xInfo, extraParamsVals, y, yInfo, result, resultShapeInfoBuffer, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);
#endif
}


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
void   NativeOps::execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                                  float *x,
                                  Nd4jLong xStride,
                                  float *result,
                                  Nd4jLong resultStride,
                                  float scalar,
                                  float *extraParams,
                                  Nd4jLong n) {
    NativeOpExcutioner<float>::execScalar(opNum,
                                          x,
                                          xStride,
                                          result,
                                          resultStride,
                                          scalar,
                                          extraParams,
                                          n);

}

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
void NativeOps::execScalarFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *x,
        Nd4jLong *xShapeInfo,
        float *result,
        Nd4jLong *resultShapeInfo,
        float scalar,
        float *extraParams) {
    NativeOpExcutioner<float>::execScalar(opNum,x,resultShapeInfo,result,resultShapeInfo,scalar,extraParams);

}

void NativeOps::execScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        Nd4jLong *xShapeInfo,
        float16 *result,
        Nd4jLong *resultShapeInfo,
        float scalar,
        float16 *extraParams) {
    // no-op
}

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
void NativeOps::execScalarFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *x,
        Nd4jLong *xShapeInfo,
        float *result,
        Nd4jLong *resultShapeInfo,
        float scalar,
        float *extraParams,
        Nd4jLong *xIndexes,
        Nd4jLong *resultIndexes) {
    NativeOpExcutioner<float>::execScalar(
            opNum,
            x,
            xShapeInfo,
            result,
            resultShapeInfo,
            scalar,
            extraParams,
            xIndexes,
            resultIndexes);

}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
float   NativeOps::execSummaryStatsScalarFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *x,
        Nd4jLong *xShapeInfo,
        float *extraParams,bool biasCorrected) {
    return NativeOpExcutioner<float>::execSummaryStatsScalar(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            biasCorrected);
}

float   NativeOps::execSummaryStatsScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        Nd4jLong *xShapeInfo,
        float16 *extraParams,bool biasCorrected) {
    // no-op
    return 0.0;
}

void   NativeOps::execScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        Nd4jLong xStride,
        float16 *result,
        Nd4jLong resultStride,
        float scalar,
        float16 *extraParams,
        Nd4jLong n) {
    // no-op
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 */
void   NativeOps::execSummaryStatsFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *x,
        Nd4jLong *xShapeInfo,
        float *extraParams,
        float *result,
        Nd4jLong *resultShapeInfo,bool biasCorrected) {
    NativeOpExcutioner<float>::execSummaryStats(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            biasCorrected);
}


void   NativeOps::execSummaryStatsHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        Nd4jLong *xShapeInfo,
        float16 *extraParams,
        float16 *result,
        Nd4jLong *resultShapeInfo,bool biasCorrected) {
    // no-op
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @param result
 * @param resultShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execSummaryStatsFloat(Nd4jPointer *extraPointers,int opNum,float *x,
                                        Nd4jLong *xShapeInfo,
                                        float *extraParams,
                                        float *result,
                                        Nd4jLong *resultShapeInfo,
                                        int *dimension, int dimensionLength,bool biasCorrected) {
    NativeOpExcutioner<float>::execSummaryStats(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            biasCorrected);
}


void   NativeOps::execSummaryStatsHalf(Nd4jPointer *extraPointers,int opNum,float16 *x,
                                       Nd4jLong *xShapeInfo,
                                       float16 *extraParams,
                                       float16 *result,
                                       Nd4jLong *resultShapeInfo,
                                       int *dimension, int dimensionLength,bool biasCorrected) {
    // no-op
}
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
void   NativeOps::execTransformFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *dx,
        Nd4jLong xStride,
        float *result,
        Nd4jLong resultStride,
        float *extraParams, Nd4jLong n) {
    NativeOpExcutioner<float>::execTransform(opNum,dx,xStride,result,resultStride,extraParams,n);
}

void   NativeOps::execTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *dx,
        Nd4jLong xStride,
        float16 *result,
        Nd4jLong resultStride,
        float16 *extraParams, Nd4jLong n) {
    // no-op
}

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
void   NativeOps::execTransformFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *dx,
        Nd4jLong *xShapeInfo,
        float *result,
        Nd4jLong *resultShapeInfo,
        float *extraParams) {

    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);

    NativeOpExcutioner<float>::execTransform(
            opNum,
            dx,
            xShapeInfo,
            result,
            resultShapeInfo,
            extraParams, tadShapeInfo, tadOffsets);
}

void   NativeOps::execTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *dx,
        Nd4jLong *xShapeInfo,
        float16 *result,
        Nd4jLong *resultShapeInfo,
        float16 *extraParams) {

}

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
void   NativeOps::execTransformFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        float *dx,
        Nd4jLong *xShapeInfo,
        float *result,
        Nd4jLong *resultShapeInfo,
        float *extraParams,
        Nd4jLong *xIndexes,
        Nd4jLong *resultIndexes) {
    NativeOpExcutioner<float>::execTransform(
            opNum,
            dx,
            xShapeInfo,
            result,
            resultShapeInfo,
            extraParams,
            xIndexes,
            resultIndexes, nullptr, nullptr);
}

void   NativeOps::execTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *dx,
        Nd4jLong *xShapeInfo,
        float16 *result,
        Nd4jLong *resultShapeInfo,
        float16 *extraParams,
        Nd4jLong *xIndexes,
        Nd4jLong *resultIndexes) {
    // no-op
}



template <typename T>
void flattenGeneric(Nd4jPointer *extraPointers,
                    int offset,
                    char order,
                    T *result,
                    Nd4jLong *resultShapeInfo,
                    T *input,
                    Nd4jLong *inputShapeInfo) {
    int numOnes = 0;
    auto shape = shape::shapeOf(inputShapeInfo);
    int wholeRank = shape::rank(inputShapeInfo);
    for(int i = 0; i < wholeRank; i++) {
        if(shape[i] == 1)
            numOnes++;
    }



    //start at the given offset
    result += offset;
    char inputOrder = shape::order(inputShapeInfo);
    auto len = shape::length(inputShapeInfo);
    auto resultEleStride = shape::elementWiseStride(resultShapeInfo);
    auto inputEleStride = shape::elementWiseStride(inputShapeInfo);
    Nd4jLong numTads, stride;
    int dimension, dimensionLength;
    int rank = shape::rank(inputShapeInfo);
    auto xStride = shape::stride(inputShapeInfo);
    auto xShape = shape::shapeOf(inputShapeInfo);

    dimensionLength = 1;
    if(order == 'f') {
        dimension = 0;
    }
    else {
        dimension = rank - 1;
    }
    stride  = xStride[dimension];
    // numTads is product of length of all dimensions excluding
    // the one we do the tad on
    numTads = 1;
    for (int i = 0; i < rank; i++) {
        if (i != dimension)
            numTads *= xShape[i];
    }

    if (inputOrder == order) {
        if (resultEleStride == 1 && inputEleStride == 1) {
            memcpy(result, input, len* sizeof(T));
        }
        else if (resultEleStride >= 1 && inputEleStride >= 1) {
            if (len < ELEMENT_THRESHOLD) {
#pragma omp simd
                for (int i = 0; i < len; i++) {
                    result[i * resultEleStride] = input[i * inputEleStride];
                }
            }
            else {
#pragma omp parallel for simd
                for (int i = 0; i < len; i++) {
                    result[i * resultEleStride] = input[i * inputEleStride];
                }
            }
        }
        else {
            int idx = 0;
            Nd4jLong coord[MAX_RANK];

            // FIXME: result[idx++] is bad idea, because of possible negative EWS
            if(order == 'f') {
                for(int i = 0; i < len; i++) {
                    shape::ind2sub(rank, xShape, i, coord);
                    auto offset = shape::getOffset(0,xShape,xStride,coord,rank);
                    result[idx++] = input[offset];

                }
            }
            else {
                for(int i = 0; i < len; i++) {
                    shape::ind2subC(rank, xShape, i, coord);
                    auto offset = shape::getOffset(0,xShape,xStride,coord,rank);
                    result[idx++] = input[offset];

                }
            }
        }
    }
    else {
        int rank = shape::rank(inputShapeInfo);
        auto xShape = shape::shapeOf(inputShapeInfo);
        auto tadShape = xShape[dimension];
        shape::TAD tad(inputShapeInfo,&dimension,dimensionLength);
        tad.createTadOnlyShapeInfo();
#pragma omp  parallel for schedule(guided) default(shared)
        for(int i = 0; i < numTads; i++) {

            Nd4jLong resultOffset;

            if (order == 'f') {
                // 1. get c ordering coordinates
                auto cIndexCoordinates = new Nd4jLong[rank - 1];
                int divisor = 1;
                for (int dim = rank - 1; dim > 0; dim--) {
                    cIndexCoordinates[dim - 1] = (i / divisor) % xShape[dim];
                    divisor *= xShape[dim];
                }


                // 2. convert to f ordering index
                int fIndex = 0;
                int multiplier = 1;
                for (int dim = 1; dim <= rank - 1; dim++) {
                    fIndex += cIndexCoordinates[dim - 1] * multiplier;
                    multiplier *= xShape[dim];
                }

                resultOffset = fIndex * tadShape;
                delete[] cIndexCoordinates;

            }
            else {
                resultOffset = i *  tadShape;
            }

            auto tadOffset = tad.tadOffset(i);
            for( int j = 0; j < tadShape; j++) {

                // TAD are returned in C ordering always
                result[resultOffset + j] = input[tadOffset + j * stride];

            }
        }
    }
}



/**
  * Concatneate multi array of the same shape together
  * along a particular dimension
  */
void NativeOps::concatFloat(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        float *result,
        Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    nd4j::SpecialMethods<float>::concatCpuGeneric(
            dimension,
            numArrays,
            data,
            inputShapeInfo,
            result,
            resultShapeInfo);

}


void NativeOps::concatHalf(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        float16 *result,
        Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    nd4j::SpecialMethods<float16>::concatCpuGeneric(
            dimension,
            numArrays,
            data,
            inputShapeInfo,
            result,
            resultShapeInfo);
}
/**
    * Concatneate multi array of the same shape together
    * along a particular dimension
    */
void NativeOps::concatDouble(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        double *result,
        Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    nd4j::SpecialMethods<double>::concatCpuGeneric(
            dimension,
            numArrays,
            data,
            inputShapeInfo,
            result,
            resultShapeInfo);

}

void NativeOps::specialConcatFloat(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        float *result,
        Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    nd4j::SpecialMethods<float>::concatCpuGeneric(
            dimension,
            numArrays,
            data,
            inputShapeInfo,
            result,
            resultShapeInfo);

}


void NativeOps::specialConcatHalf(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        float16 *result,
        Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    nd4j::SpecialMethods<float16>::concatCpuGeneric(
            dimension,
            numArrays,
            data,
            inputShapeInfo,
            result,
            resultShapeInfo);
}
/**
    * Concatneate multi array of the same shape together
    * along a particular dimension
    */
void NativeOps::specialConcatDouble(
        Nd4jPointer *extraPointers,
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        double *result,
        Nd4jLong *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    nd4j::SpecialMethods<double>::concatCpuGeneric(
            dimension,
            numArrays,
            data,
            inputShapeInfo,
            result,
            resultShapeInfo);

}

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
void NativeOps::flattenFloat(
        Nd4jPointer *extraPointers,
        int offset,
        char order,
        float *result,
        Nd4jLong *resultShapeInfo,
        float *input,
        Nd4jLong *inputShapeInfo) {
    flattenGeneric<float>(
            extraPointers,
            offset,
            order,
            result,
            resultShapeInfo,
            input,
            inputShapeInfo);
}

void NativeOps::flattenHalf(
        Nd4jPointer *extraPointers,
        int offset,
        char order,
        float16 *result,
        Nd4jLong *resultShapeInfo,
        float16 *input,
        Nd4jLong *inputShapeInfo) {
    // no-op
}

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
void NativeOps::flattenDouble(
        Nd4jPointer *extraPointers,
        int offset,
        char order,
        double *result,
        Nd4jLong *resultShapeInfo,
        double *input,
        Nd4jLong *inputShapeInfo) {
    flattenGeneric<double>(
            extraPointers,
            offset,
            order,
            result,
            resultShapeInfo,
            input,
            inputShapeInfo);
}

/**
 * This is dummy method for JNI compatibility
 * Since we'll use this from java, jni compiler would like to have method no matter what.
 */
void NativeOps::initializeDevicesAndFunctions() {

}

void NativeOps::initializeFunctions(Nd4jPointer *functions) {
    nd4j::BlasHelper::getInstance()->initializeFunctions(functions);
}

/**
       * This method acquires memory chunk of requested size on host side
       *
       * @param pointer pointer that'll be used for allocation
       * @param memorySize memory size, in bytes
       * @param flags optional parameter
       */
Nd4jPointer NativeOps::mallocHost(Nd4jLong memorySize, int flags) {
    Nd4jPointer pointer = (Nd4jPointer) malloc(memorySize);
    if (pointer == 0)
        return 0L;
    return pointer;
}

/**
 * This method acquires memory chunk of requested size on specified device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
 * @param flags optional parameter
 */
Nd4jPointer NativeOps::mallocDevice(Nd4jLong memorySize, Nd4jPointer ptrToDeviceId, int flags) {
    // not supported
    return 0L;
}

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
int NativeOps::freeHost(Nd4jPointer pointer) {
    free(reinterpret_cast<void *>(pointer));
    return 1L;
}

/**
 * This method releases previously allocated memory space on device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param pointer pointer that'll be freed
 * @param ptrToDeviceId pointer to deviceId.
 */
int NativeOps::freeDevice(Nd4jPointer pointer, Nd4jPointer ptrToDeviceId) {
    // not supported
    return 0L;
}


/**
 * Returns the maximum number open mp threads
 */
int NativeOps::ompGetMaxThreads() {
    return omp_get_max_threads();
}

/**
 * Returns the number open mp threads
 */
int NativeOps::ompGetNumThreads() {
    return omp_get_num_threads();
}

/**
 * Sets the number of openmp threads
 */
void NativeOps::setOmpNumThreads(int threads) {
    omp_set_num_threads(threads);

}

Nd4jPointer NativeOps::createContext() {
    return 0L;
}

Nd4jPointer NativeOps::createStream() {
    return 0L;
}

Nd4jPointer NativeOps::createEvent() {
    return 0L;
}

int NativeOps::getDeviceMajor(Nd4jPointer ptrToDeviceId) {
    return 0;
}

int NativeOps::getDeviceMinor(Nd4jPointer ptrToDeviceId) {
    return 0;
}

int NativeOps::registerEvent(Nd4jPointer event, Nd4jPointer stream) {
    return 0L;
}

int NativeOps::setDevice(Nd4jPointer ptrToDeviceId) {
    return 0L;
}

Nd4jLong NativeOps::getDeviceFreeMemory(Nd4jPointer ptrToDeviceId) {
    return 0L;
}

Nd4jLong NativeOps::getDeviceTotalMemory(Nd4jPointer ptrToDeviceId) {
    return 0L;
}

int NativeOps::memcpy(Nd4jPointer dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::memcpyAsync(Nd4jPointer dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::memset(Nd4jPointer dst, int value, Nd4jLong size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::memsetAsync(Nd4jPointer dst, int value, Nd4jLong size,  int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::destroyEvent(Nd4jPointer event) {
    return 0L;
}

int NativeOps::streamSynchronize(Nd4jPointer stream) {
    return 0L;
}

int NativeOps::eventSynchronize(Nd4jPointer event) {
    return 0L;
}

int NativeOps::getAvailableDevices() {
    return 0L;
}

void NativeOps::enableDebugMode(bool reallyEnable) {
    nd4j::Environment::getInstance()->setDebug(reallyEnable);
}

void NativeOps::enableVerboseMode(bool reallyEnable) {
    nd4j::Environment::getInstance()->setVerbose(reallyEnable);
}

void NativeOps::setGridLimit(int gridSize) {
    // no-op
}

void NativeOps::tadOnlyShapeInfo(Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *target, Nd4jLong *offsets) {
    shape::TAD tad;
    tad.init(xShapeInfo, dimension, dimensionLength);
    //tad->setOutputBuffer(target);
    tad.createTadOnlyShapeInfo();
    tad.createOffsets();


    std::memcpy(reinterpret_cast<void *>(target), tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));
    std::memcpy(reinterpret_cast<void *>(offsets), tad.tadOffsets, tad.numTads * sizeof(Nd4jLong));
}

int NativeOps::memcpyConstantAsync(Nd4jLong dst, Nd4jPointer src, Nd4jLong size, int flags, Nd4jPointer reserved) {
    // no-op
    return 0L;
}

Nd4jPointer NativeOps::getConstantSpace() {
    // no-op
    return 0L;
}

template<typename T>
void pullRowsGeneric(T *x,
                     Nd4jLong *xShapeInfo,
                     T *z,
                     Nd4jLong *zShapeInfo,
                     const int n,
                     Nd4jLong *indexes,
                     Nd4jLong *tadShapeInfo,
                     Nd4jLong *tadOffsets,
                     Nd4jLong *zTadShapeInfo,
                     Nd4jLong *zTadOffsets) {
    const auto xEWS = shape::elementWiseStride(tadShapeInfo);
    const auto zEWS = shape::elementWiseStride(zTadShapeInfo);
    const auto tadLength = shape::length(tadShapeInfo);

    int elementsPerThread = n / TAD_THRESHOLD;
    int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
    _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

#pragma omp parallel for num_threads(_threads) if (n > 1) schedule(guided) default(shared)
    for (int idx = 0; idx < n; idx++) {
        Nd4jLong xTadOffsetForBlock = tadOffsets[indexes[idx]];
        Nd4jLong zTadOffsetForBlock = zTadOffsets[idx];

        T *rX = x + xTadOffsetForBlock;
        T *rZ = z + zTadOffsetForBlock;

        if (xEWS == 1 && zEWS == 1) {

#pragma omp simd
            for (int i = 0; i < tadLength; i++ ) {
                rZ[i] = rX[i];
            }
        } else if (xEWS >= 1 && zEWS >= 1) {

#pragma omp simd
            for (int i = 0; i < tadLength; i++ ) {
                rZ[i * zEWS] = rX[i * xEWS];
            }
        } else {
            auto zShape = shape::shapeOf(zTadShapeInfo);
            auto zStride = shape::stride(zTadShapeInfo);
            auto xShape = shape::shapeOf(tadShapeInfo);
            auto xStride = shape::stride(tadShapeInfo);
            auto zRank = shape::rank(zTadShapeInfo);
            auto tadRank = shape::rank(tadShapeInfo);

            Nd4jLong xCoord[MAX_RANK];
            Nd4jLong zCoord[MAX_RANK];

            for (int i = 0; i < tadLength; i++) {
                shape::ind2subC(tadRank,xShape, i, xCoord);
                shape::ind2subC(zRank,zShape, i, zCoord);

                auto xOffset = shape::getOffset(xTadOffsetForBlock, xShape, xStride, xCoord, tadRank);
                auto zOffset = shape::getOffset(zTadOffsetForBlock, zShape, zStride, zCoord, zRank);
                z[zOffset] = x[xOffset];
            }
        }
    }
}

void NativeOps::pullRowsHalf(Nd4jPointer *extraPointers, float16 *x, Nd4jLong *xShapeInfo, float16 *z, Nd4jLong *zShapeInfo, Nd4jLong n, Nd4jLong *indexes, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {
    // no-op
}

void NativeOps::pullRowsFloat(Nd4jPointer *extraPointers, float *x, Nd4jLong *xShapeInfo, float *z, Nd4jLong *zShapeInfo, Nd4jLong n, Nd4jLong *indexes, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {
    pullRowsGeneric<float>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

void NativeOps::pullRowsDouble(Nd4jPointer *extraPointers, double *x, Nd4jLong *xShapeInfo, double *z, Nd4jLong *zShapeInfo, Nd4jLong n, Nd4jLong *indexes, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {
    pullRowsGeneric<double>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

template<typename T>
void tearGeneric(T *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    const Nd4jLong tadLength = shape::length(tadShapeInfo);
    auto tadEWS = shape::elementWiseStride(tadShapeInfo);
    auto zEWS = shape::elementWiseStride(zShapeInfo);
    auto tadRank = shape::rank(tadShapeInfo);
    auto zRank = shape::rank(zShapeInfo);
    auto tadShape = shape::shapeOf(tadShapeInfo);
    auto tadStride = shape::stride(tadShapeInfo);
    auto zShape = shape::shapeOf(zShapeInfo);
    auto zStride = shape::stride(zShapeInfo);
    auto numTads = shape::length(xShapeInfo) / tadLength;

#pragma omp parallel for schedule(guided) default(shared)
    for (Nd4jLong i = 0; i < numTads; i++) {
        T *z = reinterpret_cast<T *>(targets[i]);
        T *s = x + tadOffsets[i];

        if (zEWS == 1 && tadEWS == 1) {
#pragma omp simd
            for (Nd4jLong j = 0; j < tadLength; j++) {
                z[j] = s[j];
            }
        } else if (zEWS > 0 && tadEWS > 0) {
#pragma omp simd
            for (Nd4jLong j = 0; j < tadLength; j++) {
                z[j * zEWS] = s[j * tadEWS];
            }
        } else {
            Nd4jLong xCoord[MAX_RANK];
            Nd4jLong zCoord[MAX_RANK];

            for (Nd4jLong j = 0; j < tadLength; j++) {
                shape::ind2sub(tadRank,tadShape, j, xCoord);
                shape::ind2sub(zRank, zShape, j, zCoord);

                auto xOffset = shape::getOffset(0, tadShape, tadStride, xCoord, tadRank);
                auto zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                z[zOffset] = s[xOffset];
            }
        }
    }
}

void NativeOps::tearDouble(Nd4jPointer *extraPointers, double *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    tearGeneric<double>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

void NativeOps::tearFloat(Nd4jPointer *extraPointers, float *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    tearGeneric<float>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

void NativeOps::tearHalf(Nd4jPointer *extraPointers, float16 *x, Nd4jLong *xShapeInfo, Nd4jPointer *targets, Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
    tearGeneric<float16>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

void NativeOps::averageHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jLong length, bool propagate) {
    // no-op
}

void NativeOps::averageFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jLong length, bool propagate) {
    auto x = reinterpret_cast<float **>(dx);
    nd4j::SpecialMethods<float>::averageGeneric(x, dz, n, length, propagate);
}

void NativeOps::averageDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jLong length, bool propagate) {
    auto x = reinterpret_cast<double **>(dx);
    nd4j::SpecialMethods<double>::averageGeneric(x, dz, n, length, propagate);
}

void NativeOps::accumulateHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jLong length) {
    auto x = reinterpret_cast<float16 **>(dx);
    nd4j::SpecialMethods<float16>::accumulateGeneric(x, dz, n, length);
}

void NativeOps::accumulateFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jLong length) {
    auto x = reinterpret_cast<float **>(dx);
    nd4j::SpecialMethods<float>::accumulateGeneric(x, dz, n, length);
}

void NativeOps::accumulateDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jLong length) {
    auto x = reinterpret_cast<double **>(dx);
    nd4j::SpecialMethods<double>::accumulateGeneric(x, dz, n, length);
}

void NativeOps::enableP2P(bool enable) {
    // no-op
}

void NativeOps::encodeThresholdP1Half(Nd4jPointer *extraPointers, float16 *dx, Nd4jLong N, int *dz, float threshold) {
    // TODO: to be implemented
}

void NativeOps::encodeThresholdP1Float(Nd4jPointer *extraPointers, float *dx, Nd4jLong N, int *dz, float threshold) {
    // TODO: to be implemented
}

void NativeOps::encodeThresholdP1Double(Nd4jPointer *extraPointers, double *dx, Nd4jLong N, int *dz, float threshold) {
    // TODO: to be implemented
}


void NativeOps::encodeThresholdP2Int(Nd4jPointer *extraPointers, int *dx, Nd4jLong N, int *dz) {
    // TODO: to be implemented
}

void NativeOps::encodeThresholdP3Float(Nd4jPointer *extraPointers, float *dx, int *offsets, Nd4jLong N, int *dz){
    // offsets won't be used here

    // TODO: to be implemented
}

void NativeOps::encodeThresholdP3Double(Nd4jPointer *extraPointers, double *dx, int *offsets, Nd4jLong N, int *dz){
    // offsets won't be used here

    // TODO: to be implemented
}

void NativeOps::encodeThresholdP3Half(Nd4jPointer *extraPointers, float16 *dx, int *offsets, Nd4jLong N, int *dz){
    // offsets won't be used here

    // TODO: to be implemented
}

void NativeOps::decodeThresholdFloat(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float *dz){
    // TODO: to be implemented
}

void NativeOps::decodeThresholdHalf(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float16 *dz){
    // TODO: to be implemented
}

void NativeOps::decodeThresholdDouble(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, double *dz){
    // TODO: to be implemented
}

bool NativeOps::isP2PAvailable() {
    // always TRUE for cpu backend
    return true;
}

void NativeOps::checkP2P() {
    // no-op
}

template<typename T>
void shuffleGeneric(T **dX, Nd4jLong **xShapeInfo, T **dZ, Nd4jLong **zShapeInfo, int N, int *shuffleMap, Nd4jLong **tadOnlyShapeInfo, Nd4jLong **tadOffsets) {


#pragma omp parallel for if (N > 1) default(shared)
    for (int f = 0; f < N; f++) {
        auto x = reinterpret_cast<T *>(dX[f]);
        //auto z = reinterpret_cast<T *>(dZ[f]);

        auto tadOffset = reinterpret_cast<Nd4jLong *>(tadOffsets[f]);


        const auto tadLength = shape::length(tadOnlyShapeInfo[f]);
        auto tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
        auto tadRank = shape::rank(tadOnlyShapeInfo[f]);
        auto numTads = shape::length(xShapeInfo[f]) / tadLength;

        auto tadShape = shape::shapeOf(tadOnlyShapeInfo[f]);
        auto tadStride = shape::stride(tadOnlyShapeInfo[f]);

        //printf("Array: [%i], tadEWS: [%i], tadLength: [%i]\n", f, tadEWS, tadLength);

        // TODO: omp *probably* has no sense here, since 99% of uses for this method will be inside DataSet. but worth a check

        for (Nd4jLong r = 0; r < numTads; r++) {
            if (shuffleMap[r] < 0)
                continue;

            auto oldOffset = tadOffset[r];
            auto newOffset = tadOffset[shuffleMap[r]];

            auto rX = x + oldOffset;
            auto rY = x + newOffset;

            if (tadEWS == 1) {

#pragma omp simd
                for (Nd4jLong i = 0; i < tadLength; i++) {
                    nd4j::math::nd4j_swap<T>(rX[i], rY[i]);
                }

            } else {
                // ind2sub branch
#pragma omp parallel for schedule(guided) if (N == 1 && tadLength > 512) default(shared)
                for (Nd4jLong i = 0; i < tadLength; i++) {
                    Nd4jLong xCoord[MAX_RANK];
                    Nd4jLong yCoord[MAX_RANK];

                    shape::ind2subC(tadRank,tadShape, i, xCoord);
                    shape::ind2subC(tadRank,tadShape, i, yCoord);

                    auto xOffset = shape::getOffset(oldOffset, tadShape, tadStride, xCoord, tadRank);
                    auto yOffset = shape::getOffset(newOffset, tadShape, tadStride, yCoord, tadRank);

                    nd4j::math::nd4j_swap<T>(x[xOffset], x[yOffset]);
                }

            }

        }

    }
}

void NativeOps::shuffleFloat(Nd4jPointer *extras,
                             Nd4jPointer *dx,
                             Nd4jPointer *xShapeInfo,
                             Nd4jPointer *dz,
                             Nd4jPointer *zShapeInfo,
                             int N,
                             int *shuffleMap,
                             Nd4jPointer *tadShapeInfo,
                             Nd4jPointer *tadOffsets) {
    auto x = reinterpret_cast<float **>(dx);
    auto z = reinterpret_cast<float **>(dz);
    auto xShape = reinterpret_cast<Nd4jLong **>(xShapeInfo);
    auto zShape = reinterpret_cast<Nd4jLong **>(zShapeInfo);
    auto tadOnlyShapeInfo = reinterpret_cast<Nd4jLong **>(tadShapeInfo);
    auto tadOffset = reinterpret_cast<Nd4jLong **>(tadOffsets);

    shuffleGeneric<float>(x,
                          xShape,
                          z,
                          zShape,
                          N,
                          shuffleMap,
                          tadOnlyShapeInfo,
                          tadOffset);
}

void NativeOps::shuffleDouble(Nd4jPointer *extras,
                              Nd4jPointer *dx,
                              Nd4jPointer *xShapeInfo,
                              Nd4jPointer *dz,
                              Nd4jPointer *zShapeInfo,
                              int N,
                              int *shuffleMap,
                              Nd4jPointer *tadShapeInfo,
                              Nd4jPointer *tadOffsets) {
    auto x = reinterpret_cast<double **>(dx);
    auto z = reinterpret_cast<double **>(dz);
    auto xShape = reinterpret_cast<Nd4jLong **>(xShapeInfo);
    auto zShape = reinterpret_cast<Nd4jLong **>(zShapeInfo);
    auto tadOnlyShapeInfo = reinterpret_cast<Nd4jLong **>(tadShapeInfo);
    auto tadOffset = reinterpret_cast<Nd4jLong **>(tadOffsets);

    shuffleGeneric<double>(x,
                           xShape,
                           z,
                           zShape,
                           N,
                           shuffleMap,
                           tadOnlyShapeInfo,
                           tadOffset);
}

void NativeOps::shuffleHalf(Nd4jPointer *extras,
                            Nd4jPointer *dx,
                            Nd4jPointer *xShapeInfo,
                            Nd4jPointer *dz,
                            Nd4jPointer *zShapeInfo,
                            int N,
                            int *shuffleMap,
                            Nd4jPointer *tadShapeInfo,
                            Nd4jPointer *tadOffsets) {
    // no-op
}

void NativeOps::execMetaPredicateReduceFloat(Nd4jPointer *extras,
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
                                             bool scalarReturned) {
    // no-op
}

bool NativeOps::isExperimentalEnabled() {
    return experimentalSupport;
}

void NativeOps::execMetaPredicateShapeFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float *dx, Nd4jLong *xShapeInfo, float *dy, Nd4jLong *yShapeInfo, float *dz, Nd4jLong *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
    // no-op;
}

void NativeOps::setOmpMinThreads(int threads) {
    // TODO: to be implemented
}

void NativeOps::execMetaPredicateStridedFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float *dx, Nd4jLong xStride, float *dy, Nd4jLong yStride, float *dz, Nd4jLong zStride, float *extraA, float *extraB, float scalarA, float scalarB) {
    // no-op
}

void NativeOps::execMetaPredicateShapeDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, double *dx, Nd4jLong *xShapeInfo, double *dy, Nd4jLong *yShapeInfo, double *dz, Nd4jLong *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB) {
    // no-op;
}

void NativeOps::execMetaPredicateStridedDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, double *dx, Nd4jLong xStride, double *dy, Nd4jLong yStride, double *dz, Nd4jLong zStride, double *extraA, double *extraB, double scalarA, double scalarB) {
    // no-op
}

void NativeOps::execMetaPredicateShapeHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float16 *dx, Nd4jLong *xShapeInfo, float16 *dy, Nd4jLong *yShapeInfo, float16 *dz, Nd4jLong *zShapeInfo, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    // no-op;
}

void NativeOps::execMetaPredicateStridedHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float16 *dx, Nd4jLong xStride, float16 *dy, Nd4jLong yStride, float16 *dz, Nd4jLong zStride, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    // no-op
}

int NativeOps::getDevice() {
    return 0;
}


void NativeOps::execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                                float *x,
                                Nd4jLong *xShapeInfo,
                                float *z,
                                Nd4jLong *zShapeInfo,
                                float *scalars,
                                float *extraParams,
                                int *dimension,
                                int dimensionLength) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    auto tadShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[2]);
    auto tadOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[3]);

    NativeOpExcutioner<float>::execScalar(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            z,
            zShapeInfo,
            scalars,
            dimension,
            dimensionLength,
            tadShapeInfo,
            tadOffsets,
            tadShapeInfoZ,
            tadOffsetsZ);
}

void NativeOps::execScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                 double *x,
                                 Nd4jLong *xShapeInfo,
                                 double *z,
                                 Nd4jLong *zShapeInfo,
                                 double *scalars,
                                 double *extraParams,
                                 int *dimension,
                                 int dimensionLength) {
    auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
    auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[1]);
    auto tadShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[2]);
    auto tadOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[3]);

    NativeOpExcutioner<double>::execScalar(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            z,
            zShapeInfo,
            scalars,
            dimension,
            dimensionLength,
            tadShapeInfo,
            tadOffsets,
            tadShapeInfoZ,
            tadOffsetsZ);
}

void NativeOps::execScalarHalf(Nd4jPointer *extraPointers,int opNum,
                               float16 *x,
                               Nd4jLong *xShapeInfo,
                               float16 *z,
                               Nd4jLong *zShapeInfo,
                               float16 *scalars,
                               float16 *extraParams,
                               int *dimension,
                               int dimensionLength) {
//    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
//    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
//    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[2]);
//    int *tadOffsetsZ = reinterpret_cast<int *>(extraPointers[3]);
    // no-op yet, halfs
}

const char * NativeOps::getDeviceName(Nd4jPointer ptrToDeviceId) {
    if (!nameSet) {
        name = reinterpret_cast<char *>(malloc(256 * sizeof(char)));

        CHECK_ALLOC(name, "Failed to allocate new string buffer");

        std::memset(name, 0, 256 * sizeof(char));
        nameSet = true;

        // TODO: provide proper CPU model name here
        sprintf(name, "x86-compatible CPU");
    }


    return name;
}


void NativeOps::execAggregateFloat(Nd4jPointer *extraPointers,int opNum,
                                   float **arguments,
                                   int numArguments,
                                   Nd4jLong **shapeArguments,
                                   int numShapeArguments,
                                   int *indexArguments,
                                   int numIndexArguments,
                                   int **intArrays,
                                   int numIntArrays,
                                   float *realArguments,
                                   int numRealArguments) {

    NativeOpExcutioner<float>::execAggregate(opNum,
                                             arguments,
                                             numArguments,
                                             shapeArguments,
                                             numShapeArguments,
                                             indexArguments,
                                             numIndexArguments,
                                             intArrays,
                                             numIntArrays,
                                             realArguments,
                                             numRealArguments);
}

void NativeOps::execAggregateDouble(Nd4jPointer *extraPointers,int opNum,
                                    double **arguments,
                                    int numArguments,
                                    Nd4jLong **shapeArguments,
                                    int numShapeArguments,
                                    int *indexArguments,
                                    int numIndexArguments,
                                    int **intArrays,
                                    int numIntArrays,
                                    double *realArguments,
                                    int numRealArguments) {

    NativeOpExcutioner<double>::execAggregate(opNum,
                                              arguments,
                                              numArguments,
                                              shapeArguments,
                                              numShapeArguments,
                                              indexArguments,
                                              numIndexArguments,
                                              intArrays,
                                              numIntArrays,
                                              realArguments,
                                              numRealArguments);
}

void NativeOps::execAggregateHalf(Nd4jPointer *extraPointers,int opNum,
                                  float16 **arguments,
                                  int numArguments,
                                  Nd4jLong **shapeArguments,
                                  int numShapeArguments,
                                  int *indexArguments,
                                  int numIndexArguments,
                                  int **intArrays,
                                  int numIntArrays,
                                  float16 *realArguments,
                                  int numRealArguments) {

    // TODO: add this at some point
    //NativeOpExcutioner<float16>::execAggregate(opNum, arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
}



void NativeOps::execAggregateBatchFloat(Nd4jPointer *extraPointers,
                                        int numAggregates,
                                        int opNum,
                                        int maxArgs,
                                        int maxShapes,
                                        int maxIntArrays,
                                        int maxIntArraySize,
                                        int maxIdx,
                                        int maxReals,
                                        void *ptrToArguments) {

    //nd4j_printf("numAggregates: [%i]; opNum: [%i]; maxArgs: [%i]; maxShapes: [%i]; maxIntArrays: [%i]; maxIntArraySize: [%i]; maxIdx: [%i]; maxReals: [%i];\n", numAggregates, opNum, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals);

    // probably, we don't want too much threads as usually
    int _threads = nd4j::math::nd4j_min<int>(numAggregates, omp_get_max_threads());

    nd4j::PointersHelper<float> helper(ptrToArguments,
                                       numAggregates,
                                       maxArgs,
                                       maxShapes,
                                       maxIntArrays,
                                       maxIntArraySize,
                                       maxIdx,
                                       maxReals);

    // special case here, we prefer spread arrangement here, all threads are detached from each other
#pragma omp parallel for num_threads(_threads) schedule(guided) proc_bind(close) default(shared)
    for (int i = 0; i < numAggregates; i++) {
        auto intArrays = new int *[maxIntArrays];

        auto arguments = helper.getArguments(i);
        auto shapes = helper.getShapeArguments(i);
        auto idxArg = helper.getIndexArguments(i);
        auto realArg = helper.getRealArguments(i);

        for (int e = 0; e < maxIntArrays; e++) {
            intArrays[e] = helper.getIntArrayArguments(i, e);
        }

        execAggregateFloat(extraPointers,
                           opNum,
                           arguments,
                           helper.getNumArguments(i),
                           shapes,
                           helper.getNumShapeArguments(i),
                           idxArg,
                           helper.getNumIndexArguments(i),
                           reinterpret_cast<int **>(intArrays),
                           helper.getNumIntArrayArguments(i),
                           realArg,
                           helper.getNumRealArguments(i));

        delete [] intArrays;
    }
}


void NativeOps::execAggregateBatchDouble(Nd4jPointer *extraPointers,
                                         int numAggregates,
                                         int opNum,
                                         int maxArgs,
                                         int maxShapes,
                                         int maxIntArrays,
                                         int maxIntArraySize,
                                         int maxIdx,
                                         int maxReals,
                                         void *ptrToArguments) {

    // probably, we don't want too much threads as usually
    int _threads = nd4j::math::nd4j_min<int>(numAggregates, omp_get_max_threads());

    nd4j::PointersHelper<double> helper(ptrToArguments,
                                        numAggregates,
                                        maxArgs,
                                        maxShapes,
                                        maxIntArrays,
                                        maxIntArraySize,
                                        maxIdx,
                                        maxReals);

    // special case here, we prefer spread arrangement here, all threads are detached from each other
#pragma omp parallel for num_threads(_threads) schedule(guided) proc_bind(spread) default(shared)
    for (int i = 0; i < numAggregates; i++) {
        auto intArrays = new int *[maxIntArrays];

        auto arguments = helper.getArguments(i);
        auto shapes = helper.getShapeArguments(i);
        auto idxArg = helper.getIndexArguments(i);
        auto realArg = helper.getRealArguments(i);

        for (int e = 0; e < maxIntArrays; e++) {
            intArrays[e] = helper.getIntArrayArguments(i, e);
        }

        execAggregateDouble(extraPointers,
                            opNum,
                            arguments,
                            helper.getNumArguments(i),
                            shapes,
                            helper.getNumShapeArguments(i),
                            idxArg,
                            helper.getNumIndexArguments(i),
                            intArrays,
                            helper.getNumIntArrayArguments(i),
                            realArg,
                            helper.getNumRealArguments(i));

        delete [] intArrays;
    }


}

void NativeOps::execAggregateBatchHalf(Nd4jPointer *extraPointers,
                                       int numAggregates,
                                       int opNum,
                                       int maxArgs,
                                       int maxShapes,
                                       int maxIntArrays,
                                       int maxIntArraySize,
                                       int maxIdx,
                                       int maxReals,
                                       void *ptrToArguments) {
    // TODO: add support for fp16
}

void NativeOps::execRandomFloat(Nd4jPointer *extraPointers,
                                int opNum,
                                Nd4jPointer state,
                                float *z,
                                Nd4jLong *zShapeBuffer,
                                float *extraArguments) {
    NativeOpExcutioner<float>::execRandom(opNum,
                                          state,
                                          z,
                                          zShapeBuffer,
                                          extraArguments);
}

void NativeOps::execRandomFloat(Nd4jPointer *extraPointers,
                                int opNum,
                                Nd4jPointer state,
                                float *x,
                                Nd4jLong *xShapeBuffer,
                                float *y,
                                Nd4jLong *yShapeBuffer,
                                float *z,
                                Nd4jLong *zShapeBuffer,
                                float *extraArguments) {
    NativeOpExcutioner<float>::execRandom(opNum,
                                          state,
                                          x,
                                          xShapeBuffer,
                                          y,
                                          yShapeBuffer,
                                          z,
                                          zShapeBuffer,
                                          extraArguments);
}

void NativeOps::execRandomFloat(Nd4jPointer *extraPointers,
                                int opNum,
                                Nd4jPointer state,
                                float *x,
                                Nd4jLong *xShapeBuffer,
                                float *z,
                                Nd4jLong *zShapeBuffer, float *extraArguments) {
    NativeOpExcutioner<float>::execRandom(opNum,
                                          state,
                                          x,
                                          xShapeBuffer,
                                          z,
                                          zShapeBuffer,
                                          extraArguments);
}


void NativeOps::execRandomDouble(Nd4jPointer *extraPointers,
                                 int opNum,
                                 Nd4jPointer state,
                                 double *z,
                                 Nd4jLong *zShapeBuffer,
                                 double *extraArguments) {
    NativeOpExcutioner<double>::execRandom(opNum,
                                           state,
                                           z,
                                           zShapeBuffer,
                                           extraArguments);
}

void NativeOps::execRandomDouble(Nd4jPointer *extraPointers,
                                 int opNum,
                                 Nd4jPointer state,
                                 double *x,
                                 Nd4jLong *xShapeBuffer,
                                 double *y,
                                 Nd4jLong *yShapeBuffer,
                                 double *z,
                                 Nd4jLong *zShapeBuffer,
                                 double *extraArguments) {
    NativeOpExcutioner<double>::execRandom(opNum,
                                           state,
                                           x,
                                           xShapeBuffer,
                                           y,
                                           yShapeBuffer,
                                           z,
                                           zShapeBuffer,
                                           extraArguments);
}

void NativeOps::execRandomDouble(Nd4jPointer *extraPointers,
                                 int opNum,
                                 Nd4jPointer state,
                                 double *x,
                                 Nd4jLong *xShapeBuffer,
                                 double *z,
                                 Nd4jLong *zShapeBuffer,
                                 double *extraArguments) {
    NativeOpExcutioner<double>::execRandom(opNum,
                                           state,
                                           x,
                                           xShapeBuffer,
                                           z,
                                           zShapeBuffer,
                                           extraArguments);
}


void NativeOps::execRandomHalf(Nd4jPointer *extraPointers,
                               int opNum,
                               Nd4jPointer state,
                               float16 *z,
                               Nd4jLong *zShapeBuffer,
                               float16 *extraArguments) {
    //NativeOpExcutioner<float16>::execRandom(opNum, state, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomHalf(Nd4jPointer *extraPointers,
                               int opNum,
                               Nd4jPointer state,
                               float16 *x,
                               Nd4jLong *xShapeBuffer,
                               float16 *y,
                               Nd4jLong *yShapeBuffer,
                               float16 *z,
                               Nd4jLong *zShapeBuffer,
                               float16 *extraArguments) {
    //NativeOpExcutioner<float16>::execRandom(opNum, state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomHalf(Nd4jPointer *extraPointers,
                               int opNum,
                               Nd4jPointer state,
                               float16 *x,
                               Nd4jLong *xShapeBuffer,
                               float16 *z,
                               Nd4jLong *zShapeBuffer,
                               float16 *extraArguments) {
    //NativeOpExcutioner<float16>::execRandom(opNum, state, x, xShapeBuffer, z, zShapeBuffer, extraArguments);
}



Nd4jPointer NativeOps::initRandom(Nd4jPointer *extraPointers, long seed, long bufferSize, Nd4jPointer ptrToBuffer) {
    auto ptrBuf = reinterpret_cast<long *>(ptrToBuffer);
    auto buffer = new nd4j::random::RandomBuffer(seed, bufferSize, reinterpret_cast<uint64_t *>(ptrBuf));

    nd4j::random::Xoroshiro128 generator(buffer);
    generator.refreshBuffer();

    return (Nd4jPointer) buffer;
}

void NativeOps::refreshBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom) {
    auto buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (ptrRandom);

    buffer->setSeed(seed);
    buffer->setOffset(0);
    nd4j::random::Xoroshiro128 generator(buffer);
    generator.refreshBuffer();
}

void NativeOps::reSeedBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom) {
    auto buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (ptrRandom);

    buffer->reSeed(seed);
}


void NativeOps::destroyRandom(Nd4jPointer ptrBuffer) {
    auto buffer = reinterpret_cast<nd4j::random::RandomBuffer *>(ptrBuffer);
    delete buffer;
}




/**
    * Return the length of a shape buffer
    * based on the pointer
    * @param buffer  the buffer pointer to check
    * @return
    */
int NativeOps::lengthForShapeBufferPointer(Nd4jPointer buffer) {
    auto shapeBuffer = reinterpret_cast<Nd4jLong *>(buffer);
    return shape::shapeInfoLength(shape::rank(shapeBuffer));
}


/**
  * The pointer to get the address for
  *
  * @param address the address to get the pointer
  * @return the pointer for the given address
  */

Nd4jPointer NativeOps::pointerForAddress(Nd4jLong address) {
    return reinterpret_cast<Nd4jPointer >(address);
}

void NativeOps::sortFloat(Nd4jPointer *extraPointers, float *x, Nd4jLong *xShapeInfo, bool descending) {
    NativeOpExcutioner<float>::execSort(x, xShapeInfo, descending);
}

void NativeOps::sortDouble(Nd4jPointer *extraPointers, double *x, Nd4jLong *xShapeInfo, bool descending) {
    NativeOpExcutioner<double>::execSort(x, xShapeInfo, descending);
}

void NativeOps::sortHalf(Nd4jPointer *extraPointers, float16 *x, Nd4jLong *xShapeInfo, bool descending) {
    //NativeOpExcutioner<float16>::execSort(x, xShapeInfo, descending);
}

void NativeOps::sortTadFloat(Nd4jPointer *extraPointers, float *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
    NativeOpExcutioner<float>::execSort(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

void NativeOps::sortTadDouble(Nd4jPointer *extraPointers, double *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
    NativeOpExcutioner<double>::execSort(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

void NativeOps::sortTadHalf(Nd4jPointer *extraPointers, float16 *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
    //NativeOpExcutioner<float16>::execSort(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

void NativeOps::sortCooIndicesFloat(Nd4jPointer *extraPointers, Nd4jLong *indices, float *values, Nd4jLong length, int rank) {
    NativeOpExcutioner<float>::execSortCooIndices(indices, values, length, rank);
}

void NativeOps::sortCooIndicesDouble(Nd4jPointer *extraPointers, Nd4jLong *indices, double *values, Nd4jLong length, int rank) {
    NativeOpExcutioner<double >::execSortCooIndices(indices, values, length, rank);
}

void NativeOps::sortCooIndicesHalf(Nd4jPointer *extraPointers, Nd4jLong *indices, float16 *values, Nd4jLong length, int rank) {
    //   NativeOpExcutioner<float>::execSortCooIndices(indices, values, length, rank);
}

Nd4jLong NativeOps::encodeBitmapFloat(Nd4jPointer *extraPointers, float *dx, Nd4jLong N, int *dz, float threshold) {
    return NativeOpExcutioner<float>::encodeBitmap(dx, N, dz, threshold);
}

Nd4jLong NativeOps::encodeBitmapDouble(Nd4jPointer *extraPointers, double *dx, Nd4jLong N, int *dz, float threshold) {
    return NativeOpExcutioner<double>::encodeBitmap(dx, N, dz, threshold);
}

Nd4jLong NativeOps::encodeBitmapHalf(Nd4jPointer *extraPointers, float16 *dx, Nd4jLong N, int *dz, float threshold) {
    //return NativeOpExcutioner<float16>::encodeBitmap(dx, N, dz, threshold);
    return 0L;
}

void NativeOps::decodeBitmapFloat(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float *dz) {
    NativeOpExcutioner<float>::decodeBitmap(dx, N, dz);
}

void NativeOps::decodeBitmapDouble(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, double *dz) {
    NativeOpExcutioner<double>::decodeBitmap(dx, N, dz);
}

void NativeOps::decodeBitmapHalf(Nd4jPointer *extraPointers, void *dx, Nd4jLong N, float16 *dz) {
    //NativeOpExcutioner<float16>::decodeBitmap(dx, N, dz);
}


Nd4jLong* NativeOps::mmapFile(Nd4jPointer *extraPointers, const char *fileName, Nd4jLong length) {
    auto result = new Nd4jLong[2];errno = 0;

#if defined(_WIN32) || defined(_WIN64)
    _mmap(result, static_cast<size_t>(length), fileName);
#else
    int fd = open(fileName, O_RDWR, 0);// checking for failed fopen
    if (fd < 0) {
        nd4j_printf("Errno: %i\n", errno);
        throw std::runtime_error("Failed to open file for MMAP");
    }
    void * ptr = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

// check for failed allocation
    if (ptr == MAP_FAILED)
        return nullptr;

    result[0] = (Nd4jLong) ptr;
    result[1] = fd;

#endif

    return result;

}

void NativeOps::munmapFile(Nd4jPointer *extraPointers, Nd4jLong *ptrMap, Nd4jLong length) {
    munmap((Nd4jPointer) ptrMap[0], length);
#if defined(_WIN32) || defined(_WIN64)
    CloseHandle(reinterpret_cast<HANDLE>(ptrMap[1]));
#else
    close((int) ptrMap[1]);
#endif

    delete[] ptrMap;
}

nd4j::graph::ResultWrapper* NativeOps::executeFlatGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
    return nd4j::graph::GraphExecutioner<float>::executeFlatBuffer(flatBufferPointer);
}

nd4j::graph::ResultWrapper* NativeOps::executeFlatGraphHalf(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
    return nd4j::graph::GraphExecutioner<float16>::executeFlatBuffer(flatBufferPointer);
}

nd4j::graph::ResultWrapper* NativeOps::executeFlatGraphDouble(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
    return nd4j::graph::GraphExecutioner<double>::executeFlatBuffer(flatBufferPointer);
}

Nd4jPointer NativeOps::executeProtoGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer protoBufferPointer) {
    return nullptr;
}

Nd4jPointer NativeOps::executeProtoGraphFloat(Nd4jPointer *extraPointers, const char *fileName) {
    return nullptr;
}

const char* NativeOps::getAllCustomOps() {
    return nd4j::ops::OpRegistrator::getInstance()->getAllCustomOperations();
}

template <typename T>
FORCEINLINE int estimateThresholdGeneric(Nd4jPointer *extraPointers, Nd4jPointer x, int N, T threshold) {
    auto buffer = reinterpret_cast<T *>(x);

    int span = (N / 6) + 8;
    int cnt = 0;

#pragma omp parallel reduction(+:cnt)
    {
        int tid = omp_get_thread_num();
        int start = span * tid;
        int stop = span * (tid + 1);
        if (stop > N)
            stop = N;

#pragma omp simd
        for (int e = start; e < stop; e++) {
            auto v = nd4j::math::nd4j_abs<T>(buffer[e]);
            if (v >= threshold)
                cnt++;
        }
    }

    return cnt;
}

int NativeOps::estimateThresholdFloat(Nd4jPointer *extraPointers, Nd4jPointer x, int N, float threshold) {
    return estimateThresholdGeneric<float>(extraPointers, x, N, threshold);
}

int NativeOps::estimateThresholdDouble(Nd4jPointer *extraPointers, Nd4jPointer x, int N, float threshold) {
    return estimateThresholdGeneric<double>(extraPointers, x, N, threshold);
}


int NativeOps::estimateThresholdHalf(Nd4jPointer *extraPointers, Nd4jPointer x, int N, float threshold) {
    return estimateThresholdGeneric<float16>(extraPointers, x, N, threshold);
}


void NativeOps::deleteShapeList(Nd4jPointer shapeList) {
    auto list = reinterpret_cast<nd4j::ShapeList*>(shapeList);

    list->destroy();
    delete list;
}

template<typename T>
nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, T* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    nd4j::graph::VariableSpace<T> varSpace;
    Context<T> block(2, &varSpace);
    nd4j::ShapeList inShapes;

    for (int e = 0; e < numIArgs; e++)
        block.getIArguments()->push_back(iArgs[e]);

    for (int e = 0; e < numTArgs; e++)
        block.getTArguments()->push_back(tArgs[e]);

    for (int e = 0; e < numInputShapes; e++) {
        auto shape_ = reinterpret_cast<Nd4jLong *>(inputShapes[e]);

        // we shouldn't copy buffer if that's empty array
        T *buffer_ = nd4j::ArrayOptions::arrayType(shape_) == ArrayType::EMPTY ? nullptr : reinterpret_cast<T *>(inputBuffers[e]);

        auto array = new nd4j::NDArray<T>(buffer_, shape_);
        array->triggerAllocationFlag(false, false);

        // block should contain references to proper variable
        varSpace.putVariable(1, e, array);
        block.pickInput(1, e);

        inShapes.push_back(shape_);
    }

    auto shapeList = op->calculateOutputShape(&inShapes, block);

    if (varSpace.workspace() != nullptr)
        shapeList->detach();

    return shapeList;
}

nd4j::ShapeList* NativeOps::calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

    return _calculateOutputShapes<float>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

    return _calculateOutputShapes<float16>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

    return _calculateOutputShapes<double>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

template<typename T>
nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* inputShapes, int numInputShapes, T* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    Context<T> block(1);
    nd4j::ShapeList inShapes;

    for (int e = 0; e < numIArgs; e++)
        block.getIArguments()->push_back(iArgs[e]);

    for (int e = 0; e < numTArgs; e++)
        block.getTArguments()->push_back(tArgs[e]);

    for (int e = 0; e < numInputShapes; e++)
        inShapes.push_back(reinterpret_cast<Nd4jLong *>(inputShapes[e]));

    auto shapeList = op->calculateOutputShape(&inShapes, block);

    return shapeList;
}

nd4j::ShapeList* NativeOps::calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

    return _calculateOutputShapes<float>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

    return _calculateOutputShapes<float16>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

    return _calculateOutputShapes<double>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

template<typename T>
Nd4jStatus realExec(nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, T* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace) {
    if (op == nullptr)
        nd4j_printf("Can't find requested operation: [%lld]\n", hash);

    // we're using the same fake nodeId everywhere here

    std::vector<nd4j::NDArray<T>*> inputs(numInputs);
    std::vector<nd4j::NDArray<T>*> outputs(numOutputs);
    std::vector<T> ttArgs(numTArgs);
    std::vector<Nd4jLong> iiArgs(numIArgs);

    // filling block now with inputs
    for (int e = 0; e < numInputs; e++) {
        auto shape = reinterpret_cast<Nd4jLong *>(inputShapes[e]);
        T *buffer = nd4j::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : reinterpret_cast<T *>(inputBuffers[e]);

        inputs[e] = new nd4j::NDArray<T>(buffer, shape);
    }

    // if not inplace - transferring output arrays

    if (!isInplace)
        for (int e = 0; e < numOutputs; e++) {
            // we want to keep original output shape intact
            auto shape = shape::copyShape(reinterpret_cast<Nd4jLong *>(outputShapes[e]));
            T *buffer = nd4j::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : reinterpret_cast<T *>(outputBuffers[e]);

            auto array = new nd4j::NDArray<T>(buffer, shape);
            outputs[e] = array;

            // and we want to release shape copy once we're done
            array->triggerAllocationFlag(false, true);
        }

    for (int e = 0; e < numIArgs; e++)
        iiArgs[e] = iArgs[e];


    for (int e = 0; e < numTArgs; e++)
        ttArgs[e] = tArgs[e];


    // hypothetically at this point we have everything filled
    auto result = op->execute(inputs, outputs, ttArgs, iiArgs, isInplace);
    //auto result = op->execute(inputs, ttArgs, iiArgs, isInplace);


    if (!isInplace)
        for (int e = 0; e < numOutputs; e++) {
            //shape::printShapeInfoLinear("JVM output shape", (int *) outputShapes[e]);
            //shape::printShapeInfoLinear("C++ output shape", (int *) outputs[e]->shapeInfo());
            //outputs[e]->printIndexedBuffer("C++ raw output");
            //outputs[e]->printBuffer("C++ indexed output");

            if (outputs[e]->ordering() != shape::order(reinterpret_cast<Nd4jLong *>(outputShapes[e])))
                outputs[e]->streamline(shape::order(reinterpret_cast<Nd4jLong *>(outputShapes[e])));
        }

/*
    if (!isInplace) {
        if (result->size() != numOutputs) {
            return ND4J_STATUS_BAD_OUTPUT;
        }

        for (int e = 0; e < numOutputs; e++) {
            auto buffer = (T *) outputBuffers[e];
            auto shape = (int *) outputShapes[e];
            nd4j::NDArray<T> tmp(buffer, shape);

            if (tmp.lengthOf() != result->at(e)->lengthOf()) {
                nd4j_printf("Provided output array for [%s] has length of %i, but actual result has length of %i\n", op->getOpName()->c_str(), tmp.lengthOf(), result->at(e)->lengthOf());
                return ND4J_STATUS_BAD_OUTPUT;
            }

            tmp.assign(result->at(e));
        }
    } else {
        // if op is inplace, our ResultSet holds pointers
        result->purge();
    }


    delete result;

*/

    for (auto v: inputs)
        delete v;

    for (auto v: outputs)
        delete v;

    return Status::OK();
}

template Nd4jStatus realExec<float16>(nd4j::ops::DeclarableOp<float16>*, Nd4jPointer*, Nd4jLong, Nd4jPointer*, Nd4jPointer*, int, Nd4jPointer*, Nd4jPointer*, int, float16*, int, Nd4jLong*, int, bool);
template Nd4jStatus realExec<float> (nd4j::ops::DeclarableOp<float>*, Nd4jPointer*, Nd4jLong, Nd4jPointer*, Nd4jPointer*, int, Nd4jPointer*, Nd4jPointer*, int, float*, int, Nd4jLong*, int, bool);
template Nd4jStatus realExec<double>(nd4j::ops::DeclarableOp<double>*, Nd4jPointer*, Nd4jLong, Nd4jPointer*, Nd4jPointer*, int, Nd4jPointer*, Nd4jPointer*, int, double*, int, Nd4jLong*, int, bool);


int NativeOps::execCustomOpFloat(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

    return realExec<float>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::execCustomOpDouble(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

    return realExec<double>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::execCustomOpHalf(Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float16* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool isInplace) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

    return realExec<float16>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}


int NativeOps::registerGraphFloat(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer) {
    auto graph = nd4j::graph::GraphExecutioner<float>::importFromFlatPointer(flatBufferPointer);

    nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

    return ND4J_STATUS_OK;
}

int NativeOps::registerGraphDouble(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer) {
    auto graph = nd4j::graph::GraphExecutioner<double>::importFromFlatPointer(flatBufferPointer);

    nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

    return ND4J_STATUS_OK;
}

int NativeOps::registerGraphHalf(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer flatBufferPointer) {
    auto graph = nd4j::graph::GraphExecutioner<float16>::importFromFlatPointer(flatBufferPointer);

    nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

    return ND4J_STATUS_OK;
}

template <typename T>
static VariablesSet<T>* executeStoredGraphT(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    auto graph = nd4j::graph::GraphHolder::getInstance()->pullGraph<T>(graphId);
    auto varSpace = graph->getVariableSpace()->clone();

    std::vector<nd4j::NDArray<T> *> handles;

    for (int e = 0; e < numInputs; e++) {
        auto idx = inputIndices[e];

        // we'll delete this array later, together with cloned VariableSpace
        auto array = new nd4j::NDArray<T>(reinterpret_cast<T *>(inputBuffers[e]), reinterpret_cast<Nd4jLong *>(inputShapes[e]));
        handles.emplace_back(array);

        if (varSpace->hasVariable(idx)) {
            auto var = varSpace->getVariable(idx);
            if (var->hasNDArray())
                delete var->getNDArray();

            var->setNDArray(array);
        } else
            varSpace->putVariable(idx, array);
    }

    auto result = nd4j::graph::GraphExecutioner<T>::execute(graph, varSpace);
    auto varSet = new nd4j::graph::VariablesSet<T>(result);

    if (result == ND4J_STATUS_OK) {
        // pull back results, and provide them
        auto outputs = graph->fetchOutputs();
        for (int e = 0; e < outputs->size(); e++) {
            // we're only getting variable ID/Index from original grap. values will be taken from cloned workspace
            std::pair<int, int> varId(outputs->at(e)->id(), outputs->at(e)->index());

            auto var = varSpace->getVariable(varId);

            varSet->push_back(var->clone());
        }

        delete outputs;
    }

    delete varSpace;

    return varSet;
}

VariablesSet<float>* NativeOps::executeStoredGraphFloat(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    return executeStoredGraphT<float>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

VariablesSet<float16>* NativeOps::executeStoredGraphHalf(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    return executeStoredGraphT<float16>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

VariablesSet<double>* NativeOps::executeStoredGraphDouble(Nd4jPointer *extraPointers, Nd4jLong graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    return executeStoredGraphT<double>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

int NativeOps::unregisterGraph(Nd4jPointer *extraPointers, Nd4jLong graphId) {

    nd4j::graph::GraphHolder::getInstance()->dropGraphAny(graphId);

    return ND4J_STATUS_OK;
}

void NativeOps::deletePointerArray(Nd4jPointer pointer) {
    auto ptr = reinterpret_cast<Nd4jPointer *>(pointer);
    delete[] ptr;
}

void NativeOps::deleteIntArray(Nd4jPointer pointer) {
    auto ptr = reinterpret_cast<int *>(pointer);
    delete[] ptr;
}

void NativeOps::deleteLongArray(Nd4jPointer pointer) {
    auto ptr = reinterpret_cast<Nd4jLong *>(pointer);
    delete[] ptr;
}

template <typename T>
static void deleteVariablesSetT(Nd4jPointer pointer) {
    auto ptr = reinterpret_cast<nd4j::graph::VariablesSet<T>*>(pointer);
    delete ptr;
}

void NativeOps::deleteVariablesSetFloat(Nd4jPointer pointer) {
    deleteVariablesSetT<float>(pointer);
}

void NativeOps::deleteVariablesSetHalf(Nd4jPointer pointer) {
    deleteVariablesSetT<float16>(pointer);
}

void NativeOps::deleteVariablesSetDouble(Nd4jPointer pointer) {
    deleteVariablesSetT<double>(pointer);
}

const char* NativeOps::getAllOperations() {
    return nd4j::OpTracker::getInstance()->exportOperations();
}

Nd4jPointer NativeOps::getGraphStateHalf(Nd4jLong id) {
    return (Nd4jPointer) new nd4j::graph::GraphState<float16>(id);
}

Nd4jPointer NativeOps::getGraphStateFloat(Nd4jLong id) {
    return (Nd4jPointer) new nd4j::graph::GraphState<float>(id);
}

Nd4jPointer NativeOps::getGraphStateDouble(Nd4jLong id) {
    return (Nd4jPointer) new nd4j::graph::GraphState<double>(id);
}

void NativeOps::deleteGraphStateHalf(Nd4jPointer state) {
    auto stateP = reinterpret_cast<nd4j::graph::GraphState<float16> *>(state);
    delete stateP;
}

void NativeOps::deleteGraphStateFloat(Nd4jPointer state) {
    auto stateP = reinterpret_cast<nd4j::graph::GraphState<float> *>(state);
    delete stateP;
}

void NativeOps::deleteGraphStateDouble(Nd4jPointer state) {
    auto stateP = reinterpret_cast<nd4j::graph::GraphState<double> *>(state);
    delete stateP;
}

template <typename T>
Nd4jStatus execCustomOpWithScope(Nd4jPointer *extraPointers, nd4j::graph::GraphState<T> *state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    /**
     * That's basically exec, with VariableSpace provided in GraphState:
     * depending on operation (i.e. while of if), different logic executors could be used
     */

    auto graph = state->graph();
    auto varSpace = state->variableSpace();

    // Node is dynamically created, and has nothing beyond it: only inputs and outputs
    // this node has id of 0, and inputs are
    Node<T> node(OpType_LOGIC, opHash, 0);

    // mapping inputs
    for (int e = 0; e < numInputs; e++) {
        auto buffer = reinterpret_cast<T *>(inputBuffers[e]);
        auto shapeInfo = reinterpret_cast<Nd4jLong *>(inputShapes[e]);

        auto array = new nd4j::NDArray<T>(buffer, shapeInfo, varSpace->workspace());

        // now we just put array to VarSpace
        varSpace->putVariable(0, e, array);
        node.pickInput(0, e);
    }

    // mapping scopes
    for (int e = 0; e < numScopes; e++) {
        // we should check scope existence in GraphState/Graph
        int scopeId = (int) scopes[e];
        if (!state->hasScope(scopeId)) {
            // nd4j_printf("execCustomOpWithScope: referenced scope [%i] doesn't exist\n", scopeId);
            return Status::THROW();
        }
        node.pickInput(scopeId, 0);
    }

    auto result = LogicExecutor<T>::processNode(graph, &node);
    if (result != Status::OK())
        return result;

    // mapping outputs

    for (int e = 0; e < numOutputs; e++) {
        auto buffer = reinterpret_cast<T *>(outputBuffers[e]);
        auto shapeInfo = reinterpret_cast<Nd4jLong *>(outputShapes[e]);

        NDArray<T> array(buffer, shapeInfo, varSpace->workspace());

        // now we just put array to VarSpace to the same ID
        //varSpace->putVariable(0, e, array);

        auto t = varSpace->getVariable(0, e)->getNDArray();
        array.assign(t);
    }

    // removing input variables
    for (int e = 0; e < numInputs; e++) {
        varSpace->dropVariable(0, e);
    }


    // after some bla-bla-bla we should have Graph and Node for current op
    return Status::OK();
}

Nd4jStatus NativeOps::execCustomOpWithScopeHalf(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    return execCustomOpWithScope<float16>(extraPointers, reinterpret_cast<nd4j::graph::GraphState<float16> *>(state), opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}

Nd4jStatus NativeOps::execCustomOpWithScopeFloat(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    return execCustomOpWithScope<float>(extraPointers, reinterpret_cast<nd4j::graph::GraphState<float> *>(state), opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}

Nd4jStatus NativeOps::execCustomOpWithScopeDouble(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jLong opHash, Nd4jLong *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    return execCustomOpWithScope<double>(extraPointers, reinterpret_cast<nd4j::graph::GraphState<double> *>(state), opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}

void NativeOps::deleteResultWrapper(Nd4jPointer ptr) {
    // just 0 room for compiler s@!t
    auto p = reinterpret_cast<nd4j::graph::ResultWrapper *>(ptr);
    delete p;
}

/*
 * TypeDef:
 *     void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, long N, int dstType, Nd4jPointer z);
 */
void NativeOps::convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, Nd4jLong N, int dstType, Nd4jPointer z) {
    auto dx = reinterpret_cast<void *>(x);
    auto dz = reinterpret_cast<void *>(z);

    if (srcType == ND4J_FLOAT8) {
        if (dstType == ND4J_FLOAT8) {
            // convertGeneric<double, nd4j::float8>(dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<nd4j::float8, nd4j::int8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<nd4j::float8, nd4j::uint8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<nd4j::float8, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<nd4j::float8, nd4j::int16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<nd4j::float8, nd4j::uint16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<nd4j::float8, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<nd4j::float8, double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT8) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGeneric<nd4j::int8, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            //convertGeneric<nd4j::int8, nd4j::int8>(dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<nd4j::int8, nd4j::uint8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<nd4j::int8, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<nd4j::int8, nd4j::int16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<nd4j::int8, nd4j::uint16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: eventually we might want to add it
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<nd4j::int8, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<nd4j::int8, double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_UINT8) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGeneric<nd4j::uint8, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<nd4j::uint8, nd4j::int8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<nd4j::uint8, nd4j::uint8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<nd4j::uint8, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<nd4j::uint8, nd4j::int16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<nd4j::uint8, nd4j::uint16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: still might want to add
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<nd4j::uint8, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<nd4j::uint8, double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT16) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGeneric<float16, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<float16, nd4j::int8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<float16, nd4j::uint8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<float16, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<float16, nd4j::int16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<float16, nd4j::uint16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO: .... ^^^
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<float16, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<float16, double>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_THRESHOLD) {
            nd4j::TypeCast::convertToThreshold<float16>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT16) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGeneric<nd4j::int16, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<nd4j::int16, nd4j::int8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<nd4j::int16, nd4j::uint8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<nd4j::int16, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<nd4j::int16, nd4j::int16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<nd4j::int16, nd4j::uint16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {
            // TODO...
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<nd4j::int16, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<nd4j::int16, double>(nullptr, dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT24) {

    } else if (srcType == ND4J_FLOAT32) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGeneric<float, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<float, nd4j::int8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<float, nd4j::uint8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<float, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<float, nd4j::int16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<float, nd4j::uint16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertGeneric<float, double>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_THRESHOLD) {
            nd4j::TypeCast::convertToThreshold<float>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_DOUBLE) {
        if (dstType == ND4J_FLOAT8) {
            nd4j::TypeCast::convertGeneric<double, nd4j::float8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            nd4j::TypeCast::convertGeneric<double, nd4j::int8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            nd4j::TypeCast::convertGeneric<double, nd4j::uint8>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertGeneric<double, float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            nd4j::TypeCast::convertGeneric<double, nd4j::int16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            nd4j::TypeCast::convertGeneric<double, nd4j::uint16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertGeneric<double, float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            //
        } else if (dstType == ND4J_THRESHOLD) {
            nd4j::TypeCast::convertToThreshold<double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_THRESHOLD) {
        if (dstType == ND4J_FLOAT16) {
            nd4j::TypeCast::convertFromThreshold<float16>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_FLOAT32) {
            nd4j::TypeCast::convertFromThreshold<float>(nullptr, dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            nd4j::TypeCast::convertFromThreshold<double>(nullptr, dx, N, dz);
        } else {
            nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else {
        nd4j_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
}

template <typename T> int decompressParallelGeneric(Nd4jPointer* arrays, int arrayCount, Nd4jPointer output) {
    FloatBits fb;
    auto z = reinterpret_cast<T *>(output);

    // we use 3 as offset, since first 12 bytes are occupied with header

#pragma omp parallel
    {
        int* x = reinterpret_cast<int *>(arrays[0]);
        int threadCount = omp_get_num_threads();
        int localPart = x[1];
        if (threadCount > 0)
            localPart /= threadCount;

        int threadID = omp_get_thread_num();
        int upBound = localPart * (1 + threadID);
        int lowBound = 1 + threadID * localPart;
        if (threadID == threadCount - 1)
            upBound = localPart * (1 + threadID) + x[1] % threadCount;

//#pragma omp parallel for schedule(guided)
        for (int i = 0; i < arrayCount; i++) {
            x = reinterpret_cast<int *>(arrays[i]);
            int limit = x[0];
            fb.i_ = x[2];
            float threshold = fb.f_;
            int flimit = limit + 4;
//#pragma omp parallel for schedule(guided)
            for (int e = 4; e < flimit; e++) {
                int el = x[e];
                int ael = nd4j::math::nd4j_abs<int>(el);
                if (ael < lowBound && ael > upBound) continue;
                ael -= 1;
                z[ael] += el > 0 ? threshold : -threshold;
            }    //arrays
        }
    }
}

int NativeOps::decompressParallel(Nd4jPointer* arrays, int arrayCount, Nd4jPointer output) {
    return decompressParallelGeneric<float>(arrays, arrayCount, output);
}

template int decompressParallelGeneric<float16>(Nd4jPointer* arrays, int arrayCount, Nd4jPointer output);
template int decompressParallelGeneric<float>(Nd4jPointer* arrays, int arrayCount, Nd4jPointer output);
template int decompressParallelGeneric<double>(Nd4jPointer* arrays, int arrayCount, Nd4jPointer output);

template void flattenGeneric<float16>(Nd4jPointer*, int, char, float16*, Nd4jLong*, float16*, Nd4jLong*);
template void flattenGeneric<float>(Nd4jPointer*, int, char, float*, Nd4jLong*, float*, Nd4jLong*);
template void flattenGeneric<double>(Nd4jPointer*, int, char, double*, Nd4jLong*, double*, Nd4jLong*);;

template void pullRowsGeneric<float16>(float16*, Nd4jLong*, float16*, Nd4jLong*, const int, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*);
template void pullRowsGeneric<float>(float*, Nd4jLong*, float*, Nd4jLong*, const int, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*);
template void pullRowsGeneric<double>(double*, Nd4jLong*, double*, Nd4jLong*, const int, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*);

template void tearGeneric<float16>(float16*, Nd4jLong*, Nd4jPointer*, Nd4jLong*, Nd4jLong*, Nd4jLong*);
template void tearGeneric<float>(float*, Nd4jLong*, Nd4jPointer*, Nd4jLong*, Nd4jLong*, Nd4jLong*);
template void tearGeneric<double>(double*, Nd4jLong*, Nd4jPointer*, Nd4jLong*, Nd4jLong*, Nd4jLong*);

template void shuffleGeneric<float16>(float16**, Nd4jLong**, float16**, Nd4jLong**, int, int*, Nd4jLong**, Nd4jLong**);
template void shuffleGeneric<float>(float**, Nd4jLong**, float**, Nd4jLong**, int, int*, Nd4jLong**, Nd4jLong**);
template void shuffleGeneric<double>(double**, Nd4jLong**, double**, Nd4jLong**, int, int*, Nd4jLong**, Nd4jLong**);



