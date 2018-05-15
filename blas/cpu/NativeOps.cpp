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


#include <layers/layers_factory.h>
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
                                                int *xShapeInfo,
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
                                        int *xShapeInfo,
                                        double *extraParams,
                                        double *result,
                                        int *resultShapeInfo,
                                        int *dimension,
                                        int dimensionLength) {
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);
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
                                      int *xShapeInfo,
                                      double *y,
                                      int *yShapeInfo,
                                      double *result,
                                      int *resultShape,
                                      int *dimension, int dimensionLength) {
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[2]);
    Nd4jIndex *tadOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[3]);
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
                                              int xStride,
                                              double *y,
                                              int yStride,
                                              double *result,
                                              int resultStride,
                                              double *extraParams,
                                              Nd4jIndex n) {
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
        int *xShapeInfo,
        double *y,
        int *yShapeInfo,
        double *result,
        int *resultShapeInfo,
        double *extraParams,
        int *xIndexes,
        int *yIndexes,
        int *resultIndexes) {
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
        int *xShapeInfo,
        double *y,
        int *yShapeInfo,
        double *result,
        int *resultShapeInfo,
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
        int *xShapeInfo,
        double *extraParams,
        double *result,
        int *resultShapeInfo) {
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
                                   int *xShapeInfo,
                                   double *extraParams,
                                   double *result,
                                   int *resultShapeInfo,
                                   int *dimension,int dimensionLength) {
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);
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
                                 int *xShapeInfo,
                                 float16 *extraParams,
                                 float16 *result,
                                 int *resultShapeInfo,
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
                                         int *xShapeInfo,
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
                                    int *xShapeInfo,
                                    double *extraParams,
                                    double *y,
                                    int *yShapeInfo,
                                    double *result,
                                    int *resultShapeInfo) {
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
                                            int *xShapeInfo,
                                            double *extraParams,
                                            double *y,
                                            int *yShapeInfo) {
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
                                    int *xShapeInfo,
                                    double *extraParams,
                                    double *y,
                                    int *yShapeInfo,
                                    double *result,
                                    int *resultShapeInfo,
                                    int *dimension,
                                    int dimensionLength) {

    if (extraPointers == nullptr || extraPointers[2] == 0) {
        NativeOpExcutioner<double>::execReduce3(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength);
    } else {
        // going tad-way
        int *tadShapeInfo = reinterpret_cast<int *> (extraPointers[0]);
        Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);

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
        int xStride,
        double *result,
        int resultStride,
        double scalar,
        double *extraParams,
        Nd4jIndex n) {
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
        int *xShapeInfo,
        double *result,
        int *resultShapeInfo,
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
        int *xShapeInfo,
        double *result,
        int *resultShapeInfo,
        double scalar,
        double *extraParams,
        Nd4jIndex n,
        int *xIndexes,
        int *resultIndexes) {
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
                                                 int *xShapeInfo,
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
                                         int *xShapeInfo,
                                         double *extraParams,
                                         double *result,
                                         int *resultShapeInfo,bool biasCorrected) {
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
                                         int *xShapeInfo,
                                         double *extraParams,
                                         double *result,
                                         int *resultShapeInfo,
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
                                      int xStride,
                                      double *result,
                                      int resultStride,
                                      double *extraParams, Nd4jIndex n) {
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
        Nd4jPointer *extraPointers
        , int opNum,
        double *dx,
        int *xShapeInfo,
        double *result,
        int *resultShapeInfo,
        double *extraParams) {

    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);

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
        int *xShapeInfo,
        double *result,
        int *resultShapeInfo,
        double *extraParams,
        int *xIndexes,
        int *resultIndexes) {
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
                                              int *xShapeInfo,
                                              float *extraParams) {
    return NativeOpExcutioner<float>::execIndexReduceScalar(opNum,x,xShapeInfo,extraParams);
}

float   NativeOps::execIndexReduceScalarHalf(Nd4jPointer *extraPointers, int opNum,
                                             float16 *x,
                                             int *xShapeInfo,
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
                                       int *xShapeInfo,
                                       float *extraParams,
                                       float *result,
                                       int *resultShapeInfo,
                                       int *dimension, int dimensionLength) {
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);
    NativeOpExcutioner<float>::execIndexReduce(opNum,x,xShapeInfo,extraParams,result,resultShapeInfo,dimension,dimensionLength,tadShapeInfo, tadOffsets);
}

void   NativeOps::execIndexReduceHalf(Nd4jPointer *extraPointers, int opNum,
                                      float16 *x,
                                      int *xShapeInfo,
                                      float16 *extraParams,
                                      float16 *result,
                                      int *resultShapeInfo,
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
                                     int *xShapeInfo,
                                     float *y,
                                     int *yShapeInfo,
                                     float *result,int *resultShapeInfo,
                                     int *dimension, int dimensionLength) {
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[2]);
    Nd4jIndex *tadOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[3]);
    NativeOpExcutioner<float>::execBroadcast(opNum,x,xShapeInfo,y,yShapeInfo,result, resultShapeInfo, dimension,dimensionLength,
                                             tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

void   NativeOps::execBroadcastHalf(Nd4jPointer *extraPointers,int opNum,
                                    float16 *x,
                                    int *xShapeInfo,
                                    float16 *y,
                                    int *yShapeInfo,
                                    float16 *result,int *resultShapeInfo,
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
        int xStride,
        float *y,
        int yStride,
        float *result,
        int resultStride,
        float *extraParams, Nd4jIndex n) {
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
        int xStride,
        float16 *y,
        int yStride,
        float16 *result,
        int resultStride,
        float16 *extraParams, Nd4jIndex n) {
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
        int *xShapeInfo,
        float *y,
        int *yShapeInfo,
        float *result,
        int *resultShapeInfo,
        float *extraParams,
        int *xIndexes,
        int *yIndexes,
        int *resultIndexes) {
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
        int *xShapeInfo,
        float16 *y,
        int *yShapeInfo,
        float16 *result,
        int *resultShapeInfo,
        float16 *extraParams,
        int *xIndexes,
        int *yIndexes,
        int *resultIndexes) {
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
        int *xShapeInfo,
        float *y,
        int *yShapeInfo,
        float *result,
        int * resultShapeInfo,
        float *extraParams) {
    NativeOpExcutioner<float>::execPairwiseTransform(opNum,dx,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,extraParams);
}

void NativeOps::execPairwiseTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *dx,
        int *xShapeInfo,
        float16 *y,
        int *yShapeInfo,
        float16 *result,
        int *resultShapeInfo,
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
                                  int *xShapeInfo,
                                  float *extraParams,
                                  float *result,
                                  int *resultShapeInfo) {
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
                                 int *xShapeInfo,
                                 float16 *extraParams,
                                 float16 *result,
                                 int *resultShapeInfo) {
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
        int *xShapeInfo,
        float *extraParams,
        float *result,
        int *resultShapeInfo,
        int *dimension,
        int dimensionLength) {
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);
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
        int *xShapeInfo,
        float *extraParams) {
    return NativeOpExcutioner<float>::execReduceScalar(opNum,x,xShapeInfo,extraParams);
}

float NativeOps::execReduceScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        int *xShapeInfo,
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
        int *xShapeInfo,
        float *extraParams,
        float *y,
        int *yShapeInfo,
        float *result,
        int *resultShapeInfo) {
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
        int *xShapeInfo,
        float16 *extraParamsVals,
        float16 *y,
        int *yShapeInfo,
        float16 *result,
        int *resultShapeInfo) {
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
                                          int *xShapeInfo,
                                          float *extraParams,
                                          float *y,
                                          int *yShapeInfo) {
    return NativeOpExcutioner<float>::execReduce3Scalar(opNum,x,xShapeInfo,extraParams,y,yShapeInfo);
}

float   NativeOps::execReduce3ScalarHalf(Nd4jPointer *extraPointers,int opNum,
                                         float16 *x,
                                         int *xShapeInfo,
                                         float16 *extraParams,
                                         float16 *y,
                                         int *yShapeInfo) {
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
                                   int *xShapeInfo,
                                   float *extraParams,
                                   float *y,
                                   int *yShapeInfo,
                                   float *result,
                                   int *resultShapeInfo,
                                   int *dimension,
                                   int dimensionLength) {
    if (extraPointers == nullptr || extraPointers[2] == nullptr) {
        NativeOpExcutioner<float>::execReduce3(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength);
    } else {
        // going tad-way
        int *tadShapeInfo = reinterpret_cast<int *> (extraPointers[0]);
        Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);

        NativeOpExcutioner<float>::execReduce3TAD(opNum, x, xShapeInfo, extraParams, y, yShapeInfo, result, resultShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets);
    }

}

void   NativeOps::execReduce3Half(Nd4jPointer *extraPointers,int opNum,
                                  float16 *x,
                                  int *xShapeInfo,
                                  float16 *extraParams,
                                  float16 *y,
                                  int *yShapeInfo,
                                  float16 *result,
                                  int *resultShapeInfo,
                                  int *dimension,
                                  int dimensionLength) {
    // no-op
}

void NativeOps::execReduce3AllDouble(Nd4jPointer *extraPointers,
                          int opNum,
                          double *x,
                          int *xInfo,
                          double *extraParamsVals,
                          double *y,
                          int *yInfo,
                          double *result,
                          int *resultShapeInfoBuffer,
                          int *dimension,
                          int dimensionLength,
                          int *xTadShapeInfo,
                          Nd4jIndex *xOffsets,
                          int *yTadShapeInfo,
                          Nd4jIndex *yOffsets) {

    NativeOpExcutioner<double>::execReduce3All(opNum, x, xInfo, extraParamsVals, y, yInfo, result, resultShapeInfoBuffer, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);
}

void NativeOps::execReduce3AllFloat(Nd4jPointer *extraPointers,
                         int opNum,
                         float *x,
                         int *xInfo,
                         float *extraParamsVals,
                         float *y,
                         int *yInfo,
                         float *result,
                         int *resultShapeInfoBuffer,
                         int *dimension,
                         int dimensionLength,
                         int *xTadShapeInfo,
                         Nd4jIndex *xOffsets,
                         int *yTadShapeInfo,
                         Nd4jIndex *yOffsets) {

    NativeOpExcutioner<float>::execReduce3All(opNum, x, xInfo, extraParamsVals, y, yInfo, result, resultShapeInfoBuffer, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);
}

void NativeOps::execReduce3AllHalf(Nd4jPointer *extraPointers,
                        int opNum,
                        float16 *x,
                        int *xInfo,
                        float16 *extraParamsVals,
                        float16 *y,
                        int *yInfo,
                        float16 *result,
                        int *resultShapeInfoBuffer,
                        int *dimension,
                        int dimensionLength,
                        int *xTadShapeInfo,
                        Nd4jIndex *xOffsets,
                        int *yTadShapeInfo,
                        Nd4jIndex *yOffsets) {

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
                                  int xStride,
                                  float *result,
                                  int resultStride,
                                  float scalar,
                                  float *extraParams,
                                  Nd4jIndex n) {
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
        int *xShapeInfo,
        float *result,
        int *resultShapeInfo,
        float scalar,
        float *extraParams) {
    NativeOpExcutioner<float>::execScalar(opNum,x,resultShapeInfo,result,resultShapeInfo,scalar,extraParams);

}

void NativeOps::execScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        int *xShapeInfo,
        float16 *result,
        int *resultShapeInfo,
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
        int *xShapeInfo,
        float *result,
        int *resultShapeInfo,
        float scalar,
        float *extraParams,
        int *xIndexes,
        int *resultIndexes) {
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
        int *xShapeInfo,
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
        int *xShapeInfo,
        float16 *extraParams,bool biasCorrected) {
    // no-op
    return 0.0;
}

void   NativeOps::execScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *x,
        int xStride,
        float16 *result,
        int resultStride,
        float scalar,
        float16 *extraParams,
        Nd4jIndex n) {
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
        int *xShapeInfo,
        float *extraParams,
        float *result,
        int *resultShapeInfo,bool biasCorrected) {
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
        int *xShapeInfo,
        float16 *extraParams,
        float16 *result,
        int *resultShapeInfo,bool biasCorrected) {
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
                                        int *xShapeInfo,
                                        float *extraParams,
                                        float *result,
                                        int *resultShapeInfo,
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
                                       int *xShapeInfo,
                                       float16 *extraParams,
                                       float16 *result,
                                       int *resultShapeInfo,
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
        int xStride,
        float *result,
        int resultStride,
        float *extraParams, Nd4jIndex n) {
    NativeOpExcutioner<float>::execTransform(opNum,dx,xStride,result,resultStride,extraParams,n);
}

void   NativeOps::execTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        float16 *dx,
        int xStride,
        float16 *result,
        int resultStride,
        float16 *extraParams, Nd4jIndex n) {
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
        int *xShapeInfo,
        float *result,
        int *resultShapeInfo,
        float *extraParams) {

    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);

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
        int *xShapeInfo,
        float16 *result,
        int *resultShapeInfo,
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
        int *xShapeInfo,
        float *result,
        int *resultShapeInfo,
        float *extraParams,
        int *xIndexes,
        int *resultIndexes) {
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
        int *xShapeInfo,
        float16 *result,
        int *resultShapeInfo,
        float16 *extraParams,
        int *xIndexes,
        int *resultIndexes) {
    // no-op
}



template <typename T>
void flattenGeneric(Nd4jPointer *extraPointers,
                    int offset,
                    char order,
                    T *result,
                    int *resultShapeInfo,
                    T *input,
                    int *inputShapeInfo) {
    int numOnes = 0;
    int *shape = shape::shapeOf(inputShapeInfo);
    int wholeRank = shape::rank(inputShapeInfo);
    for(int i = 0; i < wholeRank; i++) {
        if(shape[i] == 1)
            numOnes++;
    }



    //start at the given offset
    result += offset;
    char inputOrder = shape::order(inputShapeInfo);
    int len = shape::length(inputShapeInfo);
    int resultEleStride = shape::elementWiseStride(resultShapeInfo);
    int inputEleStride = shape::elementWiseStride(inputShapeInfo);
    int numTads, stride, dimension, dimensionLength;
    int rank = shape::rank(inputShapeInfo);
    int *xStride = shape::stride(inputShapeInfo);
    int *xShape = shape::shapeOf(inputShapeInfo);

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
            int rank = shape::rank(inputShapeInfo);
            int *coord = new int[rank];
            int *xShape = shape::shapeOf(inputShapeInfo);
            int *xStride = shape::stride(inputShapeInfo);
            int len = shape::length(inputShapeInfo);
            // FIXME: result[idx++] is bad idea, because of possible negative EWS
            if(order == 'f') {
                for(int i = 0; i < len; i++) {
                    shape::ind2sub(rank, xShape, i, coord);
                    int offset = shape::getOffset(0,xShape,xStride,coord,rank);
                    result[idx++] = input[offset];

                }
            }
            else {
                for(int i = 0; i < len; i++) {
                    shape::ind2subC(rank, xShape, i, coord);
                    int offset = shape::getOffset(0,xShape,xStride,coord,rank);
                    result[idx++] = input[offset];

                }
            }

            delete[] coord;
        }
    }
    else {
        int rank = shape::rank(inputShapeInfo);
        int *xShape = shape::shapeOf(inputShapeInfo);
        int tadShape = xShape[dimension];
        shape::TAD tad(inputShapeInfo,&dimension,dimensionLength);
        tad.createTadOnlyShapeInfo();
#pragma omp  parallel for schedule(guided) default(shared)
        for(int i = 0; i < numTads; i++) {

            int resultOffset;

            if (order == 'f') {
                // 1. get c ordering coordinates
                int *cIndexCoordinates = new int[rank - 1];
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

            int tadOffset = tad.tadOffset(i);
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
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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
        int *resultShapeInfo,
        float *input,
        int *inputShapeInfo) {
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
        int *resultShapeInfo,
        float16 *input,
        int *inputShapeInfo) {
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
        int *resultShapeInfo,
        double *input,
        int *inputShapeInfo) {
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
    BlasHelper::getInstance()->initializeFunctions(functions);
}

/**
       * This method acquires memory chunk of requested size on host side
       *
       * @param pointer pointer that'll be used for allocation
       * @param memorySize memory size, in bytes
       * @param flags optional parameter
       */
Nd4jPointer NativeOps::mallocHost(Nd4jIndex memorySize, int flags) {
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
Nd4jPointer NativeOps::mallocDevice(Nd4jIndex memorySize, Nd4jPointer ptrToDeviceId, int flags) {
    // not supported
    return 0L;
}

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
int NativeOps::freeHost(Nd4jPointer pointer) {
    free((void *) pointer);
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

Nd4jIndex NativeOps::getDeviceFreeMemory(Nd4jPointer ptrToDeviceId) {
    return 0L;
}

Nd4jIndex NativeOps::getDeviceTotalMemory(Nd4jPointer ptrToDeviceId) {
    return 0L;
}

int NativeOps::memcpy(Nd4jPointer dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::memcpyAsync(Nd4jPointer dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::memset(Nd4jPointer dst, int value, Nd4jIndex size, int flags, Nd4jPointer reserved) {
    return 0L;
}

int NativeOps::memsetAsync(Nd4jPointer dst, int value, Nd4jIndex size,  int flags, Nd4jPointer reserved) {
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

void NativeOps::tadOnlyShapeInfo(int *xShapeInfo, int *dimension, int dimensionLength, int *target, Nd4jIndex *offsets) {
    shape::TAD *tad = new shape::TAD();
    tad->init(xShapeInfo, dimension, dimensionLength);
    //tad->setOutputBuffer(target);
    tad->createTadOnlyShapeInfo();
    tad->createOffsets();


    std::memcpy((void *) target, tad->tadOnlyShapeInfo, (tad->tadOnlyShapeInfo[0] * 2 + 4) * sizeof(int));
    std::memcpy((void *) offsets, tad->tadOffsets, tad->numTads * sizeof(Nd4jIndex));

    delete tad;
}

int NativeOps::memcpyConstantAsync(Nd4jIndex dst, Nd4jPointer src, Nd4jIndex size, int flags, Nd4jPointer reserved) {
    // no-op
    return 0L;
}

Nd4jPointer NativeOps::getConstantSpace() {
    // no-op
    return 0L;
}

template<typename T>
void pullRowsGeneric(T *x,
                     int *xShapeInfo,
                     T *z,
                     int *zShapeInfo,
                     const int n,
                     int *indexes,
                     int *tadShapeInfo,
                     Nd4jIndex *tadOffsets,
                     int *zTadShapeInfo,
                     Nd4jIndex *zTadOffsets) {
    const int xEWS = shape::elementWiseStride(tadShapeInfo);
    const int zEWS = shape::elementWiseStride(zTadShapeInfo);
    const int tadLength = shape::length(tadShapeInfo);

    int elementsPerThread = n / TAD_THRESHOLD;
    int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
    _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

#pragma omp parallel for num_threads(_threads) if (n > 1) schedule(guided) default(shared)
    for (int idx = 0; idx < n; idx++) {
        int xTadOffsetForBlock = tadOffsets[indexes[idx]];
        int zTadOffsetForBlock = zTadOffsets[idx];

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
            int *zShape = shape::shapeOf(zTadShapeInfo);
            int *zStride = shape::stride(zTadShapeInfo);
            int *xShape = shape::shapeOf(tadShapeInfo);
            int *xStride = shape::stride(tadShapeInfo);
            int zRank = shape::rank(zTadShapeInfo);
            int tadRank = shape::rank(tadShapeInfo);

            int xCoord[MAX_RANK];
            int zCoord[MAX_RANK];

            for (int i = 0; i < tadLength; i++) {
                shape::ind2subC(tadRank,xShape, i, xCoord);
                shape::ind2subC(zRank,zShape, i, zCoord);
                Nd4jIndex xOffset = shape::getOffset(xTadOffsetForBlock, xShape, xStride, xCoord, tadRank);
                Nd4jIndex zOffset = shape::getOffset(zTadOffsetForBlock, zShape, zStride, zCoord, zRank);
                z[zOffset] = x[xOffset];
            }
        }
    }
}

void NativeOps::pullRowsHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, float16 *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *zTadShapeInfo, Nd4jIndex *zTadOffsets) {
    // no-op
}

void NativeOps::pullRowsFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, float *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *zTadShapeInfo, Nd4jIndex *zTadOffsets) {
    pullRowsGeneric<float>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

void NativeOps::pullRowsDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, double *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *zTadShapeInfo, Nd4jIndex *zTadOffsets) {
    pullRowsGeneric<double>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

template<typename T>
void tearGeneric(T *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {

    const Nd4jIndex tadLength = shape::length(tadShapeInfo);
    int tadEWS = shape::elementWiseStride(tadShapeInfo);
    int zEWS = shape::elementWiseStride(zShapeInfo);
    int tadRank = shape::rank(tadShapeInfo);
    int zRank = shape::rank(zShapeInfo);
    int *tadShape = shape::shapeOf(tadShapeInfo);
    int *tadStride = shape::stride(tadShapeInfo);
    int *zShape = shape::shapeOf(zShapeInfo);
    int *zStride = shape::stride(zShapeInfo);
    Nd4jIndex numTads = shape::length(xShapeInfo) / tadLength;

#pragma omp parallel for schedule(guided) default(shared)
    for (Nd4jIndex i = 0; i < numTads; i++) {
        T *z = (T *) targets[i];
        T *s = x + tadOffsets[i];

        if (zEWS == 1 && tadEWS == 1) {
#pragma omp simd
            for (Nd4jIndex j = 0; j < tadLength; j++) {
                z[j] = s[j];
            }
        } else if (zEWS > 0 && tadEWS > 0) {
#pragma omp simd
            for (Nd4jIndex j = 0; j < tadLength; j++) {
                z[j * zEWS] = s[j * tadEWS];
            }
        } else {
            int xCoord[MAX_RANK];
            int zCoord[MAX_RANK];

            for (Nd4jIndex j = 0; j < tadLength; j++) {
                shape::ind2sub(tadRank,tadShape, j, xCoord);
                shape::ind2sub(zRank, zShape, j, zCoord);

                Nd4jIndex xOffset = shape::getOffset(0, tadShape, tadStride, xCoord, tadRank);
                Nd4jIndex zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                z[zOffset] = s[xOffset];
            }
        }
    }
}

void NativeOps::tearDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
    tearGeneric<double>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

void NativeOps::tearFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
    tearGeneric<float>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

void NativeOps::tearHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, Nd4jPointer *targets, int *zShapeInfo, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
    tearGeneric<float16>(x, xShapeInfo, targets, zShapeInfo, tadShapeInfo, tadOffsets);
}

void NativeOps::averageHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jIndex length, bool propagate) {
    // no-op
}

void NativeOps::averageFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jIndex length, bool propagate) {
    float **x = reinterpret_cast<float **>(dx);
    nd4j::SpecialMethods<float>::averageGeneric(x, dz, n, length, propagate);
}

void NativeOps::averageDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jIndex length, bool propagate) {
    double **x = reinterpret_cast<double **>(dx);
    nd4j::SpecialMethods<double>::averageGeneric(x, dz, n, length, propagate);
}

void NativeOps::accumulateHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jIndex length) {
    float16 **x = reinterpret_cast<float16 **>(dx);
    nd4j::SpecialMethods<float16>::accumulateGeneric(x, dz, n, length);
}

void NativeOps::accumulateFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jIndex length) {
    float **x = reinterpret_cast<float **>(dx);
    nd4j::SpecialMethods<float>::accumulateGeneric(x, dz, n, length);
}

void NativeOps::accumulateDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jIndex length) {
    double **x = reinterpret_cast<double **>(dx);
    nd4j::SpecialMethods<double>::accumulateGeneric(x, dz, n, length);
}

void NativeOps::enableP2P(bool enable) {
    // no-op
}

void NativeOps::encodeThresholdP1Half(Nd4jPointer *extraPointers, float16 *dx, Nd4jIndex N, int *dz, float threshold) {
    // TODO: to be implemented
}

void NativeOps::encodeThresholdP1Float(Nd4jPointer *extraPointers, float *dx, Nd4jIndex N, int *dz, float threshold) {
    // TODO: to be implemented
}

void NativeOps::encodeThresholdP1Double(Nd4jPointer *extraPointers, double *dx, Nd4jIndex N, int *dz, float threshold) {
    // TODO: to be implemented
}


void NativeOps::encodeThresholdP2Int(Nd4jPointer *extraPointers, int *dx, Nd4jIndex N, int *dz) {
    // TODO: to be implemented
}

void NativeOps::encodeThresholdP3Float(Nd4jPointer *extraPointers, float *dx, int *offsets, Nd4jIndex N, int *dz){
    // offsets won't be used here

    // TODO: to be implemented
}

void NativeOps::encodeThresholdP3Double(Nd4jPointer *extraPointers, double *dx, int *offsets, Nd4jIndex N, int *dz){
    // offsets won't be used here

    // TODO: to be implemented
}

void NativeOps::encodeThresholdP3Half(Nd4jPointer *extraPointers, float16 *dx, int *offsets, Nd4jIndex N, int *dz){
    // offsets won't be used here

    // TODO: to be implemented
}

void NativeOps::decodeThresholdFloat(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float *dz){
    // TODO: to be implemented
}

void NativeOps::decodeThresholdHalf(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float16 *dz){
    // TODO: to be implemented
}

void NativeOps::decodeThresholdDouble(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, double *dz){
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
void shuffleGeneric(T **dX, int **xShapeInfo, T **dZ, int **zShapeInfo, int N, int *shuffleMap, int **tadOnlyShapeInfo, Nd4jIndex **tadOffsets) {


#pragma omp parallel for if (N > 1) default(shared)
    for (int f = 0; f < N; f++) {
        T *x = (T *) dX[f];
        //T *z = (T *) dZ[f];

        Nd4jIndex *tadOffset = (Nd4jIndex *) tadOffsets[f];


        const Nd4jIndex tadLength = shape::length(tadOnlyShapeInfo[f]);
        int tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
        int tadRank = shape::rank(tadOnlyShapeInfo[f]);
        int numTads = shape::length(xShapeInfo[f]) / tadLength;

        int *tadShape = shape::shapeOf(tadOnlyShapeInfo[f]);
        int *tadStride = shape::stride(tadOnlyShapeInfo[f]);

        //printf("Array: [%i], tadEWS: [%i], tadLength: [%i]\n", f, tadEWS, tadLength);

        // TODO: omp *probably* has no sense here, since 99% of uses for this method will be inside DataSet. but worth a check

        for (Nd4jIndex r = 0; r < numTads; r++) {
            if (shuffleMap[r] < 0)
                continue;

            Nd4jIndex oldOffset = tadOffset[r];
            Nd4jIndex newOffset = tadOffset[shuffleMap[r]];

            T *rX = x + oldOffset;
            T *rY = x + newOffset;

            if (tadEWS == 1) {

#pragma omp simd
                for (Nd4jIndex i = 0; i < tadLength; i++) {
                    nd4j::math::nd4j_swap<T>(rX[i], rY[i]);
                }

            } else {
                // ind2sub branch
#pragma omp parallel for schedule(guided) if (N == 1 && tadLength > 512) default(shared)
                for (Nd4jIndex i = 0; i < tadLength; i++) {
                    int xCoord[MAX_RANK];
                    int yCoord[MAX_RANK];

                    shape::ind2subC(tadRank,tadShape, i, xCoord);
                    shape::ind2subC(tadRank,tadShape, i, yCoord);

                    Nd4jIndex xOffset = shape::getOffset(oldOffset, tadShape, tadStride, xCoord, tadRank);
                    Nd4jIndex yOffset = shape::getOffset(newOffset, tadShape, tadStride, yCoord, tadRank);

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
    float **x = reinterpret_cast<float **>(dx);
    float **z = reinterpret_cast<float **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    Nd4jIndex **tadOffset = reinterpret_cast<Nd4jIndex **>(tadOffsets);

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
    double **x = reinterpret_cast<double **>(dx);
    double **z = reinterpret_cast<double **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    Nd4jIndex **tadOffset = reinterpret_cast<Nd4jIndex **>(tadOffsets);

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
                                             int *xShapeInfo,
                                             float *dy,
                                             int *yShapeInfo,
                                             float *dz,
                                             int *zShapeInfo,
                                             int *dimension,
                                             int dimensionLength,
                                             int *tadShapeInfo,
                                             Nd4jIndex *tadOffsets,
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

void NativeOps::execMetaPredicateShapeFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
    // no-op;
}

void NativeOps::setOmpMinThreads(int threads) {
    // TODO: to be implemented
}

void NativeOps::execMetaPredicateStridedFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int xStride, float *dy, int yStride, float *dz, int zStride, float *extraA, float *extraB, float scalarA, float scalarB) {
    // no-op
}

void NativeOps::execMetaPredicateShapeDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, double *dx, int *xShapeInfo, double *dy, int *yShapeInfo, double *dz, int *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB) {
    // no-op;
}

void NativeOps::execMetaPredicateStridedDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, double *dx, int xStride, double *dy, int yStride, double *dz, int zStride, double *extraA, double *extraB, double scalarA, double scalarB) {
    // no-op
}

void NativeOps::execMetaPredicateShapeHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float16 *dx, int *xShapeInfo, float16 *dy, int *yShapeInfo, float16 *dz, int *zShapeInfo, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    // no-op;
}

void NativeOps::execMetaPredicateStridedHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float16 *dx, int xStride, float16 *dy, int yStride, float16 *dz, int zStride, float16 *extraA, float16 *extraB, float scalarA, float scalarB) {
    // no-op
}

int NativeOps::getDevice() {
    return 0;
}


void NativeOps::execScalarFloat(Nd4jPointer *extraPointers,int opNum,
                                float *x,
                                int *xShapeInfo,
                                float *z,
                                int *zShapeInfo,
                                float *scalars,
                                float *extraParams,
                                int *dimension,
                                int dimensionLength) {
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[2]);
    Nd4jIndex *tadOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[3]);

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
                                 int *xShapeInfo,
                                 double *z,
                                 int *zShapeInfo,
                                 double *scalars,
                                 double *extraParams,
                                 int *dimension,
                                 int dimensionLength) {
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    Nd4jIndex *tadOffsets = reinterpret_cast<Nd4jIndex *>(extraPointers[1]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[2]);
    Nd4jIndex *tadOffsetsZ = reinterpret_cast<Nd4jIndex *>(extraPointers[3]);

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
                               int *xShapeInfo,
                               float16 *z,
                               int *zShapeInfo,
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
        name = (char *) malloc(256 * sizeof(char));

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
                                   int **shapeArguments,
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
                                    int **shapeArguments,
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
                                  int **shapeArguments,
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
#pragma omp parallel for num_threads(_threads) schedule(guided) proc_bind(spread) default(shared)
    for (int i = 0; i < numAggregates; i++) {
        int **intArrays = new int *[maxIntArrays];

        float **arguments = helper.getArguments(i);
        int **shapes = helper.getShapeArguments(i);
        int *idxArg = helper.getIndexArguments(i);
        float *realArg = helper.getRealArguments(i);

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
                           (int **) intArrays,
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
        int **intArrays = new int *[maxIntArrays];

        double **arguments = helper.getArguments(i);
        int **shapes = helper.getShapeArguments(i);
        int *idxArg = helper.getIndexArguments(i);
        double *realArg = helper.getRealArguments(i);

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
                                int *zShapeBuffer,
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
                                int *xShapeBuffer,
                                float *y,
                                int *yShapeBuffer,
                                float *z,
                                int *zShapeBuffer,
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
                                int *xShapeBuffer,
                                float *z,
                                int *zShapeBuffer, float *extraArguments) {
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
                                 int *zShapeBuffer,
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
                                 int *xShapeBuffer,
                                 double *y,
                                 int *yShapeBuffer,
                                 double *z,
                                 int *zShapeBuffer,
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
                                 int *xShapeBuffer,
                                 double *z,
                                 int *zShapeBuffer,
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
                               int *zShapeBuffer,
                               float16 *extraArguments) {
    //NativeOpExcutioner<float16>::execRandom(opNum, state, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomHalf(Nd4jPointer *extraPointers,
                               int opNum,
                               Nd4jPointer state,
                               float16 *x,
                               int *xShapeBuffer,
                               float16 *y,
                               int *yShapeBuffer,
                               float16 *z,
                               int *zShapeBuffer,
                               float16 *extraArguments) {
    //NativeOpExcutioner<float16>::execRandom(opNum, state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
}

void NativeOps::execRandomHalf(Nd4jPointer *extraPointers,
                               int opNum,
                               Nd4jPointer state,
                               float16 *x,
                               int *xShapeBuffer,
                               float16 *z,
                               int *zShapeBuffer,
                               float16 *extraArguments) {
    //NativeOpExcutioner<float16>::execRandom(opNum, state, x, xShapeBuffer, z, zShapeBuffer, extraArguments);
}



Nd4jPointer NativeOps::initRandom(Nd4jPointer *extraPointers, long seed, long bufferSize, Nd4jPointer ptrToBuffer) {
    long *ptrBuf = reinterpret_cast<long *>(ptrToBuffer);
    nd4j::random::RandomBuffer *buffer = new nd4j::random::RandomBuffer(seed, bufferSize, (uint64_t *) ptrBuf);

    nd4j::random::Xoroshiro128 generator(buffer);
    generator.refreshBuffer();

    return (Nd4jPointer) buffer;
}

void NativeOps::refreshBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom) {
    nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (ptrRandom);

    buffer->setSeed(seed);
    buffer->setOffset(0);
    nd4j::random::Xoroshiro128 generator(buffer);
    generator.refreshBuffer();
}

void NativeOps::reSeedBuffer(Nd4jPointer *extraPointers, long seed, Nd4jPointer ptrRandom) {
    nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (ptrRandom);

    buffer->reSeed(seed);
}


void NativeOps::destroyRandom(Nd4jPointer ptrBuffer) {
    nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *>(ptrBuffer);
    delete buffer;
}


/**
 *
 * @param npyArray
 * @return
 */
Nd4jPointer NativeOps::shapeBufferForNumpy(Nd4jPointer npyArray) {
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    unsigned int *shape = new unsigned int[arr.shape.size()];
    for(unsigned int i = 0; i < arr.shape.size(); i++) {
        shape[i] = arr.shape[i];
    }

    int *shapeBuffer = shape::shapeBufferOfNpy(arr.shape.size(),
                                               shape,
                                               arr.fortranOrder);
    delete[] shape;
    return reinterpret_cast<Nd4jPointer>(shapeBuffer);
}


/**
 *
 * @param npyArray
 * @return
 */
Nd4jPointer NativeOps::dataPointForNumpy(Nd4jPointer npyArray) {
    //char *buff = reinterpret_cast<char *>(npyArray);
    //printf("Pointer contents %s\n",buff);
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    cnpy::NpyArray *arrPointer = &arr;
    char *data = arrPointer->data;
    if(arrPointer->wordSize == sizeof(float)) {
        float *floatData = reinterpret_cast<float *>(data);
        return reinterpret_cast<Nd4jPointer>(floatData);
    }
    else if(arrPointer->wordSize == sizeof(double)) {
        double *doubleData = reinterpret_cast<double *>(data);
        return reinterpret_cast<Nd4jPointer >(doubleData);
    }

    return reinterpret_cast<Nd4jPointer >(0);
}

/**
 * Load a numpy array from a file
 * and return it as an Nd4jPointer
 * @param path
 * @return
 */
Nd4jPointer NativeOps::numpyFromFile(std::string path) {
    char *numpyBuffer = cnpy::loadFile(path.data());
    return reinterpret_cast<Nd4jPointer >(numpyBuffer);
}

/**
    * Return the length of a shape buffer
    * based on the pointer
    * @param buffer  the buffer pointer to check
    * @return
    */
int NativeOps::lengthForShapeBufferPointer(Nd4jPointer buffer) {
    int *shapeBuffer = reinterpret_cast<int *>(buffer);
    return shape::shapeInfoLength(shape::rank(shapeBuffer));
}

/**
  * Get the element size for a numpy array
  * @param npyArray  the numpy array's address
  * to get the length for
  * @return
  */
int NativeOps::elementSizeForNpyArray(Nd4jPointer npyArray) {
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    cnpy::NpyArray *arrPointer = &arr;
    int size = arrPointer->wordSize;
   // arrPointer->destruct();
    return size;
}

void NativeOps::releaseNumpy(Nd4jPointer npyArray) {
    free((void *) npyArray);
}

/**
  * The pointer to get the address for
  *
  * @param address the address to get the pointer
  * @return the pointer for the given address
  */

Nd4jPointer NativeOps::pointerForAddress(Nd4jIndex address) {
    return reinterpret_cast<Nd4jPointer >(address);
}

void NativeOps::sortFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, bool descending) {
    NativeOpExcutioner<float>::execSort(x, xShapeInfo, descending);
}

void NativeOps::sortDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, bool descending) {
    NativeOpExcutioner<double>::execSort(x, xShapeInfo, descending);
}

void NativeOps::sortHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, bool descending) {
    //NativeOpExcutioner<float16>::execSort(x, xShapeInfo, descending);
}

void NativeOps::sortTadFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending) {
    NativeOpExcutioner<float>::execSort(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

void NativeOps::sortTadDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending) {
    NativeOpExcutioner<double>::execSort(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

void NativeOps::sortTadHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending) {
    //NativeOpExcutioner<float16>::execSort(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

void NativeOps::sortCooIndicesFloat(Nd4jPointer *extraPointers, int *indices, float *values, Nd4jIndex length, int rank) {
    NativeOpExcutioner<float>::execSortCooIndices(indices, values, length, rank);
}

void NativeOps::sortCooIndicesDouble(Nd4jPointer *extraPointers, int *indices, double *values, Nd4jIndex length, int rank) {
    NativeOpExcutioner<double >::execSortCooIndices(indices, values, length, rank);
}

void NativeOps::sortCooIndicesHalf(Nd4jPointer *extraPointers, int *indices, float16 *values, Nd4jIndex length, int rank) {
 //   NativeOpExcutioner<float>::execSortCooIndices(indices, values, length, rank);
}

Nd4jIndex NativeOps::encodeBitmapFloat(Nd4jPointer *extraPointers, float *dx, Nd4jIndex N, int *dz, float threshold) {
    return NativeOpExcutioner<float>::encodeBitmap(dx, N, dz, threshold);
}

Nd4jIndex NativeOps::encodeBitmapDouble(Nd4jPointer *extraPointers, double *dx, Nd4jIndex N, int *dz, float threshold) {
    return NativeOpExcutioner<double>::encodeBitmap(dx, N, dz, threshold);
}

Nd4jIndex NativeOps::encodeBitmapHalf(Nd4jPointer *extraPointers, float16 *dx, Nd4jIndex N, int *dz, float threshold) {
    //return NativeOpExcutioner<float16>::encodeBitmap(dx, N, dz, threshold);
    return 0L;
}

void NativeOps::decodeBitmapFloat(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float *dz) {
    NativeOpExcutioner<float>::decodeBitmap(dx, N, dz);
}

void NativeOps::decodeBitmapDouble(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, double *dz) {
    NativeOpExcutioner<double>::decodeBitmap(dx, N, dz);
}

void NativeOps::decodeBitmapHalf(Nd4jPointer *extraPointers, void *dx, Nd4jIndex N, float16 *dz) {
    //NativeOpExcutioner<float16>::decodeBitmap(dx, N, dz);
}


Nd4jIndex* NativeOps::mmapFile(Nd4jPointer *extraPointers, const char *fileName, Nd4jIndex length) {
auto result = new Nd4jIndex[2];
errno = 0;

#if defined(_WIN32) || defined(_WIN64)
    _mmap(result, static_cast<size_t>(length), fileName);
#else
int fd = open(fileName, O_RDWR, 0);
// checking for failed fopen
if (fd < 0) {
    nd4j_printf("Errno: %i\n", errno);
    throw std::runtime_error("Failed to open file for MMAP");
}

void * ptr = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

// check for failed allocation
if (ptr == MAP_FAILED)
    return nullptr;

result[0] = (Nd4jIndex) ptr;
result[1] = fd;

#endif


return result;


}

void NativeOps::munmapFile(Nd4jPointer *extraPointers, Nd4jIndex *ptrMap, Nd4jIndex length) {
munmap((Nd4jPointer) ptrMap[0], length);

#if defined(_WIN32) || defined(_WIN64)
    CloseHandle(reinterpret_cast<HANDLE>(ptrMap[1]));
#else
    close((int) ptrMap[1]);
#endif

delete[] ptrMap;
}

Nd4jPointer NativeOps::executeFlatGraphFloat(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
    return nd4j::graph::GraphExecutioner<float>::executeFlatBuffer(flatBufferPointer);
}

Nd4jPointer NativeOps::executeFlatGraphHalf(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
    return nd4j::graph::GraphExecutioner<float16>::executeFlatBuffer(flatBufferPointer);
}

Nd4jPointer NativeOps::executeFlatGraphDouble(Nd4jPointer *extraPointers, Nd4jPointer flatBufferPointer) {
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


void NativeOps::deleteShapeList(Nd4jPointer shapeList) {
    nd4j::ShapeList* list = reinterpret_cast<nd4j::ShapeList*>(shapeList);

    list->destroy();
    delete list;
}

template<typename T>
nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, T* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    nd4j::graph::VariableSpace<T> varSpace;
    Context<T> block(2, &varSpace);
    nd4j::ShapeList inShapes;

    for (int e = 0; e < numIArgs; e++)
        block.getIArguments()->push_back(iArgs[e]);

    for (int e = 0; e < numTArgs; e++)
        block.getTArguments()->push_back(tArgs[e]);

    for (int e = 0; e < numInputShapes; e++) {
        auto shape_ = (int *) inputShapes[e];
        auto buffer_ = (T *) inputBuffers[e];
        auto array = new NDArray<T>(buffer_, shape_);
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

nd4j::ShapeList* NativeOps::calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

    return _calculateOutputShapes<float>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

    return _calculateOutputShapes<float16>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

    return _calculateOutputShapes<double>(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

template<typename T>
nd4j::ShapeList* _calculateOutputShapes(Nd4jPointer* extraPointers, nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* inputShapes, int numInputShapes, T* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    Context<T> block(1);
    nd4j::ShapeList inShapes;

    for (int e = 0; e < numIArgs; e++)
        block.getIArguments()->push_back(iArgs[e]);

    for (int e = 0; e < numTArgs; e++)
        block.getTArguments()->push_back(tArgs[e]);

    for (int e = 0; e < numInputShapes; e++)
        inShapes.push_back((int *) inputShapes[e]);

    auto shapeList = op->calculateOutputShape(&inShapes, block);

    return shapeList;
}

nd4j::ShapeList* NativeOps::calculateOutputShapesFloat(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputShapes, int numInputShapes, float* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

    return _calculateOutputShapes<float>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesHalf(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputShapes, int numInputShapes, float16* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

    return _calculateOutputShapes<float16>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

nd4j::ShapeList* NativeOps::calculateOutputShapesDouble(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, int *iArgs, int numIArgs) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

    return _calculateOutputShapes<double>(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
}

template<typename T>
Nd4jStatus realExec(nd4j::ops::DeclarableOp<T>* op, Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, T* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace) {
    if (op == nullptr)
        nd4j_printf("Can't find requested operation: [%lld]\n", hash);

    // we're using the same fake nodeId everywhere here

    std::vector<nd4j::NDArray<T>*> inputs(numInputs);
    std::vector<nd4j::NDArray<T>*> outputs(numOutputs);
    std::vector<T> ttArgs(numTArgs);
    std::vector<int> iiArgs(numIArgs);

    // filling block now with inputs
    for (int e = 0; e < numInputs; e++) {
        auto buffer = (T *) inputBuffers[e];
        auto shape = (int *) inputShapes[e];

        inputs[e] = new nd4j::NDArray<T>(buffer, shape);
    }

    // if not inplace - transferring output arrays

    if (!isInplace)
        for (int e = 0; e < numOutputs; e++) {
            auto buffer = (T *) outputBuffers[e];

            // we want to keep original output shape intact
            auto shape = shape::copyShape((int *) outputShapes[e]);

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

            if (outputs[e]->ordering() != shape::order((int *) outputShapes[e]))
                outputs[e]->streamline(shape::order((int *) outputShapes[e]));
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

template Nd4jStatus realExec<float16>(nd4j::ops::DeclarableOp<float16>*, Nd4jPointer*, Nd4jIndex, Nd4jPointer*, Nd4jPointer*, int, Nd4jPointer*, Nd4jPointer*, int, float16*, int, int*, int, bool);
template Nd4jStatus realExec<float> (nd4j::ops::DeclarableOp<float>*, Nd4jPointer*, Nd4jIndex, Nd4jPointer*, Nd4jPointer*, int, Nd4jPointer*, Nd4jPointer*, int, float*, int, int*, int, bool);
template Nd4jStatus realExec<double>(nd4j::ops::DeclarableOp<double>*, Nd4jPointer*, Nd4jIndex, Nd4jPointer*, Nd4jPointer*, int, Nd4jPointer*, Nd4jPointer*, int, double*, int, int*, int, bool);


int NativeOps::execCustomOpFloat(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(hash);

    return realExec<float>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::execCustomOpDouble(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, double* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationDouble(hash);

    return realExec<double>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}

int NativeOps::execCustomOpHalf(Nd4jPointer* extraPointers, Nd4jIndex hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputs, Nd4jPointer* outputBuffers, Nd4jPointer* outputShapes, int numOutputs, float16* tArgs, int numTArgs, int *iArgs, int numIArgs, bool isInplace) {
    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationHalf(hash);

    return realExec<float16>(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs, tArgs, numTArgs, iArgs, numIArgs, isInplace);
}


int NativeOps::registerGraphFloat(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer flatBufferPointer) {
    auto graph = nd4j::graph::GraphExecutioner<float>::importFromFlatPointer(flatBufferPointer);

    nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

    return ND4J_STATUS_OK;
}

int NativeOps::registerGraphDouble(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer flatBufferPointer) {
    auto graph = nd4j::graph::GraphExecutioner<double>::importFromFlatPointer(flatBufferPointer);

    nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

    return ND4J_STATUS_OK;
}

int NativeOps::registerGraphHalf(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer flatBufferPointer) {
    auto graph = nd4j::graph::GraphExecutioner<float16>::importFromFlatPointer(flatBufferPointer);

    nd4j::graph::GraphHolder::getInstance()->registerGraph(graphId, graph);

    return ND4J_STATUS_OK;
}

template <typename T>
static VariablesSet<T>* executeStoredGraphT(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    auto graph = nd4j::graph::GraphHolder::getInstance()->pullGraph<T>(graphId);
    auto varSpace = graph->getVariableSpace()->clone();

    std::vector<nd4j::NDArray<T> *> handles;

    for (int e = 0; e < numInputs; e++) {
        auto idx = inputIndices[e];

        // we'll delete this array later, together with cloned VariableSpace
        auto array = new nd4j::NDArray<T>((T *) inputBuffers[e], (int *) inputShapes[e]);
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

VariablesSet<float>* NativeOps::executeStoredGraphFloat(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    return executeStoredGraphT<float>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

VariablesSet<float16>* NativeOps::executeStoredGraphHalf(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    return executeStoredGraphT<float16>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

VariablesSet<double>* NativeOps::executeStoredGraphDouble(Nd4jPointer *extraPointers, Nd4jIndex graphId, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int* inputIndices, int numInputs) {
    return executeStoredGraphT<double>(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
}

int NativeOps::unregisterGraph(Nd4jPointer *extraPointers, Nd4jIndex graphId) {

    nd4j::graph::GraphHolder::getInstance()->dropGraphAny(graphId);

    return ND4J_STATUS_OK;
}

void NativeOps::deletePointerArray(Nd4jPointer pointer) {
    Nd4jPointer *ptr = reinterpret_cast<Nd4jPointer *>(pointer);
    delete[] ptr;
}

void NativeOps::deleteIntArray(Nd4jPointer pointer) {
    int *ptr = reinterpret_cast<int *>(pointer);
    delete[] ptr;
}

template <typename T>
static void deleteVariablesSetT(Nd4jPointer pointer) {
    nd4j::graph::VariablesSet<T>* ptr = reinterpret_cast<nd4j::graph::VariablesSet<T>*>(pointer);
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

Nd4jPointer NativeOps::getGraphStateHalf(Nd4jIndex id) {
    return (Nd4jPointer) new nd4j::graph::GraphState<float16>(id);
}

Nd4jPointer NativeOps::getGraphStateFloat(Nd4jIndex id) {
    return (Nd4jPointer) new nd4j::graph::GraphState<float>(id);
}

Nd4jPointer NativeOps::getGraphStateDouble(Nd4jIndex id) {
    return (Nd4jPointer) new nd4j::graph::GraphState<double>(id);
}

void NativeOps::deleteGraphStateHalf(Nd4jPointer state) {
    auto stateP = (nd4j::graph::GraphState<float16> *) state;
    delete stateP;
}

void NativeOps::deleteGraphStateFloat(Nd4jPointer state) {
    auto stateP = (nd4j::graph::GraphState<float> *) state;
    delete stateP;
}

void NativeOps::deleteGraphStateDouble(Nd4jPointer state) {
    auto stateP = (nd4j::graph::GraphState<double> *) state;
    delete stateP;
}

template <typename T>
Nd4jStatus execCustomOpWithScope(Nd4jPointer *extraPointers, nd4j::graph::GraphState<T> *state, Nd4jIndex opHash, Nd4jIndex *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
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
        auto buffer = (T *) inputBuffers[e];
        auto shapeInfo = (int *) inputShapes[e];

        auto array = new NDArray<T>(buffer, shapeInfo, varSpace->workspace());
        
        // now we just put array to VarSpace
        varSpace->putVariable(0, e, array);
        node.pickInput(0, e);
    }

    // mapping scopes
    for (int e = 0; e < numScopes; e++) {
        // we should check scope existence in GraphState/Graph
        int scopeId = (int) scopes[e];
        if (!state->hasScope(scopeId)) {
            nd4j_printf("execCustomOpWithScope: referenced scope [%i] doesn't exist\n", scopeId);
            return Status::THROW();
        }
        node.pickInput(scopeId, 0);
    }

    auto result = LogicExecutor<T>::processNode(graph, &node);
    if (result != Status::OK())
        return result;

    // mapping outputs

    for (int e = 0; e < numOutputs; e++) {
        auto buffer = (T *) outputBuffers[e];
        auto shapeInfo = (int *) outputShapes[e];

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

Nd4jStatus NativeOps::execCustomOpWithScopeHalf(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jIndex opHash, Nd4jIndex *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    return execCustomOpWithScope<float16>(extraPointers, (nd4j::graph::GraphState<float16> *) state, opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}

Nd4jStatus NativeOps::execCustomOpWithScopeFloat(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jIndex opHash, Nd4jIndex *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    return execCustomOpWithScope<float>(extraPointers, (nd4j::graph::GraphState<float> *) state, opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}

Nd4jStatus NativeOps::execCustomOpWithScopeDouble(Nd4jPointer *extraPointers, Nd4jPointer state, Nd4jIndex opHash, Nd4jIndex *scopes, int numScopes, Nd4jPointer *inputBuffers, Nd4jPointer *inputShapes, int numInputs, Nd4jPointer *outputBuffers, Nd4jPointer *outputShapes, int numOutputs) {
    return execCustomOpWithScope<double>(extraPointers, (nd4j::graph::GraphState<double> *) state, opHash, scopes, numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes, numOutputs);
}


template void flattenGeneric<float16>(Nd4jPointer*, int, char, float16*, int*, float16*, int*);
template void flattenGeneric<float>(Nd4jPointer*, int, char, float*, int*, float*, int*);
template void flattenGeneric<double>(Nd4jPointer*, int, char, double*, int*, double*, int*);;

template void pullRowsGeneric<float16>(float16*, int*, float16*, int*, const int, int*, int*, Nd4jIndex*, int*, Nd4jIndex*);
template void pullRowsGeneric<float>(float*, int*, float*, int*, const int, int*, int*, Nd4jIndex*, int*, Nd4jIndex*);
template void pullRowsGeneric<double>(double*, int*, double*, int*, const int, int*, int*, Nd4jIndex*, int*, Nd4jIndex*);

template void tearGeneric<float16>(float16*, int*, Nd4jPointer*, int*, int*, Nd4jIndex*);
template void tearGeneric<float>(float*, int*, Nd4jPointer*, int*, int*, Nd4jIndex*);
template void tearGeneric<double>(double*, int*, Nd4jPointer*, int*, int*, Nd4jIndex*);

template void shuffleGeneric<float16>(float16**, int**, float16**, int**, int, int*, int**, Nd4jIndex**);
template void shuffleGeneric<float>(float**, int**, float**, int**, int, int*, int**, Nd4jIndex**);
template void shuffleGeneric<double>(double**, int**, double**, int**, int, int*, int**, Nd4jIndex**);



