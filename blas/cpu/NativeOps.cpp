//
// Created by agibsonccc on 2/21/16.
//

#include "../NativeOps.h"
#include "../NativeOpExcutioner.h"
#include <pointercast.h>
#include <pairwise_util.h>
#include <templatemath.h>
#include <types/float8.h>
#include <type_conversions.h>

#ifdef __EXPERIMENTAL__
bool experimentalSupport = true;
#else
bool experimentalSupport = false;
#endif


/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
double   NativeOps::execIndexReduceScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                                Nd4jPointer x,
                                                Nd4jPointer xShapeInfo,
                                                Nd4jPointer extraParams) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	return NativeOpExcutioner<double>::execIndexReduceScalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer);

}

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
void   NativeOps::execIndexReduceDouble(Nd4jPointer *extraPointers,int opNum,
                                        Nd4jPointer x,
                                        Nd4jPointer xShapeInfo,
                                        Nd4jPointer extraParams,
                                        Nd4jPointer result,
                                        Nd4jPointer resultShapeInfoBuffer,
                                        Nd4jPointer dimension, int dimensionLength) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
	NativeOpExcutioner<double>::execIndexReduce(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer,dimensionPointer,dimensionLength,tadShapeInfo,tadOffsets);
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
                                      Nd4jPointer x,
                                      Nd4jPointer xShapeInfo,
                                      Nd4jPointer y,
                                      Nd4jPointer yShapeInfo,
                                      Nd4jPointer result,
                                      Nd4jPointer resultShapeInfo,
                                      Nd4jPointer dimension, int dimensionLength) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *yPointer = reinterpret_cast<double *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
	NativeOpExcutioner<double>::execBroadcast(
            opNum,
            xPointer,
            xShapeInfoPointer,
            yPointer,
            yShapeInfoPointer,
            resultPointer,
            dimensionPointer,
            dimensionLength, tadShapeInfo, tadOffsets);

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
void   NativeOps::execPairwiseTransformDouble(Nd4jPointer *extraPointers,int opNum,
                                              Nd4jPointer dx,
                                              int xStride,
                                              Nd4jPointer y,
                                              int yStride,
                                              Nd4jPointer result,
                                              int resultStride,
                                              Nd4jPointer extraParams, Nd4jIndex n) {
    double *xPointer = reinterpret_cast<double *>(dx);
    double *yPointer = reinterpret_cast<double *>(y);
    double *resultPointer = reinterpret_cast<double *>(result);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	NativeOpExcutioner<double>::execPairwiseTransform(opNum,xPointer,xStride,yPointer,yStride,resultPointer,resultStride,extraParamsPointer,n);
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
        Nd4jPointer dx,
        Nd4jPointer xShapeInfo,
        Nd4jPointer y,
        Nd4jPointer yShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer extraParams,
        Nd4jPointer xIndexes,
        Nd4jPointer yIndexes,
        Nd4jPointer resultIndexes) {
    double *xPointer = reinterpret_cast<double *>(dx);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *yPointer = reinterpret_cast<double *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
    int *xIndexesPointer = reinterpret_cast<int *>(xIndexes);
    int *yIndexesPointer = reinterpret_cast<int *>(yIndexes);
    int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);
	NativeOpExcutioner<double>::execPairwiseTransform(
            opNum,
            xPointer,
            xShapeInfoPointer,
            yPointer,
            yShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            extraParamsPointer,
            xIndexesPointer,
            yIndexesPointer,
            resultIndexesPointer);
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
        Nd4jPointer dx,
        Nd4jPointer  xShapeInfo,
        Nd4jPointer y,
        Nd4jPointer  yShapeInfo,
        Nd4jPointer result,
        Nd4jPointer  resultShapeInfo,
        Nd4jPointer extraParams) {
    double *xPointer = reinterpret_cast<double *>(dx);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *yPointer = reinterpret_cast<double *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	NativeOpExcutioner<double>::execPairwiseTransform(
            opNum,
            xPointer,
            xShapeInfoPointer,
            yPointer,
            yShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            extraParamsPointer);
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
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer extraParams,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
    resultPointer[0] = NativeOpExcutioner<double>::execReduceScalar(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer);

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
                                   Nd4jPointer x,
                                   Nd4jPointer xShapeInfo,
                                   Nd4jPointer extraParams,
                                   Nd4jPointer result,
                                   Nd4jPointer resultShapeInfo,
                                   Nd4jPointer dimension,int dimensionLength) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);

	NativeOpExcutioner<double>::execReduce(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer,dimensionPointer,dimensionLength, tadShapeInfo, tadOffsets);

}

void   NativeOps::execReduceHalf(Nd4jPointer *extraPointers,int opNum,
                                   Nd4jPointer x,
                                   Nd4jPointer xShapeInfo,
                                   Nd4jPointer extraParams,
                                   Nd4jPointer result,
                                   Nd4jPointer resultShapeInfo,
                                   Nd4jPointer dimension,int dimensionLength) {
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
                                         Nd4jPointer x,
                                         Nd4jPointer xShapeInfo,
                                         Nd4jPointer extraParams) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
    return NativeOpExcutioner<double>::execReduceScalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer);
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
                                    Nd4jPointer x,
                                    Nd4jPointer xShapeInfo,
                                    Nd4jPointer extraParamsVals,
                                    Nd4jPointer y,
                                    Nd4jPointer yShapeInfo,
                                    Nd4jPointer result,
                                    Nd4jPointer resultShapeInfo) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *yPointer = reinterpret_cast<double *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParamsVals);
	NativeOpExcutioner<double>::execReduce3(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,yPointer,yShapeInfoPointer,resultPointer,resultShapeInfoPointer);
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
                                            Nd4jPointer x,
                                            Nd4jPointer xShapeInfo,
                                            Nd4jPointer extraParamsVals,
                                            Nd4jPointer y,
                                            Nd4jPointer yShapeInfo) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *yPointer = reinterpret_cast<double *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParamsVals);
    return NativeOpExcutioner<double>::execReduce3Scalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,yPointer,yShapeInfoPointer);
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
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execReduce3Double(Nd4jPointer *extraPointers,int opNum,
                                    Nd4jPointer x,
                                    Nd4jPointer xShapeInfo,
                                    Nd4jPointer extraParamsVals,
                                    Nd4jPointer y,
                                    Nd4jPointer yShapeInfo,
                                    Nd4jPointer result,
                                    Nd4jPointer resultShapeInfoBuffer,
                                    Nd4jPointer dimension,
                                    int dimensionLength) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *yPointer = reinterpret_cast<double *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParamsVals);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
	NativeOpExcutioner<double>::execReduce3(opNum,
                                                          xPointer,
                                                          xShapeInfoPointer,
                                                          extraParamsPointer,
                                                          yPointer,
                                                          yShapeInfoPointer,
                                                          resultPointer,
                                                          resultShapeInfoPointer,
                                                          dimensionPointer,
                                                          dimensionLength);

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
        Nd4jPointer x,
        int xStride,
        Nd4jPointer result,
        int resultStride,
        double scalar,
        Nd4jPointer extraParams,
        Nd4jIndex n) {
    double *xPointer = reinterpret_cast<double *>(x);
    double *resultPointer = reinterpret_cast<double *>(result);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	NativeOpExcutioner<double>::execScalar(
            opNum,
            xPointer,
            xStride,
            resultPointer,
            resultStride,
            scalar,
            extraParamsPointer,
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
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        double scalar,
        Nd4jPointer extraParams) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	NativeOpExcutioner<double>::execScalar(
            opNum,
            xPointer,
            xShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            scalar,
            extraParamsPointer);
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
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        double scalar,
        Nd4jPointer extraParams,
        Nd4jIndex n,
        Nd4jPointer xIndexes,
        Nd4jPointer resultIndexes) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
    int *xIndexesPointer = reinterpret_cast<int *>(xIndexes);
    int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);
	NativeOpExcutioner<double>::execScalar(
            opNum,
            xPointer,
            xShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            scalar,
            extraParamsPointer,
            xIndexesPointer,
            resultIndexesPointer);

}
/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
double   NativeOps::execSummaryStatsScalarDouble(Nd4jPointer *extraPointers, int opNum,Nd4jPointer x,
                                                 Nd4jPointer xShapeInfo,
                                                 Nd4jPointer extraParams,bool biasCorrected) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	return NativeOpExcutioner<double>::execSummaryStatsScalar(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,biasCorrected);
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
                                         Nd4jPointer x,
                                         Nd4jPointer xShapeInfo,
                                         Nd4jPointer extraParams,
                                         Nd4jPointer result,
                                         Nd4jPointer resultShapeInfo,bool biasCorrected) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	NativeOpExcutioner<double>::execSummaryStats(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            resultPointer,
            resultShapeInfoPointer,
            biasCorrected);
}
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
void   NativeOps::execSummaryStatsDouble(Nd4jPointer *extraPointers, int opNum,Nd4jPointer x,
                                         Nd4jPointer xShapeInfo,
                                         Nd4jPointer extraParams,
                                         Nd4jPointer result,
                                         Nd4jPointer resultShapeInfoBuffer,
                                         Nd4jPointer dimension, int dimensionLength,bool biasCorrected) {
    double *xPointer = reinterpret_cast<double *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
	NativeOpExcutioner<double>::execSummaryStats(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            resultPointer,
            resultShapeInfoPointer,
            dimensionPointer,
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
                                      Nd4jPointer dx,
                                      int xStride,
                                      Nd4jPointer result,
                                      int resultStride,
                                      Nd4jPointer extraParams, Nd4jIndex n) {
    double *xPointer = reinterpret_cast<double *>(dx);
    double *resultPointer = reinterpret_cast<double *>(result);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	NativeOpExcutioner<double>::execTransform(opNum,xPointer,xStride,resultPointer,resultStride,extraParamsPointer,n);
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
        Nd4jPointer dx,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer extraParams) {
    double *xPointer = reinterpret_cast<double *>(dx);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	NativeOpExcutioner<double>::execTransform(
            opNum,
            xPointer,
            xShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            extraParamsPointer);
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
        Nd4jPointer dx,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer extraParams,
        Nd4jPointer xIndexes,
        Nd4jPointer resultIndexes) {
    double *xPointer = reinterpret_cast<double *>(dx);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    double *resultPointer = reinterpret_cast<double *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
    Nd4jIndex *xIndexesPointer = reinterpret_cast<Nd4jIndex*>(xIndexes);
    Nd4jIndex *resultIndexesPointer = reinterpret_cast<Nd4jIndex *>(resultIndexes);
	NativeOpExcutioner<double>::execTransform(
            opNum,
            xPointer,
            xShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            extraParamsPointer,
            xIndexesPointer,
            resultIndexesPointer);

}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
float   NativeOps::execIndexReduceScalarFloat(Nd4jPointer *extraPointers, int opNum,
                                              Nd4jPointer x,
                                              Nd4jPointer xShapeInfo,
                                              Nd4jPointer extraParams) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	return NativeOpExcutioner<float>::execIndexReduceScalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer);
}

float   NativeOps::execIndexReduceScalarHalf(Nd4jPointer *extraPointers, int opNum,
                                              Nd4jPointer x,
                                              Nd4jPointer xShapeInfo,
                                              Nd4jPointer extraParams) {
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
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execIndexReduceFloat(Nd4jPointer *extraPointers, int opNum,
                                       Nd4jPointer x,
                                       Nd4jPointer xShapeInfo,
                                       Nd4jPointer extraParams,
                                       Nd4jPointer result,
                                       Nd4jPointer resultShapeInfoBuffer,
                                       Nd4jPointer dimension, int dimensionLength) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);

	NativeOpExcutioner<float>::execIndexReduce(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer,dimensionPointer,dimensionLength,tadShapeInfo, tadOffsets);
}

void   NativeOps::execIndexReduceHalf(Nd4jPointer *extraPointers, int opNum,
                                       Nd4jPointer x,
                                       Nd4jPointer xShapeInfo,
                                       Nd4jPointer extraParams,
                                       Nd4jPointer result,
                                       Nd4jPointer resultShapeInfoBuffer,
                                       Nd4jPointer dimension, int dimensionLength) {
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
                                     Nd4jPointer x,
                                     Nd4jPointer xShapeInfo,
                                     Nd4jPointer y,
                                     Nd4jPointer yShapeInfo,
                                     Nd4jPointer result,Nd4jPointer resultShapeInfo,
                                     Nd4jPointer dimension, int dimensionLength) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *yPointer = reinterpret_cast<float *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
	NativeOpExcutioner<float>::execBroadcast(opNum,xPointer,xShapeInfoPointer,yPointer,yShapeInfoPointer,resultPointer,dimensionPointer,dimensionLength,
                                                           tadShapeInfo, tadOffsets);

}

void   NativeOps::execBroadcastHalf(Nd4jPointer *extraPointers,int opNum,
                                     Nd4jPointer x,
                                     Nd4jPointer xShapeInfo,
                                     Nd4jPointer y,
                                     Nd4jPointer yShapeInfo,
                                     Nd4jPointer result,Nd4jPointer resultShapeInfo,
                                     Nd4jPointer dimension, int dimensionLength) {
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
        Nd4jPointer dx,
        int xStride,
        Nd4jPointer y,
        int yStride,
        Nd4jPointer result,
        int resultStride,
        Nd4jPointer extraParams, Nd4jIndex n) {
    float *xPointer = reinterpret_cast<float *>(dx);
    float *yPointer = reinterpret_cast<float *>(y);
    float *resultPointer = reinterpret_cast<float *>(result);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	NativeOpExcutioner<float>::execPairwiseTransform(
            opNum,
            xPointer,
            xStride,
            yPointer,
            yStride,
            resultPointer,
            resultStride,
            extraParamsPointer,
            n);
}

void   NativeOps::execPairwiseTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer dx,
        int xStride,
        Nd4jPointer y,
        int yStride,
        Nd4jPointer result,
        int resultStride,
        Nd4jPointer extraParams, Nd4jIndex n) {
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
        Nd4jPointer dx,
        Nd4jPointer xShapeInfo,
        Nd4jPointer y,
        Nd4jPointer yShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer extraParams,
        Nd4jPointer xIndexes,
        Nd4jPointer yIndexes,
        Nd4jPointer resultIndexes) {
    float *xPointer = reinterpret_cast<float *>(dx);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *yPointer = reinterpret_cast<float *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
    int *xIndexesPointer = reinterpret_cast<int *>(xIndexes);
    int *yIndexesPointer = reinterpret_cast<int *>(yIndexes);
    int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);
	NativeOpExcutioner<float>::execPairwiseTransform(
            opNum,
            xPointer,
            xShapeInfoPointer,
            yPointer,
            yShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            extraParamsPointer,
            xIndexesPointer,
            yIndexesPointer,
            resultIndexesPointer);

}

void NativeOps::execPairwiseTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer dx,
        Nd4jPointer xShapeInfo,
        Nd4jPointer y,
        Nd4jPointer yShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer extraParams,
        Nd4jPointer xIndexes,
        Nd4jPointer yIndexes,
        Nd4jPointer resultIndexes) {
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
        Nd4jPointer dx,
        Nd4jPointer  xShapeInfo,
        Nd4jPointer y,
        Nd4jPointer  yShapeInfo,
        Nd4jPointer result,
        Nd4jPointer  resultShapeInfo,
        Nd4jPointer extraParams) {
    float *xPointer = reinterpret_cast<float *>(dx);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *yPointer = reinterpret_cast<float *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	NativeOpExcutioner<float>::execPairwiseTransform(opNum,xPointer,xShapeInfoPointer,yPointer,yShapeInfoPointer,resultPointer,resultShapeInfoPointer,extraParamsPointer);

}

void NativeOps::execPairwiseTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer dx,
        Nd4jPointer  xShapeInfo,
        Nd4jPointer y,
        Nd4jPointer  yShapeInfo,
        Nd4jPointer result,
        Nd4jPointer  resultShapeInfo,
        Nd4jPointer extraParams) {
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
                                  Nd4jPointer x,
                                  Nd4jPointer xShapeInfo,
                                  Nd4jPointer extraParams,
                                  Nd4jPointer result,
                                  Nd4jPointer resultShapeInfo) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
    int *dimension = new int[1];
    dimension[0] = MAX_DIMENSION;
	NativeOpExcutioner<float>::execReduce(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer,dimension,1,
                                                        nullptr, nullptr);
    delete[] dimension;
}

void   NativeOps::execReduceHalf(Nd4jPointer *extraPointers,int opNum,
                                  Nd4jPointer x,
                                  Nd4jPointer xShapeInfo,
                                  Nd4jPointer extraParams,
                                  Nd4jPointer result,
                                  Nd4jPointer resultShapeInfo) {
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
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer extraParams,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer dimension,
        int dimensionLength) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);

	NativeOpExcutioner<float>::execReduce(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            resultPointer,
            resultShapeInfoPointer,
            dimensionPointer,
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
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer extraParams) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
    return NativeOpExcutioner<float>::execReduceScalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer);
}

float NativeOps::execReduceScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer extraParams) {
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
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer extraParamsVals,
        Nd4jPointer y,
        Nd4jPointer yShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *yPointer = reinterpret_cast<float *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParamsVals);
	NativeOpExcutioner<float>::execReduce3(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            yPointer,
            yShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer);

}

void   NativeOps::execReduce3Half(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer extraParamsVals,
        Nd4jPointer y,
        Nd4jPointer yShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo) {
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
                                          Nd4jPointer x,
                                          Nd4jPointer xShapeInfo,
                                          Nd4jPointer extraParamsVals,
                                          Nd4jPointer y,
                                          Nd4jPointer yShapeInfo) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *yPointer = reinterpret_cast<float *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParamsVals);
	return NativeOpExcutioner<float>::execReduce3Scalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,yPointer,yShapeInfoPointer);
}

float   NativeOps::execReduce3ScalarHalf(Nd4jPointer *extraPointers,int opNum,
                                          Nd4jPointer x,
                                          Nd4jPointer xShapeInfo,
                                          Nd4jPointer extraParamsVals,
                                          Nd4jPointer y,
                                          Nd4jPointer yShapeInfo) {
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
 * @param resultShapeInfoBuffer
 * @param dimension
 * @param dimensionLength
 */
void   NativeOps::execReduce3Float(Nd4jPointer *extraPointers,int opNum,
                                   Nd4jPointer x,
                                   Nd4jPointer xShapeInfo,
                                   Nd4jPointer extraParamsVals,
                                   Nd4jPointer y,
                                   Nd4jPointer yShapeInfo,
                                   Nd4jPointer result,
                                   Nd4jPointer resultShapeInfoBuffer,
                                   Nd4jPointer dimension,
                                   int dimensionLength) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *yPointer = reinterpret_cast<float *>(y);
    int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParamsVals);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
	NativeOpExcutioner<float>::execReduce3(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            yPointer,
            yShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            dimensionPointer,
            dimensionLength);

}

void   NativeOps::execReduce3Half(Nd4jPointer *extraPointers,int opNum,
                                   Nd4jPointer x,
                                   Nd4jPointer xShapeInfo,
                                   Nd4jPointer extraParamsVals,
                                   Nd4jPointer y,
                                   Nd4jPointer yShapeInfo,
                                   Nd4jPointer result,
                                   Nd4jPointer resultShapeInfoBuffer,
                                   Nd4jPointer dimension,
                                   int dimensionLength) {
    // no-op
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
                                  Nd4jPointer x,
                                  int xStride,
                                  Nd4jPointer result,
                                  int resultStride,
                                  double scalar,
                                  Nd4jPointer extraParams,
                                  Nd4jIndex n) {
    float *xPointer = reinterpret_cast<float *>(x);
    float *resultPointer = reinterpret_cast<float *>(result);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	NativeOpExcutioner<float>::execScalar(opNum,xPointer,xStride,resultPointer,resultStride,scalar,extraParamsPointer,n);

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
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        float scalar,
        Nd4jPointer extraParams) {
    float *xPointer = reinterpret_cast<float *>(x);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	NativeOpExcutioner<float>::execScalar(opNum,xPointer,resultShapeInfoPointer,resultPointer,resultShapeInfoPointer,scalar,extraParamsPointer);

}

void NativeOps::execScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        float scalar,
        Nd4jPointer extraParams) {
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
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        double scalar,
        Nd4jPointer extraParams,
        Nd4jPointer xIndexes,
        Nd4jPointer resultIndexes) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
    int *xIndexesPointer = reinterpret_cast<int *>(xIndexes);
    int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);
	NativeOpExcutioner<float>::execScalar(
            opNum,
            xPointer,
            xShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            scalar,
            extraParamsPointer,
            xIndexesPointer,
            resultIndexesPointer);

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
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer extraParams,bool biasCorrected) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
    return NativeOpExcutioner<float>::execSummaryStatsScalar(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            biasCorrected);
}

float   NativeOps::execSummaryStatsScalarHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer extraParams,bool biasCorrected) {
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
 */
void   NativeOps::execSummaryStatsFloat(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer extraParams,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,bool biasCorrected) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	NativeOpExcutioner<float>::execSummaryStats(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            resultPointer,
            resultShapeInfoPointer,
            biasCorrected);
}


void   NativeOps::execSummaryStatsHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer x,
        Nd4jPointer xShapeInfo,
        Nd4jPointer extraParams,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,bool biasCorrected) {
    // no-op
}

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
void   NativeOps::execSummaryStatsFloat(Nd4jPointer *extraPointers,int opNum,Nd4jPointer x,
                                        Nd4jPointer xShapeInfo,
                                        Nd4jPointer extraParams,
                                        Nd4jPointer result,
                                        Nd4jPointer resultShapeInfoBuffer,
                                        Nd4jPointer dimension, int dimensionLength,bool biasCorrected) {
    float *xPointer = reinterpret_cast<float *>(x);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
	NativeOpExcutioner<float>::execSummaryStats(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            resultPointer,
            resultShapeInfoPointer,
            dimensionPointer,
            dimensionLength,
            biasCorrected);

}


void   NativeOps::execSummaryStatsHalf(Nd4jPointer *extraPointers,int opNum,Nd4jPointer x,
                                        Nd4jPointer xShapeInfo,
                                        Nd4jPointer extraParams,
                                        Nd4jPointer result,
                                        Nd4jPointer resultShapeInfoBuffer,
                                        Nd4jPointer dimension, int dimensionLength,bool biasCorrected) {
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
        Nd4jPointer dx,
        int xStride,
        Nd4jPointer result,
        int resultStride,
        Nd4jPointer extraParams, Nd4jIndex n) {
    float *xPointer = reinterpret_cast<float *>(dx);
    float *resultPointer = reinterpret_cast<float *>(result);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	NativeOpExcutioner<float>::execTransform(opNum,xPointer,xStride,resultPointer,resultStride,extraParamsPointer,n);
}

void   NativeOps::execTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer dx,
        int xStride,
        Nd4jPointer result,
        int resultStride,
        Nd4jPointer extraParams, Nd4jIndex n) {
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
        Nd4jPointer dx,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer extraParams) {
    float *xPointer = reinterpret_cast<float *>(dx);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	NativeOpExcutioner<float>::execTransform(
            opNum,
            xPointer,
            xShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            extraParamsPointer);
}

void   NativeOps::execTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer dx,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer extraParams) {

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
        Nd4jPointer dx,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer extraParams,
        Nd4jPointer xIndexes,
        Nd4jPointer resultIndexes) {
    float *xPointer = reinterpret_cast<float *>(dx);
    int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
    float *resultPointer = reinterpret_cast<float *>(result);
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
    Nd4jIndex *xIndexesPointer = reinterpret_cast<Nd4jIndex*>(xIndexes);
    Nd4jIndex *resultIndexesPointer = reinterpret_cast<Nd4jIndex *>(resultIndexes);
	NativeOpExcutioner<float>::execTransform(
            opNum,
            xPointer,
            xShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            extraParamsPointer,
            xIndexesPointer,
            resultIndexesPointer);

}

void   NativeOps::execTransformHalf(
        Nd4jPointer *extraPointers,
        int opNum,
        Nd4jPointer dx,
        Nd4jPointer xShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer extraParams,
        Nd4jPointer xIndexes,
        Nd4jPointer resultIndexes) {
    // no-op
}



template <typename T>
void flattenGeneric(Nd4jPointer *extraPointers,
                    int offset,
                    char order,
                    Nd4jPointer result,
                    Nd4jPointer resultShapeInfo,
                    Nd4jPointer input,
                    Nd4jPointer inputShapeInfo) {
    T *resultPointer = reinterpret_cast<T *>(result);
    int *resultShapeInfoBufferPointer = reinterpret_cast<int *>(resultShapeInfo);
    T *inputPointer = reinterpret_cast<T *>(input);
    int *inputShapeInfoPointer = reinterpret_cast<int *>(inputShapeInfo);
    int numOnes = 0;
    int *shape = shape::shapeOf(inputShapeInfoPointer);
    int wholeRank = shape::rank(inputShapeInfoPointer);
    for(int i = 0; i < wholeRank; i++) {
        if(shape[i] == 1)
            numOnes++;
    }



    //start at the given offset
    resultPointer += offset;
    char inputOrder = shape::order(inputShapeInfoPointer);
    int len = shape::length(inputShapeInfoPointer);
    int resultEleStride = shape::elementWiseStride(resultShapeInfoBufferPointer);
    int inputEleStride = shape::elementWiseStride(inputShapeInfoPointer);
    int numTads, stride, dimension, dimensionLength;
    int rank = shape::rank(inputShapeInfoPointer);
    int *xStride = shape::stride(inputShapeInfoPointer);
    int *xShape = shape::shapeOf(inputShapeInfoPointer);

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
            memcpy(resultPointer, inputPointer, len* sizeof(T));
        }
        else if (resultEleStride >= 1 && inputEleStride >= 1) {
            if (len < 8000) {
#pragma omp simd
                for (int i = 0; i < len; i++) {
                    resultPointer[i * resultEleStride] = inputPointer[i * inputEleStride];
                }
            }
            else {
#pragma omp parallel for simd
                for (int i = 0; i < len; i++) {
                    resultPointer[i * resultEleStride] = inputPointer[i * inputEleStride];
                }
            }
        }
        else {
            int idx = 0;
            int rank = shape::rank(inputShapeInfoPointer);
            int *coord = new int[rank];
            int *xShape = shape::shapeOf(inputShapeInfoPointer);
            int *xStride = shape::stride(inputShapeInfoPointer);
            int len = shape::length(inputShapeInfoPointer);
            if(order == 'f') {
                for(int i = 0; i < len; i++) {
                    shape::ind2sub(rank, xShape, i, coord);
                    int offset = shape::getOffset(0,xShape,xStride,coord,rank);
                    resultPointer[idx++] = inputPointer[offset];

                }
            }
            else {
                for(int i = 0; i < len; i++) {
                    shape::ind2subC(rank, xShape, i, coord);
                    int offset = shape::getOffset(0,xShape,xStride,coord,rank);
                    resultPointer[idx++] = inputPointer[offset];

                }
            }

            delete[] coord;
        }
    }
    else {
        int rank = shape::rank(inputShapeInfoPointer);
        int *xShape = shape::shapeOf(inputShapeInfoPointer);
        int tadShape = xShape[dimension];
        shape::TAD tad(inputShapeInfoPointer,&dimension,dimensionLength);
        tad.createTadOnlyShapeInfo();
#pragma omp  parallel  for
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
                resultPointer[resultOffset + j] = inputPointer[tadOffset + j * stride];

            }
        }


    }
}


/**
  * Concatneate multi array of the same shape together
  * along a particular dimension
  */
template <typename T>
void concatGeneric(
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo) {
    int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
    //number of total arrays, every other dimension should be the same
    T **dataBuffers = reinterpret_cast<T **>(data);
    int **inputShapeInfoPointers = reinterpret_cast<int **>(inputShapeInfo);
    T *resultPointer = reinterpret_cast<T *>(result);

    bool allC = true;
    bool allScalar = true;

    //nothing to concat
    if(numArrays == 1)
        return;

    //detect whether all arrays are c ordered or not
    //Also detect whether they are all scalars
    for(int i = 0; i < numArrays; i++) {
        allC &= (shape::order(inputShapeInfoPointers[i]) == 'c');
        allScalar &= (shape::isScalar(inputShapeInfoPointers[i]));
    }

    //we are merging all scalars
    if(allScalar) {
        for(int i = 0; i < numArrays; i++) {
            resultPointer[i] = dataBuffers[i][0];
        }
        return;
    }


    int length = shape::length(resultShapeInfoPointer);


    if(allC && dimension == 0 && shape::order(resultShapeInfoPointer) == 'c') {
        int currBuffer = 0;
        int currBufferOffset = 0;
        for(int i = 0; i <  length; i++) {
            resultPointer[i] = dataBuffers[currBuffer][currBufferOffset++];
            if(currBufferOffset >= shape::length(inputShapeInfoPointers[currBuffer])) {
                currBuffer++;
                currBufferOffset = 0;
            }
        }

        return;
    }

    int resultStride = shape::elementWiseStride(resultShapeInfoPointer);
    //vector case
    if(shape::isVector(resultShapeInfoPointer)) {
        int idx = 0;
        if(resultStride == 1) {
            for(int i = 0; i < numArrays; i++) {
                if(shape::isVector(inputShapeInfoPointers[i]) || shape::order(inputShapeInfoPointers[i]) == shape::order(resultShapeInfoPointer)) {
                    Nd4jIndex  currArrLength = shape::length(inputShapeInfoPointers[i]);
                    Nd4jIndex eleStride = shape::elementWiseStride(inputShapeInfoPointers[i]);
                    if(eleStride == 1) {
                        for(Nd4jIndex arrIdx = 0; arrIdx < currArrLength; arrIdx++) {
                            if(idx >= shape::length(resultShapeInfoPointer)) {
                                break;
                            }
                            resultPointer[idx] = dataBuffers[i][arrIdx];
                            idx++;
                        }
                    }
                    else {
                        for(Nd4jIndex arrIdx = 0; arrIdx < currArrLength; arrIdx++) {
                            resultPointer[idx] = dataBuffers[i][arrIdx * eleStride];
                            if(idx >= shape::length(resultShapeInfoPointer)) {
                                break;
                            }

                            idx++;

                        }
                    }
                }
                    //non vector or different order (element wise stride can't be used)
                else {
                    int *coordsUse = new int[shape::rank(inputShapeInfoPointers[i])];
                    Nd4jIndex  currArrLength = shape::length(inputShapeInfoPointers[i]);
                    for(Nd4jIndex arrIdx = 0; arrIdx < currArrLength; arrIdx++) {
                        shape::ind2subC(shape::rank(inputShapeInfoPointers[i]),shape::shapeOf(inputShapeInfoPointers[i]),arrIdx,coordsUse);
                        Nd4jIndex offset = shape::getOffset(0,shape::shapeOf(inputShapeInfoPointers[i]),shape::stride(inputShapeInfoPointers[i]),coordsUse,shape::rank(inputShapeInfoPointers[i]));
                        resultPointer[idx] = dataBuffers[i][offset];
                        if(idx >= shape::length(resultShapeInfoPointer)) {
                            break;
                        }

                        idx++;

                    }

                    delete[] coordsUse;
                }


            }
        }
        else {
            for(int i = 0; i < numArrays; i++) {
                if(shape::isVector(inputShapeInfoPointers[i]) || shape::order(inputShapeInfoPointers[i]) == shape::order(resultShapeInfoPointer)) {
                    Nd4jIndex  currArrLength = shape::length(inputShapeInfoPointers[i]);
                    Nd4jIndex eleStride = shape::elementWiseStride(inputShapeInfoPointers[i]);
                    if(eleStride == 1) {
                        for(Nd4jIndex arrIdx = 0; arrIdx < currArrLength; arrIdx++) {
                            if(idx >= shape::length(resultShapeInfoPointer)) {
                                break;
                            }
                            resultPointer[idx * resultStride] = dataBuffers[i][arrIdx];
                            idx++;

                        }
                    }
                    else {
                        for(Nd4jIndex arrIdx = 0; arrIdx < currArrLength; arrIdx++) {
                            if(idx >= shape::length(resultShapeInfoPointer)) {
                                break;
                            }
                            resultPointer[idx * resultStride] = dataBuffers[i][arrIdx * eleStride];
                            idx++;
                        }
                    }

                }
                //non vector or different order (element wise stride can't be used)
                else {
                    int *coordsUse = new int[shape::rank(inputShapeInfoPointers[i])];
                    Nd4jIndex  currArrLength = shape::length(inputShapeInfoPointers[i]);

                    for(Nd4jIndex arrIdx = 0; arrIdx < currArrLength; arrIdx++) {
                        shape::ind2subC(shape::rank(inputShapeInfoPointers[i]),shape::shapeOf(inputShapeInfoPointers[i]),arrIdx,coordsUse);
                        Nd4jIndex offset = shape::getOffset(0,shape::shapeOf(inputShapeInfoPointers[i]),shape::stride(inputShapeInfoPointers[i]),coordsUse,shape::rank(inputShapeInfoPointers[i]));
                        resultPointer[idx] = dataBuffers[i][offset];
                        if(idx >= shape::length(resultShapeInfoPointer)) {
                            break;
                        }

                        idx++;

                    }

                    delete[] coordsUse;
                }

            }
        }

        return;
    }


    //tad shape information for result
    shape::TAD resultTad(resultShapeInfoPointer,&dimension,1);
    resultTad.createTadOnlyShapeInfo();
    resultTad.createOffsets();
    int resultTadEleStride = shape::elementWiseStride(resultTad.tadOnlyShapeInfo);

    int arrOffset = 0;
    int tadEleStride = shape::elementWiseStride(resultTad.tadOnlyShapeInfo);
    for(int i = 0; i < numArrays; i++) {
        //tad info for the current array
        shape::TAD arrTad(inputShapeInfoPointers[i],&dimension,1);
        arrTad.createTadOnlyShapeInfo();
        arrTad.createOffsets();

        //element wise stride and length for tad of current array
        int arrTadEleStride = shape::elementWiseStride(arrTad.tadOnlyShapeInfo);
        int arrTadLength = shape::length(arrTad.tadOnlyShapeInfo);
        for(int j = 0; j < arrTad.numTads; j++) {
            T *arrTadData = dataBuffers[i] + arrTad.tadOffsets[j];
            //result tad offset + the current offset for each tad + array offset (matches current array)
            T *currResultTadWithOffset = resultPointer  + resultTad.tadOffsets[j];
            //ensure we start at the proper index, we need to move the starting index forward relative to the desired array offset
            int* sub = shape::ind2subC(shape::rank(resultTad.tadOnlyShapeInfo),shape::shapeOf(resultTad.tadOnlyShapeInfo),arrOffset);
            Nd4jIndex baseOffset = shape::getOffset(0,shape::shapeOf(resultTad.tadOnlyShapeInfo),shape::stride(resultTad.tadOnlyShapeInfo),sub,shape::rank(resultTad.tadOnlyShapeInfo));
            delete[] sub;
            currResultTadWithOffset += baseOffset;
            if(arrTadEleStride > 0 && shape::order(resultShapeInfoPointer) == shape::order(arrTad.tadOnlyShapeInfo)) {
                if(arrTadEleStride == 1 && resultTadEleStride == 1) {
                    //iterate over the specified chunk of the tad
                    for(int k = 0; k < arrTadLength; k++) {
                        currResultTadWithOffset[k] = arrTadData[k];
                    }

                } //element wise stride isn't 1 for both can't use memcpy
                else if(tadEleStride > 0 && shape::order(resultShapeInfoPointer) == shape::order(arrTad.tadOnlyShapeInfo)) {
                    for(int k = 0; k < arrTadLength; k++) {
                        currResultTadWithOffset[k * tadEleStride] = arrTadData[k * arrTadEleStride];
                    }
                }
            }
            else {
                int idx = 0;
                //use element wise stride for result but not this tad
                if(tadEleStride > 0 && shape::order(resultShapeInfoPointer) == shape::order(arrTad.tadOnlyShapeInfo)) {
                    if(arrTad.wholeThing) {
                        for(int k = 0; k < shape::length(arrTad.tadOnlyShapeInfo); k++) {
                            currResultTadWithOffset[idx *resultTadEleStride] = arrTadData[k];

                        }
                    }
                    else {
                        int shapeIter[MAX_RANK];
                        int coord[MAX_RANK];
                        int dim;
                        int rankIter = shape::rank(arrTad.tadOnlyShapeInfo);
                        int xStridesIter[MAX_RANK];
                        if (PrepareOneRawArrayIter<T>(rankIter,
                                                      shape::shapeOf(arrTad.tadOnlyShapeInfo),
                                                      arrTadData,
                                                      shape::stride(arrTad.tadOnlyShapeInfo),
                                                      &rankIter,
                                                      shapeIter,
                                                      &arrTadData,
                                                      xStridesIter) >= 0) {
                            ND4J_RAW_ITER_START(dim, shape::rank(arrTad.tadOnlyShapeInfo), coord, shapeIter); {
                                    /* Process the innermost dimension */
                                    currResultTadWithOffset[idx *resultTadEleStride] = arrTadData[0];
                                }
                            ND4J_RAW_ITER_ONE_NEXT(dim,
                                                   rankIter,
                                                   coord,
                                                   shapeIter,
                                                   arrTadData,
                                                   xStridesIter);

                        }
                        else {
                            printf("Unable to prepare array\n");
                        }


                    }

                }
                    //don't use element wise stride for either
                else {

                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int xStridesIter[MAX_RANK];
                    int resultStridesIter[MAX_RANK];
                    int *xShape = shape::shapeOf(arrTad.tadOnlyShapeInfo);
                    int *xStride = shape::stride(arrTad.tadOnlyShapeInfo);
                    int *resultStride = shape::stride(resultTad.tadOnlyShapeInfo);
                    int rank = shape::rank(arrTad.tadOnlyShapeInfo);
                    if (PrepareTwoRawArrayIter<T>(rank,
                                                  xShape,
                                                  arrTadData,
                                                  xStride,
                                                  currResultTadWithOffset,
                                                  resultStride,
                                                  &rank,
                                                  shapeIter,
                                                  &arrTadData,
                                                  xStridesIter,
                                                  &currResultTadWithOffset,
                                                  resultStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                currResultTadWithOffset[0] = arrTadData[0];
                            } ND4J_RAW_ITER_TWO_NEXT(
                                dim,
                                rank,
                                coord,
                                shapeIter,
                                arrTadData,
                                xStridesIter,
                                currResultTadWithOffset,
                                resultStridesIter);


                    }
                }
            }

        }

        arrOffset += shape::length(arrTad.tadOnlyShapeInfo);
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
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    concatGeneric<float>(
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
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    // no-op
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
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
    concatGeneric<double>(
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
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer input,
        Nd4jPointer inputShapeInfo) {
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
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer input,
        Nd4jPointer inputShapeInfo) {
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
        Nd4jPointer result,
        Nd4jPointer resultShapeInfo,
        Nd4jPointer input,
        Nd4jPointer inputShapeInfo) {
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

Nd4jPointer NativeOps::createBlasHandle() {
    return 0L;
}

int NativeOps::registerEvent(Nd4jPointer event, Nd4jPointer stream) {
    return 0L;
}

int NativeOps::setBlasStream(Nd4jPointer handle, Nd4jPointer stream) {
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
    // no-op?
}

void NativeOps::enableVerboseMode(bool reallyEnable) {
    // no-op?
}

void NativeOps::setGridLimit(int gridSize) {
    // no-op
}

void NativeOps::tadOnlyShapeInfo(Nd4jPointer xShapeInfo, Nd4jPointer dimension, int dimensionLength, Nd4jPointer targetBuffer, Nd4jPointer offsetsBuffer) {
    int *hostXShapeInfo = reinterpret_cast<int *>(xShapeInfo);
    int *dimensionPointer = reinterpret_cast<int *>(dimension);
    int *target = reinterpret_cast<int *>(targetBuffer);
    int *offsets = reinterpret_cast<int *>(offsetsBuffer);


    shape::TAD *tad = new shape::TAD();
    tad->init(hostXShapeInfo, dimensionPointer, dimensionLength);
    //tad->setOutputBuffer(target);
    tad->createTadOnlyShapeInfo();
    tad->createOffsets();


    std::memcpy((void *) target, tad->tadOnlyShapeInfo, (tad->tadOnlyShapeInfo[0] * 2 + 4) * sizeof(int));
    std::memcpy((void *) offsets, tad->tadOffsets, tad->numTads * sizeof(int));

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
                     int *tadOffsets) {
    const int xEWS = shape::elementWiseStride(tadShapeInfo);
    const int zEWS = shape::elementWiseStride(zShapeInfo);
    const int tadLength = shape::length(tadShapeInfo);

#pragma omp parallel for schedule(guided) if (n > 16)
    for (int idx = 0; idx < n; idx++) {
        int tadOffsetForBlock = tadOffsets[indexes[idx]];

        T *rX = x + tadOffsetForBlock;
        T *rZ = z + idx * tadLength;

        if (xEWS == 1 && zEWS == 1) {

#pragma omp simd
            for (int i = 0; i < tadLength; i++ ) {
                rZ[i] = rX[i];
            }
        } else {

#pragma omp simd
            for (int i = 0; i < tadLength; i++ ) {
                rZ[i * zEWS] = rX[i * xEWS];
            }
        }
    }
}

void NativeOps::pullRowsHalf(Nd4jPointer *extraPointers, Nd4jPointer x, Nd4jPointer xShapeInfo, Nd4jPointer z, Nd4jPointer zShapeInfo, int n, Nd4jPointer indexes,  Nd4jPointer tadShapeInfo, Nd4jPointer tadOffsets) {
    // no-op
}

void NativeOps::pullRowsFloat(Nd4jPointer *extraPointers, Nd4jPointer x, Nd4jPointer xShapeInfo, Nd4jPointer z, Nd4jPointer zShapeInfo, int n, Nd4jPointer indexes,  Nd4jPointer tadShapeInfo, Nd4jPointer tadOffsets) {
    float *xBuffer = reinterpret_cast<float *>(x);
    float *zBuffer = reinterpret_cast<float *>(z);
    int *zShape = reinterpret_cast<int *>(zShapeInfo);
    int *xShape = reinterpret_cast<int *>(xShapeInfo);

    int *index = reinterpret_cast<int *>(indexes);
    int *tadOnlyShapeInfo = reinterpret_cast<int *>(tadShapeInfo);
    int *tadOffset = reinterpret_cast<int *>(tadOffsets);

    pullRowsGeneric<float>(xBuffer, xShape, zBuffer, zShape, n, index, tadOnlyShapeInfo, tadOffset);
}

void NativeOps::pullRowsDouble(Nd4jPointer *extraPointers, Nd4jPointer x, Nd4jPointer xShapeInfo, Nd4jPointer z, Nd4jPointer zShapeInfo, int n, Nd4jPointer indexes, Nd4jPointer tadShapeInfo, Nd4jPointer tadOffsets) {
    double *xBuffer = reinterpret_cast<double *>(x);
    double *zBuffer = reinterpret_cast<double *>(z);
    int *zShape = reinterpret_cast<int *>(zShapeInfo);
    int *xShape = reinterpret_cast<int *>(xShapeInfo);

    int *index = reinterpret_cast<int *>(indexes);
    int *tadOnlyShapeInfo = reinterpret_cast<int *>(tadShapeInfo);
    int *tadOffset = reinterpret_cast<int *>(tadOffsets);

    pullRowsGeneric<double>(xBuffer, xShape, zBuffer, zShape, n, index, tadOnlyShapeInfo, tadOffset);
}


template<typename T>
void averageGeneric(T **x, T *z, int n, const Nd4jIndex length, bool propagate) {

// aggregation step
// TODO: this step should be improved, to exploit SIMD
#pragma omp parallel for simd schedule(guided)
    for (Nd4jIndex i = 0; i < length; i++) {
        z[i] = 0.0;

        for (int ar = 0; ar < n; ar++) {
            z[i] += x[ar][i];
        }
    }

//div step
    if (length > 8000) {
#pragma omp parallel for simd schedule(guided)
        for (Nd4jIndex i = 0; i < length; i++) {
            z[i] /= n;
        }
    } else {
#pragma omp simd
        for (Nd4jIndex i = 0; i < length; i++) {
            z[i] /= n;
        }
    }

//propagation step
    if (propagate) {
#pragma omp parallel for if (n > 4 || length > 8000)
        for(int ar = 0; ar < n; ar++) {

#pragma omp simd
            for (Nd4jIndex i = 0; i < length; i++) {
                x[ar][i] = z[i];
            }
        }
    }
}

void NativeOps::averageHalf(Nd4jPointer *extras, Nd4jPointer dx, Nd4jPointer dz, int n, Nd4jIndex length, bool propagate) {
    // no-op
}

void NativeOps::averageFloat(Nd4jPointer *extras, Nd4jPointer dx, Nd4jPointer dz, int n, Nd4jIndex length, bool propagate) {
    float **x = reinterpret_cast<float **>(dx);
    float *z = reinterpret_cast<float *>(dz);

    averageGeneric<float>(x, z, n, length, propagate);
}

void NativeOps::averageDouble(Nd4jPointer *extras, Nd4jPointer dx, Nd4jPointer dz, int n, Nd4jIndex length, bool propagate) {
    double **x = reinterpret_cast<double **>(dx);
    double *z = reinterpret_cast<double *>(dz);

    averageGeneric<double>(x, z, n, length, propagate);
}

void NativeOps::enableP2P(bool enable) {
    // no-op
}

bool NativeOps::isP2PAvailable() {
    // always TRUE for cpu backend
    return true;
}

void NativeOps::checkP2P() {
    // no-op
}

template<typename T>
void shuffleGeneric(T **dX, int **xShapeInfo, T **dZ, int **zShapeInfo, int N, int *shuffleMap, int **tadOnlyShapeInfo, int **tadOffsets) {


#pragma omp parallel for if (N > 1)
    for (int f = 0; f < N; f++) {
        T *x = (T *) dX[f];
        //T *z = (T *) dZ[f];

        int *tadOffset = (int *) tadOffsets[f];


        const int tadLength = shape::length(tadOnlyShapeInfo[f]);
        int tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
        int tadRank = shape::rank(tadOnlyShapeInfo[f]);
        int numTads = shape::length(xShapeInfo[f]) / tadLength;

        int *tadShape = shape::shapeOf(tadOnlyShapeInfo[f]);
        int *tadStride = shape::stride(tadOnlyShapeInfo[f]);

        //printf("Array: [%i], tadEWS: [%i], tadLength: [%i]\n", f, tadEWS, tadLength);

        // TODO: omp *probably* has no sense here, since 99% of uses for this method will be inside DataSet. but worth a check

        for (Nd4jIndex r = 0; r < numTads / 2; r++) {
            int oldOffset = tadOffset[r];
            int newOffset = tadOffset[shuffleMap[r]];

            T *rX = x + oldOffset;
            T *rY = x + newOffset;

            if (tadEWS == 1) {

#pragma omp simd
                    for (Nd4jIndex i = 0; i < tadLength; i++) {
                        nd4j::math::nd4j_swap<T>(rX[i], rY[i]);
                    }

            } else {
                // ind2sub branch
                int xCoord[MAX_RANK];
                int yCoord[MAX_RANK];

#pragma omp parallel for schedule(guided) if (N == 1 && tadLength > 512)
                for (Nd4jIndex i = 0; i < tadLength; i++) {
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

void NativeOps::shuffleFloat(Nd4jPointer *extras, Nd4jPointer dx, Nd4jPointer xShapeInfo, Nd4jPointer dz, Nd4jPointer zShapeInfo, int N, Nd4jPointer shuffleMap, Nd4jPointer tadShapeInfo, Nd4jPointer tadOffsets) {
    float **x = reinterpret_cast<float **>(dx);
    float **z = reinterpret_cast<float **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int *shuffle = reinterpret_cast<int *>(shuffleMap);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    int **tadOffset = reinterpret_cast<int **>(tadOffsets);

    shuffleGeneric<float>(x, xShape, z, zShape, N, shuffle, tadOnlyShapeInfo, tadOffset);
}

void NativeOps::shuffleDouble(Nd4jPointer *extras, Nd4jPointer dx, Nd4jPointer xShapeInfo, Nd4jPointer dz, Nd4jPointer zShapeInfo, int N, Nd4jPointer shuffleMap, Nd4jPointer tadShapeInfo, Nd4jPointer tadOffsets) {
    double **x = reinterpret_cast<double **>(dx);
    double **z = reinterpret_cast<double **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int *shuffle = reinterpret_cast<int *>(shuffleMap);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    int **tadOffset = reinterpret_cast<int **>(tadOffsets);

    shuffleGeneric<double>(x, xShape, z, zShape, N, shuffle, tadOnlyShapeInfo, tadOffset);
}

void NativeOps::shuffleHalf(Nd4jPointer *extras, Nd4jPointer dx, Nd4jPointer xShapeInfo, Nd4jPointer dz, Nd4jPointer zShapeInfo, int N, Nd4jPointer shuffleMap, Nd4jPointer tadShapeInfo, Nd4jPointer tadOffsets) {
    // no-op
}

void NativeOps::execMetaPredicateReduceFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jPointer dx, Nd4jPointer xShapeInfo, Nd4jPointer dy, Nd4jPointer yShapeInfo, Nd4jPointer dz, Nd4jPointer zShapeInfo, Nd4jPointer dimension, int dimensionLength, Nd4jPointer tadShapeInfo, Nd4jPointer tadOffsets, Nd4jPointer extraA, Nd4jPointer extraB, float scalarA, float scalarB, bool scalarReturned) {
    // no-op
}

bool NativeOps::isExperimentalEnabled() {
    return experimentalSupport;
}

void NativeOps::execMetaPredicateShapeFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, Nd4jPointer dx, Nd4jPointer xShapeInfo, Nd4jPointer dy, Nd4jPointer yShapeInfo, Nd4jPointer dz, Nd4jPointer zShapeInfo, Nd4jPointer extraA, Nd4jPointer extraB, float scalarA, float scalarB) {
    // no-op;
}

void NativeOps::setOmpMinThreads(int threads) {
    // TODO: to be implemented
}

void NativeOps::execMetaPredicateStridedFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, Nd4jPointer dx, int xStride, Nd4jPointer dy, int yStride, Nd4jPointer dz, int zStride, Nd4jPointer extraA, Nd4jPointer extraB, float scalarA, float scalarB) {
    // no-op
}

void NativeOps::execMetaPredicateShapeDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, Nd4jPointer dx, Nd4jPointer xShapeInfo, Nd4jPointer dy, Nd4jPointer yShapeInfo, Nd4jPointer dz, Nd4jPointer zShapeInfo, Nd4jPointer extraA, Nd4jPointer extraB, double scalarA, double scalarB) {
    // no-op;
}

void NativeOps::execMetaPredicateStridedDouble(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, Nd4jPointer dx, int xStride, Nd4jPointer dy, int yStride, Nd4jPointer dz, int zStride, Nd4jPointer extraA, Nd4jPointer extraB, double scalarA, double scalarB) {
    // no-op
}

void NativeOps::execMetaPredicateShapeHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, Nd4jPointer dx, Nd4jPointer xShapeInfo, Nd4jPointer dy, Nd4jPointer yShapeInfo, Nd4jPointer dz, Nd4jPointer zShapeInfo, Nd4jPointer extraA, Nd4jPointer extraB, float scalarA, float scalarB) {
    // no-op;
}

void NativeOps::execMetaPredicateStridedHalf(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, Nd4jPointer dx, int xStride, Nd4jPointer dy, int yStride, Nd4jPointer dz, int zStride, Nd4jPointer extraA, Nd4jPointer extraB, float scalarA, float scalarB) {
    // no-op
}