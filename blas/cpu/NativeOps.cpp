//
// Created by agibsonccc on 2/21/16.
//

int tad_threshold = 1;
int element_threshold = 32;

#include "../NativeOps.h"
#include "../NativeOpExcutioner.h"
#include <pointercast.h>
#include <pairwise_util.h>
#include <templatemath.h>
#include <types/float8.h>
#include <type_conversions.h>
#include <aggregates.h>
#include <helper_ptrmap.h>

char *name;
bool nameSet = false;


#ifdef __EXPERIMENTAL__
bool experimentalSupport = true;
#else
bool experimentalSupport = false;
#endif


void NativeOps::setElementThreshold(int num) {
	if (num > 0)
		element_threshold = num;
}

void NativeOps::setTADThreshold(int num) {
	if (num > 0)
		tad_threshold = num;
}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
double   NativeOps::execIndexReduceScalarDouble(Nd4jPointer *extraPointers,int opNum,
                                                double *x,
                                                int *xShapeInfo,
                                                double *extraParams) {
	return NativeOpExcutioner<double>::execIndexReduceScalar(opNum,x,xShapeInfo,extraParams);

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
                                        int *dimension, int dimensionLength) {
    int *tadShapeInfo = reinterpret_cast<int *>(extraPointers[0]);
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
	NativeOpExcutioner<double>::execIndexReduce(opNum,x,xShapeInfo,extraParams,result,resultShapeInfo,dimension,dimensionLength,tadShapeInfo,tadOffsets);
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
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[2]);
    int *tadOffsetsZ = reinterpret_cast<int *>(extraPointers[3]);
	NativeOpExcutioner<double>::execBroadcast(
            opNum,
            x,
            xShapeInfo,
            y,
            yShapeInfo,
            result,
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
void   NativeOps::execPairwiseTransformDouble(Nd4jPointer *extraPointers,int opNum,
                                              double *dx,
                                              int xStride,
                                              double *y,
                                              int yStride,
                                              double *result,
                                              int resultStride,
                                              double *extraParams, Nd4jIndex n) {
	NativeOpExcutioner<double>::execPairwiseTransform(opNum,dx,xStride,y,yStride,result,resultStride,extraParams,n);
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
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
	NativeOpExcutioner<double>::execReduce(opNum,x,xShapeInfo,extraParams,result,resultShapeInfo,dimension,dimensionLength, tadShapeInfo, tadOffsets);
}

void   NativeOps::execReduceHalf(Nd4jPointer *extraPointers,int opNum,
                                   float16 *x,
                                   int *xShapeInfo,
                                   float16 *extraParams,
                                   float16 *result,
                                   int *resultShapeInfo,
                                   int *dimension,int dimensionLength) {
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
	NativeOpExcutioner<double>::execReduce3(opNum,x,xShapeInfo,extraParams,y,yShapeInfo,result,resultShapeInfo);
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
	NativeOpExcutioner<double>::execReduce3(opNum,
                                                          x,
                                                          xShapeInfo,
                                                          extraParams,
                                                          y,
                                                          yShapeInfo,
                                                          result,
                                                          resultShapeInfo,
                                                          dimension,
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
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);

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
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
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
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[2]);
    int *tadOffsetsZ = reinterpret_cast<int *>(extraPointers[3]);
	NativeOpExcutioner<float>::execBroadcast(opNum,x,xShapeInfo,y,yShapeInfo,result,dimension,dimensionLength,
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
	NativeOpExcutioner<float>::execReduce(opNum,x,xShapeInfo,extraParams,result,resultShapeInfo,dimension,1, nullptr, nullptr);
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
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
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
	NativeOpExcutioner<float>::execReduce3(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            y,
            yShapeInfo,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength);

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
	NativeOpExcutioner<float>::execScalar(opNum,x,xStride,result,resultStride,scalar,extraParams,n);

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
        double scalar,
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
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);

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
template <typename T>
void concatGeneric(
        int dimension,
        int numArrays,
        Nd4jPointer *data,
        Nd4jPointer *inputShapeInfo,
        T *result,
        int *resultShapeInfo) {
    //number of total arrays, every other dimension should be the same
    T **dataBuffers = reinterpret_cast<T **>(data);
    int **inputShapeInfoPointers = reinterpret_cast<int **>(inputShapeInfo);

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
            result[i] = dataBuffers[i][0];
        }
        return;
    }


    int length = shape::length(resultShapeInfo);


    if(allC && dimension == 0 && shape::order(resultShapeInfo) == 'c') {
        int currBuffer = 0;
        int currBufferOffset = 0;
        for(int i = 0; i <  length; i++) {
            result[i] = dataBuffers[currBuffer][currBufferOffset++];
            if(currBufferOffset >= shape::length(inputShapeInfoPointers[currBuffer])) {
                currBuffer++;
                currBufferOffset = 0;
            }
        }

        return;
    }

    int resultStride = shape::elementWiseStride(resultShapeInfo);
    //vector case
    if(shape::isVector(resultShapeInfo)) {
        int idx = 0;
        if(resultStride == 1) {
            for(int i = 0; i < numArrays; i++) {
                if(shape::isVector(inputShapeInfoPointers[i]) || shape::order(inputShapeInfoPointers[i]) == shape::order(resultShapeInfo)) {
                    Nd4jIndex  currArrLength = shape::length(inputShapeInfoPointers[i]);
                    Nd4jIndex eleStride = shape::elementWiseStride(inputShapeInfoPointers[i]);
                    if(eleStride == 1) {
                        for(Nd4jIndex arrIdx = 0; arrIdx < currArrLength; arrIdx++) {
                            if(idx >= shape::length(resultShapeInfo)) {
                                break;
                            }
                            result[idx] = dataBuffers[i][arrIdx];
                            idx++;
                        }
                    }
                    else {
                        for(Nd4jIndex arrIdx = 0; arrIdx < currArrLength; arrIdx++) {
                            result[idx] = dataBuffers[i][arrIdx * eleStride];
                            if(idx >= shape::length(resultShapeInfo)) {
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
                        result[idx] = dataBuffers[i][offset];
                        if(idx >= shape::length(resultShapeInfo)) {
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
                if(shape::isVector(inputShapeInfoPointers[i]) || shape::order(inputShapeInfoPointers[i]) == shape::order(resultShapeInfo)) {
                    Nd4jIndex  currArrLength = shape::length(inputShapeInfoPointers[i]);
                    Nd4jIndex eleStride = shape::elementWiseStride(inputShapeInfoPointers[i]);
                    if(eleStride == 1) {
                        for(Nd4jIndex arrIdx = 0; arrIdx < currArrLength; arrIdx++) {
                            if(idx >= shape::length(resultShapeInfo)) {
                                break;
                            }
                            result[idx * resultStride] = dataBuffers[i][arrIdx];
                            idx++;

                        }
                    }
                    else {
                        for(Nd4jIndex arrIdx = 0; arrIdx < currArrLength; arrIdx++) {
                            if(idx >= shape::length(resultShapeInfo)) {
                                break;
                            }
                            result[idx * resultStride] = dataBuffers[i][arrIdx * eleStride];
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
                        result[idx] = dataBuffers[i][offset];
                        if(idx >= shape::length(resultShapeInfo)) {
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
    shape::TAD resultTad(resultShapeInfo,&dimension,1);
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
            T *currResultTadWithOffset = result  + resultTad.tadOffsets[j];
            //ensure we start at the proper index, we need to move the starting index forward relative to the desired array offset
            int* sub = shape::ind2subC(shape::rank(resultTad.tadOnlyShapeInfo),shape::shapeOf(resultTad.tadOnlyShapeInfo),arrOffset);
            Nd4jIndex baseOffset = shape::getOffset(0,shape::shapeOf(resultTad.tadOnlyShapeInfo),shape::stride(resultTad.tadOnlyShapeInfo),sub,shape::rank(resultTad.tadOnlyShapeInfo));
            delete[] sub;
            currResultTadWithOffset += baseOffset;
            if(arrTadEleStride > 0 && shape::order(resultShapeInfo) == shape::order(arrTad.tadOnlyShapeInfo)) {
                if(arrTadEleStride == 1 && resultTadEleStride == 1) {
                    //iterate over the specified chunk of the tad
                    for(int k = 0; k < arrTadLength; k++) {
                        currResultTadWithOffset[k] = arrTadData[k];
                    }

                } //element wise stride isn't 1 for both can't use memcpy
                else if(tadEleStride > 0 && shape::order(resultShapeInfo) == shape::order(arrTad.tadOnlyShapeInfo)) {
                    for(int k = 0; k < arrTadLength; k++) {
                        currResultTadWithOffset[k * tadEleStride] = arrTadData[k * arrTadEleStride];
                    }
                }
            }
            else {
                int idx = 0;
                //use element wise stride for result but not this tad
                if(tadEleStride > 0 && shape::order(resultShapeInfo) == shape::order(arrTad.tadOnlyShapeInfo)) {
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
        float *result,
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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
        float16 *result,
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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
        double *result,
        int *resultShapeInfo, Nd4jPointer *tadPointers, Nd4jPointer *offsetPointers) {
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

int NativeOps::getDeviceMajor(Nd4jPointer ptrToDeviceId) {
    return 0;
}

int NativeOps::getDeviceMinor(Nd4jPointer ptrToDeviceId) {
    return 0;
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

void NativeOps::tadOnlyShapeInfo(int *xShapeInfo, int *dimension, int dimensionLength, int *target, int *offsets) {
    shape::TAD *tad = new shape::TAD();
    tad->init(xShapeInfo, dimension, dimensionLength);
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
                     int *tadOffsets,
                     int *zTadShapeInfo,
                     int *zTadOffsets) {
    const int xEWS = shape::elementWiseStride(tadShapeInfo);
    const int zEWS = shape::elementWiseStride(zTadShapeInfo);
    const int tadLength = shape::length(tadShapeInfo);

    int elementsPerThread = n / TAD_THRESHOLD;
    int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
    _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

#pragma omp parallel for num_threads(_threads) if (n > 1) schedule(guided) default(shared)
    for (int idx = 0; idx < n; idx++) {
        int tadOffsetForBlock = tadOffsets[indexes[idx]];

        T *rX = x + tadOffsetForBlock;
        T *rZ = z + zTadOffsets[idx];

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

void NativeOps::pullRowsHalf(Nd4jPointer *extraPointers, float16 *x, int *xShapeInfo, float16 *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, int *tadOffsets, int *zTadShapeInfo, int *zTadOffsets) {
    // no-op
}

void NativeOps::pullRowsFloat(Nd4jPointer *extraPointers, float *x, int *xShapeInfo, float *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, int *tadOffsets, int *zTadShapeInfo, int *zTadOffsets) {
    pullRowsGeneric<float>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}

void NativeOps::pullRowsDouble(Nd4jPointer *extraPointers, double *x, int *xShapeInfo, double *z, int *zShapeInfo, int n, int *indexes, int *tadShapeInfo, int *tadOffsets, int *zTadShapeInfo, int *zTadOffsets) {
    pullRowsGeneric<double>(x, xShapeInfo, z, zShapeInfo, n, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
}


template<typename T>
void averageGeneric(T **x, T *z, int n, const Nd4jIndex length, bool propagate) {

// aggregation step
// TODO: this step should be improved, to exploit SIMD
#pragma omp parallel for schedule(guided) default(shared)
    for (Nd4jIndex i = 0; i < length; i++) {
        z[i] = 0.0;

#pragma omp simd
        for (int ar = 0; ar < n; ar++) {
            z[i] += x[ar][i];
        }
    }

//div step
    if (length > ELEMENT_THRESHOLD) {
#pragma omp parallel for simd schedule(guided) default(shared)
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
#pragma omp parallel for if (n > 4 || length > ELEMENT_THRESHOLD) default(shared)
        for(int ar = 0; ar < n; ar++) {

#pragma omp simd
            for (Nd4jIndex i = 0; i < length; i++) {
                x[ar][i] = z[i];
            }
        }
    }
}

void NativeOps::averageHalf(Nd4jPointer *extras, Nd4jPointer *dx, float16 *dz, int n, Nd4jIndex length, bool propagate) {
    // no-op
}

void NativeOps::averageFloat(Nd4jPointer *extras, Nd4jPointer *dx, float *dz, int n, Nd4jIndex length, bool propagate) {
    float **x = reinterpret_cast<float **>(dx);
    averageGeneric<float>(x, dz, n, length, propagate);
}

void NativeOps::averageDouble(Nd4jPointer *extras, Nd4jPointer *dx, double *dz, int n, Nd4jIndex length, bool propagate) {
    double **x = reinterpret_cast<double **>(dx);
    averageGeneric<double>(x, dz, n, length, propagate);
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


#pragma omp parallel for if (N > 1) default(shared)
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

        for (Nd4jIndex r = 0; r < numTads; r++) {
            if (shuffleMap[r] < 0)
                continue;

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

void NativeOps::shuffleFloat(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    float **x = reinterpret_cast<float **>(dx);
    float **z = reinterpret_cast<float **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    int **tadOffset = reinterpret_cast<int **>(tadOffsets);

    shuffleGeneric<float>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);
}

void NativeOps::shuffleDouble(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    double **x = reinterpret_cast<double **>(dx);
    double **z = reinterpret_cast<double **>(dz);
    int **xShape = reinterpret_cast<int **>(xShapeInfo);
    int **zShape = reinterpret_cast<int **>(zShapeInfo);
    int **tadOnlyShapeInfo = reinterpret_cast<int **>(tadShapeInfo);
    int **tadOffset = reinterpret_cast<int **>(tadOffsets);

    shuffleGeneric<double>(x, xShape, z, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset);
}

void NativeOps::shuffleHalf(Nd4jPointer *extras, Nd4jPointer *dx, Nd4jPointer *xShapeInfo, Nd4jPointer *dz, Nd4jPointer *zShapeInfo, int N, int *shuffleMap, Nd4jPointer *tadShapeInfo, Nd4jPointer *tadOffsets) {
    // no-op
}

void NativeOps::execMetaPredicateReduceFloat(Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets, float *extraA, float *extraB, float scalarA, float scalarB, bool scalarReturned) {
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
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[2]);
    int *tadOffsetsZ = reinterpret_cast<int *>(extraPointers[3]);

    NativeOpExcutioner<float>::execScalar(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            z,
            zShapeInfo,
            scalars,
            dimension,
            dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
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
    int *tadOffsets = reinterpret_cast<int *>(extraPointers[1]);
    int *tadShapeInfoZ = reinterpret_cast<int *>(extraPointers[2]);
    int *tadOffsetsZ = reinterpret_cast<int *>(extraPointers[3]);

    NativeOpExcutioner<double>::execScalar(
            opNum,
            x,
            xShapeInfo,
            extraParams,
            z,
            zShapeInfo,
            scalars,
            dimension,
            dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
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

    NativeOpExcutioner<float>::execAggregate(opNum, arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
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

    NativeOpExcutioner<double>::execAggregate(opNum, arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
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



void NativeOps::execAggregateBatchFloat(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments) {

    // probably, we don't want too much threads as usually
    int _threads = nd4j::math::nd4j_min<int>(numAggregates, omp_get_max_threads());

    nd4j::PointersHelper<float> helper(ptrToArguments, numAggregates, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals);

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

        execAggregateFloat(extraPointers, opNum, arguments, helper.getNumArguments(i), shapes, helper.getNumShapeArguments(i), idxArg, helper.getNumIndexArguments(i), (int **) intArrays, helper.getNumIntArrayArguments(i), realArg, helper.getNumRealArguments(i));

        delete [] intArrays;
    }
}


void NativeOps::execAggregateBatchDouble(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments) {

    // probably, we don't want too much threads as usually
    int _threads = nd4j::math::nd4j_min<int>(numAggregates, omp_get_max_threads());

    nd4j::PointersHelper<double> helper(ptrToArguments, numAggregates, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIdx, maxReals);

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

        execAggregateDouble(extraPointers, opNum, arguments, helper.getNumArguments(i), shapes, helper.getNumShapeArguments(i), idxArg, helper.getNumIndexArguments(i), (int **) intArrays, helper.getNumIntArrayArguments(i), realArg, helper.getNumRealArguments(i));

        delete [] intArrays;
    }


}

void NativeOps::execAggregateBatchHalf(Nd4jPointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments) {
    // TODO: add support for fp16
}