//
// Created by agibsonccc on 2/21/16.
//

#include "../NativeOps.h"
#include "../NativeOpExcutioner.h"
#include <shape.h>
class DoubleNativeOpExecutioner : public NativeOpExcutioner<double> {
private:
	static DoubleNativeOpExecutioner *INSTANCE;
public:
	static DoubleNativeOpExecutioner * getInstance() {
		if(INSTANCE == NULL)
			INSTANCE = new DoubleNativeOpExecutioner();
		return INSTANCE;
	}
};

class FloatNativeOpExecutioner : public NativeOpExcutioner<float> {
private:
	static FloatNativeOpExecutioner *INSTANCE;
public:
	static FloatNativeOpExecutioner * getInstance() {
		if(INSTANCE == NULL)
			INSTANCE = new FloatNativeOpExecutioner();
		return INSTANCE;
	}
};


/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 */
double   NativeOps::execIndexReduceScalarDouble(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParams) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	return DoubleNativeOpExecutioner::getInstance()->execIndexReduceScalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer);

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
void   NativeOps::execIndexReduceDouble(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParams,
		long result,
		long resultShapeInfoBuffer,
		long dimension, int dimensionLength) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	DoubleNativeOpExecutioner::getInstance()->execIndexReduce(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer,dimensionPointer,dimensionLength);


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
void   NativeOps::execBroadcastDouble(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long y,
		long yShapeInfo,
		long result,
		long resultShapeInfo,
		long dimension, int dimensionLength) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	DoubleNativeOpExecutioner::getInstance()->execBroadcast(
			opNum,
            xPointer,
            xShapeInfoPointer,
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
 * @param dx
 * @param xStride
 * @param y
 * @param yStride
 * @param result
 * @param resultStride
 * @param extraParams
 * @param n
 */
void   NativeOps::execPairwiseTransformDouble(long *extraPointers,int opNum,
		long dx,
		int xStride,
		long y,
		int yStride,
		long result,
		int resultStride,
		long extraParams, int n) {
	double *xPointer = reinterpret_cast<double *>(dx);
	double *yPointer = reinterpret_cast<double *>(y);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	DoubleNativeOpExecutioner::getInstance()->execPairwiseTransform(opNum,xPointer,xStride,yPointer,yStride,resultPointer,resultStride,extraParamsPointer,n);
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
        long *extraPointers,
        int opNum,
		long dx,
		long xShapeInfo,
		long y,
		long yShapeInfo,
		long result,
		long resultShapeInfo,
		long extraParams,
		long xIndexes,
		long yIndexes,
		long resultIndexes) {
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
	DoubleNativeOpExecutioner::getInstance()->execPairwiseTransform(
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
        long *extraPointers,
        int opNum,
		long dx,
		long  xShapeInfo,
		long y,
		long  yShapeInfo,
		long result,
		long  resultShapeInfo,
		long extraParams) {
	double *xPointer = reinterpret_cast<double *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	DoubleNativeOpExecutioner::getInstance()->execPairwiseTransform(
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
        long *extraPointers,
        int opNum,
		long x,
		long xShapeInfo,
		long extraParams,
		long result,
		long resultShapeInfo) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	resultPointer[0] = DoubleNativeOpExecutioner::getInstance()->execReduceScalar(
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
void   NativeOps::execReduceDouble(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParams,
		long result,
		long resultShapeInfo,
		long dimension,int dimensionLength) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	DoubleNativeOpExecutioner::getInstance()->execReduce(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer,dimensionPointer,dimensionLength);

}

/**
 *
 * @param opNum
 * @param x
 * @param xShapeInfo
 * @param extraParams
 * @return
 */
double NativeOps::execReduceScalarDouble(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParams) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	return DoubleNativeOpExecutioner::getInstance()->execReduceScalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer);
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
void   NativeOps::execReduce3Double(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParamsVals,
		long y,
		long yShapeInfo,
		long result,
		long resultShapeInfo) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParamsVals);
	DoubleNativeOpExecutioner::getInstance()->execReduce3(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,yPointer,yShapeInfoPointer,resultPointer,resultShapeInfoPointer);
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
double   NativeOps::execReduce3ScalarDouble(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParamsVals,
		long y,
		long yShapeInfo) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParamsVals);
	return DoubleNativeOpExecutioner::getInstance()->execReduce3Scalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,yPointer,yShapeInfoPointer);
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
void   NativeOps::execReduce3Double(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParamsVals,
		long y,
		long yShapeInfo,
		long result,
		long resultShapeInfoBuffer,
		long dimension,
		int dimensionLength) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *yPointer = reinterpret_cast<double *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParamsVals);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	DoubleNativeOpExecutioner::getInstance()->execReduce3(opNum,
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
        long *extraPointers,
        int opNum,
		long x,
		int xStride,
		long result,
		int resultStride,
		double scalar,
		long extraParams,
		int n) {
	double *xPointer = reinterpret_cast<double *>(x);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	DoubleNativeOpExecutioner::getInstance()->execScalar(
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
        long *extraPointers,
        int opNum,
		long x,
		long xShapeInfo,
		long result,
		long resultShapeInfo,
		double scalar,
		long extraParams) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	DoubleNativeOpExecutioner::getInstance()->execScalar(
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
        long *extraPointers,
        int opNum,
		long x,
		long xShapeInfo,
		long result,
		long resultShapeInfo,
		double scalar,
		long extraParams,
		int n,
		long xIndexes,
		long resultIndexes) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *xIndexesPointer = reinterpret_cast<int *>(xIndexes);
	int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);
	DoubleNativeOpExecutioner::getInstance()->execScalar(
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
double   NativeOps::execSummaryStatsScalarDouble(long *extraPointers,int opNum,long x,
		long xShapeInfo,
		long extraParams) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	return DoubleNativeOpExecutioner::getInstance()->execSummaryStatsScalar(
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
void   NativeOps::execSummaryStatsDouble(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParams,
		long result,
		long resultShapeInfo) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	DoubleNativeOpExecutioner::getInstance()->execSummaryStats(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer);
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
void   NativeOps::execSummaryStatsDouble(long *extraPointers,int opNum,long x,
		long xShapeInfo,
		long extraParams,
		long result,
		long resultShapeInfoBuffer,
		long dimension, int dimensionLength) {
	double *xPointer = reinterpret_cast<double *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	DoubleNativeOpExecutioner::getInstance()->execSummaryStats(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer,dimensionPointer,dimensionLength);

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
void   NativeOps::execTransformDouble(long *extraPointers,int opNum,
		long dx,
		int xStride,
		long result,
		int resultStride,
		long extraParams, int n) {
	double *xPointer = reinterpret_cast<double *>(dx);
	double *resultPointer = reinterpret_cast<double *>(result);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	DoubleNativeOpExecutioner::getInstance()->execTransform(opNum,xPointer,xStride,resultPointer,resultStride,extraParamsPointer,n);
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
		long *extraPointers
		,int opNum,
		long dx,
		long xShapeInfo,
		long result,
		long resultShapeInfo,
		long extraParams) {
	double *xPointer = reinterpret_cast<double *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	DoubleNativeOpExecutioner::getInstance()->execTransform(
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
        long *extraPointers,
        int opNum,
		long dx,
		long xShapeInfo,
		long result,
		long resultShapeInfo,
		long extraParams,
		long xIndexes,
		long resultIndexes) {
	double *xPointer = reinterpret_cast<double *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	double *resultPointer = reinterpret_cast<double *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	double *extraParamsPointer = reinterpret_cast<double *>(extraParams);
	int *xIndexesPointer = reinterpret_cast<int *>(xIndexes);
	int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);
	DoubleNativeOpExecutioner::getInstance()->execTransform(
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
float   NativeOps::execIndexReduceScalarFloat(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParams) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	return FloatNativeOpExecutioner::getInstance()->execIndexReduceScalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer);
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
void   NativeOps::execIndexReduceFloat(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParams,
		long result,
		long resultShapeInfoBuffer,
		long dimension, int dimensionLength) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	FloatNativeOpExecutioner::getInstance()->execIndexReduce(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer,dimensionPointer,dimensionLength);


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
void   NativeOps::execBroadcastFloat(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long y,
		long yShapeInfo,
		long result,
		long resultShapeInfo,
		long dimension, int dimensionLength) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	FloatNativeOpExecutioner::getInstance()->execBroadcast(opNum,xPointer,xShapeInfoPointer,yPointer,yShapeInfoPointer,resultPointer,resultShapeInfoPointer,dimensionPointer,dimensionLength);

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
        long *extraPointers,
        int opNum,
		long dx,
		int xStride,
		long y,
		int yStride,
		long result,
		int resultStride,
		long extraParams, int n) {
	float *xPointer = reinterpret_cast<float *>(dx);
	float *yPointer = reinterpret_cast<float *>(y);
	float *resultPointer = reinterpret_cast<float *>(result);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	FloatNativeOpExecutioner::getInstance()->execPairwiseTransform(opNum,xPointer,xStride,yPointer,yStride,resultPointer,resultStride,extraParamsPointer,n);
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
        long *extraPointers,
        int opNum,
		long dx,
		long xShapeInfo,
		long y,
		long yShapeInfo,
		long result,
		long resultShapeInfo,
		long extraParams,
		long xIndexes,
		long yIndexes,
		long resultIndexes) {
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
	FloatNativeOpExecutioner::getInstance()->execPairwiseTransform(
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
void NativeOps::execPairwiseTransformFloat(
        long *extraPointers,
        int opNum,
		long dx,
		long  xShapeInfo,
		long y,
		long  yShapeInfo,
		long result,
		long  resultShapeInfo,
		long extraParams) {
	float *xPointer = reinterpret_cast<float *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	FloatNativeOpExecutioner::getInstance()->execPairwiseTransform(opNum,xPointer,xShapeInfoPointer,yPointer,yShapeInfoPointer,resultPointer,resultShapeInfoPointer,extraParamsPointer);

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
void   NativeOps::execReduceFloat(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParams,
		long result,
		long resultShapeInfo) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
    int *dimension = (int *) malloc(sizeof(int));
    dimension[0] = shape::MAX_DIMENSION;
	FloatNativeOpExecutioner::getInstance()->execReduce(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer,dimension,1);
    free(dimension);
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
        long *extraPointers,
        int opNum,
		long x,
		long xShapeInfo,
		long extraParams,
		long result,
		long resultShapeInfo,
		long dimension,
        int dimensionLength) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	FloatNativeOpExecutioner::getInstance()->execReduce(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            resultPointer,
            resultShapeInfoPointer,
            dimensionPointer,
            dimensionLength);

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
		long *extraPointers,
		int opNum,
		long x,
		long xShapeInfo,
		long extraParams) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	return FloatNativeOpExecutioner::getInstance()->execReduceScalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer);
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
        long *extraPointers,
        int opNum,
		long x,
		long xShapeInfo,
		long extraParamsVals,
		long y,
		long yShapeInfo,
		long result,
		long resultShapeInfo) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParamsVals);
	FloatNativeOpExecutioner::getInstance()->execReduce3(
            opNum,
            xPointer,
            xShapeInfoPointer,
            extraParamsPointer,
            yPointer,
            yShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer);

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
float   NativeOps::execReduce3ScalarFloat(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParamsVals,
		long y,
		long yShapeInfo) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParamsVals);
	return FloatNativeOpExecutioner::getInstance()->execReduce3Scalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,yPointer,yShapeInfoPointer);
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
void   NativeOps::execReduce3Float(long *extraPointers,int opNum,
		long x,
		long xShapeInfo,
		long extraParamsVals,
		long y,
		long yShapeInfo,
		long result,
		long resultShapeInfoBuffer,
		long dimension,
		int dimensionLength) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *yPointer = reinterpret_cast<float *>(y);
	int *yShapeInfoPointer = reinterpret_cast<int *>(yShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParamsVals);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	FloatNativeOpExecutioner::getInstance()->execReduce3(
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
void   NativeOps::execScalarFloat(long *extraPointers,int opNum,
		long x,
		int xStride,
		long result,
		int resultStride,
		double scalar,
		long extraParams,
		int n) {
	float *xPointer = reinterpret_cast<float *>(x);
	float *resultPointer = reinterpret_cast<float *>(result);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	FloatNativeOpExecutioner::getInstance()->execScalar(opNum,xPointer,xStride,resultPointer,resultStride,scalar,extraParamsPointer,n);

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
        long *extraPointers,
        int opNum,
		long x,
		long xShapeInfo,
		long result,
		long resultShapeInfo,
		float scalar,
		long extraParams) {
	float *xPointer = reinterpret_cast<float *>(x);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	FloatNativeOpExecutioner::getInstance()->execScalar(opNum,xPointer,resultShapeInfoPointer,resultPointer,resultShapeInfoPointer,scalar,extraParamsPointer);

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
		long *extraPointers,
		int opNum,
		long x,
		long xShapeInfo,
		long result,
		long resultShapeInfo,
		double scalar,
		long extraParams,
		long xIndexes,
		long resultIndexes) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *xIndexesPointer = reinterpret_cast<int *>(xIndexes);
	int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);
	FloatNativeOpExecutioner::getInstance()->execScalar(
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
		long *extraPointers,
		int opNum,
		long x,
		long xShapeInfo,
		long extraParams) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	return FloatNativeOpExecutioner::getInstance()->execSummaryStatsScalar(opNum,xPointer,xShapeInfoPointer,extraParamsPointer);
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
		long *extraPointers,
		int opNum,
		long x,
		long xShapeInfo,
		long extraParams,
		long result,
		long resultShapeInfo) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	FloatNativeOpExecutioner::getInstance()->execSummaryStats(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer);
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
void   NativeOps::execSummaryStatsFloat(long *extraPointers,int opNum,long x,
		long xShapeInfo,
		long extraParams,
		long result,
		long resultShapeInfoBuffer,
		long dimension, int dimensionLength) {
	float *xPointer = reinterpret_cast<float *>(x);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
		float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfoBuffer);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *dimensionPointer = reinterpret_cast<int *>(dimension);
	FloatNativeOpExecutioner::getInstance()->execSummaryStats(opNum,xPointer,xShapeInfoPointer,extraParamsPointer,resultPointer,resultShapeInfoPointer,dimensionPointer,dimensionLength);

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
        long *extraPointers,
        int opNum,
		long dx,
		int xStride,
		long result,
		int resultStride,
		long extraParams, int n) {
	float *xPointer = reinterpret_cast<float *>(dx);
	float *resultPointer = reinterpret_cast<float *>(result);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	FloatNativeOpExecutioner::getInstance()->execTransform(opNum,xPointer,xStride,resultPointer,resultStride,extraParamsPointer,n);
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
		long *extraPointers,
		int opNum,
		long dx,
		long xShapeInfo,
		long result,
		long resultShapeInfo,
		long extraParams) {
	float *xPointer = reinterpret_cast<float *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	FloatNativeOpExecutioner::getInstance()->execTransform(opNum,xPointer,xShapeInfoPointer,resultPointer,resultShapeInfoPointer,extraParamsPointer);
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
		long *extraPointers,
		int opNum,
		long dx,
		long xShapeInfo,
		long result,
		long resultShapeInfo,
		long extraParams,
		long xIndexes,
		long resultIndexes) {
	float *xPointer = reinterpret_cast<float *>(dx);
	int *xShapeInfoPointer = reinterpret_cast<int *>(xShapeInfo);
	float *resultPointer = reinterpret_cast<float *>(result);
	int *resultShapeInfoPointer = reinterpret_cast<int *>(resultShapeInfo);
	float *extraParamsPointer = reinterpret_cast<float *>(extraParams);
	int *xIndexesPointer = reinterpret_cast<int *>(xIndexes);
	int *resultIndexesPointer = reinterpret_cast<int *>(resultIndexes);
	FloatNativeOpExecutioner::getInstance()->execTransform(
            opNum,
            xPointer,
            xShapeInfoPointer,
            resultPointer,
            resultShapeInfoPointer,
            extraParamsPointer,
            xIndexesPointer,
            resultIndexesPointer);

}
