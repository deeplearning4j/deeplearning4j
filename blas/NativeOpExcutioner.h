//
// Created by agibsonccc on 1/28/16.
//

#ifndef NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
#define NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H

#include <broadcasting.h>
#include <indexreduce.h>
#include <pairwise_transform.h>
#include <reduce.h>
#include <reduce3.h>
#include <summarystatsreduce.h>
#include <transform.h>
#include <scalar.h>
#include <aggregates.h>
#include <pointercast.h>
/**
 * Native op executioner:
 *
 */
template <typename T>
class NativeOpExcutioner {
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
    static T execIndexReduceScalar(int opNum,
                            T *x,
                            int *xShapeInfo,
                            T *extraParams) {
        return functions::indexreduce::IndexReduce<T>::execScalar(opNum, x,xShapeInfo,extraParams);
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
	static void execIndexReduce(int opNum,
                         T *x,
                         int *xShapeInfo,
                         T *extraParams,
                         T *result,
                         int *resultShapeInfoBuffer,
                         int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets) {
        functions::indexreduce::IndexReduce<T>::exec(opNum, x,xShapeInfo,extraParams,result,resultShapeInfoBuffer,dimension,dimensionLength, tadShapeInfo, tadOffsets);
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
	static void execBroadcast(int opNum,
                       T *x,
                       int *xShapeInfo,
                       T *y,
                       int *yShapeInfo,
                       T *result,
                       int *dimension, int dimensionLength, int *tadOnlyShapeInfo, int *tadOffsets, int *tadOnlyShapeInfoZ, int *tadOffsetsZ) {
		functions::broadcast::Broadcast<T>::exec(opNum, x, xShapeInfo, y, yShapeInfo, result, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
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
	static void execPairwiseTransform(int opNum,
                               T *dx,
                               int xStride,
                               T *y,
                               int yStride,
                               T *result,
                               int resultStride,
                               T *extraParams, Nd4jIndex n) {
		functions::pairwise_transforms::PairWiseTransform<T>::exec(opNum,
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
  * @param xStride
  * @param y
  * @param yStride
  * @param result
  * @param resultStride
  * @param extraParams
  * @param n
  */
	static void execPairwiseTransform(int opNum,
                               T *dx,
                               int *xShapeInfo,
                               T *y,
                               int *yShapeInfo,
                               T *result,
                               int *resultShapeInfo,
                               T *extraParams) {
		functions::pairwise_transforms::PairWiseTransform<T>::exec(opNum,
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
                               T *dx,
                               int *xShapeInfo,
                               T *y,
                               int *yShapeInfo,
                               T *result,
                               int *resultShapeInfo,
                               T *extraParams,
                               int *xIndexes,
                               int *yIndexes,
                               int *resultIndexes) {
		functions::pairwise_transforms::PairWiseTransform<T>::exec(opNum,
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
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
	static void execReduce(int opNum,
                    T *x,
                    int *xShapeInfo,
                    T *extraParams,
                    T *result,
                    int *resultShapeInfo,
                    int *dimension,
                    int dimensionLength, int *tadShapeInfo, int *tadOffsets) {
		functions::reduce::ReduceFunction<T>::exec(opNum, x,xShapeInfo,extraParams,result,resultShapeInfo,dimension,dimensionLength, tadShapeInfo, tadOffsets);
    }

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
	static T execReduceScalar(int opNum,
                       T *x,
                       int *xShapeInfo,
                       T *extraParams) {
        return functions::reduce::ReduceFunction<T>::execScalar(opNum, x,xShapeInfo,extraParams);
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
	static void execReduce3(int opNum,
                     T *x,
                     int *xShapeInfo,
                     T *extraParamsVals,
                     T *y,
                     int *yShapeInfo,
                     T *result, int *resultShapeInfo) {
        functions::reduce3::Reduce3<T>::exec(opNum, x,xShapeInfo,extraParamsVals,y,yShapeInfo, result, resultShapeInfo, nullptr, 1);
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
	static T execReduce3Scalar(int opNum,
                        T *x,
                        int *xShapeInfo,
                        T *extraParamsVals,
                        T *y,
                        int *yShapeInfo) {
        return functions::reduce3::Reduce3<T>::execScalar(opNum,x,xShapeInfo,extraParamsVals,y,yShapeInfo);
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
	static void execReduce3(int opNum,
                     T *x,
                     int *xShapeInfo,
                     T *extraParamsVals,
                     T *y,
                     int *yShapeInfo,
                     T *result,
                     int *resultShapeInfoBuffer,
                     int *dimension,
                     int dimensionLength) {
        functions::reduce3::Reduce3<T>::exec(opNum, x,xShapeInfo,extraParamsVals,y,yShapeInfo,result,resultShapeInfoBuffer,dimension,dimensionLength);
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
	static void execScalar(int opNum,
                    T *x,
                    int xStride,
                    T *result,
                    int resultStride,
                    T scalar,
                    T *extraParams,
                    Nd4jIndex n) {
		functions::scalar::ScalarTransform<T>::transform(opNum, x,xStride,result,resultStride,scalar,extraParams,n);
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
	static void execScalar(int opNum,
                    T *x,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T scalar,
                    T *extraParams) {
		functions::scalar::ScalarTransform<T>::transform(opNum,
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
 * @param xStride
 * @param result
 * @param resultStride
 * @param scalar
 * @param extraParams
 * @param n
 */
	static void execScalar(int opNum,
                    T *x,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T scalar,
                    T *extraParams,
                    int *xIndexes,
                    int *resultIndexes) {
		functions::scalar::ScalarTransform<T>::transform(opNum,
			x,
			xShapeInfo,
			result,
			resultShapeInfo,
			scalar,
			extraParams,
			xIndexes,
			resultIndexes);
    }

	static void execScalar(int opNum,
						   T *x,
						   int *xShapeInfo,
						   T *extraParams,
						   T *z,
						   int *zShapeInfo,
						   T *scalars,
						   int *dimension,
						   int dimensionLength,
						   int *tadShapeInfo,
						   int *tadOffsets,
						   int *tadShapeInfoZ,
						   int *tadOffsetsZ) {
		functions::scalar::ScalarTransform<T>::transform(opNum,
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
                          T *x,
                          int *xShapeInfo,
                          T *extraParams,
                          T *result,
                          int *resultShapeInfo,bool biasCorrected) {
        functions::summarystats::SummaryStatsReduce<T>::exec(opNum, biasCorrected, x,xShapeInfo,extraParams,result,resultShapeInfo, nullptr, 1);
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
	static T execSummaryStatsScalar(int opNum,
                             T *x,
                             int *xShapeInfo,
                             T *extraParams,bool biasCorrected) {
        return functions::summarystats::SummaryStatsReduce<T>::execScalar(opNum, biasCorrected, x,xShapeInfo,extraParams);
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
	static void execSummaryStats(int opNum,T *x,
                          int *xShapeInfo,
                          T *extraParams,
                          T *result,
                          int *resultShapeInfoBuffer,
                          int *dimension, int dimensionLength, bool biasCorrected) {
        functions::summarystats::SummaryStatsReduce<T>::exec(opNum, biasCorrected, x,
                 xShapeInfo,
                 extraParams,
                 result,
                 resultShapeInfoBuffer,
                 dimension,
                 dimensionLength);
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
	static void execTransform(int opNum,
                       T *dx,
                       int xStride,
                       T *result,
                       int resultStride,
                       T *extraParams,
                       Nd4jIndex n) {
		functions::transform::Transform<T>::exec(opNum, dx,
                        xStride,
                        result,
                        resultStride,
                        extraParams,
                        n);

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
	static void execTransform(int opNum,
                       T *dx,
                       int *xShapeInfo,
                       T *result,
                       int *resultShapeInfo,
                       T *extraParams, int *tadShapeInfo, int *tadOffsets) {
		functions::transform::Transform<T>::exec(opNum, dx,
                        xShapeInfo,
                        result,
                        resultShapeInfo,
                        extraParams, tadShapeInfo, tadOffsets);
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
	static void execTransform(int opNum,
                       T *dx,
                       int *xShapeInfo,
                       T *result,
                       int *resultShapeInfo,
                       T *extraParams,
                       int *xIndexes,
                       int *resultIndexes, int *tadShapeInfo, int *tadOffsets) {
		functions::transform::Transform<T>::exec(opNum, dx,
                        xShapeInfo,
                        result,
                        resultShapeInfo,
                        extraParams,
                        xIndexes,
                        resultIndexes, tadShapeInfo, tadOffsets);

    }


	static void execAggregate(int opNum, T **arguments, int numArguments,  int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
		functions::aggregate::AggregatedFunction<T>::exec(opNum, arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
	}
};


#endif //NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
