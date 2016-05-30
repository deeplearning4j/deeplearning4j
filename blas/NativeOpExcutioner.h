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
#include <pointercast.h>
/**
 * Native op executioner:
 *
 */
template <typename T>
class NativeOpExcutioner {
private:
    functions::indexreduce::IndexReduceOpFactory<T> *indexReduceOpFactory = new functions::indexreduce::IndexReduceOpFactory<T>();
    functions::reduce3::Reduce3OpFactory<T> *reduce3OpFactory = new functions::reduce3::Reduce3OpFactory<T>();
    functions::summarystats::SummaryStatsReduceOpFactory<T> *summaryStatsReduceOpFactory = new functions::summarystats::SummaryStatsReduceOpFactory<T>();

public:
    ~NativeOpExcutioner() {
        delete indexReduceOpFactory;
        delete reduce3OpFactory;
        delete summaryStatsReduceOpFactory;
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
    T execIndexReduceScalar(int opNum,
                            T *x,
                            int *xShapeInfo,
                            T *extraParams) {
        functions::indexreduce::IndexReduce<T> *op = indexReduceOpFactory->getOp(opNum);
        T ret = op->execScalar(x,xShapeInfo,extraParams);
        delete op;
        return ret;

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
    void execIndexReduce(int opNum,
                         T *x,
                         int *xShapeInfo,
                         T *extraParams,
                         T *result,
                         int *resultShapeInfoBuffer,
                         int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets) {
        functions::indexreduce::IndexReduce<T> *op = indexReduceOpFactory->getOp(opNum);
        op->exec(x,xShapeInfo,extraParams,result,resultShapeInfoBuffer,dimension,dimensionLength, tadShapeInfo, tadOffsets);
        delete op;
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
    void execBroadcast(int opNum,
                       T *x,
                       int *xShapeInfo,
                       T *y,
                       int *yShapeInfo,
                       T *result,
                       int *dimension, int dimensionLength, int *tadOnlyShapeInfo, int *tadOffsets) {
		functions::broadcast::Broadcast<T>::exec(opNum, x, xShapeInfo, y, yShapeInfo, result, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets);
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
    void execPairwiseTransform(int opNum,
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
    void execPairwiseTransform(int opNum,
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
    void execPairwiseTransform(int opNum,
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
    void execReduce(int opNum,
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
    T execReduceScalar(int opNum,
                       T *x,
                       int *xShapeInfo,
                       T *extraParams) {
        T ret = functions::reduce::ReduceFunction<T>::execScalar(opNum, x,xShapeInfo,extraParams);
        return ret;
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
    void execReduce3(int opNum,
                     T *x,
                     int *xShapeInfo,
                     T *extraParamsVals,
                     T *y,
                     int *yShapeInfo,
                     T *result, int *resultShapeInfo) {
        functions::reduce3::Reduce3<T> *reduce3 = reduce3OpFactory->getOp(opNum);
        reduce3->exec(x,xShapeInfo,extraParamsVals,y,yShapeInfo,result,resultShapeInfo);
        delete reduce3;

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
    T execReduce3Scalar(int opNum,
                        T *x,
                        int *xShapeInfo,
                        T *extraParamsVals,
                        T *y,
                        int *yShapeInfo) {
        functions::reduce3::Reduce3<T> *reduce3 = reduce3OpFactory->getOp(opNum);
        T ret = reduce3->execScalar(x,xShapeInfo,extraParamsVals,y,yShapeInfo);
        delete reduce3;
        return ret;

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
    void execReduce3(int opNum,
                     T *x,
                     int *xShapeInfo,
                     T *extraParamsVals,
                     T *y,
                     int *yShapeInfo,
                     T *result,
                     int *resultShapeInfoBuffer,
                     int *dimension,
                     int dimensionLength) {
        functions::reduce3::Reduce3<T> *reduce3 = reduce3OpFactory->getOp(opNum);
        reduce3->exec(x,xShapeInfo,extraParamsVals,y,yShapeInfo,result,resultShapeInfoBuffer,dimension,dimensionLength);
        delete reduce3;

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
    void execScalar(int opNum,
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
    void execScalar(int opNum,
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
    void execScalar(int opNum,
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

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void execSummaryStats(int opNum,
                          T *x,
                          int *xShapeInfo,
                          T *extraParams,
                          T *result,
                          int *resultShapeInfo,bool biasCorrected) {
        functions::summarystats::SummaryStatsReduce<T> *op = summaryStatsReduceOpFactory->getOp(opNum,biasCorrected);
        op->exec(x,xShapeInfo,extraParams,result,resultShapeInfo);
        delete op;
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
    T execSummaryStatsScalar(int opNum,
                             T *x,
                             int *xShapeInfo,
                             T *extraParams,bool biasCorrected) {
        functions::summarystats::SummaryStatsReduce<T> *op = summaryStatsReduceOpFactory->getOp(opNum,biasCorrected);
        T ret = op->execScalar(x,xShapeInfo,extraParams);
        delete op;
        return ret;
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
    void execSummaryStats(int opNum,T *x,
                          int *xShapeInfo,
                          T *extraParams,
                          T *result,
                          int *resultShapeInfoBuffer,
                          int *dimension, int dimensionLength, bool biasCorrected) {
        functions::summarystats::SummaryStatsReduce<T> *op = summaryStatsReduceOpFactory->getOp(opNum,biasCorrected);
        op->exec(x,
                 xShapeInfo,
                 extraParams,
                 result,
                 resultShapeInfoBuffer,
                 dimension,
                 dimensionLength);
        delete op;

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
    void execTransform(int opNum,
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
    void execTransform(int opNum,
                       T *dx,
                       int *xShapeInfo,
                       T *result,
                       int *resultShapeInfo,
                       T *extraParams) {
		functions::transform::Transform<T>::exec(opNum, dx,
                        xShapeInfo,
                        result,
                        resultShapeInfo,
                        extraParams);
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
    void execTransform(int opNum,
                       T *dx,
                       int *xShapeInfo,
                       T *result,
                       int *resultShapeInfo,
                       T *extraParams,
                       Nd4jIndex *xIndexes,
                       Nd4jIndex *resultIndexes) {
		functions::transform::Transform<T>::exec(opNum, dx,
                        xShapeInfo,
                        result,
                        resultShapeInfo,
                        extraParams,
                        xIndexes,
                        resultIndexes);

    }


};


#endif //NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
