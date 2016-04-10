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
    functions::broadcast::BroadcastOpFactory<T> *broadcastOpFactory = new functions::broadcast::BroadcastOpFactory<T>();
    functions::indexreduce::IndexReduceOpFactory<T> *indexReduceOpFactory = new functions::indexreduce::IndexReduceOpFactory<T>();
    functions::pairwise_transforms::PairWiseTransformOpFactory<T> *pairWiseTransformOpFactory = new functions::pairwise_transforms::PairWiseTransformOpFactory<T>();
    functions::reduce::ReduceOpFactory<T> *reduceOpFactory = new functions::reduce::ReduceOpFactory<T>();
    functions::reduce3::Reduce3OpFactory<T> *reduce3OpFactory = new functions::reduce3::Reduce3OpFactory<T>();
    functions::scalar::ScalarOpFactory<T> *scalarOpFactory = new functions::scalar::ScalarOpFactory<T>();
    functions::summarystats::SummaryStatsReduceOpFactory<T> *summaryStatsReduceOpFactory = new functions::summarystats::SummaryStatsReduceOpFactory<T>();
    functions::transform::TransformOpFactory<T> *transformOpFactory = new functions::transform::TransformOpFactory<T>();

public:
    ~NativeOpExcutioner() {
        delete broadcastOpFactory;
        delete indexReduceOpFactory;
        delete pairWiseTransformOpFactory;
        delete reduceOpFactory;
        delete reduce3OpFactory;
        delete scalarOpFactory;
        delete summaryStatsReduceOpFactory;
        delete transformOpFactory;
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
                         int *dimension, int dimensionLength) {
        functions::indexreduce::IndexReduce<T> *op = indexReduceOpFactory->getOp(opNum);
        op->exec(x,xShapeInfo,extraParams,result,resultShapeInfoBuffer,dimension,dimensionLength);
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
                       int *dimension, int dimensionLength) {

        functions::broadcast::Broadcast<T> *broadcast = broadcastOpFactory->getOp(opNum);
        broadcast->exec(x, xShapeInfo, y, yShapeInfo, result, dimension, dimensionLength);
        delete broadcast;
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
        functions::pairwise_transforms::PairWiseTransform<T> *op = pairWiseTransformOpFactory->getOp(opNum);
        op->exec(
                dx,
                xStride,
                y,
                yStride,
                result,
                resultStride,
                extraParams,
                n);
        delete op;
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
        functions::pairwise_transforms::PairWiseTransform<T> *op = pairWiseTransformOpFactory->getOp(opNum);
        op->exec(dx,
                 xShapeInfo,
                 y,
                 yShapeInfo,
                 result,
                 resultShapeInfo,
                 extraParams);
        delete op;
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
        functions::pairwise_transforms::PairWiseTransform<T> *op = pairWiseTransformOpFactory->getOp(opNum);
        op->exec(dx,
                 xShapeInfo,
                 y,
                 yShapeInfo,
                 result,
                 resultShapeInfo,
                 extraParams,
                 xIndexes,
                 yIndexes,
                 resultIndexes);
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
    void execReduce(int opNum,
                    T *x,
                    int *xShapeInfo,
                    T *extraParams,
                    T *result,
                    int *resultShapeInfo,
                    int *dimension,
                    int dimensionLength) {
        functions::reduce::ReduceFunction<T> *reduceFunction = reduceOpFactory->create(opNum);
        reduceFunction->exec(x,xShapeInfo,extraParams,result,resultShapeInfo,dimension,dimensionLength);
        delete reduceFunction;
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
        functions::reduce::ReduceFunction<T> *reduceFunction = reduceOpFactory->create(opNum);
        T ret = reduceFunction->execScalar(x,xShapeInfo,extraParams);
        delete reduceFunction;
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
        functions::scalar::ScalarTransform<T> *scalarTransform = scalarOpFactory->getOp(opNum);
        scalarTransform->transform(x,xStride,result,resultStride,scalar,extraParams,n);
        delete scalarTransform;


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
        functions::scalar::ScalarTransform<T> *scalarTransform = scalarOpFactory->getOp(opNum);
        scalarTransform->transform(x,
                                   xShapeInfo,
                                   result,
                                   resultShapeInfo,
                                   scalar,
                                   extraParams);
        delete scalarTransform;


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
        functions::scalar::ScalarTransform<T> *scalarTransform = scalarOpFactory->getOp(opNum);
        scalarTransform->transform(x,
                                   xShapeInfo,
                                   result,
                                   resultShapeInfo,
                                   scalar,
                                   extraParams,
                                   xIndexes,
                                   resultIndexes);
        delete scalarTransform;


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
    void execSummaryStats(OpType opNum,
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
    T execSummaryStatsScalar(OpType opNum,
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
        functions::transform::Transform<T> *transform = transformOpFactory->getOp(opNum);
        transform->exec(dx,
                        xStride,
                        result,
                        resultStride,
                        extraParams,
                        n);
        delete transform;

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
        functions::transform::Transform<T> *transform = transformOpFactory->getOp(opNum);
        transform->exec(dx,
                        xShapeInfo,
                        result,
                        resultShapeInfo,
                        extraParams);
        delete transform;

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
        functions::transform::Transform<T> *transform = transformOpFactory->getOp(opNum);
        transform->exec(dx,
                        xShapeInfo,
                        result,
                        resultShapeInfo,
                        extraParams,
                        xIndexes,
                        resultIndexes);
        delete transform;

    }


};


#endif //NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
