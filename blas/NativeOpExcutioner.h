//
// Created by agibsonccc on 1/28/16.
//

#ifndef NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
#define NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H

#include <loops/broadcasting.h>
#include <loops/indexreduce.h>
#include <loops/pairwise_transform.h>
#include <loops/reduce.h>
#include <loops/reduce3.h>
#include <loops/summarystatsreduce.h>
#include <loops/transform.h>
#include <loops/scalar.h>
#include <loops/aggregates.h>
#include <loops/random.h>
#include <pointercast.h>
#include <ops/specials.h>
#include <ops/specials_sparse.h>
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
    static T execIndexReduceScalar(int opNum, T *x, int *xShapeInfo, T *extraParams);

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
                                int *dimension,
                                int dimensionLength,
                                int *tadShapeInfo,
                                Nd4jIndex *tadOffsets);

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
                              T *result, int *resultShapeInfo,
                              int *dimension,
                              int dimensionLength,
                              int *tadOnlyShapeInfo,
                              Nd4jIndex *tadOffsets,
                              int *tadOnlyShapeInfoZ,
                              Nd4jIndex *tadOffsetsZ);

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
                                      T *extraParams, Nd4jIndex n);

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
                                      T *extraParams);

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
                                      int *resultIndexes);

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
                           int dimensionLength,
                           int *tadShapeInfo,
                           Nd4jIndex *tadOffsets);

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
                              T *extraParams);

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
                            T *result, int *resultShapeInfo);    


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
                               int *yShapeInfo);

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
                            int dimensionLength);

    static void execReduce3All(int opNum,
                            T *x,
                            int *xShapeInfo,
                            T *extraParamsVals,
                            T *y,
                            int *yShapeInfo,
                            T *result,
                            int *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength,
                            int *xTadShapeInfo,
                            Nd4jIndex *xOffsets,
                            int *yTadShapeInfo,
                            Nd4jIndex *yOffsets);

    static void execReduce3TAD(int opNum,
                            T *x,
                            int *xShapeInfo,
                            T *extraParamsVals,
                            T *y,
                            int *yShapeInfo,
                            T *result,
                            int *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets);

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
                           Nd4jIndex n);

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
                           T *extraParams);

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
                           int *resultIndexes);

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
                           Nd4jIndex *tadOffsets,
                           int *tadShapeInfoZ,
                           Nd4jIndex *tadOffsetsZ);

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
                                 int *resultShapeInfo,bool biasCorrected);

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
                                    T *extraParams,bool biasCorrected);

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
                              T *dx,
                              int xStride,
                              T *result,
                              int resultStride,
                              T *extraParams,
                              Nd4jIndex n);
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
                              int *tadShapeInfo,
                              Nd4jIndex *tadOffsets);
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
                              int *resultIndexes,
                              int *tadShapeInfo,
                              Nd4jIndex *tadOffsets);

    static void execAggregate(int opNum,
                              T **arguments,
                              int numArguments,
                              int **shapeArguments,
                              int numShapeArguments,
                              int *indexArguments,
                              int numIndexArguments,
                              int **intArrays,
                              int numIntArrays,
                              T *realArguments,
                              int numRealArguments);

    static void execRandom(int opNum,
                           Nd4jPointer state,
                           T *z,
                           int *zShapeBuffer, T *extraArguments);

    static void execRandom(int opNum,
                           Nd4jPointer state,
                           T *x,
                           int *xShapeBuffer,
                           T *z,
                           int *zShapeBuffer, T *extraArguments);

    static void execRandom(int opNum,
                           Nd4jPointer state,
                           T *x,
                           int *xShapeBuffer,
                           T *y, int *yShapeBuffer,
                           T *z, int *zShapeBuffer,
                           T *extraArguments);

    inline static void execSort(T *x, int *xShapeInfo, bool descending) {
        nd4j::SpecialMethods<T>::sortGeneric(x, xShapeInfo, descending);
    }

    static void execSort(T *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending) {
        nd4j::SpecialMethods<T>::sortTadGeneric(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
    }

    inline static void execSortCooIndices(int *indices, T *values, Nd4jIndex length, int rank) {
        nd4j::sparse::SparseUtils<T>::sortCooIndicesGeneric(indices, values, length, rank);
    }


    inline static Nd4jIndex encodeBitmap(T *dx, Nd4jIndex N, int *dz, float threshold) {
        return nd4j::SpecialMethods<T>::encodeBitmapGeneric(dx, N, dz, threshold);
    }

    inline static void decodeBitmap(void *dx, Nd4jIndex N, T *dz) {
        nd4j::SpecialMethods<T>::decodeBitmapGeneric(dx, N, dz);
    }

};


#endif //NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
