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
    static T execIndexReduceScalar(int opNum, T *x, Nd4jLong *xShapeInfo, T *extraParams);

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
                                Nd4jLong *xShapeInfo,
                                T *extraParams,
                                T *result,
                                Nd4jLong *resultShapeInfoBuffer,
                                int *dimension,
                                int dimensionLength,
                                Nd4jLong *tadShapeInfo,
                                Nd4jLong *tadOffsets);

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
                              Nd4jLong *xShapeInfo,
                              T *y,
                              Nd4jLong *yShapeInfo,
                              T *result, 
                              Nd4jLong *resultShapeInfo,
                              int *dimension,
                              int dimensionLength,
                              Nd4jLong *tadOnlyShapeInfo,
                              Nd4jLong *tadOffsets,
                              Nd4jLong *tadOnlyShapeInfoZ,
                              Nd4jLong *tadOffsetsZ);

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
                                      Nd4jLong xStride,
                                      T *y,
                                      Nd4jLong yStride,
                                      T *result,
                                      Nd4jLong resultStride,
                                      T *extraParams, Nd4jLong n);

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
                                      Nd4jLong *xShapeInfo,
                                      T *y,
                                      Nd4jLong *yShapeInfo,
                                      T *result,
                                      Nd4jLong *resultShapeInfo,
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
                                      Nd4jLong *xShapeInfo,
                                      T *y,
                                      Nd4jLong *yShapeInfo,
                                      T *result,
                                      Nd4jLong *resultShapeInfo,
                                      T *extraParams,
                                      Nd4jLong *xIndexes,
                                      Nd4jLong *yIndexes,
                                      Nd4jLong *resultIndexes);

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
                           Nd4jLong *xShapeInfo,
                           T *extraParams,
                           T *result,
                           Nd4jLong *resultShapeInfo,
                           int *dimension,
                           int dimensionLength,
                           Nd4jLong *tadShapeInfo,
                           Nd4jLong *tadOffsets);

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
                              Nd4jLong *xShapeInfo,
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
                            Nd4jLong *xShapeInfo,
                            T *extraParamsVals,
                            T *y,
                            Nd4jLong *yShapeInfo,
                            T *result, 
                            Nd4jLong *resultShapeInfo);    


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
                               Nd4jLong *xShapeInfo,
                               T *extraParamsVals,
                               T *y,
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
    static void execReduce3(int opNum,
                            T *x,
                            Nd4jLong *xShapeInfo,
                            T *extraParamsVals,
                            T *y,
                            Nd4jLong *yShapeInfo,
                            T *result,
                            Nd4jLong *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength);

    static void execReduce3All(int opNum,
                            T *x,
                            Nd4jLong *xShapeInfo,
                            T *extraParamsVals,
                            T *y,
                            Nd4jLong *yShapeInfo,
                            T *result,
                            Nd4jLong *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength,
                            Nd4jLong *xTadShapeInfo,
                            Nd4jLong *xOffsets,
                            Nd4jLong *yTadShapeInfo,
                            Nd4jLong *yOffsets);

    static void execReduce3TAD(int opNum,
                            T *x,
                            Nd4jLong *xShapeInfo,
                            T *extraParamsVals,
                            T *y,
                            Nd4jLong *yShapeInfo,
                            T *result,
                            Nd4jLong *resultShapeInfoBuffer,
                            int *dimension,
                            int dimensionLength, 
                            Nd4jLong *tadShapeInfo, 
                            Nd4jLong *tadOffsets);

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
                           Nd4jLong xStride,
                           T *result,
                           Nd4jLong resultStride,
                           T scalar,
                           T *extraParams,
                           Nd4jLong n);

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
                           Nd4jLong *xShapeInfo,
                           T *result,
                           Nd4jLong *resultShapeInfo,
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
                           Nd4jLong *xShapeInfo,
                           T *result,
                           Nd4jLong *resultShapeInfo,
                           T scalar,
                           T *extraParams,
                           Nd4jLong *xIndexes,
                           Nd4jLong *resultIndexes);

    static void execScalar(int opNum,
                           T *x,
                           Nd4jLong *xShapeInfo,
                           T *extraParams,
                           T *z,
                           Nd4jLong *zShapeInfo,
                           T *scalars,
                           int *dimension,
                           int dimensionLength,
                           Nd4jLong *tadShapeInfo,
                           Nd4jLong *tadOffsets,
                           Nd4jLong *tadShapeInfoZ,
                           Nd4jLong *tadOffsetsZ);

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
                                 Nd4jLong *xShapeInfo,
                                 T *extraParams,
                                 T *result,
                                 Nd4jLong *resultShapeInfo,
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
    static T execSummaryStatsScalar(int opNum,
                                    T *x,
                                    Nd4jLong *xShapeInfo,
                                    T *extraParams,
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
    static void execSummaryStats(int opNum,T *x,
                                 Nd4jLong *xShapeInfo,
                                 T *extraParams,
                                 T *result,
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
    static void execTransform(int opNum,
                              T *dx,
                              Nd4jLong xStride,
                              T *result,
                              Nd4jLong resultStride,
                              T *extraParams,
                              Nd4jLong n);
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
                              Nd4jLong *xShapeInfo,
                              T *result,
                              Nd4jLong *resultShapeInfo,
                              T *extraParams,
                              Nd4jLong *tadShapeInfo,
                              Nd4jLong *tadOffsets);
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
                              Nd4jLong *xShapeInfo,
                              T *result,
                              Nd4jLong *resultShapeInfo,
                              T *extraParams,
                              Nd4jLong *xIndexes,
                              Nd4jLong *resultIndexes,
                              Nd4jLong *tadShapeInfo,
                              Nd4jLong *tadOffsets);

    static void execAggregate(int opNum,
                              T **arguments,
                              int numArguments,
                              Nd4jLong **shapeArguments,
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
                           Nd4jLong *zShapeBuffer, 
                           T *extraArguments);

    static void execRandom(int opNum,
                           Nd4jPointer state,
                           T *x,
                           Nd4jLong *xShapeBuffer,
                           T *z,
                           Nd4jLong *zShapeBuffer, 
                           T *extraArguments);

    static void execRandom(int opNum,
                           Nd4jPointer state,
                           T *x,
                           Nd4jLong *xShapeBuffer,
                           T *y, 
                           Nd4jLong *yShapeBuffer,
                           T *z, 
                           Nd4jLong *zShapeBuffer,
                           T *extraArguments);

    inline static void execSort(T *x, Nd4jLong *xShapeInfo, bool descending) {
        nd4j::SpecialMethods<T>::sortGeneric(x, xShapeInfo, descending);
    }

    static void execSort(T *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
        nd4j::SpecialMethods<T>::sortTadGeneric(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
    }

    inline static void execSortCooIndices(Nd4jLong *indices, T *values, Nd4jLong length, int rank) {
        nd4j::sparse::SparseUtils<T>::sortCooIndicesGeneric(indices, values, length, rank);
    }


    inline static Nd4jLong encodeBitmap(T *dx, Nd4jLong N, int *dz, float threshold) {
        return nd4j::SpecialMethods<T>::encodeBitmapGeneric(dx, N, dz, threshold);
    }

    inline static void decodeBitmap(void *dx, Nd4jLong N, T *dz) {
        nd4j::SpecialMethods<T>::decodeBitmapGeneric(dx, N, dz);
    }

};


#endif //NATIVEOPERATIONS_NATIVEOPEXCUTIONER_H
