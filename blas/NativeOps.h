//
// Created by agibsonccc on 2/21/16.
//

#ifndef NATIVEOPERATIONS_NATIVEOPS_H
#define NATIVEOPERATIONS_NATIVEOPS_H


class NativeOps {


public:
    /**
       *
       * @param opNum
       * @param x
       * @param xShapeInfo
       * @param extraParams
       */
    double   execIndexReduceScalarDouble(long *extraPointers,int opNum,
                                   long x,
                                   long xShapeInfo,
                                   long extraParams);

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
    void   execIndexReduceDouble(long *extraPointers,int opNum,
                           long x,
                           long xShapeInfo,
                           long extraParams,
                           long result,
                           long resultShapeInfoBuffer,
                           long dimension, int dimensionLength);
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
    void   execBroadcastDouble(long *extraPointers,int opNum,
                         long x,
                         long xShapeInfo,
                         long y,
                         long yShapeInfo,
                         long result,
                         long resultShapeInfo,
                         long dimension, int dimensionLength);



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
    void   execPairwiseTransformDouble(long *extraPointers,int opNum,
                                 long dx,
                                 int xStride,
                                 long y,
                                 int yStride,
                                 long result,
                                 int resultStride,
                                 long extraParams, int n);

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
    void execPairwiseTransformDouble(long *extraPointers,int opNum,
                               long dx,
                               long xShapeInfo,
                               long y,
                               long yShapeInfo,
                               long result,
                               long resultShapeInfo,
                               long extraParams,
                               int n,
                               long xIndexes,
                               long yIndexes,
                               long resultIndexes);

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
    void execPairwiseTransformDouble(long *extraPointers,int opNum,
                               long dx,
                               long  xShapeInfo,
                               long y,
                               long  yShapeInfo,
                               long result,
                               long  resultShapeInfo,
                               long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceDouble(long *extraPointers,int opNum,
                      long x,
                      long xShapeInfo,
                      long extraParams,
                      long result,
                      long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceDouble(long *extraPointers,int opNum,
                      long x,
                      long xShapeInfo,
                      long extraParams,
                      long result,
                      long resultShapeInfo,
                      long dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    double execReduceScalarDouble(long *extraPointers,int opNum,
                            long x,
                            long xShapeInfo,
                            long extraParams);

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
    void   execReduce3Double(long *extraPointers,int opNum,
                       long x,
                       long xShapeInfo,
                       long extraParamsVals,
                       long y,
                       long yShapeInfo,
                       long result,
                       long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    double   execReduce3ScalarDouble(long *extraPointers,int opNum,
                               long x,
                               long xShapeInfo,
                               long extraParamsVals,
                               long y,
                               long yShapeInfo);
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
    void   execReduce3Double(long *extraPointers,int opNum,
                       long x,
                       long xShapeInfo,
                       long extraParamsVals,
                       long y,
                       long yShapeInfo,
                       long result,
                       long resultShapeInfoBuffer,
                       long dimension,
                       int dimensionLength);
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
    void   execScalarDouble(long *extraPointers,int opNum,
                      long x,
                      int xStride,
                      long result,
                      int resultStride,
                      double scalar,
                      long extraParams,
                      int n);

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
    void execScalarDouble(long *extraPointers,int opNum,
                    long x,
                    long xShapeInfo,
                    long result,
                    long resultShapeInfo,
                    double scalar,
                    long extraParams,
                    int n);

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
    void execScalarDouble(long *extraPointers,int opNum,
                    long x,
                    long xShapeInfo,
                    long result,
                    long resultShapeInfo,
                    double scalar,
                    long extraParams,
                    int n,
                    long xIndexes,
                    long resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    double   execSummaryStatsScalarDouble(long *extraPointers,int opNum,long x,
                                    long xShapeInfo,
                                    long extraParams);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execSummaryStatsDouble(long *extraPointers,int opNum,
                            long x,
                            long xShapeInfo,
                            long extraParams,
                            long result,
                            long resultShapeInfo);
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
    void   execSummaryStatsDouble(long *extraPointers,int opNum,long x,
                            long xShapeInfo,
                            long extraParams,
                            long result,
                            long resultShapeInfoBuffer,
                            long dimension, int dimensionLength);
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
    void   execTransformDouble(long *extraPointers,int opNum,
                         long dx,
                         int xStride,
                         long result,
                         int resultStride,
                         long extraParams, int n);

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
    void   execTransformDouble(long *extraPointers,int opNum,
                         long dx,
                         long xShapeInfo,
                         long result,
                         long resultShapeInfo,
                         long extraParams);

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
    void   execTransformDouble(long *extraPointers,int opNum,
                         long dx,
                         long xShapeInfo,
                         long result,
                         long resultShapeInfo,
                         long extraParams,
                         int n,
                         long xIndexes,
                         long resultIndexes);

    /**
    *
    * @param opNum
    * @param x
    * @param xShapeInfo
    * @param extraParams
    */
    double   execIndexReduceScalarFloat(long *extraPointers,int opNum,
                                        long x,
                                        long xShapeInfo,
                                        long extraParams);

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
    void   execIndexReduceFloat(long *extraPointers,int opNum,
                                long x,
                                long xShapeInfo,
                                long extraParams,
                                long result,
                                long resultShapeInfoBuffer,
                                long dimension, int dimensionLength);
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
    void   execBroadcastFloat(long *extraPointers,int opNum,
                              long x,
                              long xShapeInfo,
                              long y,
                              long yShapeInfo,
                              long result,
                              long resultShapeInfo,
                              long dimension, int dimensionLength);



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
    void   execPairwiseTransformFloat(long *extraPointers,int opNum,
                                      long dx,
                                      int xStride,
                                      long y,
                                      int yStride,
                                      long result,
                                      int resultStride,
                                      long extraParams, int n);

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
    void execPairwiseTransformFloat(long *extraPointers,int opNum,
                                    long dx,
                                    long xShapeInfo,
                                    long y,
                                    long yShapeInfo,
                                    long result,
                                    long resultShapeInfo,
                                    long extraParams,
                                    int n,
                                    long xIndexes,
                                    long yIndexes,
                                    long resultIndexes);

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
    void execPairwiseTransformFloat(long *extraPointers,int opNum,
                                    long dx,
                                    long  xShapeInfo,
                                    long y,
                                    long  yShapeInfo,
                                    long result,
                                    long  resultShapeInfo,
                                    long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(long *extraPointers,int opNum,
                           long x,
                           long xShapeInfo,
                           long extraParams,
                           long result,
                           long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(long *extraPointers,int opNum,
                           long x,
                           long xShapeInfo,
                           long extraParams,
                           long result,
                           long resultShapeInfo,
                           long dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    double execReduceScalarFloat(long *extraPointers,int opNum,
                                 long x,
                                 long xShapeInfo,
                                 long extraParams);

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
    void   execReduce3Float(long *extraPointers,int opNum,
                            long x,
                            long xShapeInfo,
                            long extraParamsVals,
                            long y,
                            long yShapeInfo,
                            long result,
                            long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    float   execReduce3ScalarFloat(long *extraPointers,int opNum,
                                    long x,
                                    long xShapeInfo,
                                    long extraParamsVals,
                                    long y,
                                    long yShapeInfo);
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
    void   execReduce3Float(long *extraPointers,int opNum,
                            long x,
                            long xShapeInfo,
                            long extraParamsVals,
                            long y,
                            long yShapeInfo,
                            long result,
                            long resultShapeInfoBuffer,
                            long dimension,
                            int dimensionLength);
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
    void   execScalarFloat(long *extraPointers,int opNum,
                           long x,
                           int xStride,
                           long result,
                           int resultStride,
                           double scalar,
                           long extraParams,
                           int n);

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
    void execScalarFloat(long *extraPointers,int opNum,
                         long x,
                         long xShapeInfo,
                         long result,
                         long resultShapeInfo,
                         double scalar,
                         long extraParams,
                         int n);

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
    void execScalarFloat(long *extraPointers,int opNum,
                         long x,
                         long xShapeInfo,
                         long result,
                         long resultShapeInfo,
                         double scalar,
                         long extraParams,
                         int n,
                         long xIndexes,
                         long resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    float   execSummaryStatsScalarFloat(long *extraPointers,int opNum,long x,
                                         long xShapeInfo,
                                         long extraParams);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execSummaryStatsFloat(long *extraPointers,int opNum,
                                 long x,
                                 long xShapeInfo,
                                 long extraParams,
                                 long result,
                                 long resultShapeInfo);
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
    void   execSummaryStatsFloat(long *extraPointers,int opNum,long x,
                                 long xShapeInfo,
                                 long extraParams,
                                 long result,
                                 long resultShapeInfoBuffer,
                                 long dimension, int dimensionLength);
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
    void   execTransformFloat(long *extraPointers,int opNum,
                              long dx,
                              int xStride,
                              long result,
                              int resultStride,
                              long extraParams, int n);

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
    void   execTransformFloat(long *extraPointers,int opNum,
                              long dx,
                              long xShapeInfo,
                              long result,
                              long resultShapeInfo,
                              long extraParams, int n);

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
    void   execTransformFloat(long *extraPointers,int opNum,
                              long dx,
                              long xShapeInfo,
                              long result,
                              long resultShapeInfo,
                              long extraParams,
                              int n,
                              long xIndexes,
                              long resultIndexes);
};


#endif //NATIVEOPERATIONS_NATIVEOPS_H
