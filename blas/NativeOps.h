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
    double   execIndexReduceScalarDouble(int opNum,
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
    void   execIndexReduceDouble(int opNum,
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
    void   execBroadcastDouble(int opNum,
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
    void   execPairwiseTransformDouble(int opNum,
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
    void execPairwiseTransformDouble(int opNum,
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
    void execPairwiseTransformDouble(int opNum,
                               long dx,
                               long  xShapeInfo,
                               long y,
                               long  yShapeInfo,
                               long result,
                               long  resultShapeInfo,
                               long extraParams, int n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceDouble(int opNum,
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
    void   execReduceDouble(int opNum,
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
    double execReduceScalarDouble(int opNum,
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
    void   execReduce3Double(int opNum,
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
    double   execReduce3ScalarDouble(int opNum,
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
    void   execReduce3Double(int opNum,
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
    void   execScalarDouble(int opNum,
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
    void execScalarDouble(int opNum,
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
    void execScalarDouble(int opNum,
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
    double   execSummaryStatsScalarDouble(int opNum,long x,
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
    void   execSummaryStatsDouble(int opNum,
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
    void   execSummaryStatsDouble(int opNum,long x,
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
    void   execTransformDouble(int opNum,
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
    void   execTransformDouble(int opNum,
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
    void   execTransformDouble(int opNum,
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
    double   execIndexReduceScalarFloat(int opNum,
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
    void   execIndexReduceFloat(int opNum,
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
    void   execBroadcastFloat(int opNum,
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
    void   execPairwiseTransformFloat(int opNum,
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
    void execPairwiseTransformFloat(int opNum,
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
    void execPairwiseTransformFloat(int opNum,
                                    long dx,
                                    long  xShapeInfo,
                                    long y,
                                    long  yShapeInfo,
                                    long result,
                                    long  resultShapeInfo,
                                    long extraParams, int n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(int opNum,
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
    void   execReduceFloat(int opNum,
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
    double execReduceScalarFloat(int opNum,
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
    void   execReduce3Float(int opNum,
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
    double   execReduce3ScalarFloat(int opNum,
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
    void   execReduce3Float(int opNum,
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
    void   execScalarFloat(int opNum,
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
    void execScalarFloat(int opNum,
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
    void execScalarFloat(int opNum,
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
    double   execSummaryStatsScalarFloat(int opNum,long x,
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
    void   execSummaryStatsFloat(int opNum,
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
    void   execSummaryStatsFloat(int opNum,long x,
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
    void   execTransformFloat(int opNum,
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
    void   execTransformFloat(int opNum,
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
    void   execTransformFloat(int opNum,
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
