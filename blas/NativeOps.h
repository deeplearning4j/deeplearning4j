//
// Created by agibsonccc on 2/21/16.
//

#ifndef NATIVEOPERATIONS_NATIVEOPS_H
#define NATIVEOPERATIONS_NATIVEOPS_H

#ifndef thread_local
# if __STDC_VERSION__ >= 201112 && !defined __STDC_NO_THREADS__
#  define thread_local _Thread_local
# elif defined _WIN32 && ( \
       defined _MSC_VER || \
       defined __ICL || \
       defined __DMC__ || \
       defined __BORLANDC__ )
#  define thread_local __declspec(thread)
/* note that ICC (linux) and Clang are covered by __GNUC__ */
# elif defined __GNUC__ || \
       defined __SUNPRO_C || \
       defined __xlC__
#  define thread_local __thread
# else
#  error "Cannot define thread_local"
# endif
#endif



class NativeOps {


public:
    /**
       *
       * @param opNum
       * @param x
       * @param xShapeInfo
       * @param extraParams
       */
    double   execIndexReduceScalarDouble(long long *extraPointers,int opNum,
                                         long long x,
                                         long long xShapeInfo,
                                         long long extraParams);

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
    void   execIndexReduceDouble(long long *extraPointers,int opNum,
                                 long long x,
                                 long long xShapeInfo,
                                 long long extraParams,
                                 long long result,
                                 long long resultShapeInfoBuffer,
                                 long long dimension, int dimensionLength);
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
    void   execBroadcastDouble(long long *extraPointers,int opNum,
                               long long x,
                               long long xShapeInfo,
                               long long y,
                               long long yShapeInfo,
                               long long result,
                               long long resultShapeInfo,
                               long long dimension, int dimensionLength);



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
    void   execPairwiseTransformDouble(long long *extraPointers,int opNum,
                                       long long dx,
                                       int xStride,
                                       long long y,
                                       int yStride,
                                       long long result,
                                       int resultStride,
                                       long long extraParams, int n);

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
    void execPairwiseTransformDouble(long long *extraPointers,
                                     int opNum,
                                     long long dx,
                                     long long xShapeInfo,
                                     long long y,
                                     long long yShapeInfo,
                                     long long result,
                                     long long resultShapeInfo,
                                     long long extraParams,
                                     long long xIndexes,
                                     long long yIndexes,
                                     long long resultIndexes);

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
    void execPairwiseTransformDouble(
            long long *extraPointers,
            int opNum,
            long long dx,
            long long  xShapeInfo,
            long long y,
            long long  yShapeInfo,
            long long result,
            long long  resultShapeInfo,
            long long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceDouble(long long *extraPointers,int opNum,
                            long long x,
                            long long xShapeInfo,
                            long long extraParams,
                            long long result,
                            long long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceDouble(long long *extraPointers,int opNum,
                            long long x,
                            long long xShapeInfo,
                            long long extraParams,
                            long long result,
                            long long resultShapeInfo,
                            long long dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    double execReduceScalarDouble(long long *extraPointers,int opNum,
                                  long long x,
                                  long long xShapeInfo,
                                  long long extraParams);

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
    void   execReduce3Double(long long *extraPointers,int opNum,
                             long long x,
                             long long xShapeInfo,
                             long long extraParamsVals,
                             long long y,
                             long long yShapeInfo,
                             long long result,
                             long long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    double   execReduce3ScalarDouble(long long *extraPointers,int opNum,
                                     long long x,
                                     long long xShapeInfo,
                                     long long extraParamsVals,
                                     long long y,
                                     long long yShapeInfo);
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
    void   execReduce3Double(long long *extraPointers,int opNum,
                             long long x,
                             long long xShapeInfo,
                             long long extraParamsVals,
                             long long y,
                             long long yShapeInfo,
                             long long result,
                             long long resultShapeInfoBuffer,
                             long long dimension,
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
    void   execScalarDouble(long long *extraPointers,int opNum,
                            long long x,
                            int xStride,
                            long long result,
                            int resultStride,
                            double scalar,
                            long long extraParams,
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
    void execScalarDouble(long long *extraPointers,int opNum,
                          long long x,
                          long long xShapeInfo,
                          long long result,
                          long long resultShapeInfo,
                          double scalar,
                          long long extraParams);

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
    void execScalarDouble(long long *extraPointers,int opNum,
                          long long x,
                          long long xShapeInfo,
                          long long result,
                          long long resultShapeInfo,
                          double scalar,
                          long long extraParams,
                          int n,
                          long long xIndexes,
                          long long resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    double   execSummaryStatsScalarDouble(long long *extraPointers,int opNum,long long x,
                                          long long xShapeInfo,
                                          long long extraParams,bool biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execSummaryStatsDouble(long long *extraPointers,int opNum,
                                  long long x,
                                  long long xShapeInfo,
                                  long long extraParams,
                                  long long result,
                                  long long resultShapeInfo,bool biasCorrected);
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
    void   execSummaryStatsDouble(long long *extraPointers,int opNum,long long x,
                                  long long xShapeInfo,
                                  long long extraParams,
                                  long long result,
                                  long long resultShapeInfoBuffer,
                                  long long dimension, int dimensionLength,bool biasCorrected);
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
    void   execTransformDouble(long long *extraPointers,int opNum,
                               long long dx,
                               int xStride,
                               long long result,
                               int resultStride,
                               long long extraParams, int n);

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
    void   execTransformDouble(long long *extraPointers,int opNum,
                               long long dx,
                               long long xShapeInfo,
                               long long result,
                               long long resultShapeInfo,
                               long long extraParams);

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
    void   execTransformDouble(long long *extraPointers,int opNum,
                               long long dx,
                               long long xShapeInfo,
                               long long result,
                               long long resultShapeInfo,
                               long long extraParams,
                               long long xIndexes,
                               long long resultIndexes);

    /**
    *
    * @param opNum
    * @param x
    * @param xShapeInfo
    * @param extraParams
    */
    float   execIndexReduceScalarFloat(long long *extraPointers,
                                       int opNum,
                                       long long x,
                                       long long xShapeInfo,
                                       long long extraParams);

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
    void   execIndexReduceFloat(long long *extraPointers,int opNum,
                                long long x,
                                long long xShapeInfo,
                                long long extraParams,
                                long long result,
                                long long resultShapeInfoBuffer,
                                long long dimension, int dimensionLength);
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
    void   execBroadcastFloat(long long *extraPointers,int opNum,
                              long long x,
                              long long xShapeInfo,
                              long long y,
                              long long yShapeInfo,
                              long long result,
                              long long resultShapeInfo,
                              long long dimension, int dimensionLength);



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
    void   execPairwiseTransformFloat(long long *extraPointers,int opNum,
                                      long long dx,
                                      int xStride,
                                      long long y,
                                      int yStride,
                                      long long result,
                                      int resultStride,
                                      long long extraParams, int n);

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
    void execPairwiseTransformFloat(long long *extraPointers,int opNum,
                                    long long dx,
                                    long long xShapeInfo,
                                    long long y,
                                    long long yShapeInfo,
                                    long long result,
                                    long long resultShapeInfo,
                                    long long extraParams,
                                    long long xIndexes,
                                    long long yIndexes,
                                    long long resultIndexes);

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
    void execPairwiseTransformFloat(long long *extraPointers,int opNum,
                                    long long dx,
                                    long long  xShapeInfo,
                                    long long y,
                                    long long  yShapeInfo,
                                    long long result,
                                    long long  resultShapeInfo,
                                    long long extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(long long *extraPointers,int opNum,
                           long long x,
                           long long xShapeInfo,
                           long long extraParams,
                           long long result,
                           long long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execReduceFloat(long long *extraPointers,int opNum,
                           long long x,
                           long long xShapeInfo,
                           long long extraParams,
                           long long result,
                           long long resultShapeInfo,
                           long long dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    float execReduceScalarFloat(long long *extraPointers,int opNum,
                                long long x,
                                long long xShapeInfo,
                                long long extraParams);

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
    void   execReduce3Float(long long *extraPointers,int opNum,
                            long long x,
                            long long xShapeInfo,
                            long long extraParamsVals,
                            long long y,
                            long long yShapeInfo,
                            long long result,
                            long long resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    float   execReduce3ScalarFloat(long long *extraPointers,int opNum,
                                   long long x,
                                   long long xShapeInfo,
                                   long long extraParamsVals,
                                   long long y,
                                   long long yShapeInfo);
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
    void   execReduce3Float(long long *extraPointers,int opNum,
                            long long x,
                            long long xShapeInfo,
                            long long extraParamsVals,
                            long long y,
                            long long yShapeInfo,
                            long long result,
                            long long resultShapeInfoBuffer,
                            long long dimension,
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
    void   execScalarFloat(long long *extraPointers,int opNum,
                           long long x,
                           int xStride,
                           long long result,
                           int resultStride,
                           double scalar,
                           long long extraParams,
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
    void execScalarFloat(long long *extraPointers,int opNum,
                         long long x,
                         long long xShapeInfo,
                         long long result,
                         long long resultShapeInfo,
                         float scalar,
                         long long extraParams);

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
    void execScalarFloat(long long *extraPointers,int opNum,
                         long long x,
                         long long xShapeInfo,
                         long long result,
                         long long resultShapeInfo,
                         double scalar,
                         long long extraParams,
                         long long xIndexes,
                         long long resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    float   execSummaryStatsScalarFloat(long long *extraPointers,int opNum,long long x,
                                        long long xShapeInfo,
                                        long long extraParams,bool biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    void   execSummaryStatsFloat(long long *extraPointers,int opNum,
                                 long long x,
                                 long long xShapeInfo,
                                 long long extraParams,
                                 long long result,
                                 long long resultShapeInfo,bool biasCorrected);
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
    void   execSummaryStatsFloat(long long *extraPointers,int opNum,long long x,
                                 long long xShapeInfo,
                                 long long extraParams,
                                 long long result,
                                 long long resultShapeInfoBuffer,
                                 long long dimension, int dimensionLength,bool biasCorrected);
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
    void   execTransformFloat(long long *extraPointers,int opNum,
                              long long dx,
                              int xStride,
                              long long result,
                              int resultStride,
                              long long extraParams, int n);

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
    void   execTransformFloat(long long *extraPointers,int opNum,
                              long long dx,
                              long long xShapeInfo,
                              long long result,
                              long long resultShapeInfo,
                              long long extraParams);

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
    void   execTransformFloat(long long *extraPointers,int opNum,
                              long long dx,
                              long long xShapeInfo,
                              long long result,
                              long long resultShapeInfo,
                              long long extraParams,
                              long long xIndexes,
                              long long resultIndexes);
};


#endif //NATIVEOPERATIONS_NATIVEOPS_H
