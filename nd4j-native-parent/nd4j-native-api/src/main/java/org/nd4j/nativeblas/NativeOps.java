package org.nd4j.nativeblas;


import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;



/**
 * Native interface for 
 * op execution on cpu
 * @author Adam Gibson
 */
@Platform(include="NativeOps.h",link = "libnd4j")
public class NativeOps extends Pointer {

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native double   execIndexReduceScalarDouble(long[] extraPointers,int opNum,
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
    public native void   execIndexReduceDouble(long[] extraPointers,int opNum,
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
    public native void   execBroadcastDouble(long[] extraPointers,int opNum,
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
    public native void   execPairwiseTransformDouble(long[] extraPointers,int opNum,
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
    public native void execPairwiseTransformDouble(long[] extraPointers,int opNum,
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
    public native void execPairwiseTransformDouble(long[] extraPointers,int opNum,
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
    public native void   execReduceDouble(long[] extraPointers,int opNum,
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
    public native void   execReduceDouble(long[] extraPointers,int opNum,
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
    public native double execReduceScalarDouble(long[] extraPointers,int opNum,
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
    public native void   execReduce3Double(long[] extraPointers,int opNum,
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
    public native double   execReduce3ScalarDouble(long[] extraPointers,int opNum,
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
    public native void   execReduce3Double(long[] extraPointers,int opNum,
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
    public native void   execScalarDouble(long[] extraPointers,int opNum,
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
    public native void execScalarDouble(long[] extraPointers,int opNum,
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
    public native void execScalarDouble(long[] extraPointers,int opNum,
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
    public native double   execSummaryStatsScalarDouble(long[] extraPointers,int opNum,long x,
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
    public native void   execSummaryStatsDouble(long[] extraPointers,int opNum,
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
    public native void   execSummaryStatsDouble(long[] extraPointers,int opNum,long x,
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
    public native void   execTransformDouble(long[] extraPointers,int opNum,
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
    public native void   execTransformDouble(long[] extraPointers,int opNum,
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
    public native void   execTransformDouble(long[] extraPointers,int opNum,
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
    public native double   execIndexReduceScalarFloat(long[] extraPointers,int opNum,
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
    public native void   execIndexReduceFloat(long[] extraPointers,int opNum,
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
    public native void   execBroadcastFloat(long[] extraPointers,int opNum,
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
    public native void   execPairwiseTransformFloat(long[] extraPointers,int opNum,
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
    public native void execPairwiseTransformFloat(long[] extraPointers,int opNum,
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
    public native void execPairwiseTransformFloat(long[] extraPointers,int opNum,
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
    public native void   execReduceFloat(long[] extraPointers,int opNum,
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
    public native void   execReduceFloat(long[] extraPointers,int opNum,
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
    public native double execReduceScalarFloat(long[] extraPointers,int opNum,
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
    public native void   execReduce3Float(long[] extraPointers,int opNum,
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
    public native double   execReduce3ScalarFloat(long[] extraPointers,int opNum,
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
    public native void   execReduce3Float(long[] extraPointers,int opNum,
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
    public native void   execScalarFloat(long[] extraPointers,int opNum,
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
    public native void execScalarFloat(long[] extraPointers,int opNum,
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
    public native void execScalarFloat(long[] extraPointers,int opNum,
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
    public native double   execSummaryStatsScalarFloat(long[] extraPointers,int opNum,long x,
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
    public native void   execSummaryStatsFloat(long[] extraPointers,int opNum,
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
    public native void   execSummaryStatsFloat(long[] extraPointers,int opNum,long x,
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
    public native void   execTransformFloat(long[] extraPointers,int opNum,
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
    public native void   execTransformFloat(long[] extraPointers,int opNum,
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
    public native void   execTransformFloat(long[] extraPointers,int opNum,
                                            long dx,
                                            long xShapeInfo,
                                            long result,
                                            long resultShapeInfo,
                                            long extraParams,
                                            int n,
                                            long xIndexes,
                                            long resultIndexes);







}
