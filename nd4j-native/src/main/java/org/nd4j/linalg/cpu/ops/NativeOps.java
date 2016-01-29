package org.nd4j.linalg.cpu.ops;

/**
 * Created by agibsonccc on 1/28/16.
 */
public class NativeOps {
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execIndexReduceScalar(int opNum,
                                               double[] x,
                                               int[] xShapeInfo,
                                               double[] extraParams,
                                               double[] result,
                                               int[] resultShapeInfo);

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
    public native void   execIndexReduce(int opNum,
                                         double[] x,
                                         int[] xShapeInfo,
                                         double[] extraParams,
                                         double[] result,
                                         int[] resultShapeInfoBuffer,
                                         int[] dimension, int dimensionLength);
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
    public native void   execBroadcast(int opNum,
                                       double[] x,
                                       int[] xShapeInfo,
                                       double[] y,
                                       int[] yShapeInfo,
                                       double[] result,
                                       int[] resultShapeInfo,
                                       int[] dimension, int dimensionLength);

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
    public native void   execPairwiseTransform(int opNum,
                                               double[] dx,
                                               int xStride,
                                               double[] y,
                                               int yStride,
                                               double[] result,
                                               int resultStride,
                                               double[] extraParams, int n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduce(int opNum,
                                    double[] x,
                                    int[] xShapeInfo,
                                    double[] extraParams,
                                    double[] result,
                                    int[] resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
   public native double execReduceScalar(int opNum,
                            double[] x,
                            int[] xShapeInfo,
                            double[] extraParams);

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
    public native void   execReduce3(int opNum,
                                     double[] x,
                                     int[] xShapeInfo,
                                     double[] extraParamsVals,
                                     double[] y,
                                     int[] yShapeInfo,
                                     double[] result, int[] resultShapeInfo);
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
    public native void   execReduce3(int opNum,
                                     double[] x,
                                     int[] xShapeInfo,
                                     double[] extraParamsVals,
                                     double[] y,
                                     int[] yShapeInfo,
                                     double[] result,
                                     int[] resultShapeInfoBuffer,
                                     int[] dimension,
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
    public native void   execScalar(int opNum,
                                    double[] x,
                                    int xStride,
                                    double[] result,
                                    int resultStride,
                                    double scalar,
                                    double[] extraParams,
                                    int n);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execSummaryStats(int opNum,double[] x,
                                          int[] xShapeInfo,
                                          double[] extraParams,
                                          double[] result,
                                          int[] resultShapeInfo);
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
    public native void   execSummaryStats(int opNum,double[] x,
                                          int[] xShapeInfo,
                                          double[] extraParams,
                                          double[] result,
                                          int[] resultShapeInfoBuffer,
                                          int[] dimension, int dimensionLength);
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
    public native void   execTransform(int opNum,double[] dx, int xStride, double[] result, int resultStride,
                                       double[] extraParams, int n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execIndexReduceScalar(int opNum,
                                               float[] x,
                                               int[] xShapeInfo,
                                               float[] extraParams,
                                               float[] result,
                                               int[] resultShapeInfo);

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
    public native void   execIndexReduce(int opNum,
                                         float[] x,
                                         int[] xShapeInfo,
                                         float[] extraParams,
                                         float[] result,
                                         int[] resultShapeInfoBuffer,
                                         int[] dimension, int dimensionLength);
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
    public native void   execBroadcast(int opNum,
                                       float[] x,
                                       int[] xShapeInfo,
                                       float[] y,
                                       int[] yShapeInfo,
                                       float[] result,
                                       int[] resultShapeInfo,
                                       int[] dimension, int dimensionLength);

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
    public native void   execPairwiseTransform(int opNum,
                                               float[] dx,
                                               int xStride,
                                               float[] y,
                                               int yStride,
                                               float[] result,
                                               int resultStride,
                                               float[] extraParams, int n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduce(int opNum,
                                    float[] x,
                                    int[] xShapeInfo,
                                    float[] extraParams,
                                    float[] result,
                                    int[] resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public native float execReduceScalar(int opNum,
                                         float[] x,
                                         int[] xShapeInfo,
                                         float[] extraParams);

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
    public native void   execReduce3(int opNum,
                                     float[] x,
                                     int[] xShapeInfo,
                                     float[] extraParamsVals,
                                     float[] y,
                                     int[] yShapeInfo,
                                     float[] result, int[] resultShapeInfo);
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
    public native void   execReduce3(int opNum,
                                     float[] x,
                                     int[] xShapeInfo,
                                     float[] extraParamsVals,
                                     float[] y,
                                     int[] yShapeInfo,
                                     float[] result,
                                     int[] resultShapeInfoBuffer,
                                     int[] dimension,
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
    public native void   execScalar(int opNum,
                                    float[] x,
                                    int xStride,
                                    float[] result,
                                    int resultStride,
                                    float scalar,
                                    float[] extraParams,
                                    int n);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execSummaryStats(int opNum,float[] x,
                                          int[] xShapeInfo,
                                          float[] extraParams,
                                          float[] result,
                                          int[] resultShapeInfo);
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
    public native void   execSummaryStats(int opNum,float[] x,
                                          int[] xShapeInfo,
                                          float[] extraParams,
                                          float[] result,
                                          int[] resultShapeInfoBuffer,
                                          int[] dimension, int dimensionLength);
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
    public native void   execTransform(int opNum,float[] dx, int xStride, float[] result, int resultStride,
                                       float[] extraParams, int n);
}
