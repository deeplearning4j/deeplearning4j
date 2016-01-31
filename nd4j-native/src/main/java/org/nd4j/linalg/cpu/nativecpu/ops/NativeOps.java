package org.nd4j.linalg.cpu.nativecpu.ops;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * Native interface for 
 * op execution on cpu
 * @author Adam Gibson
 */
public class NativeOps {
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native double   execIndexReduceScalar(int opNum,
                                               DoubleBuffer x,
                                               IntBuffer xShapeInfo,
                                               DoubleBuffer extraParams);

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
                                         DoubleBuffer x,
                                         IntBuffer xShapeInfo,
                                         DoubleBuffer extraParams,
                                         DoubleBuffer result,
                                         IntBuffer resultShapeInfoBuffer,
                                         IntBuffer dimension, int dimensionLength);
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
                                       DoubleBuffer x,
                                       IntBuffer xShapeInfo,
                                       DoubleBuffer y,
                                       IntBuffer yShapeInfo,
                                       DoubleBuffer result,
                                       IntBuffer resultShapeInfo,
                                       IntBuffer dimension, int dimensionLength);

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
                                               DoubleBuffer dx,
                                               int xStride,
                                               DoubleBuffer y,
                                               int yStride,
                                               DoubleBuffer result,
                                               int resultStride,
                                               DoubleBuffer extraParams, int n);

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
                                    DoubleBuffer x,
                                    IntBuffer xShapeInfo,
                                    DoubleBuffer extraParams,
                                    DoubleBuffer result,
                                    IntBuffer resultShapeInfo);

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
                                    DoubleBuffer x,
                                    IntBuffer xShapeInfo,
                                    DoubleBuffer extraParams,
                                    DoubleBuffer result,
                                    IntBuffer resultShapeInfo,
                                    IntBuffer dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public native double execReduceScalar(int opNum,
                                          DoubleBuffer x,
                                          IntBuffer xShapeInfo,
                                          DoubleBuffer extraParams);

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
                                     DoubleBuffer x,
                                     IntBuffer xShapeInfo,
                                     DoubleBuffer extraParamsVals,
                                     DoubleBuffer y,
                                     IntBuffer yShapeInfo,
                                     DoubleBuffer result,
                                     IntBuffer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    public native double   execReduce3Scalar(int opNum,
                                             DoubleBuffer x,
                                             IntBuffer xShapeInfo,
                                             DoubleBuffer extraParamsVals,
                                             DoubleBuffer y,
                                             IntBuffer yShapeInfo);
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
                                     DoubleBuffer x,
                                     IntBuffer xShapeInfo,
                                     DoubleBuffer extraParamsVals,
                                     DoubleBuffer y,
                                     IntBuffer yShapeInfo,
                                     DoubleBuffer result,
                                     IntBuffer resultShapeInfoBuffer,
                                     IntBuffer dimension,
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
    public native double   execScalar(int opNum,
                                      DoubleBuffer x,
                                      int xStride,
                                      DoubleBuffer result,
                                      int resultStride,
                                      double scalar,
                                      DoubleBuffer extraParams,
                                      int n);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native double   execSummaryStatsScalar(int opNum,DoubleBuffer x,
                                                  IntBuffer xShapeInfo,
                                                  DoubleBuffer extraParams);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execSummaryStats(int opNum,DoubleBuffer x,
                                          IntBuffer xShapeInfo,
                                          DoubleBuffer extraParams,
                                          DoubleBuffer result,
                                          IntBuffer resultShapeInfo);
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
    public native void   execSummaryStats(int opNum,DoubleBuffer x,
                                          IntBuffer xShapeInfo,
                                          DoubleBuffer extraParams,
                                          DoubleBuffer result,
                                          IntBuffer resultShapeInfoBuffer,
                                          IntBuffer dimension, int dimensionLength);
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
    public native void   execTransform(int opNum,
                                       DoubleBuffer dx,
                                       int xStride,
                                       DoubleBuffer result,
                                       int resultStride,
                                       DoubleBuffer extraParams, int n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native float   execIndexReduceScalar(int opNum,
                                               FloatBuffer x,
                                               IntBuffer xShapeInfo,
                                               FloatBuffer extraParams);

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
                                         FloatBuffer x,
                                         IntBuffer xShapeInfo,
                                         FloatBuffer extraParams,
                                         FloatBuffer result,
                                         IntBuffer resultShapeInfoBuffer,
                                         IntBuffer dimension, int dimensionLength);
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
                                       FloatBuffer x,
                                       IntBuffer xShapeInfo,
                                       FloatBuffer y,
                                       IntBuffer yShapeInfo,
                                       FloatBuffer result,
                                       IntBuffer resultShapeInfo,
                                       IntBuffer dimension, int dimensionLength);

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
                                               FloatBuffer dx,
                                               int xStride,
                                               FloatBuffer y,
                                               int yStride,
                                               FloatBuffer result,
                                               int resultStride,
                                               FloatBuffer extraParams, int n);


    /**
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduce(int opNum,
                                    FloatBuffer x,
                                    IntBuffer xShapeInfo,
                                    FloatBuffer extraParams,
                                    FloatBuffer result,
                                    IntBuffer resultShapeInfo,IntBuffer dimension,int dimensionLength);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public native float execReduceScalar(int opNum,
                                         FloatBuffer x,
                                         IntBuffer xShapeInfo,
                                         FloatBuffer extraParams);

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
                                     FloatBuffer x,
                                     IntBuffer xShapeInfo,
                                     FloatBuffer extraParamsVals,
                                     FloatBuffer y,
                                     IntBuffer yShapeInfo,
                                     FloatBuffer result, IntBuffer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    public native float   execReduce3Scalar(int opNum,
                                            FloatBuffer x,
                                            IntBuffer xShapeInfo,
                                            FloatBuffer extraParamsVals,
                                            FloatBuffer y,
                                            IntBuffer yShapeInfo);
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
                                     FloatBuffer x,
                                     IntBuffer xShapeInfo,
                                     FloatBuffer extraParamsVals,
                                     FloatBuffer y,
                                     IntBuffer yShapeInfo,
                                     FloatBuffer result,
                                     IntBuffer resultShapeInfoBuffer,
                                     IntBuffer dimension,
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
    public native float   execScalar(int opNum,
                                     FloatBuffer x,
                                     int xStride,
                                     FloatBuffer result,
                                     int resultStride,
                                     float scalar,
                                     FloatBuffer extraParams,
                                     int n);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native float   execSummaryStatsScalar(int opNum,FloatBuffer x,
                                                 IntBuffer xShapeInfo,
                                                 FloatBuffer extraParams);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execSummaryStats(int opNum,FloatBuffer x,
                                          IntBuffer xShapeInfo,
                                          FloatBuffer extraParams,
                                          FloatBuffer result,
                                          IntBuffer resultShapeInfo);
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
    public native void   execSummaryStats(int opNum,FloatBuffer x,
                                          IntBuffer xShapeInfo,
                                          FloatBuffer extraParams,
                                          FloatBuffer result,
                                          IntBuffer resultShapeInfoBuffer,
                                          IntBuffer dimension, int dimensionLength);
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
    public native void   execTransform(int opNum,
                                       FloatBuffer dx,
                                       int xStride,
                                       FloatBuffer result,
                                       int resultStride,
                                       FloatBuffer extraParams, int n);

}
