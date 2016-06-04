package org.nd4j.nativeblas;


import java.util.Properties;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.annotation.Platform;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Native interface for
 * op execution on cpu
 * @author Adam Gibson
 */
@Platform(include = "NativeOps.h", compiler = "cpp11", link = "nd4j", library = "jnind4j")
public class NativeOps extends Pointer {
    private static Logger log = LoggerFactory.getLogger(NativeOps.class);
    static {
        // using our custom platform properties from resources, and on user request,
        // load in priority libraries found in the library path over bundled ones
        String platform = Loader.getPlatform();
        Properties properties = Loader.loadProperties(platform + "-nd4j", platform);
        properties.remove("platform.preloadpath");
        String s = System.getProperty("org.nd4j.nativeblas.pathsfirst", "false").toLowerCase();
        boolean pathsFirst = s.equals("true") || s.equals("t") || s.equals("");
        Loader.load(NativeOps.class, properties, pathsFirst);
    }

    public NativeOps() {
        allocate();
        initializeDevicesAndFunctions();
        int numThreads;
        String numThreadsString = System.getenv("OMP_NUM_THREADS");
        if(numThreadsString != null && !numThreadsString.isEmpty()) {
            numThreads = Integer.parseInt(numThreadsString);
            setOmpNumThreads(numThreads);
        }
        else
            setOmpNumThreads(Runtime.getRuntime().availableProcessors());

        log.debug("Number of threads used for linear algebra " + ompGetNumThreads());

    }
    private native void allocate();


    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native double   execIndexReduceScalarDouble(PointerPointer extraPointers, int opNum,
                                                       Pointer x,
                                                       Pointer xShapeInfo,
                                                       Pointer extraParams);

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
    public native void   execIndexReduceDouble(PointerPointer extraPointers, int opNum,
                                               Pointer x,
                                               Pointer xShapeInfo,
                                               Pointer extraParams,
                                               Pointer result,
                                               Pointer resultShapeInfoBuffer,
                                               Pointer dimension, int dimensionLength);
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
    public native void   execBroadcastDouble(PointerPointer extraPointers,
                                             int opNum,
                                             Pointer x,
                                             Pointer xShapeInfo,
                                             Pointer y,
                                             Pointer yShapeInfo,
                                             Pointer result,
                                             Pointer resultShapeInfo,
                                             Pointer dimension, int dimensionLength);



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
    public native void   execPairwiseTransformDouble(PointerPointer extraPointers, int opNum,
                                                     Pointer dx,
                                                     int xStride,
                                                     Pointer y,
                                                     int yStride,
                                                     Pointer result,
                                                     int resultStride,
                                                     Pointer extraParams, long n);

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
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    public native void execPairwiseTransformDouble(PointerPointer extraPointers,
                                                   int opNum,
                                                   Pointer dx,
                                                   Pointer xShapeInfo,
                                                   Pointer y,
                                                   Pointer yShapeInfo,
                                                   Pointer result,
                                                   Pointer resultShapeInfo,
                                                   Pointer extraParams,
                                                   Pointer xIndexes,
                                                   Pointer yIndexes,
                                                   Pointer resultIndexes);

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
     */
    public native void execPairwiseTransformDouble(
            PointerPointer extraPointers,
            int opNum,
            Pointer dx,
            Pointer xShapeInfo,
            Pointer y,
            Pointer yShapeInfo,
            Pointer result,
            Pointer resultShapeInfo,
            Pointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduceDouble(PointerPointer extraPointers, int opNum,
                                          Pointer x,
                                          Pointer xShapeInfo,
                                          Pointer extraParams,
                                          Pointer result,
                                          Pointer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduceDouble(PointerPointer extraPointers, int opNum,
                                          Pointer x,
                                          Pointer xShapeInfo,
                                          Pointer extraParams,
                                          Pointer result,
                                          Pointer resultShapeInfo,
                                          Pointer dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public native  double execReduceScalarDouble(PointerPointer extraPointers, int opNum,
                                                 Pointer x,
                                                 Pointer xShapeInfo,
                                                 Pointer extraParams);

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
    public native void   execReduce3Double(PointerPointer extraPointers, int opNum,
                                           Pointer x,
                                           Pointer xShapeInfo,
                                           Pointer extraParamsVals,
                                           Pointer y,
                                           Pointer yShapeInfo,
                                           Pointer result,
                                           Pointer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    public native double   execReduce3ScalarDouble(PointerPointer extraPointers, int opNum,
                                                   Pointer x,
                                                   Pointer xShapeInfo,
                                                   Pointer extraParamsVals,
                                                   Pointer y,
                                                   Pointer yShapeInfo);
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
    public native void   execReduce3Double(PointerPointer extraPointers, int opNum,
                                           Pointer x,
                                           Pointer xShapeInfo,
                                           Pointer extraParamsVals,
                                           Pointer y,
                                           Pointer yShapeInfo,
                                           Pointer result,
                                           Pointer resultShapeInfoBuffer,
                                           Pointer dimension,
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
    public native void   execScalarDouble(PointerPointer extraPointers, int opNum,
                                          Pointer x,
                                          int xStride,
                                          Pointer result,
                                          int resultStride,
                                          double scalar,
                                          Pointer extraParams,
                                          long n);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     */
    public native void execScalarDouble(PointerPointer extraPointers, int opNum,
                                        Pointer x,
                                        Pointer xShapeInfo,
                                        Pointer result,
                                        Pointer resultShapeInfo,
                                        double scalar,
                                        Pointer extraParams);

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
    public native void execScalarDouble(PointerPointer extraPointers, int opNum,
                                        Pointer x,
                                        Pointer xShapeInfo,
                                        Pointer result,
                                        Pointer resultShapeInfo,
                                        double scalar,
                                        Pointer extraParams,
                                        long n,
                                        Pointer xIndexes,
                                        Pointer resultIndexes);
    /**
     *  @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param biasCorrected
     */
    public native double   execSummaryStatsScalarDouble(PointerPointer extraPointers,  int opNum, Pointer x,
                                                        Pointer xShapeInfo,
                                                        Pointer extraParams, boolean biasCorrected);
    /**
     *  @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     * @param biasCorrected
     */
    public native void   execSummaryStatsDouble(PointerPointer extraPointers,  int opNum,
                                                Pointer x,
                                                Pointer xShapeInfo,
                                                Pointer extraParams,
                                                Pointer result,
                                                Pointer resultShapeInfo, boolean biasCorrected);
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
    public native void   execSummaryStatsDouble(PointerPointer extraPointers, int opNum,Pointer x,
                                                Pointer xShapeInfo,
                                                Pointer extraParams,
                                                Pointer result,
                                                Pointer resultShapeInfoBuffer,
                                                Pointer dimension, int dimensionLength,boolean biasCorrected);
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
    public native void   execTransformDouble(PointerPointer extraPointers, int opNum,
                                             Pointer dx,
                                             int xStride,
                                             Pointer result,
                                             int resultStride,
                                             Pointer extraParams, long n);

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
    public native void   execTransformDouble(PointerPointer extraPointers, int opNum,
                                             Pointer dx,
                                             Pointer xShapeInfo,
                                             Pointer result,
                                             Pointer resultShapeInfo,
                                             Pointer extraParams);

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
    public native void   execTransformDouble(PointerPointer extraPointers, int opNum,
                                             Pointer dx,
                                             Pointer xShapeInfo,
                                             Pointer result,
                                             Pointer resultShapeInfo,
                                             Pointer extraParams,
                                             Pointer xIndexes,
                                             Pointer resultIndexes);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native float   execIndexReduceScalarFloat(PointerPointer extraPointers,
                                                     int opNum,
                                                     Pointer x,
                                                     Pointer xShapeInfo,
                                                     Pointer extraParams);

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
    public native void   execIndexReduceFloat(PointerPointer extraPointers, int opNum,
                                              Pointer x,
                                              Pointer xShapeInfo,
                                              Pointer extraParams,
                                              Pointer result,
                                              Pointer resultShapeInfoBuffer,
                                              Pointer dimension, int dimensionLength);
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
    public native void   execBroadcastFloat(PointerPointer extraPointers,
                                            int opNum,
                                            Pointer x,
                                            Pointer xShapeInfo,
                                            Pointer y,
                                            Pointer yShapeInfo,
                                            Pointer result,
                                            Pointer resultShapeInfo,
                                            Pointer dimension,
                                            int dimensionLength);



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
    public native void   execPairwiseTransformFloat(PointerPointer extraPointers, int opNum,
                                                    Pointer dx,
                                                    int xStride,
                                                    Pointer y,
                                                    int yStride,
                                                    Pointer result,
                                                    int resultStride,
                                                    Pointer extraParams, long n);

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
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    public native void execPairwiseTransformFloat(PointerPointer extraPointers, int opNum,
                                                  Pointer dx,
                                                  Pointer xShapeInfo,
                                                  Pointer y,
                                                  Pointer yShapeInfo,
                                                  Pointer result,
                                                  Pointer resultShapeInfo,
                                                  Pointer extraParams,
                                                  Pointer xIndexes,
                                                  Pointer yIndexes,
                                                  Pointer resultIndexes);

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
     */
    public native void execPairwiseTransformFloat(PointerPointer extraPointers, int opNum,
                                                  Pointer dx,
                                                  Pointer xShapeInfo,
                                                  Pointer y,
                                                  Pointer yShapeInfo,
                                                  Pointer result,
                                                  Pointer resultShapeInfo,
                                                  Pointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduceFloat(PointerPointer extraPointers, int opNum,
                                         Pointer x,
                                         Pointer xShapeInfo,
                                         Pointer extraParams,
                                         Pointer result,
                                         Pointer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execReduceFloat(PointerPointer extraPointers, int opNum,
                                         Pointer x,
                                         Pointer xShapeInfo,
                                         Pointer extraParams,
                                         Pointer result,
                                         Pointer resultShapeInfo,
                                         Pointer dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public native float execReduceScalarFloat(PointerPointer extraPointers, int opNum,
                                              Pointer x,
                                              Pointer xShapeInfo,
                                              Pointer extraParams);

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
    public native void   execReduce3Float(PointerPointer extraPointers, int opNum,
                                          Pointer x,
                                          Pointer xShapeInfo,
                                          Pointer extraParamsVals,
                                          Pointer y,
                                          Pointer yShapeInfo,
                                          Pointer result,
                                          Pointer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    public native float   execReduce3ScalarFloat(PointerPointer extraPointers, int opNum,
                                                 Pointer x,
                                                 Pointer xShapeInfo,
                                                 Pointer extraParamsVals,
                                                 Pointer y,
                                                 Pointer yShapeInfo);
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
    public native void   execReduce3Float(PointerPointer extraPointers, int opNum,
                                          Pointer x,
                                          Pointer xShapeInfo,
                                          Pointer extraParamsVals,
                                          Pointer y,
                                          Pointer yShapeInfo,
                                          Pointer result,
                                          Pointer resultShapeInfoBuffer,
                                          Pointer dimension,
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
    public native void   execScalarFloat(PointerPointer extraPointers, int opNum,
                                         Pointer x,
                                         int xStride,
                                         Pointer result,
                                         int resultStride,
                                         double scalar,
                                         Pointer extraParams,
                                         long n);

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
    public native void execScalarFloat(PointerPointer extraPointers, int opNum,
                                       Pointer x,
                                       Pointer xShapeInfo,
                                       Pointer result,
                                       Pointer resultShapeInfo,
                                       float scalar,
                                       Pointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param xIndexes
     * @param resultIndexes
     */
    public native void execScalarFloat(PointerPointer extraPointers, int opNum,
                                       Pointer x,
                                       Pointer xShapeInfo,
                                       Pointer result,
                                       Pointer resultShapeInfo,
                                       double scalar,
                                       Pointer extraParams,
                                       Pointer xIndexes,
                                       Pointer resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native float   execSummaryStatsScalarFloat(PointerPointer extraPointers, int opNum,Pointer x,
                                                      Pointer xShapeInfo,
                                                      Pointer extraParams,boolean biasCorrected);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public native void   execSummaryStatsFloat(PointerPointer extraPointers, int opNum,
                                               Pointer x,
                                               Pointer xShapeInfo,
                                               Pointer extraParams,
                                               Pointer result,
                                               Pointer resultShapeInfo,boolean biasCorrected);
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
    public native void   execSummaryStatsFloat(PointerPointer extraPointers, int opNum,Pointer x,
                                               Pointer xShapeInfo,
                                               Pointer extraParams,
                                               Pointer result,
                                               Pointer resultShapeInfoBuffer,
                                               Pointer dimension, int dimensionLength,boolean biasCorrected);
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
    public native void   execTransformFloat(PointerPointer extraPointers, int opNum,
                                            Pointer dx,
                                            int xStride,
                                            Pointer result,
                                            int resultStride,
                                            Pointer extraParams, long n);

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
    public native void   execTransformFloat(
            PointerPointer extraPointers,
            int opNum,
            Pointer dx,
            Pointer xShapeInfo,
            Pointer result,
            Pointer resultShapeInfo,
            Pointer extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param resultIndexes
     */
    public native void   execTransformFloat(PointerPointer extraPointers,
                                            int opNum,
                                            Pointer dx,
                                            Pointer xShapeInfo,
                                            Pointer result,
                                            Pointer resultShapeInfo,
                                            Pointer extraParams,
                                            Pointer xIndexes,
                                            Pointer resultIndexes);


    /**
     * Append an input array
     * to the end of a flat array
     * in a particular order
     * @param offset the offset of the array to start at
     * @param order the order
     * @param result the result array
     * @param resultShapeInfo the shape info for te array
     * @param input the input for the array
     * @param inputShapeInfo the shape information for that array
     */
    public native void flattenFloat(
            PointerPointer extraPointers,
            int offset,
            char order,
            Pointer result,
            Pointer resultShapeInfo,
            Pointer input,
            Pointer inputShapeInfo);


    /**
     * Append an input array
     * to the end of a flat array
     * in a particular order
     * @param offset the offset of the array to start at
     * @param order the order
     * @param result the result array
     * @param resultShapeInfo the shape info for te array
     * @param input the input for the array
     * @param inputShapeInfo the shape information for that array
     */
    public native void flattenDouble(PointerPointer extraPointers,
                                     int offset,
                                     char order,
                                     Pointer result,
                                     Pointer resultShapeInfo,
                                     Pointer input,
                                     Pointer inputShapeInfo);

    /**
     *
     * @param dimension
     * @param data
     * @param inputShapeInfo
     * @param result
     * @param resultShapeInfo
     */
    public native void concatDouble(
            PointerPointer extraPointers,
            int dimension,
            int numArrays,
            PointerPointer data,
            PointerPointer inputShapeInfo,
            Pointer result,
            Pointer resultShapeInfo,
            PointerPointer tadPointers,
            PointerPointer tadOffsets);

    /**
     *
     * @param dimension
     * @param data
     * @param inputShapeInfo
     * @param result
     * @param resultShapeInfo
     */
    public native void concatFloat(
            PointerPointer extraPointers,
            int dimension,
            int numArrays,
            PointerPointer data,
            PointerPointer inputShapeInfo,
            Pointer result,
            Pointer resultShapeInfo,
            PointerPointer tadPointers,
            PointerPointer tadOffsets);

    /**
     * Gets the number of open mp threads
     * @return
     */
    public native int ompGetNumThreads();

    /**
     * Sets the number of openmp threads
     * @param threads
     */
    public native void setOmpNumThreads(int threads);

    /**
     * NEVER EVER USE THIS METHOD OUTSIDE OF  CUDA
     */
    public native void initializeDevicesAndFunctions();

    public native Pointer mallocHost(long memorySize, int flags);

    public native Pointer mallocDevice(long memorySize, Pointer ptrToDeviceId, int flags);

    public native int freeHost(Pointer pointer);

    public native int freeDevice(Pointer pointer, Pointer deviceId);

    public native Pointer createContext();

    public native Pointer createStream();

    public native Pointer createEvent();

    public native Pointer createBlasHandle();

    public native int registerEvent(Pointer event, Pointer stream);

    public native int destroyEvent(Pointer event);

    public native int setBlasStream(Pointer handle, Pointer stream);

    public native int setDevice(Pointer ptrToDeviceId);

    public native int streamSynchronize(Pointer stream);

    public native int eventSynchronize(Pointer event);

    public native long getDeviceFreeMemory(Pointer ptrToDeviceId);

    public native long getDeviceTotalMemory(Pointer ptrToDeviceId);

    public native int memcpy(Pointer dst, Pointer src, long size, int flags, Pointer reserved);

    public native int memcpyAsync(Pointer dst, Pointer src, long size, int flags, Pointer reserved);

    public native int memcpyConstantAsync(long dst, Pointer src, long size, int flags, Pointer reserved);

    public native int memset(Pointer dst, int value, long size,  int flags, Pointer reserved);

    public native int memsetAsync(Pointer dst, int value, long size, int flags, Pointer reserved);

    public native Pointer getConstantSpace();

    public native int getAvailableDevices();

    public native void enableDebugMode(boolean reallyEnable);

    public native void enableVerboseMode(boolean reallyEnable);

    public native void setGridLimit(int gridSize);

    public native void tadOnlyShapeInfo(Pointer shapeInfo, Pointer dimension, int dimensionLength, Pointer targetBuffer, Pointer offsetsBuffer);
}
