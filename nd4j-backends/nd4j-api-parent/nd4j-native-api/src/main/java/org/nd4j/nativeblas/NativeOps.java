package org.nd4j.nativeblas;


import java.util.Properties;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.javacpp.annotation.Cast;
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
        try {
            Loader.load(NativeOps.class, properties, pathsFirst);
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException("ND4J is probably missing dependencies. For more information, please refer to: http://nd4j.org/getstarted.html", e);
        }
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
            setOmpNumThreads(getCores(Runtime.getRuntime().availableProcessors()));
            //setOmpNumThreads(4);

        log.info("Number of threads used for NativeOps: {}", ompGetMaxThreads());

    }
    private native void allocate();


    private static int getCores(int totals) {
        // that's special case for Xeon Phi
        if (totals >= 256) return  64;

        int ht_off = totals / 2; // we count off HyperThreading without any excuses
        if (ht_off <= 4) return 4; // special case for Intel i5. and nobody likes i3 anyway

        if (ht_off > 24) {
            int rounds = 0;
            while (ht_off > 24) { // we loop until final value gets below 24 cores, since that's reasonable threshold as of 2016
                if (ht_off > 24) {
                    ht_off /= 2; // we dont' have any cpus that has higher number then 24 physical cores
                    rounds++;
                }
            }
            // 20 threads is special case in this branch
            if (ht_off == 20 && rounds < 2)
                ht_off /= 2;
        } else { // low-core models are known, but there's a gap, between consumer cpus and xeons
            if (ht_off <= 6) {
                // that's more likely consumer-grade cpu, so leave this value alone
                return ht_off;
            } else {
                if (isOdd(ht_off)) // if that's odd number, it's final result
                    return ht_off;

                // 20 threads & 16 threads are special case in this branch, where we go min value
                if (ht_off == 20 || ht_off == 16)
                    ht_off /= 2;
            }
        }
        return ht_off;
    }

    private static boolean isOdd(int value) {
        return (value % 2 != 0);
    }

    /**
     * This method allows you to specify minimal number of elements per thread/block during op call
     * PLEASE NOTE: Changing this value might and will affect performance.
     *
     * @param value
     */
    public native void setElementThreshold(int value);

    /**
     * This method allows you to specify minimal number of TADs per thread/block during op call
     * PLEASE NOTE: Changing this value might and will affect performance.
     *
     * @param value
     */
    public native void setTADThreshold(int value);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native double   execIndexReduceScalarDouble(PointerPointer extraPointers, int opNum,
                                                       DoublePointer x,
                                                       IntPointer xShapeInfo,
                                                       DoublePointer extraParams);

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
                                               DoublePointer x,
                                               IntPointer xShapeInfo,
                                               DoublePointer extraParams,
                                               DoublePointer result,
                                               IntPointer resultShapeInfoBuffer,
                                               IntPointer dimension, int dimensionLength);
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
                                             DoublePointer x,
                                             IntPointer xShapeInfo,
                                             DoublePointer y,
                                             IntPointer yShapeInfo,
                                             DoublePointer result,
                                             IntPointer resultShapeInfo,
                                             IntPointer dimension, int dimensionLength);



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
                                                     DoublePointer dx,
                                                     int xStride,
                                                     DoublePointer y,
                                                     int yStride,
                                                     DoublePointer result,
                                                     int resultStride,
                                                     DoublePointer extraParams, long n);

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
                                                   DoublePointer dx,
                                                   IntPointer xShapeInfo,
                                                   DoublePointer y,
                                                   IntPointer yShapeInfo,
                                                   DoublePointer result,
                                                   IntPointer resultShapeInfo,
                                                   DoublePointer extraParams,
                                                   IntPointer xIndexes,
                                                   IntPointer yIndexes,
                                                   IntPointer resultIndexes);

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
            DoublePointer dx,
            IntPointer xShapeInfo,
            DoublePointer y,
            IntPointer yShapeInfo,
            DoublePointer result,
            IntPointer resultShapeInfo,
            DoublePointer extraParams);

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
                                          DoublePointer x,
                                          IntPointer xShapeInfo,
                                          DoublePointer extraParams,
                                          DoublePointer result,
                                          IntPointer resultShapeInfo);

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
                                          DoublePointer x,
                                          IntPointer xShapeInfo,
                                          DoublePointer extraParams,
                                          DoublePointer result,
                                          IntPointer resultShapeInfo,
                                          IntPointer dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public native  double execReduceScalarDouble(PointerPointer extraPointers, int opNum,
                                                 DoublePointer x,
                                                 IntPointer xShapeInfo,
                                                 DoublePointer extraParams);

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
                                           DoublePointer x,
                                           IntPointer xShapeInfo,
                                           DoublePointer extraParamsVals,
                                           DoublePointer y,
                                           IntPointer yShapeInfo,
                                           DoublePointer result,
                                           IntPointer resultShapeInfo);

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
                                                   DoublePointer x,
                                                   IntPointer xShapeInfo,
                                                   DoublePointer extraParamsVals,
                                                   DoublePointer y,
                                                   IntPointer yShapeInfo);
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
                                           DoublePointer x,
                                           IntPointer xShapeInfo,
                                           DoublePointer extraParamsVals,
                                           DoublePointer y,
                                           IntPointer yShapeInfo,
                                           DoublePointer result,
                                           IntPointer resultShapeInfoBuffer,
                                           IntPointer dimension,
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
                                          DoublePointer x,
                                          int xStride,
                                          DoublePointer result,
                                          int resultStride,
                                          double scalar,
                                          DoublePointer extraParams,
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
                                        DoublePointer x,
                                        IntPointer xShapeInfo,
                                        DoublePointer result,
                                        IntPointer resultShapeInfo,
                                        double scalar,
                                        DoublePointer extraParams);

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
                                        DoublePointer x,
                                        IntPointer xShapeInfo,
                                        DoublePointer result,
                                        IntPointer resultShapeInfo,
                                        double scalar,
                                        DoublePointer extraParams,
                                        long n,
                                        IntPointer xIndexes,
                                        IntPointer resultIndexes);
    /**
     *  @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param biasCorrected
     */
    public native double   execSummaryStatsScalarDouble(PointerPointer extraPointers,  int opNum, DoublePointer x,
                                                        IntPointer xShapeInfo,
                                                        DoublePointer extraParams, boolean biasCorrected);
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
                                                DoublePointer x,
                                                IntPointer xShapeInfo,
                                                DoublePointer extraParams,
                                                DoublePointer result,
                                                IntPointer resultShapeInfo, boolean biasCorrected);
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
    public native void   execSummaryStatsDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                                                IntPointer xShapeInfo,
                                                DoublePointer extraParams,
                                                DoublePointer result,
                                                IntPointer resultShapeInfoBuffer,
                                                IntPointer dimension, int dimensionLength,boolean biasCorrected);
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
                                             DoublePointer dx,
                                             int xStride,
                                             DoublePointer result,
                                             int resultStride,
                                             DoublePointer extraParams, long n);

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
                                             DoublePointer dx,
                                             IntPointer xShapeInfo,
                                             DoublePointer result,
                                             IntPointer resultShapeInfo,
                                             DoublePointer extraParams);

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
                                             DoublePointer dx,
                                             IntPointer xShapeInfo,
                                             DoublePointer result,
                                             IntPointer resultShapeInfo,
                                             DoublePointer extraParams,
                                             IntPointer xIndexes,
                                             IntPointer resultIndexes);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native float   execIndexReduceScalarFloat(PointerPointer extraPointers,
                                                     int opNum,
                                                     FloatPointer x,
                                                     IntPointer xShapeInfo,
                                                     FloatPointer extraParams);

    public native float   execIndexReduceScalarHalf(PointerPointer extraPointers,
                                                     int opNum,
                                                     @Cast("float16*") ShortPointer x,
                                                     IntPointer xShapeInfo,
                                                     @Cast("float16*") ShortPointer extraParams);

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
                                              FloatPointer x,
                                              IntPointer xShapeInfo,
                                              FloatPointer extraParams,
                                              FloatPointer results,
                                              IntPointer resultShapeInfoBuffer,
                                              IntPointer dimension, int dimensionLength);

    public native void   execIndexReduceHalf(PointerPointer extraPointers, int opNum,
                                              @Cast("float16*") ShortPointer x,
                                              IntPointer xShapeInfo,
                                              @Cast("float16*") ShortPointer extraParams,
                                              @Cast("float16*") ShortPointer results,
                                              IntPointer resultShapeInfoBuffer,
                                              IntPointer dimension, int dimensionLength);
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
                                            FloatPointer x,
                                            IntPointer xShapeInfo,
                                            FloatPointer y,
                                            IntPointer yShapeInfo,
                                            FloatPointer results,
                                            IntPointer resultShapeInfo,
                                            IntPointer dimension,
                                            int dimensionLength);

    public native void   execBroadcastHalf(PointerPointer extraPointers,
                                            int opNum,
                                            @Cast("float16*") ShortPointer x,
                                            IntPointer xShapeInfo,
                                            @Cast("float16*") ShortPointer y,
                                            IntPointer yShapeInfo,
                                            @Cast("float16*") ShortPointer results,
                                            IntPointer resultShapeInfo,
                                            IntPointer dimension,
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
                                                    FloatPointer dx,
                                                    int xStride,
                                                    FloatPointer y,
                                                    int yStride,
                                                    FloatPointer results,
                                                    int resultStride,
                                                    FloatPointer extraParams, long n);

    public native void   execPairwiseTransformHalf(PointerPointer extraPointers, int opNum,
                                                    @Cast("float16*") ShortPointer dx,
                                                    int xStride,
                                                    @Cast("float16*") ShortPointer y,
                                                    int yStride,
                                                    @Cast("float16*") ShortPointer results,
                                                    int resultStride,
                                                    @Cast("float16*") ShortPointer extraParams, long n);

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
                                                  FloatPointer dx,
                                                  IntPointer xShapeInfo,
                                                  FloatPointer y,
                                                  IntPointer yShapeInfo,
                                                  FloatPointer results,
                                                  IntPointer resultShapeInfo,
                                                  FloatPointer extraParams,
                                                  IntPointer xIndexes,
                                                  IntPointer yIndexes,
                                                  IntPointer resultIndexes);

    public native void execPairwiseTransformHalf(PointerPointer extraPointers, int opNum,
                                                  @Cast("float16*") ShortPointer dx,
                                                  IntPointer xShapeInfo,
                                                  @Cast("float16*") ShortPointer y,
                                                  IntPointer yShapeInfo,
                                                  @Cast("float16*") ShortPointer results,
                                                  IntPointer resultShapeInfo,
                                                  @Cast("float16*") ShortPointer extraParams,
                                                  IntPointer xIndexes,
                                                  IntPointer yIndexes,
                                                  IntPointer resultIndexes);

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
                                                  FloatPointer dx,
                                                  IntPointer xShapeInfo,
                                                  FloatPointer y,
                                                  IntPointer yShapeInfo,
                                                  FloatPointer results,
                                                  IntPointer resultShapeInfo,
                                                  FloatPointer extraParams);

    public native void execPairwiseTransformHalf(PointerPointer extraPointers, int opNum,
                                                  @Cast("float16*") ShortPointer dx,
                                                  IntPointer xShapeInfo,
                                                  @Cast("float16*") ShortPointer y,
                                                  IntPointer yShapeInfo,
                                                  @Cast("float16*") ShortPointer results,
                                                  IntPointer resultShapeInfo,
                                                  @Cast("float16*") ShortPointer extraParams);

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
                                         FloatPointer x,
                                         IntPointer xShapeInfo,
                                         FloatPointer extraParams,
                                         FloatPointer results,
                                         IntPointer resultShapeInfo);

    public native void   execReduceHalf(PointerPointer extraPointers, int opNum,
                                         @Cast("float16*") ShortPointer x,
                                         IntPointer xShapeInfo,
                                         @Cast("float16*") ShortPointer extraParams,
                                         @Cast("float16*") ShortPointer results,
                                         IntPointer resultShapeInfo);

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
                                         FloatPointer x,
                                         IntPointer xShapeInfo,
                                         FloatPointer extraParams,
                                         FloatPointer results,
                                         IntPointer resultShapeInfo,
                                         IntPointer dimension,int dimensionLength);

    public native void   execReduceHalf(PointerPointer extraPointers, int opNum,
                                         @Cast("float16*") ShortPointer x,
                                         IntPointer xShapeInfo,
                                         @Cast("float16*") ShortPointer extraParams,
                                         @Cast("float16*") ShortPointer results,
                                         IntPointer resultShapeInfo,
                                         IntPointer dimension,int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public native float execReduceScalarFloat(PointerPointer extraPointers, int opNum,
                                              FloatPointer x,
                                              IntPointer xShapeInfo,
                                              FloatPointer extraParams);


    public native float execReduceScalarHalf(PointerPointer extraPointers, int opNum,
                                              @Cast("float16*") ShortPointer x,
                                              IntPointer xShapeInfo,
                                              @Cast("float16*") ShortPointer extraParams);

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
                                          FloatPointer x,
                                          IntPointer xShapeInfo,
                                          FloatPointer extraParamsVals,
                                          FloatPointer y,
                                          IntPointer yShapeInfo,
                                          FloatPointer results,
                                          IntPointer resultShapeInfo);

    public native void   execReduce3Half(PointerPointer extraPointers, int opNum,
                                          @Cast("float16*") ShortPointer x,
                                          IntPointer xShapeInfo,
                                          @Cast("float16*") ShortPointer extraParamsVals,
                                          @Cast("float16*") ShortPointer y,
                                          IntPointer yShapeInfo,
                                          @Cast("float16*") ShortPointer results,
                                          IntPointer resultShapeInfo);

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
                                                 FloatPointer x,
                                                 IntPointer xShapeInfo,
                                                 FloatPointer extraParamsVals,
                                                 FloatPointer y,
                                                 IntPointer yShapeInfo);

    public native float   execReduce3ScalarHalf(PointerPointer extraPointers, int opNum,
                                                 @Cast("float16*") ShortPointer x,
                                                 IntPointer xShapeInfo,
                                                 @Cast("float16*") ShortPointer extraParamsVals,
                                                 @Cast("float16*") ShortPointer y,
                                                 IntPointer yShapeInfo);
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
                                          FloatPointer x,
                                          IntPointer xShapeInfo,
                                          FloatPointer extraParamsVals,
                                          FloatPointer y,
                                          IntPointer yShapeInfo,
                                          FloatPointer results,
                                          IntPointer resultShapeInfoBuffer,
                                          IntPointer dimension,
                                          int dimensionLength);

    public native void   execReduce3Half(PointerPointer extraPointers, int opNum,
                                          @Cast("float16*") ShortPointer x,
                                          IntPointer xShapeInfo,
                                          @Cast("float16*") ShortPointer extraParamsVals,
                                          @Cast("float16*") ShortPointer y,
                                          IntPointer yShapeInfo,
                                          @Cast("float16*") ShortPointer results,
                                          IntPointer resultShapeInfoBuffer,
                                          IntPointer dimension,
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
                                         FloatPointer x,
                                         int xStride,
                                         FloatPointer results,
                                         int resultStride,
                                         float scalar,
                                         FloatPointer extraParams,
                                         long n);

    public native void   execScalarHalf(PointerPointer extraPointers, int opNum,
                                         @Cast("float16*") ShortPointer x,
                                         int xStride,
                                         @Cast("float16*") ShortPointer results,
                                         int resultStride,
                                         float scalar,
                                         @Cast("float16*") ShortPointer extraParams,
                                         long n);

    /**
     * ScalarOp along dimension
     *
     * @param extraPointers pointers to tadShapes and tadoffsets
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param z
     * @param zShapeInfo
     * @param scalars
     * @param extraParams
     * @param dimension
     * @param dimensionLength
     */
    public native void execScalarFloat(PointerPointer extraPointers, int opNum,
                                       FloatPointer x,
                                       IntPointer xShapeInfo,
                                       FloatPointer z,
                                       IntPointer zShapeInfo,
                                       FloatPointer scalars,
                                       FloatPointer extraParams,
                                       IntPointer dimension,
                                       int dimensionLength
                                       );

    public native void execScalarDouble(PointerPointer extraPointers, int opNum,
                                       DoublePointer x,
                                       IntPointer xShapeInfo,
                                       DoublePointer z,
                                       IntPointer zShapeInfo,
                                       DoublePointer scalars,
                                       DoublePointer extraParams,
                                       IntPointer dimension,
                                       int dimensionLength
    );

    public native void execScalarHalf(PointerPointer extraPointers, int opNum,
                                        @Cast("float16*") ShortPointer x,
                                        IntPointer xShapeInfo,
                                        @Cast("float16*") ShortPointer z,
                                        IntPointer zShapeInfo,
                                        @Cast("float16*") ShortPointer scalars,
                                        @Cast("float16*") ShortPointer extraParams,
                                        IntPointer dimension,
                                        int dimensionLength
    );

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
                                       FloatPointer x,
                                       IntPointer xShapeInfo,
                                       FloatPointer results,
                                       IntPointer resultShapeInfo,
                                       float scalar,
                                       FloatPointer extraParams);

    public native void execScalarHalf(PointerPointer extraPointers, int opNum,
                                       @Cast("float16*") ShortPointer x,
                                       IntPointer xShapeInfo,
                                       @Cast("float16*") ShortPointer results,
                                       IntPointer resultShapeInfo,
                                       float scalar,
                                       @Cast("float16*") ShortPointer extraParams);


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
                                       FloatPointer x,
                                       IntPointer xShapeInfo,
                                       FloatPointer results,
                                       IntPointer resultShapeInfo,
                                       float scalar,
                                       FloatPointer extraParams,
                                       IntPointer xIndexes,
                                       IntPointer resultIndexes);
    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public native float   execSummaryStatsScalarFloat(PointerPointer extraPointers, int opNum,FloatPointer x,
                                                      IntPointer xShapeInfo,
                                                      FloatPointer extraParams,boolean biasCorrected);

    public native float   execSummaryStatsScalarHalf(PointerPointer extraPointers, int opNum,@Cast("float16*") ShortPointer x,
                                                      IntPointer xShapeInfo,
                                                      @Cast("float16*") ShortPointer extraParams,boolean biasCorrected);
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
                                               FloatPointer x,
                                               IntPointer xShapeInfo,
                                               FloatPointer extraParams,
                                               FloatPointer results,
                                               IntPointer resultShapeInfo,boolean biasCorrected);

    public native void   execSummaryStatsHalf(PointerPointer extraPointers, int opNum,
                                               @Cast("float16*") ShortPointer x,
                                               IntPointer xShapeInfo,
                                               @Cast("float16*") ShortPointer extraParams,
                                               @Cast("float16*") ShortPointer results,
                                               IntPointer resultShapeInfo,boolean biasCorrected);
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
    public native void   execSummaryStatsFloat(PointerPointer extraPointers, int opNum,FloatPointer x,
                                               IntPointer xShapeInfo,
                                               FloatPointer extraParams,
                                               FloatPointer results,
                                               IntPointer resultShapeInfoBuffer,
                                               IntPointer dimension, int dimensionLength,boolean biasCorrected);

    public native void   execSummaryStatsHalf(PointerPointer extraPointers, int opNum,@Cast("float16*") ShortPointer x,
                                               IntPointer xShapeInfo,
                                               @Cast("float16*") ShortPointer extraParams,
                                               @Cast("float16*") ShortPointer results,
                                               IntPointer resultShapeInfoBuffer,
                                               IntPointer dimension, int dimensionLength,boolean biasCorrected);
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
                                            FloatPointer dx,
                                            int xStride,
                                            FloatPointer results,
                                            int resultStride,
                                            FloatPointer extraParams, long n);

    public native void   execTransformHalf(PointerPointer extraPointers, int opNum,
                                            @Cast("float16*") ShortPointer dx,
                                            int xStride,
                                            @Cast("float16*") ShortPointer results,
                                            int resultStride,
                                            @Cast("float16*") ShortPointer extraParams, long n);

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
            FloatPointer dx,
            IntPointer xShapeInfo,
            FloatPointer results,
            IntPointer resultShapeInfo,
            FloatPointer extraParams);

    public native void   execTransformHalf(
            PointerPointer extraPointers,
            int opNum,
            @Cast("float16*") ShortPointer dx,
            IntPointer xShapeInfo,
            @Cast("float16*") ShortPointer results,
            IntPointer resultShapeInfo,
            @Cast("float16*") ShortPointer extraParams);

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
                                            FloatPointer dx,
                                            IntPointer xShapeInfo,
                                            FloatPointer results,
                                            IntPointer resultShapeInfo,
                                            FloatPointer extraParams,
                                            IntPointer xIndexes,
                                            IntPointer resultIndexes);

    public native void   execTransformHalf(PointerPointer extraPointers,
                                            int opNum,
                                            @Cast("float16*") ShortPointer dx,
                                            IntPointer xShapeInfo,
                                            @Cast("float16*") ShortPointer results,
                                            IntPointer resultShapeInfo,
                                            @Cast("float16*") ShortPointer extraParams,
                                            IntPointer xIndexes,
                                            IntPointer resultIndexes);


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
            FloatPointer results,
            IntPointer resultShapeInfo,
            FloatPointer input,
            IntPointer inputShapeInfo);

    public native void flattenHalf(
            PointerPointer extraPointers,
            int offset,
            char order,
            @Cast("float16*") ShortPointer results,
            IntPointer resultShapeInfo,
            @Cast("float16*") ShortPointer input,
            IntPointer inputShapeInfo);


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
                                     DoublePointer results,
                                     IntPointer resultShapeInfo,
                                     DoublePointer input,
                                     IntPointer inputShapeInfo);

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
            DoublePointer results,
            IntPointer resultShapeInfo,
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
            FloatPointer results,
            IntPointer resultShapeInfo,
            PointerPointer tadPointers,
            PointerPointer tadOffsets);

    public native void concatHalf(
            PointerPointer extraPointers,
            int dimension,
            int numArrays,
            PointerPointer data,
            PointerPointer inputShapeInfo,
            @Cast("float16*") ShortPointer results,
            IntPointer resultShapeInfo,
            PointerPointer tadPointers,
            PointerPointer tadOffsets);

    /**
     * Gets the maximum number of open mp threads
     * @return
     */
    public native int ompGetMaxThreads();

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
     * Sets the minimal number of openmp threads for variative methods
     * @param threads
     */
    public native void setOmpMinThreads(int threads);

    /**
     * NEVER EVER USE THIS METHOD OUTSIDE OF  CUDA
     */
    public native void initializeDevicesAndFunctions();

    public synchronized native Pointer mallocHost(long memorySize, int flags);

    public synchronized native Pointer mallocDevice(long memorySize, Pointer ptrToDeviceId, int flags);

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

    public native int getDevice();

    public native int streamSynchronize(Pointer stream);

    public native int eventSynchronize(Pointer event);

    public native long getDeviceFreeMemory(Pointer ptrToDeviceId);

    public native long getDeviceTotalMemory(Pointer ptrToDeviceId);

    public native int getDeviceMajor(Pointer ptrToDeviceId);

    public native int getDeviceMinor(Pointer ptrToDeviceId);

    public native String getDeviceName(Pointer ptrToDeviceId);

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

    public native void tadOnlyShapeInfo(IntPointer shapeInfo, IntPointer dimension, int dimensionLength, IntPointer targetBuffer, IntPointer offsetsBuffer);

    ///////////////

    public native void pullRowsFloat(PointerPointer extraPointers, FloatPointer x, IntPointer xShapeInfo, FloatPointer z, IntPointer zShapeInfo, int n, IntPointer indexes, IntPointer tadShapeInfo, IntPointer tadOffsets);

    public native void pullRowsDouble(PointerPointer extraPointers, DoublePointer x, IntPointer xShapeInfo, DoublePointer z, IntPointer zShapeInfo, int n, IntPointer indexes, IntPointer tadShapeInfo, IntPointer tadOffsets);

    public native void pullRowsHalf(PointerPointer extraPointers, @Cast("float16*") ShortPointer x, IntPointer xShapeInfo, @Cast("float16*") ShortPointer z, IntPointer zShapeInfo, int n, IntPointer indexes, IntPointer tadShapeInfo, IntPointer tadOffsets);


    ///////////////////////

    public native void averageHalf(PointerPointer extraPointers, PointerPointer x, @Cast("float16*") ShortPointer z, int n, long length, boolean propagate);

    public native void averageFloat(PointerPointer extraPointers, PointerPointer x, FloatPointer z, int n, long length, boolean propagate);

    public native void averageDouble(PointerPointer extraPointers, PointerPointer x, DoublePointer z, int n, long length, boolean propagate);

    ///////////////////////

    public native void enableP2P(boolean reallyEnable);

    public native void checkP2P();

    public native boolean isP2PAvailable();

    //

    public native void shuffleDouble(PointerPointer extraPointers, PointerPointer x, PointerPointer xShapeInfo, PointerPointer z, PointerPointer zShapeInfo, int N, IntPointer shuffleMap, PointerPointer tadShapeInfo, PointerPointer tadOffsets);

    public native void shuffleFloat(PointerPointer extraPointers, PointerPointer x, PointerPointer xShapeInfo, PointerPointer z, PointerPointer zShapeInfo, int N, IntPointer shuffleMap, PointerPointer tadShapeInfo, PointerPointer tadOffsets);

    public native void shuffleHalf(PointerPointer extraPointers, PointerPointer x, PointerPointer xShapeInfo, PointerPointer z, PointerPointer zShapeInfo, int N, IntPointer shuffleMap, PointerPointer tadShapeInfo, PointerPointer tadOffsets);

    // type conversion

    public native void convertTypes(PointerPointer extras, int srcType, Pointer x, long N, int dstType, Pointer z);

    public native boolean isExperimentalEnabled();

    // GridOps



    // MetaOps
    public native void execMetaPredicateStridedFloat(PointerPointer extras, int opTypeA, int opNumA, int opTypeB, int opNumB,
                                                         long N,
                                                         FloatPointer dx,
                                                         int xStride,
                                                         FloatPointer dy,
                                                         int yStride,
                                                         FloatPointer dz,
                                                         int zStride,
                                                         FloatPointer extraA,
                                                         FloatPointer extraB,
                                                         float scalarA,
                                                         float scalarB);

    public native void execMetaPredicateStridedDouble(PointerPointer extras, int opTypeA, int opNumA, int opTypeB, int opNumB,
                                                     long N,
                                                     DoublePointer dx,
                                                     int xStride,
                                                     DoublePointer dy,
                                                     int yStride,
                                                     DoublePointer dz,
                                                     int zStride,
                                                     DoublePointer extraA,
                                                     DoublePointer extraB,
                                                     double scalarA,
                                                     double scalarB);

    public native void execMetaPredicateStridedHalf(PointerPointer extras, int opTypeA, int opNumA, int opTypeB, int opNumB,
                                                     long N,
                                                     @Cast("float16*") ShortPointer dx,
                                                     int xStride,
                                                     @Cast("float16*") ShortPointer dy,
                                                     int yStride,
                                                     @Cast("float16*") ShortPointer dz,
                                                     int zStride,
                                                     @Cast("float16*") ShortPointer extraA,
                                                     @Cast("float16*") ShortPointer extraB,
                                                     float scalarA,
                                                     float scalarB);

    public native void execMetaPredicateShapeFloat(PointerPointer extras, int opTypeA, int opNumA, int opTypeB, int opNumB,
                                                     long N,
                                                     FloatPointer dx,
                                                     IntPointer xShape,
                                                     FloatPointer dy,
                                                     IntPointer yShape,
                                                     FloatPointer dz,
                                                     IntPointer zShape,
                                                     FloatPointer extraA,
                                                     FloatPointer extraB,
                                                     float scalarA,
                                                     float scalarB);

    public native void execMetaPredicateShapeDouble(PointerPointer extras, int opTypeA, int opNumA, int opTypeB, int opNumB,
                                                   long N,
                                                   DoublePointer dx,
                                                   IntPointer xShape,
                                                   DoublePointer dy,
                                                   IntPointer yShape,
                                                   DoublePointer dz,
                                                   IntPointer zShape,
                                                   DoublePointer extraA,
                                                   DoublePointer extraB,
                                                   double scalarA,
                                                   double scalarB);

    public native void execMetaPredicateShapeHalf(PointerPointer extras, int opTypeA, int opNumA, int opTypeB, int opNumB,
                                                   long N,
                                                   @Cast("float16*") ShortPointer dx,
                                                   IntPointer xShape,
                                                   @Cast("float16*") ShortPointer dy,
                                                   IntPointer yShape,
                                                   @Cast("float16*") ShortPointer dz,
                                                   IntPointer zShape,
                                                   @Cast("float16*") ShortPointer extraA,
                                                   @Cast("float16*") ShortPointer extraB,
                                                   float scalarA,
                                                   float scalarB);

    public native void execMetaPredicateReduceFloat(PointerPointer extras, int opTypeA, int opNumA, int opTypeB, int opNumB,
                                                    FloatPointer dx,
                                                    IntPointer xShapeInfo,
                                                    FloatPointer dy,
                                                    IntPointer yShapeInfo,
                                                    FloatPointer dz,
                                                    IntPointer zShapeInfo,
                                                    IntPointer dimension,
                                                    int dimensionLength,
                                                    IntPointer tadShapeInfo,
                                                    IntPointer tadOffsets,
                                                    FloatPointer extraA,
                                                    FloatPointer extraB,
                                                    float scalarA,
                                                    float scalarB,
                                                    boolean scalarReturned);


    /////////////////////////

    public native void execAggregateFloat(PointerPointer extras,int opNum,
                                          @Cast("float **") PointerPointer arguments,
                                          int numArguments,
                                          @Cast("int **") PointerPointer shapes,
                                          int numShapes,
                                          IntPointer indexArguments,
                                          int numIndexArguments,
                                          @Cast("int **") PointerPointer intArrays,
                                          int numIntArrays,
                                          FloatPointer realArguments,
                                          int numRealArguments);

    public native void execAggregateBatchFloat(PointerPointer extras, int numAggregates, int opNum, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, Pointer ptrToArguments);
}
