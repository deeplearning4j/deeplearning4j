package org.nd4j.nativeblas;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.Cast;


/**
 * Native interface for
 * op execution on cpu
 * @author Adam Gibson
 */
public abstract class NativeOps extends Pointer {
    public NativeOps(Pointer p) {
        super(p);
    }

    public static int getCores(int totals) {
        // that's special case for Xeon Phi
        if (totals >= 256)
            return 64;

        int ht_off = totals / 2; // we count off HyperThreading without any excuses
        if (ht_off <= 4)
            return 4; // special case for Intel i5. and nobody likes i3 anyway

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
    public abstract void setTADThreshold(int value);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public abstract double execIndexReduceScalarDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParams);

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
    public abstract void execIndexReduceDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParams, DoublePointer result,
                    IntPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength);

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
    public abstract void execBroadcastDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer y, IntPointer yShapeInfo, DoublePointer result,
                    IntPointer resultShapeInfo, IntPointer dimension, int dimensionLength);



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
    public abstract void execPairwiseTransformDouble(PointerPointer extraPointers, int opNum, DoublePointer dx,
                    int xStride, DoublePointer y, int yStride, DoublePointer result, int resultStride,
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
    public abstract void execPairwiseTransformDouble(PointerPointer extraPointers, int opNum, DoublePointer dx,
                    IntPointer xShapeInfo, DoublePointer y, IntPointer yShapeInfo, DoublePointer result,
                    IntPointer resultShapeInfo, DoublePointer extraParams, IntPointer xIndexes, IntPointer yIndexes,
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
    public abstract void execPairwiseTransformDouble(PointerPointer extraPointers, int opNum, DoublePointer dx,
                    IntPointer xShapeInfo, DoublePointer y, IntPointer yShapeInfo, DoublePointer result,
                    IntPointer resultShapeInfo, DoublePointer extraParams);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public abstract void execReduceDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParams, DoublePointer result, IntPointer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     */
    public abstract void execReduceDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParams, DoublePointer result, IntPointer resultShapeInfo,
                    IntPointer dimension, int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public abstract double execReduceScalarDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParams);

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
    public abstract void execReduce3Double(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParamsVals, DoublePointer y, IntPointer yShapeInfo,
                    DoublePointer result, IntPointer resultShapeInfo);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     */
    public abstract double execReduce3ScalarDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParamsVals, DoublePointer y, IntPointer yShapeInfo);

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
    public abstract void execReduce3Double(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParamsVals, DoublePointer y, IntPointer yShapeInfo,
                    DoublePointer result, IntPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength);

    public abstract void execReduce3AllDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParamsVals, DoublePointer y, IntPointer yShapeInfo,
                    DoublePointer result, IntPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength,
                    IntPointer xTadShape, @Cast("Nd4jIndex *") LongPointer xOffsets, IntPointer yTadShape,
                    @Cast("Nd4jIndex *") LongPointer yOffsets);

    public abstract void execReduce3AllFloat(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer extraParamsVals, FloatPointer y, IntPointer yShapeInfo,
                    FloatPointer result, IntPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength,
                    IntPointer xTadShape, @Cast("Nd4jIndex *") LongPointer xOffsets, IntPointer yTadShape,
                    @Cast("Nd4jIndex *") LongPointer yOffsets);

    public abstract void execReduce3AllHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer extraParamsVals,
                    @Cast("float16*") ShortPointer y, IntPointer yShapeInfo, @Cast("float16*") ShortPointer result,
                    IntPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength, IntPointer xTadShape,
                    @Cast("Nd4jIndex *") LongPointer xOffsets, IntPointer yTadShape,
                    @Cast("Nd4jIndex *") LongPointer yOffsets);

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
    public abstract void execScalarDouble(PointerPointer extraPointers, int opNum, DoublePointer x, int xStride,
                    DoublePointer result, int resultStride, double scalar, DoublePointer extraParams, long n);

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
    public abstract void execScalarDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer result, IntPointer resultShapeInfo, double scalar,
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
    public abstract void execScalarDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer result, IntPointer resultShapeInfo, double scalar,
                    DoublePointer extraParams, long n, IntPointer xIndexes, IntPointer resultIndexes);

    /**
     *  @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param biasCorrected
     */
    public abstract double execSummaryStatsScalarDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParams, boolean biasCorrected);

    /**
     *  @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param result
     * @param resultShapeInfo
     * @param biasCorrected
     */
    public abstract void execSummaryStatsDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParams, DoublePointer result, IntPointer resultShapeInfo,
                    boolean biasCorrected);

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
    public abstract void execSummaryStatsDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer extraParams, DoublePointer result,
                    IntPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength, boolean biasCorrected);

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
    public abstract void execTransformDouble(PointerPointer extraPointers, int opNum, DoublePointer dx, int xStride,
                    DoublePointer result, int resultStride, DoublePointer extraParams, long n);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param result
     * @param resultShapeInfo
     * @param extraParams
     */
    public abstract void execTransformDouble(PointerPointer extraPointers, int opNum, DoublePointer dx,
                    IntPointer xShapeInfo, DoublePointer result, IntPointer resultShapeInfo, DoublePointer extraParams);

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
    public abstract void execTransformDouble(PointerPointer extraPointers, int opNum, DoublePointer dx,
                    IntPointer xShapeInfo, DoublePointer result, IntPointer resultShapeInfo, DoublePointer extraParams,
                    IntPointer xIndexes, IntPointer resultIndexes);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public abstract float execIndexReduceScalarFloat(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public abstract float execIndexReduceScalarHalf(PointerPointer extraPointers, int opNum,
                    @Cast("float16*") ShortPointer x, IntPointer xShapeInfo,
                    @Cast("float16*") ShortPointer extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param results
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execIndexReduceFloat(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer extraParams, FloatPointer results,
                    IntPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param results
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execIndexReduceHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer extraParams,
                    @Cast("float16*") ShortPointer results, IntPointer resultShapeInfoBuffer, IntPointer dimension,
                    int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execBroadcastFloat(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer y, IntPointer yShapeInfo, FloatPointer results,
                    IntPointer resultShapeInfo, IntPointer dimension, int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execBroadcastHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer y, IntPointer yShapeInfo,
                    @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo, IntPointer dimension,
                    int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xStride
     * @param y
     * @param yStride
     * @param results
     * @param resultStride
     * @param extraParams
     * @param n
     */
    public abstract void execPairwiseTransformFloat(PointerPointer extraPointers, int opNum, FloatPointer dx,
                    int xStride, FloatPointer y, int yStride, FloatPointer results, int resultStride,
                    FloatPointer extraParams, long n);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xStride
     * @param y
     * @param yStride
     * @param results
     * @param resultStride
     * @param extraParams
     * @param n
     */
    public abstract void execPairwiseTransformHalf(PointerPointer extraPointers, int opNum,
                    @Cast("float16*") ShortPointer dx, int xStride, @Cast("float16*") ShortPointer y, int yStride,
                    @Cast("float16*") ShortPointer results, int resultStride,
                    @Cast("float16*") ShortPointer extraParams, long n);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    public abstract void execPairwiseTransformFloat(PointerPointer extraPointers, int opNum, FloatPointer dx,
                    IntPointer xShapeInfo, FloatPointer y, IntPointer yShapeInfo, FloatPointer results,
                    IntPointer resultShapeInfo, FloatPointer extraParams, IntPointer xIndexes, IntPointer yIndexes,
                    IntPointer resultIndexes);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param yIndexes
     * @param resultIndexes
     */
    public abstract void execPairwiseTransformHalf(PointerPointer extraPointers, int opNum,
                    @Cast("float16*") ShortPointer dx, IntPointer xShapeInfo, @Cast("float16*") ShortPointer y,
                    IntPointer yShapeInfo, @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo,
                    @Cast("float16*") ShortPointer extraParams, IntPointer xIndexes, IntPointer yIndexes,
                    IntPointer resultIndexes);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param extraParams
     */
    public abstract void execPairwiseTransformFloat(PointerPointer extraPointers, int opNum, FloatPointer dx,
                    IntPointer xShapeInfo, FloatPointer y, IntPointer yShapeInfo, FloatPointer results,
                    IntPointer resultShapeInfo, FloatPointer extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param y
     * @param yShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param extraParams
     */
    public abstract void execPairwiseTransformHalf(PointerPointer extraPointers, int opNum,
                    @Cast("float16*") ShortPointer dx, IntPointer xShapeInfo, @Cast("float16*") ShortPointer y,
                    IntPointer yShapeInfo, @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo,
                    @Cast("float16*") ShortPointer extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param results
     * @param resultShapeInfo
     */
    public abstract void execReduceFloat(PointerPointer extraPointers, int opNum, FloatPointer x, IntPointer xShapeInfo,
                    FloatPointer extraParams, FloatPointer results, IntPointer resultShapeInfo);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param results
     * @param resultShapeInfo
     */
    public abstract void execReduceHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer extraParams,
                    @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param results
     * @param resultShapeInfo
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execReduceFloat(PointerPointer extraPointers, int opNum, FloatPointer x, IntPointer xShapeInfo,
                    FloatPointer extraParams, FloatPointer results, IntPointer resultShapeInfo, IntPointer dimension,
                    int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param results
     * @param resultShapeInfo
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execReduceHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer extraParams,
                    @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo, IntPointer dimension,
                    int dimensionLength);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @return
     */
    public abstract float execReduceScalarFloat(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer extraParams);


    public abstract float execReduceScalarHalf(PointerPointer extraPointers, int opNum,
                    @Cast("float16*") ShortPointer x, IntPointer xShapeInfo,
                    @Cast("float16*") ShortPointer extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param results
     * @param resultShapeInfo
     */
    public abstract void execReduce3Float(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer extraParamsVals, FloatPointer y, IntPointer yShapeInfo,
                    FloatPointer results, IntPointer resultShapeInfo);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param results
     * @param resultShapeInfo
     */
    public abstract void execReduce3Half(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer extraParamsVals,
                    @Cast("float16*") ShortPointer y, IntPointer yShapeInfo, @Cast("float16*") ShortPointer results,
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
    public abstract float execReduce3ScalarFloat(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer extraParamsVals, FloatPointer y, IntPointer yShapeInfo);

    public abstract float execReduce3ScalarHalf(PointerPointer extraPointers, int opNum,
                    @Cast("float16*") ShortPointer x, IntPointer xShapeInfo,
                    @Cast("float16*") ShortPointer extraParamsVals, @Cast("float16*") ShortPointer y,
                    IntPointer yShapeInfo);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param results
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execReduce3Float(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer extraParamsVals, FloatPointer y, IntPointer yShapeInfo,
                    FloatPointer results, IntPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParamsVals
     * @param y
     * @param yShapeInfo
     * @param results
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     */
    public abstract void execReduce3Half(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer extraParamsVals,
                    @Cast("float16*") ShortPointer y, IntPointer yShapeInfo, @Cast("float16*") ShortPointer results,
                    IntPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xStride
     * @param results
     * @param resultStride
     * @param scalar
     * @param extraParams
     * @param n
     */
    public abstract void execScalarFloat(PointerPointer extraPointers, int opNum, FloatPointer x, int xStride,
                    FloatPointer results, int resultStride, float scalar, FloatPointer extraParams, long n);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xStride
     * @param results
     * @param resultStride
     * @param scalar
     * @param extraParams
     * @param n
     */
    public abstract void execScalarHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    int xStride, @Cast("float16*") ShortPointer results, int resultStride, float scalar,
                    @Cast("float16*") ShortPointer extraParams, long n);

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
    public abstract void execScalarFloat(PointerPointer extraPointers, int opNum, FloatPointer x, IntPointer xShapeInfo,
                    FloatPointer z, IntPointer zShapeInfo, FloatPointer scalars, FloatPointer extraParams,
                    IntPointer dimension, int dimensionLength);

    public abstract void execScalarDouble(PointerPointer extraPointers, int opNum, DoublePointer x,
                    IntPointer xShapeInfo, DoublePointer z, IntPointer zShapeInfo, DoublePointer scalars,
                    DoublePointer extraParams, IntPointer dimension, int dimensionLength);

    public abstract void execScalarHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer z, IntPointer zShapeInfo,
                    @Cast("float16*") ShortPointer scalars, @Cast("float16*") ShortPointer extraParams,
                    IntPointer dimension, int dimensionLength);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     */
    public abstract void execScalarFloat(PointerPointer extraPointers, int opNum, FloatPointer x, IntPointer xShapeInfo,
                    FloatPointer results, IntPointer resultShapeInfo, float scalar, FloatPointer extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     */
    public abstract void execScalarHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo,
                    float scalar, @Cast("float16*") ShortPointer extraParams);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param scalar
     * @param extraParams
     * @param xIndexes
     * @param resultIndexes
     */
    public abstract void execScalarFloat(PointerPointer extraPointers, int opNum, FloatPointer x, IntPointer xShapeInfo,
                    FloatPointer results, IntPointer resultShapeInfo, float scalar, FloatPointer extraParams,
                    IntPointer xIndexes, IntPointer resultIndexes);

    /**
     *
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     */
    public abstract float execSummaryStatsScalarFloat(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer extraParams, boolean biasCorrected);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param biasCorrected
     * @return
     */
    public abstract float execSummaryStatsScalarHalf(PointerPointer extraPointers, int opNum,
                    @Cast("float16*") ShortPointer x, IntPointer xShapeInfo, @Cast("float16*") ShortPointer extraParams,
                    boolean biasCorrected);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param results
     * @param resultShapeInfo
     * @param biasCorrected
     */
    public abstract void execSummaryStatsFloat(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer extraParams, FloatPointer results, IntPointer resultShapeInfo,
                    boolean biasCorrected);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param results
     * @param resultShapeInfo
     * @param biasCorrected
     */
    public abstract void execSummaryStatsHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer extraParams,
                    @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo, boolean biasCorrected);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param results
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     * @param biasCorrected
     */
    public abstract void execSummaryStatsFloat(PointerPointer extraPointers, int opNum, FloatPointer x,
                    IntPointer xShapeInfo, FloatPointer extraParams, FloatPointer results,
                    IntPointer resultShapeInfoBuffer, IntPointer dimension, int dimensionLength, boolean biasCorrected);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param x
     * @param xShapeInfo
     * @param extraParams
     * @param results
     * @param resultShapeInfoBuffer
     * @param dimension
     * @param dimensionLength
     * @param biasCorrected
     */
    public abstract void execSummaryStatsHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer extraParams,
                    @Cast("float16*") ShortPointer results, IntPointer resultShapeInfoBuffer, IntPointer dimension,
                    int dimensionLength, boolean biasCorrected);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xStride
     * @param results
     * @param resultStride
     * @param extraParams
     * @param n
     */
    public abstract void execTransformFloat(PointerPointer extraPointers, int opNum, FloatPointer dx, int xStride,
                    FloatPointer results, int resultStride, FloatPointer extraParams, long n);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xStride
     * @param results
     * @param resultStride
     * @param extraParams
     * @param n
     */
    public abstract void execTransformHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer dx,
                    int xStride, @Cast("float16*") ShortPointer results, int resultStride,
                    @Cast("float16*") ShortPointer extraParams, long n);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param extraParams
     */
    public abstract void execTransformFloat(PointerPointer extraPointers, int opNum, FloatPointer dx,
                    IntPointer xShapeInfo, FloatPointer results, IntPointer resultShapeInfo, FloatPointer extraParams);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param extraParams
     */
    public abstract void execTransformHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer dx,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo,
                    @Cast("float16*") ShortPointer extraParams);

    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param resultIndexes
     */
    public abstract void execTransformFloat(PointerPointer extraPointers, int opNum, FloatPointer dx,
                    IntPointer xShapeInfo, FloatPointer results, IntPointer resultShapeInfo, FloatPointer extraParams,
                    IntPointer xIndexes, IntPointer resultIndexes);


    /**
     *
     * @param extraPointers
     * @param opNum
     * @param dx
     * @param xShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param extraParams
     * @param xIndexes
     * @param resultIndexes
     */
    public abstract void execTransformHalf(PointerPointer extraPointers, int opNum, @Cast("float16*") ShortPointer dx,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo,
                    @Cast("float16*") ShortPointer extraParams, IntPointer xIndexes, IntPointer resultIndexes);


    /**
     *
     * @param extraPointers
     * @param offset
     * @param order
     * @param results
     * @param resultShapeInfo
     * @param input
     * @param inputShapeInfo
     */
    public abstract void flattenFloat(PointerPointer extraPointers, int offset, char order, FloatPointer results,
                    IntPointer resultShapeInfo, FloatPointer input, IntPointer inputShapeInfo);

    public abstract void flattenHalf(PointerPointer extraPointers, int offset, char order,
                    @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo,
                    @Cast("float16*") ShortPointer input, IntPointer inputShapeInfo);


    /**
     *
     * @param extraPointers
     * @param offset
     * @param order
     * @param results
     * @param resultShapeInfo
     * @param input
     * @param inputShapeInfo
     */
    public abstract void flattenDouble(PointerPointer extraPointers, int offset, char order, DoublePointer results,
                    IntPointer resultShapeInfo, DoublePointer input, IntPointer inputShapeInfo);

    /**
     *
     * @param extraPointers
     * @param dimension
     * @param numArrays
     * @param data
     * @param inputShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param tadPointers
     * @param tadOffsets
     */
    public abstract void concatDouble(PointerPointer extraPointers, int dimension, int numArrays, PointerPointer data,
                    PointerPointer inputShapeInfo, DoublePointer results, IntPointer resultShapeInfo,
                    PointerPointer tadPointers, PointerPointer tadOffsets);

    /**
     *
     * @param extraPointers
     * @param dimension
     * @param numArrays
     * @param data
     * @param inputShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param tadPointers
     * @param tadOffsets
     */
    public abstract void concatFloat(PointerPointer extraPointers, int dimension, int numArrays, PointerPointer data,
                    PointerPointer inputShapeInfo, FloatPointer results, IntPointer resultShapeInfo,
                    PointerPointer tadPointers, PointerPointer tadOffsets);


    /**
     *
     * @param extraPointers
     * @param dimension
     * @param numArrays
     * @param data
     * @param inputShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param tadPointers
     * @param tadOffsets
     */
    public abstract void concatHalf(PointerPointer extraPointers, int dimension, int numArrays, PointerPointer data,
                    PointerPointer inputShapeInfo, @Cast("float16*") ShortPointer results, IntPointer resultShapeInfo,
                    PointerPointer tadPointers, PointerPointer tadOffsets);


    public abstract void specialConcatDouble(PointerPointer extraPointers, int dimension, int numArrays,
                    PointerPointer data, PointerPointer inputShapeInfo, DoublePointer results,
                    IntPointer resultShapeInfo, PointerPointer tadPointers, PointerPointer tadOffsets);

    /**
     *
     * @param extraPointers
     * @param dimension
     * @param numArrays
     * @param data
     * @param inputShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param tadPointers
     * @param tadOffsets
     */
    public abstract void specialConcatFloat(PointerPointer extraPointers, int dimension, int numArrays,
                    PointerPointer data, PointerPointer inputShapeInfo, FloatPointer results,
                    IntPointer resultShapeInfo, PointerPointer tadPointers, PointerPointer tadOffsets);


    /**
     *
     * @param extraPointers
     * @param dimension
     * @param numArrays
     * @param data
     * @param inputShapeInfo
     * @param results
     * @param resultShapeInfo
     * @param tadPointers
     * @param tadOffsets
     */
    public abstract void specialConcatHalf(PointerPointer extraPointers, int dimension, int numArrays,
                    PointerPointer data, PointerPointer inputShapeInfo, @Cast("float16*") ShortPointer results,
                    IntPointer resultShapeInfo, PointerPointer tadPointers, PointerPointer tadOffsets);

    /**
     * Gets the maximum number of open mp threads
     * @return
     */
    public abstract int ompGetMaxThreads();

    /**
     * Gets the number of open mp threads
     * @return
     */
    public abstract int ompGetNumThreads();

    /**
     * Sets the number of openmp threads
     * @param threads
     */
    public abstract void setOmpNumThreads(int threads);

    /**
     * Sets the minimal number of openmp threads for variative methods
     * @param threads
     */
    public abstract void setOmpMinThreads(int threads);

    /**
     * NEVER EVER USE THIS METHOD OUTSIDE OF  CUDA
     */
    public abstract void initializeDevicesAndFunctions();

    public abstract void initializeFunctions(PointerPointer functions);

    public abstract Pointer mallocHost(long memorySize, int flags);

    public abstract Pointer mallocDevice(long memorySize, Pointer ptrToDeviceId, int flags);

    public abstract int freeHost(Pointer pointer);

    public abstract int freeDevice(Pointer pointer, Pointer deviceId);

    public abstract Pointer createContext();

    public abstract Pointer createStream();

    public abstract Pointer createEvent();

    public abstract int registerEvent(Pointer event, Pointer stream);

    public abstract int destroyEvent(Pointer event);

    public abstract int setDevice(Pointer ptrToDeviceId);

    public abstract int getDevice();

    public abstract int streamSynchronize(Pointer stream);

    public abstract int eventSynchronize(Pointer event);

    public abstract long getDeviceFreeMemory(Pointer ptrToDeviceId);

    public abstract long getDeviceTotalMemory(Pointer ptrToDeviceId);

    public abstract int getDeviceMajor(Pointer ptrToDeviceId);

    public abstract int getDeviceMinor(Pointer ptrToDeviceId);

    public abstract String getDeviceName(Pointer ptrToDeviceId);

    public abstract int memcpy(Pointer dst, Pointer src, long size, int flags, Pointer reserved);

    public abstract int memcpyAsync(Pointer dst, Pointer src, long size, int flags, Pointer reserved);

    public abstract int memcpyConstantAsync(long dst, Pointer src, long size, int flags, Pointer reserved);

    public abstract int memset(Pointer dst, int value, long size, int flags, Pointer reserved);

    public abstract int memsetAsync(Pointer dst, int value, long size, int flags, Pointer reserved);

    public abstract Pointer getConstantSpace();

    public abstract int getAvailableDevices();

    public abstract void enableDebugMode(boolean reallyEnable);

    public abstract void enableVerboseMode(boolean reallyEnable);

    public abstract void setGridLimit(int gridSize);

    public abstract void tadOnlyShapeInfo(IntPointer shapeInfo, IntPointer dimension, int dimensionLength,
                    IntPointer targetBuffer, @Cast("Nd4jIndex *") LongPointer offsetsBuffer);

    ///////////////

    public abstract void pullRowsFloat(PointerPointer extraPointers, FloatPointer x, IntPointer xShapeInfo,
                    FloatPointer z, IntPointer zShapeInfo, int n, IntPointer indexes, IntPointer tadShapeInfo,
                    @Cast("Nd4jIndex *") LongPointer tadOffsets, IntPointer zTadShapeInfo,
                    @Cast("Nd4jIndex *") LongPointer zTadOffsets);

    public abstract void pullRowsDouble(PointerPointer extraPointers, DoublePointer x, IntPointer xShapeInfo,
                    DoublePointer z, IntPointer zShapeInfo, int n, IntPointer indexes, IntPointer tadShapeInfo,
                    @Cast("Nd4jIndex *") LongPointer tadOffsets, IntPointer zTadShapeInfo,
                    @Cast("Nd4jIndex *") LongPointer zTadOffsets);

    public abstract void pullRowsHalf(PointerPointer extraPointers, @Cast("float16*") ShortPointer x,
                    IntPointer xShapeInfo, @Cast("float16*") ShortPointer z, IntPointer zShapeInfo, int n,
                    IntPointer indexes, IntPointer tadShapeInfo, @Cast("Nd4jIndex *") LongPointer tadOffsets,
                    IntPointer zTadShapeInfo, @Cast("Nd4jIndex *") LongPointer zTadOffsets);


    ///////////////////////

    public abstract void averageHalf(PointerPointer extraPointers, PointerPointer x, @Cast("float16*") ShortPointer z,
                    int n, long length, boolean propagate);

    public abstract void averageFloat(PointerPointer extraPointers, PointerPointer x, FloatPointer z, int n,
                    long length, boolean propagate);

    public abstract void averageDouble(PointerPointer extraPointers, PointerPointer x, DoublePointer z, int n,
                    long length, boolean propagate);

    ///////////////////////

    public abstract void accumulateHalf(PointerPointer extraPointers, PointerPointer x,
                    @Cast("float16*") ShortPointer z, int n, long length);

    public abstract void accumulateFloat(PointerPointer extraPointers, PointerPointer x, FloatPointer z, int n,
                    long length);

    public abstract void accumulateDouble(PointerPointer extraPointers, PointerPointer x, DoublePointer z, int n,
                    long length);

    ///////////////////////

    public abstract void enableP2P(boolean reallyEnable);

    public abstract void checkP2P();

    public abstract boolean isP2PAvailable();

    //

    public abstract void shuffleDouble(PointerPointer extraPointers, PointerPointer x, PointerPointer xShapeInfo,
                    PointerPointer z, PointerPointer zShapeInfo, int N, IntPointer shuffleMap,
                    PointerPointer tadShapeInfo, PointerPointer tadOffsets);

    public abstract void shuffleFloat(PointerPointer extraPointers, PointerPointer x, PointerPointer xShapeInfo,
                    PointerPointer z, PointerPointer zShapeInfo, int N, IntPointer shuffleMap,
                    PointerPointer tadShapeInfo, PointerPointer tadOffsets);

    public abstract void shuffleHalf(PointerPointer extraPointers, PointerPointer x, PointerPointer xShapeInfo,
                    PointerPointer z, PointerPointer zShapeInfo, int N, IntPointer shuffleMap,
                    PointerPointer tadShapeInfo, PointerPointer tadOffsets);

    // opType conversion

    public abstract void convertTypes(PointerPointer extras, int srcType, Pointer x, long N, int dstType, Pointer z);

    public abstract boolean isExperimentalEnabled();

    // GridOps



    // MetaOps
    public abstract void execMetaPredicateStridedFloat(PointerPointer extras, int opTypeA, int opNumA, int opTypeB,
                    int opNumB, long N, FloatPointer dx, int xStride, FloatPointer dy, int yStride, FloatPointer dz,
                    int zStride, FloatPointer extraA, FloatPointer extraB, float scalarA, float scalarB);

    public abstract void execMetaPredicateStridedDouble(PointerPointer extras, int opTypeA, int opNumA, int opTypeB,
                    int opNumB, long N, DoublePointer dx, int xStride, DoublePointer dy, int yStride, DoublePointer dz,
                    int zStride, DoublePointer extraA, DoublePointer extraB, double scalarA, double scalarB);

    public abstract void execMetaPredicateStridedHalf(PointerPointer extras, int opTypeA, int opNumA, int opTypeB,
                    int opNumB, long N, @Cast("float16*") ShortPointer dx, int xStride,
                    @Cast("float16*") ShortPointer dy, int yStride, @Cast("float16*") ShortPointer dz, int zStride,
                    @Cast("float16*") ShortPointer extraA, @Cast("float16*") ShortPointer extraB, float scalarA,
                    float scalarB);

    public abstract void execMetaPredicateShapeFloat(PointerPointer extras, int opTypeA, int opNumA, int opTypeB,
                    int opNumB, long N, FloatPointer dx, IntPointer xShape, FloatPointer dy, IntPointer yShape,
                    FloatPointer dz, IntPointer zShape, FloatPointer extraA, FloatPointer extraB, float scalarA,
                    float scalarB);

    public abstract void execMetaPredicateShapeDouble(PointerPointer extras, int opTypeA, int opNumA, int opTypeB,
                    int opNumB, long N, DoublePointer dx, IntPointer xShape, DoublePointer dy, IntPointer yShape,
                    DoublePointer dz, IntPointer zShape, DoublePointer extraA, DoublePointer extraB, double scalarA,
                    double scalarB);

    public abstract void execMetaPredicateShapeHalf(PointerPointer extras, int opTypeA, int opNumA, int opTypeB,
                    int opNumB, long N, @Cast("float16*") ShortPointer dx, IntPointer xShape,
                    @Cast("float16*") ShortPointer dy, IntPointer yShape, @Cast("float16*") ShortPointer dz,
                    IntPointer zShape, @Cast("float16*") ShortPointer extraA, @Cast("float16*") ShortPointer extraB,
                    float scalarA, float scalarB);

    public abstract void execMetaPredicateReduceFloat(PointerPointer extras, int opTypeA, int opNumA, int opTypeB,
                    int opNumB, FloatPointer dx, IntPointer xShapeInfo, FloatPointer dy, IntPointer yShapeInfo,
                    FloatPointer dz, IntPointer zShapeInfo, IntPointer dimension, int dimensionLength,
                    IntPointer tadShapeInfo, @Cast("Nd4jIndex *") LongPointer tadOffsets, FloatPointer extraA,
                    FloatPointer extraB, float scalarA, float scalarB, boolean scalarReturned);


    /////////////////////////

    public abstract void execAggregateFloat(PointerPointer extras, int opNum,
                    @Cast("float **") PointerPointer arguments, int numArguments, @Cast("int **") PointerPointer shapes,
                    int numShapes, IntPointer indexArguments, int numIndexArguments,
                    @Cast("int **") PointerPointer intArrays, int numIntArrays, FloatPointer realArguments,
                    int numRealArguments);


    public abstract void execAggregateDouble(PointerPointer extras, int opNum,
                    @Cast("double **") PointerPointer arguments, int numArguments,
                    @Cast("int **") PointerPointer shapes, int numShapes, IntPointer indexArguments,
                    int numIndexArguments, @Cast("int **") PointerPointer intArrays, int numIntArrays,
                    DoublePointer realArguments, int numRealArguments);

    public abstract void execAggregateHalf(PointerPointer extras, int opNum,
                    @Cast("float16 **") PointerPointer arguments, int numArguments,
                    @Cast("int **") PointerPointer shapes, int numShapes, IntPointer indexArguments,
                    int numIndexArguments, @Cast("int **") PointerPointer intArrays, int numIntArrays,
                    @Cast("float16*") ShortPointer realArguments, int numRealArguments);

    public abstract void execAggregateBatchFloat(PointerPointer extras, int numAggregates, int opNum, int maxArgs,
                    int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,
                    Pointer ptrToArguments);

    public abstract void execAggregateBatchDouble(PointerPointer extras, int numAggregates, int opNum, int maxArgs,
                    int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,
                    Pointer ptrToArguments);

    public abstract void execAggregateBatchHalf(PointerPointer extras, int numAggregates, int opNum, int maxArgs,
                    int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals,
                    Pointer ptrToArguments);


    public abstract void execRandomFloat(PointerPointer extraPointers, int opNum, Pointer state, FloatPointer z,
                    IntPointer zShapeBuffer, FloatPointer extraArguments);

    public abstract void execRandomFloat(PointerPointer extraPointers, int opNum, Pointer state, FloatPointer x,
                    IntPointer xShapeBuffer, FloatPointer y, IntPointer yShapeBuffer, FloatPointer z,
                    IntPointer zShapeBuffer, FloatPointer extraArguments);

    public abstract void execRandomFloat(PointerPointer extraPointers, int opNum, Pointer state, FloatPointer x,
                    IntPointer xShapeBuffer, FloatPointer z, IntPointer zShapeBuffer, FloatPointer extraArguments);


    public abstract void execRandomDouble(PointerPointer extraPointers, int opNum, Pointer state, DoublePointer z,
                    IntPointer zShapeBuffer, DoublePointer extraArguments);

    public abstract void execRandomDouble(PointerPointer extraPointers, int opNum, Pointer state, DoublePointer x,
                    IntPointer xShapeBuffer, DoublePointer y, IntPointer yShapeBuffer, DoublePointer z,
                    IntPointer zShapeBuffer, DoublePointer extraArguments);

    public abstract void execRandomDouble(PointerPointer extraPointers, int opNum, Pointer state, DoublePointer x,
                    IntPointer xShapeBuffer, DoublePointer z, IntPointer zShapeBuffer, DoublePointer extraArguments);


    public abstract void execRandomHalf(PointerPointer extraPointers, int opNum, Pointer state,
                    @Cast("float16*") ShortPointer z, IntPointer zShapeBuffer,
                    @Cast("float16*") ShortPointer extraArguments);

    public abstract void execRandomHalf(PointerPointer extraPointers, int opNum, Pointer state,
                    @Cast("float16*") ShortPointer x, IntPointer xShapeBuffer, @Cast("float16*") ShortPointer y,
                    IntPointer yShapeBuffer, @Cast("float16*") ShortPointer z, IntPointer zShapeBuffer,
                    @Cast("float16*") ShortPointer extraArguments);

    public abstract void execRandomHalf(PointerPointer extraPointers, int opNum, Pointer state,
                    @Cast("float16*") ShortPointer x, IntPointer xShapeBuffer, @Cast("float16*") ShortPointer z,
                    IntPointer zShapeBuffer, @Cast("float16*") ShortPointer extraArguments);


    public abstract Pointer initRandom(PointerPointer extraPointers, long seed, long numberOfElements,
                    Pointer pointerToBuffer);

    public abstract void refreshBuffer(PointerPointer extraPointers, long seed, Pointer pointer);

    public abstract void reSeedBuffer(PointerPointer extraPointers, long seed, Pointer pointer);

    public abstract void destroyRandom(Pointer pointer);


    /**
     *
     * @param npyArray
     * @return
     */
    public abstract Pointer dataPointForNumpy(Pointer npyArray);

    /**
     *
     * @param npyArray
     * @return
     */
    public abstract Pointer shapeBufferForNumpy(Pointer npyArray);

    /**
     * Thie method releases numpy pointer
     *
     * PLEASE NOTE: This method should be ONLY used if pointer/numpy array was originated from file
     *
     * @param npyArray
     */
    public abstract void releaseNumpy(Pointer npyArray);


    /**
     * Create a numpy array pointer
     * from a file
     * @param path the path to the file
     * @return
     */
    public abstract Pointer numpyFromFile(BytePointer path);



    /**
     * Return the length of a shape buffer
     * based on the pointer
     * @param buffer  the buffer pointer to check
     * @return
     */
    public abstract int lengthForShapeBufferPointer(Pointer buffer);

    /**
     * Calculate the element size
     * for a numpy array
     * @param npyArray the numpy array to get the
     *                 element size for
     * @return the element size for a given array
     */
    public abstract int elementSizeForNpyArray(Pointer npyArray);


    /**
     * The pointer to get the address for
     * @param address the address to get the pointer
     * @return the pointer for the given address
     */
    public abstract Pointer pointerForAddress(long address);


    public abstract void tearDouble(PointerPointer extras, DoublePointer tensor, IntPointer xShapeInfo,
                    PointerPointer targets, IntPointer zShapeInfo, IntPointer tadShapeInfo,
                    @Cast("Nd4jIndex *") LongPointer tadOffsets);

    public abstract void tearFloat(PointerPointer extras, FloatPointer tensor, IntPointer xShapeInfo,
                    PointerPointer targets, IntPointer zShapeInfo, IntPointer tadShapeInfo,
                    @Cast("Nd4jIndex *") LongPointer tadOffsets);

    public abstract void tearHalf(PointerPointer extras, @Cast("float16*") ShortPointer tensor, IntPointer xShapeInfo,
                    PointerPointer targets, IntPointer zShapeInfo, IntPointer tadShapeInfo,
                    @Cast("Nd4jIndex *") LongPointer tadOffsets);


    public abstract long encodeBitmapFloat(PointerPointer extraPointers, FloatPointer dx, long N, IntPointer dz,
                    float threshold);

    public abstract long encodeBitmapDouble(PointerPointer extraPointers, DoublePointer dx, long N, IntPointer dz,
                    float threshold);

    public abstract long encodeBitmapHalf(PointerPointer extraPointers, @Cast("float16*") ShortPointer dx, long N,
                    IntPointer dz, float threshold);


    public abstract void decodeBitmapFloat(PointerPointer extraPointers, Pointer dx, long N, FloatPointer dz);

    public abstract void decodeBitmapDouble(PointerPointer extraPointers, Pointer dx, long N, DoublePointer dz);

    public abstract void decodeBitmapHalf(PointerPointer extraPointers, Pointer dx, long N,
                    @Cast("float16*") ShortPointer dz);



    public abstract void encodeThresholdP1Float(PointerPointer extraPointers, FloatPointer dx, long N, IntPointer dz,
                    float threshold);

    public abstract void encodeThresholdP1Double(PointerPointer extraPointers, DoublePointer dx, long N, IntPointer dz,
                    float threshold);

    public abstract void encodeThresholdP1Half(PointerPointer extraPointers, @Cast("float16*") ShortPointer dx, long N,
                    IntPointer dz, float threshold);


    public abstract void encodeThresholdP2Int(PointerPointer extraPointers, IntPointer dx, long N, IntPointer dz);


    public abstract void encodeThresholdP3Float(PointerPointer extraPointers, FloatPointer dx, IntPointer offsets,
                    long N, IntPointer dz);

    public abstract void encodeThresholdP3Double(PointerPointer extraPointers, DoublePointer dx, IntPointer offsets,
                    long N, IntPointer dz);

    public abstract void encodeThresholdP3Half(PointerPointer extraPointers, @Cast("float16*") ShortPointer dx,
                    IntPointer offsets, long N, IntPointer dz);

    public abstract void decodeThresholdFloat(PointerPointer extraPointers, Pointer dx, long N, FloatPointer dz);

    public abstract void decodeThresholdDouble(PointerPointer extraPointers, Pointer dx, long N, DoublePointer dz);

    public abstract void decodeThresholdHalf(PointerPointer extraPointers, Pointer dx, long N,
                    @Cast("float16*") ShortPointer dz);


    public abstract void sortFloat(PointerPointer extraPointers, FloatPointer dx, IntPointer xShapeInfo,
                    boolean descending);

    public abstract void sortDouble(PointerPointer extraPointers, DoublePointer dx, IntPointer xShapeInfo,
                    boolean descending);

    public abstract void sortHalf(PointerPointer extraPointers, @Cast("float16*") ShortPointer dx,
                    IntPointer xShapeInfo, boolean descending);



    public abstract void sortTadFloat(PointerPointer extraPointers, FloatPointer dx, IntPointer xShapeInfo,
                    IntPointer dimension, int dimensionLength, IntPointer tadShapeInfo,
                    @Cast("Nd4jIndex *") LongPointer tadOffsets, boolean descending);

    public abstract void sortTadDouble(PointerPointer extraPointers, DoublePointer dx, IntPointer xShapeInfo,
                    IntPointer dimension, int dimensionLength, IntPointer tadShapeInfo,
                    @Cast("Nd4jIndex *") LongPointer tadOffsets, boolean descending);

    public abstract void sortTadHalf(PointerPointer extraPointers, @Cast("float16*") ShortPointer dx,
                    IntPointer xShapeInfo, IntPointer dimension, int dimensionLength, IntPointer tadShapeInfo,
                    @Cast("Nd4jIndex *") LongPointer tadOffsets, boolean descending);

    public abstract void sortCooIndicesFloat(PointerPointer extraPointers, IntPointer indices, FloatPointer values,
                    long length, int rank);

    public abstract void sortCooIndicesDouble(PointerPointer extraPointers, IntPointer indices, DoublePointer values,
                    long length, int rank);

    public abstract void sortCooIndicesHalf(PointerPointer extraPointers, IntPointer indices, @Cast("float16*") ShortPointer values, long length, int rank);

    public abstract LongPointer mmapFile(PointerPointer extraPointers, String fileName, long length);

    public abstract void munmapFile(PointerPointer extraPointers, LongPointer ptrMap, long length);

    public abstract Pointer executeFlatGraphFloat(PointerPointer extraPointers, Pointer flatBufferPointer);

    public abstract String getAllCustomOps();

    public abstract int execCustomOpFloat(PointerPointer extraPointers, long opHashCode, PointerPointer inputBuffers, PointerPointer inputShapes, int numInput, PointerPointer outputBuffers, PointerPointer outputShapes, int numOutputs, FloatPointer tArgs, int numTArgs, IntPointer iArgs, int numIArgs, boolean isInplace);
    public abstract int execCustomOpDouble(PointerPointer extraPointers, long opHashCode, PointerPointer inputBuffers, PointerPointer inputShapes, int numInput, PointerPointer outputBuffers, PointerPointer outputShapes, int numOutputs, DoublePointer tArgs, int numTArgs, IntPointer iArgs, int numIArgs, boolean isInplace);
    public abstract int execCustomOpHalf(PointerPointer extraPointers, long opHashCode, PointerPointer inputBuffers, PointerPointer inputShapes, int numInput, PointerPointer outputBuffers, PointerPointer outputShapes, int numOutputs, @Cast("float16*") ShortPointer tArgs, int numTArgs, IntPointer iArgs, int numIArgs, boolean isInplace);

    public abstract PointerPointer calculateOutputShapesFloat(PointerPointer extraPointers, long hash, PointerPointer inputShapes, int numInputShapes, FloatPointer tArgs, int numTArgs, IntPointer iArgs, int numIArgs);
    public abstract PointerPointer calculateOutputShapesHalf(PointerPointer extraPointers, long hash, PointerPointer inputShapes, int numInputShapes, @Cast("float16") ShortPointer tArgs, int numTArgs, IntPointer iArgs, int numIArgs);
    public abstract PointerPointer calculateOutputShapesDouble(PointerPointer extraPointers, long hash, PointerPointer inputShapes, int numInputShapes, DoublePointer tArgs, int numTArgs, IntPointer iArgs, int numIArgs);

    public abstract int registerGraphFloat(PointerPointer extraPointers, long graphId, Pointer flatBufferPointer);
    public abstract int registerGraphDouble(PointerPointer extraPointers, long graphId, Pointer flatBufferPointer);
    public abstract int registerGraphHalf(PointerPointer extraPointers, long graphId, Pointer flatBufferPointer);

    public abstract Pointer executeStoredGraphFloat(PointerPointer extraPointers, long graphId, PointerPointer inputBuffers, PointerPointer inputShapes, IntPointer inputIndices, int numInputs);
    public abstract Pointer executeStoredGraphDouble(PointerPointer extraPointers, long graphId, PointerPointer inputBuffers, PointerPointer inputShapes, IntPointer inputIndices, int numInputs);
    public abstract Pointer executeStoredGraphHalf(PointerPointer extraPointers, long graphId, PointerPointer inputBuffers, PointerPointer inputShapes, IntPointer inputIndices, int numInputs);

    public abstract int unregisterGraph(PointerPointer extraPointers, long graphId);

    public abstract void deleteIntArray(Pointer pointer);
    public abstract void deletePointerArray(Pointer pointer);

    public abstract void deleteVariablesSetFloat(Pointer pointer);
    public abstract void deleteVariablesSetDouble(Pointer pointer);
    public abstract void deleteVariablesSetHalf(Pointer pointer);
}
