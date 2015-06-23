package org.nd4j.linalg.cpu.javacpp;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;

/**
 * @author Adam Gibson
 */
@Platform(include="Loop.h")
public class Loop extends Pointer {
    static { Loader.load(); }

    public Loop() {
        allocate();
    }

    private native void allocate();



    public native void execFloatTransform(
            float[] data
            , int length
            , int offset,
            int resultOffset
            , int stride
            ,int resultStride
            , String  operation,
            float[] otherParams
            , float[] result);

    public native void execFloatTransform(
            float[] data,
            float[] pairData
            , int length
            , int offset,
            int yOffset,
            int resultOffset
            , int stride,
            int yStride
            ,int resultStride
            , String  operation,
            float[] otherParams
            , float[] result);

    public native void execScalarDouble(
            double[] data
            ,double[] result
            ,int length
            ,int offset,
            int resultOffset
            ,int stride
            ,int resultStride
            ,String  operation
            ,double[] otherParams);

    public native void execScalarFloat(
            float[] data
            , float[] result
            ,int length
            ,int offset,
            int resultOffset
            ,int stride
            ,int resultStride
            ,String  operation
            , float[] otherParams);

    public native void execDoubleTransform(
            double[] data
            , int length
            , int offset,
            int resultOffset,
            int stride
            ,int resultStride
            , String  operation,
            double[] otherParams
            ,double[] result);

    public native void execDoubleTransform(
            double[] data,
            double[] pairData
            , int length
            , int offset,
            int yOffset,
            int resultOffset
            , int stride,
            int yStride
            ,int resultStride
            , String  operation,
            double[] otherParams
            , double[] result);

    public native double reduce3(
            double[] data
            , double[] data2
            ,int length
            , int xOffset
            , int yOffset
            ,int xStride
            ,int yStride
            , String  operation,
                   double[] otherParams);

    public native double reduce(
            double[] data
            , int length
            , int offset
            , int stride
            , String  operation,
                  double[] otherParams);

    public native float reduce3Float(
            float[] data,
            float[] data2,
            int length,
            int xOffset,
            int yOffset,
            int xStride,
            int yStride,
            String  operation,
                        float[] otherParams);

    public native float reduceFloat(
            float[] data
            , int length
            , int offset
            , int stride
            , String  operation
            , float[] otherParams);

}
