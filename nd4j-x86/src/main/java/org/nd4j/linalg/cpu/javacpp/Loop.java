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


    /**
     * Perform a linear transform
     * over a vector
     * @param data the data to transform
     * @param length the length of the transform
     * @param offset the offset to start at
     * @param stride the stride to iterate at
     * @param operation the operation to perform
     * @param extraParams any extra parameters neccessary for the
     *                    transform (aka 2 for power)
     */
    public native void execFloatTransform(float[] data,int length,int offset,int stride,String operation,float[] extraParams);
    /**
     * Perform a linear transform
     * over a vector
     * @param data the data to transform
     * @param length the length of the transform
     * @param offset the offset to start at
     * @param stride the stride to iterate at
     * @param operation the operation to perform
     * @param extraParams any extra parameters neccessary for the
     *                    transform (aka 2 for power)
     */
    public native void execDoubleTransform(double[] data,int length,int offset,int stride,String operation,double[] extraParams);

    /**
     * A reduction operation involving 2 vectors
     * @param data the x
     * @param data2 the y
     * @param length the length of the operation
     * @param xOffset the offset for x
     * @param yOffset the offset for y
     * @param xStride the increment for x
     * @param yStride the increment for y
     * @param operation the operation to perform
     * @param otherParams extra parameters involved with the operation
     * @return the result of the reduce operation
     */
    public native double reduce3(double[] data, double[] data2,int length, int xOffset, int yOffset,int xStride,int yStride, String operation,
                   double[] otherParams);

    /**
     * A reduction operation (think sum,prod,..)
     * @param data the data to reduce
     * @param length the length of the transform
     * @param offset the offset to start at
     * @param stride the start to increment at
     * @param operation the operation to perform
     * @param otherParams other parameters involved with the
     *                    operation
     * @return the result of the reduction
     */
    public native double reduce(double[] data, int length, int offset, int stride, String operation,
                                double[] otherParams);
    /**
     * A reduction operation involving 2 vectors
     * @param data the x
     * @param data2 the y
     * @param length the length of the operation
     * @param xOffset the offset for x
     * @param yOffset the offset for y
     * @param xStride the increment for x
     * @param yStride the increment for y
     * @param operation the operation to perform
     * @param otherParams extra parameters involved with the operation
     * @return the result of the reduce operation
     */
    public native  float reduce3Float(float[] data, float[] data2,int length, int xOffset, int yOffset,int xStride,int yStride, String operation,
                                      float[] otherParams);

    /**
     * A reduction operation (think sum,prod,..)
     * @param data the data to reduce
     * @param length the length of the transform
     * @param offset the offset to start at
     * @param stride the start to increment at
     * @param operation the operation to perform
     * @param otherParams other parameters involved with the
     *                    operation
     * @return the result of the reduction
     */
    public native float reduceFloat(float[] data, int length, int offset, int stride,String operation,
                                    float[] otherParams);

}
