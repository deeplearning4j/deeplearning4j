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
}
