package org.nd4j.linalg.jcublas.util;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.ScalarOp;

import static jcuda.driver.JCudaDriver.cuMemAlloc;

/**
 * Various methods for pointer based methods (mainly for the jcuda executioner)
 *
 * @author Adam Gibson
 */
public class PointerUtil {


    //convert an object array to doubles
    public static double[] toDoubles(Object[] extraArgs) {
        double[] ret = new double[extraArgs.length];
        for (int i = 0; i < extraArgs.length; i++) {
            ret[i] = Double.valueOf(extraArgs[i].toString());
        }

        return ret;
    }


    //convert a float array to floats
    public static float[] toFloats(Object[] extraArgs) {
        float[] ret = new float[extraArgs.length];
        for (int i = 0; i < extraArgs.length; i++) {
            ret[i] = Float.valueOf(extraArgs[i].toString());
        }

        return ret;
    }


    /**
     * Compute the number of blocks that should be used for the
     * given input size and limits
     *
     * @param n          The input size
     * @param maxBlocks  The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of blocks
     */
    public static int getNumBlocks(int n, int maxBlocks, int maxThreads) {
        int blocks;
        int threads = getNumThreads(n, maxThreads);
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
        blocks = Math.min(maxBlocks, blocks);
        return blocks;
    }

    /**
     * Compute the number of threads that should be used for the
     * given input size and limits
     *
     * @param n          The input size
     * @param maxThreads The maximum number of threads
     * @return The number of threads
     */
    public static int getNumThreads(int n, int maxThreads) {
        return (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
    }

    /**
     * Returns the power of 2 that is equal to or greater than x
     *
     * @param x The input
     * @return The next power of 2
     */
    public static int nextPow2(int x) {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    /**
     * Construct and allocate a device pointer
     *
     * @param length the length of the pointer
     * @param dType  the data type to choose
     * @return the new pointer
     */
    public static CUdeviceptr constructAndAlloc(int length, int dType) {
        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, length * dType == DataBuffer.FLOAT ? Sizeof.FLOAT : Sizeof.DOUBLE);
        return deviceOutput;
    }

    public static int sizeFor(int dataType) {
        return dataType == DataBuffer.DOUBLE ? Sizeof.DOUBLE : Sizeof.FLOAT;
    }


    public static Pointer getPointer(ScalarOp scalarOp) {
        if (scalarOp.scalar() != null) {
            if (scalarOp.x().data().dataType() == DataBuffer.FLOAT)
                return Pointer.to(new float[]{scalarOp.scalar().floatValue()});
            else if (scalarOp.x().data().dataType() == DataBuffer.DOUBLE)
                return Pointer.to(new double[]{scalarOp.scalar().doubleValue()});
        }

        throw new IllegalStateException("Unable to get pointer for scalar operation " + scalarOp);
    }


}
