/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.jcublas.util;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cuComplex;
import jcuda.cuDoubleComplex;
import jcuda.driver.CUdeviceptr;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.ops.ScalarOp;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import static jcuda.driver.JCudaDriver.cuMemAlloc;

/**
 * Various methods for pointer
 * based methods (mainly for the jcuda executioner)
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


    /**
     * Get the pointer for a single complex float
     * @param x the number ot get the pointer for
     * @return the pointer for the given complex number
     */
    public static Pointer getPointer(IComplexDouble x) {
       return getPointer(cuDoubleComplex.cuCmplx(x.realComponent().doubleValue(),x.imaginaryComponent().doubleValue()));
    }
    /**
     * Get the pointer for a single complex float
     * @param x the number ot get the pointer for
     * @return the pointer for the given complex number
     */
    public static Pointer getPointer(IComplexFloat x) {
        return getPointer(cuComplex.cuCmplx(x.realComponent().floatValue(), x.imaginaryComponent().floatValue()));

    }


    /**
     * Get the pointer for a single complex float
     * @param x the number ot get the pointer for
     * @return the pointer for the given complex number
     */
    public static Pointer getPointer(cuDoubleComplex x) {
        ByteBuffer byteBufferx = ByteBuffer.allocateDirect(8 * 2);
        byteBufferx.order(ByteOrder.nativeOrder());
        java.nio.DoubleBuffer floatBufferx = byteBufferx.asDoubleBuffer();
        floatBufferx.put(0,x.x);
        floatBufferx.put(1,x.y);
        return Pointer.to(floatBufferx);
    }
    /**
     * Get the pointer for a single complex float
     * @param x the number ot get the pointer for
     * @return the pointer for the given complex number
     */
    public static Pointer getPointer(cuComplex x) {
        ByteBuffer byteBufferx = ByteBuffer.allocateDirect(4 * 2);
        byteBufferx.order(ByteOrder.nativeOrder());
        FloatBuffer floatBufferx = byteBufferx.asFloatBuffer();
        floatBufferx.put(0,x.x);
        floatBufferx.put(1,x.y);
        return Pointer.to(floatBufferx);
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
    public static CUdeviceptr constructAndAlloc(int length, DataBuffer.Type dType) {
        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, length * (dType == DataBuffer.Type.FLOAT ? Sizeof.FLOAT : Sizeof.DOUBLE));
        return deviceOutput;
    }

    public static int sizeFor(DataBuffer.Type dataType) {
        return dataType == DataBuffer.Type.DOUBLE ? Sizeof.DOUBLE : Sizeof.FLOAT;
    }


    public static Object getPointer(ScalarOp scalarOp) {
        if (scalarOp.scalar() != null) {
            if (scalarOp.x().data().dataType() == DataBuffer.Type.FLOAT)
                return new float[]{scalarOp.scalar().floatValue()};
            else if (scalarOp.x().data().dataType() == DataBuffer.Type.DOUBLE)
                return new double[]{scalarOp.scalar().doubleValue()};
        }

        throw new IllegalStateException("Unable to get pointer for scalar operation " + scalarOp);
    }

    public static Pointer getPointer(double alpha) {
        return Pointer.to(new double[]{alpha});
    }
    public static Pointer getPointer(float alpha) {
        return Pointer.to(new float[]{alpha});
    }
}
