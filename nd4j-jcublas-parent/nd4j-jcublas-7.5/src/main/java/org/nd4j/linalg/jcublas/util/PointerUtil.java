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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;

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
     * Converts a raw int buffer of the layout:
     * rank
     * shape
     * stride
     * offset
     * element wise stride
     * ordering
     * where shape and stride are both straight int pointers
     *
     * Of note here is that offset will be zero automatically
     * because the offset is handled by the
     * pointer object instead
     *
     */
    public static int[] toShapeInfoBuffer(INDArray arr,int...dimension) {
        if(dimension == null || dimension[0] == Integer.MAX_VALUE)
            return toShapeInfoBuffer(arr);
        int[] ret = new int[arr.rank() * 2 + 4];
        ret[0]= arr.rank();
        int count = 1;
        for(int i = 0; i < arr.rank(); i++) {
            ret[count++] = arr.size(i);
        }
        for(int i = 0; i < arr.rank(); i++) {
            ret[count++] = arr.stride(i);
        }

        //note here we do offset of zero due to the offset
        //already being handled by the cuda device pointer
        ret[ret.length - 3] = arr.offset();
        ret[ret.length - 2] = arr.tensorAlongDimension(0,dimension).elementWiseStride();
        if( ret[ret.length - 2]  < 0)
            throw new IllegalStateException("Found illegal element wise stride of -1");
        ret[ret.length - 1] = arr.ordering();
        return ret;
    }

    public static void printDeviceBuffer(JCudaBuffer buffer,CudaContext ctx) {
        CublasPointer pointer = new CublasPointer(buffer,ctx);

    }


    /**
     * Converts a raw int buffer of the layout:
     * rank
     * shape
     * stride
     * offset
     * element wise stride
     * ordering
     * where shape and stride are both straight int pointers
     *
     *  Of note here is that offset will be zero automatically
     * because the offset is handled by the
     * pointer object instead
     *
     */
    public static int[] toShapeInfoBuffer(INDArray arr) {
        int[] ret = new int[arr.rank() * 2 + 4];
        ret[0]= arr.rank();
        int count = 1;
        for(int i = 0; i < arr.rank(); i++) {
            ret[count++] = arr.size(i);
        }
        for(int i = 0; i < arr.rank(); i++) {
            ret[count++] = arr.stride(i);
        }

        //note here we do offset of zero due to the offset
        //already being handled by the cuda device pointer
        ret[ret.length - 3] = arr.offset();
        ret[ret.length -2] = arr.elementWiseStride();
        ret[ret.length - 1] = arr.ordering();
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
     * Returns the host pointer wrt
     * the underlying storage
     * type for the buffer
     * @param buffer the buffer to get the
     *               host pointer for
     * @return the host pointer(Pointer.to) for the underlying
     * data buffer
     */
    public static Pointer getHostPointer(DataBuffer buffer) {
        if(buffer.allocationMode() == DataBuffer.AllocationMode.DIRECT) {
            JCudaBuffer buf = (JCudaBuffer) buffer;
            return Pointer.to(buf.asNio());
        }
        else if(buffer.allocationMode() == DataBuffer.AllocationMode.HEAP) {
            if(buffer.dataType() == DataBuffer.Type.DOUBLE) {
                double[] arr = buffer.asDouble();
                return Pointer.to(arr);
            }
            else if(buffer.dataType() == DataBuffer.Type.FLOAT) {
                float[] arr = buffer.asFloat();
                return Pointer.to(arr);
            }
            else if(buffer.dataType() == DataBuffer.Type.INT) {
                int[] arr = buffer.asInt();
                return Pointer.to(arr);
            }
        }

        throw new IllegalStateException("Unable to determine host pointer");
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
            else if(scalarOp.x().data().dataType() == DataBuffer.Type.INT)
                return new int[] {scalarOp.scalar().intValue()};
        }

        throw new IllegalStateException("Unable to get pointer for scalar operation " + scalarOp);
    }

    public static Pointer getPointer(int alpha) {
        return Pointer.to(new int[]{alpha});
    }

    public static Pointer getPointer(double alpha) {
        return Pointer.to(new double[]{alpha});
    }
    public static Pointer getPointer(float alpha) {
        return Pointer.to(new float[]{alpha});
    }
}
