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

package org.nd4j.linalg.jcublas.buffer.factory;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.*;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;

/**
 * Creates cuda buffers
 *
 * @author Adam Gibson
 */
public class CudaDataBufferFactory implements DataBufferFactory {
    protected DataBuffer.AllocationMode allocationMode;
    private static Logger log = LoggerFactory.getLogger(CudaDataBufferFactory.class);

    @Override
    public void setAllocationMode(DataBuffer.AllocationMode allocationMode) {
        this.allocationMode = allocationMode;
    }

    @Override
    public DataBuffer.AllocationMode allocationMode() {
        if(allocationMode == null) {
            String otherAlloc = System.getProperty("alloc");
            if(otherAlloc.equals("heap"))
                setAllocationMode(DataBuffer.AllocationMode.HEAP);
            else if(otherAlloc.equals("direct"))
                setAllocationMode(DataBuffer.AllocationMode.DIRECT);
            else if(otherAlloc.equals("javacpp"))
                setAllocationMode(DataBuffer.AllocationMode.JAVACPP);
        }
        return allocationMode;
    }

    @Override
    public DataBuffer create(DataBuffer underlyingBuffer, long offset, long length) {
        if(underlyingBuffer.dataType() == DataBuffer.Type.DOUBLE) {
            return new CudaDoubleDataBuffer(underlyingBuffer,length,offset);
        }
        else if(underlyingBuffer.dataType() == DataBuffer.Type.FLOAT) {
            return new CudaFloatDataBuffer(underlyingBuffer,length,offset);

        }
        else if(underlyingBuffer.dataType() == DataBuffer.Type.INT) {
            return new CudaIntDataBuffer(underlyingBuffer,length,offset);
        }
        else if (underlyingBuffer.dataType() == DataBuffer.Type.HALF) {
            return new CudaHalfDataBuffer(underlyingBuffer, length, offset);
        }
        return null;
    }

    @Override
    public DataBuffer createInt(int offset, ByteBuffer buffer, int length) {
        return new CudaIntDataBuffer(buffer,length,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, ByteBuffer buffer, int length) {
        return new CudaFloatDataBuffer(buffer,length,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, ByteBuffer buffer, int length) {
        return new CudaDoubleDataBuffer(buffer,length,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, int length) {
        return new CudaDoubleDataBuffer(length,8,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, int length) {
        return new CudaFloatDataBuffer(length,4,length);
    }

    @Override
    public DataBuffer createInt(int offset, int length) {
        return new CudaIntDataBuffer(length,4,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, int[] data) {
        return new CudaDoubleDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, int[] data) {
        return new CudaFloatDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createInt(int offset, int[] data) {
        return new CudaIntDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, double[] data) {
        return new CudaDoubleDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, byte[] data, int length) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDoubleArray(data),true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, byte[] data, int length) {
        return new CudaFloatDataBuffer(ArrayUtil.toDoubleArray(data),true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, double[] data) {
        return new CudaFloatDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createInt(int offset, double[] data) {
        return new CudaIntDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, float[] data) {
        return new CudaDoubleDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, float[] data) {
        return new CudaFloatDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createInt(int offset, float[] data) {
        return new CudaIntDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, int[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, int[] data, boolean copy) {
        return new CudaFloatDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createInt(int offset, int[] data, boolean copy) {
        return new CudaIntDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, double[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, double[] data, boolean copy) {
        return new CudaFloatDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createInt(int offset, double[] data, boolean copy) {
        return new CudaIntDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, float[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, float[] data, boolean copy) {
        return new CudaFloatDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createInt(int offset, float[] data, boolean copy) {
        return new CudaIntDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createInt(ByteBuffer buffer, int length) {
        return new CudaIntDataBuffer(buffer,length);
    }

    @Override
    public DataBuffer createFloat(ByteBuffer buffer, int length) {
        return new CudaFloatDataBuffer(buffer,length);
    }

    @Override
    public DataBuffer createDouble(ByteBuffer buffer, int length) {
        return new CudaDoubleDataBuffer(buffer,length);
    }

    @Override
    public DataBuffer createDouble(long length) {
        return new CudaDoubleDataBuffer(length);
    }

    @Override
    public DataBuffer createDouble(long length, boolean initialize){
        return new CudaDoubleDataBuffer(length, initialize);
    }

    @Override
    public DataBuffer createFloat(long length) {
        return new CudaFloatDataBuffer(length);
    }

    @Override
    public DataBuffer createFloat(long length, boolean initialize){
        return new CudaFloatDataBuffer(length, initialize);
    }

    @Override
    public DataBuffer createInt(long length) {
        return new CudaIntDataBuffer(length);
    }

    @Override
    public DataBuffer createInt(long length, boolean initialize){
        return new CudaIntDataBuffer(length, initialize);
    }

    @Override
    public DataBuffer createDouble(int[] data) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDoubles(data));
    }

    @Override
    public DataBuffer createFloat(int[] data) {
        return new CudaFloatDataBuffer(ArrayUtil.toFloats(data));
    }

    @Override
    public DataBuffer createInt(int[] data) {
        return new CudaIntDataBuffer(data);
    }

    @Override
    public DataBuffer createDouble(double[] data) {
        return new CudaDoubleDataBuffer(data);
    }

    @Override
    public DataBuffer createDouble(byte[] data, int length) {
        return new CudaDoubleDataBuffer(data, length);
    }

    @Override
    public DataBuffer createFloat(byte[] data, int length) {
        return new CudaFloatDataBuffer(data, length);
    }

    @Override
    public DataBuffer createFloat(double[] data) {
        return new CudaFloatDataBuffer(ArrayUtil.toFloats(data));
    }

    @Override
    public DataBuffer createInt(double[] data) {
        return new CudaIntDataBuffer(ArrayUtil.toInts(data));
    }

    @Override
    public DataBuffer createDouble(float[] data) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDoubles(data));
    }

    @Override
    public DataBuffer createFloat(float[] data) {
        return new CudaFloatDataBuffer(data);
    }

    @Override
    public DataBuffer createInt(float[] data) {
        return new CudaIntDataBuffer(ArrayUtil.toInts(data));
    }

    @Override
    public DataBuffer createDouble(int[] data, boolean copy) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDouble(data));
    }

    @Override
    public DataBuffer createFloat(int[] data, boolean copy) {
        return new CudaFloatDataBuffer(ArrayUtil.toFloats(data));
    }

    @Override
    public DataBuffer createInt(int[] data, boolean copy) {
        return new CudaIntDataBuffer(data);
    }

    @Override
    public DataBuffer createDouble(double[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data);
    }

    @Override
    public DataBuffer createFloat(double[] data, boolean copy) {
        return new CudaFloatDataBuffer(ArrayUtil.toFloats(data));
    }

    @Override
    public DataBuffer createInt(double[] data, boolean copy) {
        return new CudaIntDataBuffer(ArrayUtil.toInts(data));
    }

    @Override
    public DataBuffer createDouble(float[] data, boolean copy) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDoubles(data));
    }

    @Override
    public DataBuffer createFloat(float[] data, boolean copy) {
        return new CudaFloatDataBuffer(data);
    }

    @Override
    public DataBuffer createInt(float[] data, boolean copy) {
        return new CudaIntDataBuffer(ArrayUtil.toInts(data));
    }

    /**
     * Create a data buffer based on the
     * given pointer, data buffer type,
     * and length of the buffer
     *
     * @param pointer the pointer to use
     * @param type    the type of buffer
     * @param length  the length of the buffer
     * @param indexer
     * @return the data buffer
     * backed by this pointer with the given
     * type and length.
     */
    @Override
    public DataBuffer create(Pointer pointer, DataBuffer.Type type, long length, Indexer indexer) {
        switch (type) {
            case INT: return new CudaIntDataBuffer(pointer,indexer,length);
            case DOUBLE: return new CudaDoubleDataBuffer(pointer,indexer,length);
            case FLOAT: return new CudaFloatDataBuffer(pointer,indexer,length);
            case HALF: return new CudaHalfDataBuffer(pointer, indexer, length);
        }
        throw new IllegalArgumentException("Illegal type " + type);
    }


    @Override
    public DataBuffer createHalf(long length) {
        return new CudaHalfDataBuffer(length);
    }

    @Override
    public DataBuffer createHalf(long length, boolean initialize) {
        return new CudaHalfDataBuffer(length, initialize);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @param copy
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(float[] data, boolean copy) {
        return new CudaHalfDataBuffer(data, copy);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @param copy
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(double[] data, boolean copy) {
        return new CudaHalfDataBuffer(data, copy);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @param copy
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int offset, double[] data, boolean copy) {
        return new CudaHalfDataBuffer(data, copy, offset);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @param copy
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int offset, float[] data, boolean copy) {
        return new CudaHalfDataBuffer(data, copy, offset);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @param copy
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int offset, int[] data, boolean copy) {
        return new CudaHalfDataBuffer(data, copy, offset);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int offset, double[] data) {
        return new CudaHalfDataBuffer(data, true, offset);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int offset, float[] data) {
        return new CudaHalfDataBuffer(data, true, offset);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int offset, int[] data) {
        return new CudaHalfDataBuffer(data, true, offset);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @param copy
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int offset, byte[] data, boolean copy) {
        return new CudaHalfDataBuffer(ArrayUtil.toFloatArray(data), copy, offset);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @param copy
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int[] data, boolean copy) {
        return new CudaHalfDataBuffer(data, copy);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(float[] data) {
        return new CudaHalfDataBuffer(data);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(double[] data) {
        return new CudaHalfDataBuffer(data);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int[] data) {
        return new CudaHalfDataBuffer(data);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @param length
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int offset, byte[] data, int length) {
        return new CudaHalfDataBuffer(ArrayUtil.toFloatArray(data), true, offset);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param length
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int offset, int length) {
        return new CudaHalfDataBuffer(length);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param buffer
     * @param length
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(ByteBuffer buffer, int length) {
        return new CudaHalfDataBuffer(buffer, length);
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param data
     * @param length
     * @return
     */
    @Override
    public DataBuffer createHalf(byte[] data, int length) {
        return new CudaHalfDataBuffer(data, length);
    }
}