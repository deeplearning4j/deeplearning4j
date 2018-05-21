/*-
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

package org.nd4j.linalg.api.buffer.factory;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.nd4j.linalg.api.buffer.*;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.util.ArrayUtil;

import java.nio.ByteBuffer;

/**
 * Normal data buffer creation
 *
 * @author Adam Gibson
 */
public class DefaultDataBufferFactory implements DataBufferFactory {
    protected DataBuffer.AllocationMode allocationMode;


    @Override
    public void setAllocationMode(DataBuffer.AllocationMode allocationMode) {
        this.allocationMode = allocationMode;
    }

    @Override
    public DataBuffer.AllocationMode allocationMode() {
        if (allocationMode == null) {
            String otherAlloc = System.getProperty("alloc");
            if (otherAlloc.equals("heap"))
                setAllocationMode(DataBuffer.AllocationMode.HEAP);
            else if (otherAlloc.equals("direct"))
                setAllocationMode(DataBuffer.AllocationMode.DIRECT);
            else if (otherAlloc.equals("javacpp"))
                setAllocationMode(DataBuffer.AllocationMode.JAVACPP);
        }
        return allocationMode;
    }

    @Override
    public DataBuffer create(DataBuffer underlyingBuffer, long offset, long length) {
        if (underlyingBuffer.dataType() == DataBuffer.Type.DOUBLE) {
            return new DoubleBuffer(underlyingBuffer, length, offset);
        } else if (underlyingBuffer.dataType() == DataBuffer.Type.FLOAT) {
            return new FloatBuffer(underlyingBuffer, length, offset);

        } else if (underlyingBuffer.dataType() == DataBuffer.Type.INT) {
            return new IntBuffer(underlyingBuffer, length, offset);
        } else if (underlyingBuffer.dataType() == DataBuffer.Type.LONG) {
            return new LongBuffer(underlyingBuffer, length, offset);
        }
        return null;
    }

    @Override
    public DataBuffer createInt(long offset, ByteBuffer buffer, int length) {
        return new IntBuffer(buffer, length, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, ByteBuffer buffer, int length) {
        return new FloatBuffer(buffer, length, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, ByteBuffer buffer, int length) {
        return new DoubleBuffer(buffer, length, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, int length) {
        return new DoubleBuffer(length, 8, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, int length) {
        return new FloatBuffer(length, 4, offset);
    }

    @Override
    public DataBuffer createInt(long offset, int length) {
        return new IntBuffer(length, 4, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, int[] data) {
        return createDouble(offset, data, true);
    }

    @Override
    public DataBuffer createFloat(long offset, int[] data) {
        FloatBuffer ret = new FloatBuffer(ArrayUtil.toFloats(data), true, offset);
        return ret;
    }

    @Override
    public DataBuffer createInt(long offset, int[] data) {
        return new IntBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, double[] data) {
        return new DoubleBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, double[] data, MemoryWorkspace workspace) {
        return new DoubleBuffer(data, true, offset, workspace);
    }

    @Override
    public DataBuffer createDouble(long offset, byte[] data, int length) {
        return createDouble(offset, ArrayUtil.toDoubleArray(data), true);
    }

    @Override
    public DataBuffer createFloat(long offset, byte[] data, int length) {
        return createFloat(offset, ArrayUtil.toFloatArray(data), true);
    }

    @Override
    public DataBuffer createFloat(long offset, double[] data) {
        return new FloatBuffer(ArrayUtil.toFloats(data), true, offset);
    }

    @Override
    public DataBuffer createInt(long offset, double[] data) {
        return new IntBuffer(ArrayUtil.toInts(data), true, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, float[] data) {
        return new DoubleBuffer(ArrayUtil.toDoubles(data), true, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, float[] data) {
        return new FloatBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, float[] data, MemoryWorkspace workspace) {
        return new FloatBuffer(data, true, offset, workspace);
    }

    @Override
    public DataBuffer createInt(long offset, float[] data) {
        return new IntBuffer(ArrayUtil.toInts(data), true, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, int[] data, boolean copy) {
        return new DoubleBuffer(ArrayUtil.toDoubles(data), true, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, int[] data, boolean copy) {
        return new FloatBuffer(ArrayUtil.toFloats(data), copy, offset);
    }

    @Override
    public DataBuffer createInt(long offset, int[] data, boolean copy) {
        return new IntBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, double[] data, boolean copy) {
        return new DoubleBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, double[] data, boolean copy) {
        return new FloatBuffer(ArrayUtil.toFloats(data), copy, offset);
    }

    @Override
    public DataBuffer createInt(long offset, double[] data, boolean copy) {
        return new IntBuffer(ArrayUtil.toInts(data), copy, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, float[] data, boolean copy) {
        return new DoubleBuffer(ArrayUtil.toDoubles(data), copy, offset);
    }



    @Override
    public DataBuffer createFloat(long offset, float[] data, boolean copy) {
        return new FloatBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createInt(long offset, float[] data, boolean copy) {
        return new IntBuffer(ArrayUtil.toInts(data), copy, offset);
    }

    @Override
    public DataBuffer createInt(ByteBuffer buffer, int length) {
        return new IntBuffer(buffer, length);
    }

    @Override
    public DataBuffer createLong(ByteBuffer buffer, int length) {
        return new LongBuffer(buffer, length);
    }

    @Override
    public DataBuffer createFloat(ByteBuffer buffer, int length) {
        return new FloatBuffer(buffer, length);
    }

    @Override
    public DataBuffer createDouble(ByteBuffer buffer, int length) {
        return new DoubleBuffer(buffer, length);
    }

    @Override
    public DataBuffer createDouble(long length) {
        return new DoubleBuffer(length);
    }

    @Override
    public DataBuffer createDouble(long length, boolean initialize) {
        return new DoubleBuffer(length, initialize);
    }

    @Override
    public DataBuffer createFloat(long length) {
        return new FloatBuffer(length);
    }

    @Override
    public DataBuffer createFloat(long length, boolean initialize) {
        return new FloatBuffer(length, initialize);
    }

    @Override
    public DataBuffer createFloat(long length, boolean initialize, MemoryWorkspace workspace) {
        return new FloatBuffer(length, initialize, workspace);
    }

    @Override
    public DataBuffer createInt(long length) {
        return new IntBuffer(length);
    }

    @Override
    public DataBuffer createInt(long length, boolean initialize) {
        return new IntBuffer(length, initialize);
    }

    @Override
    public DataBuffer createInt(long length, boolean initialize, MemoryWorkspace workspace) {
        return new IntBuffer(length, initialize, workspace);
    }

    /**
     * This method will create new DataBuffer of the same dataType & same length
     *
     * @param buffer
     * @return
     */
    @Override
    public DataBuffer createSame(DataBuffer buffer, boolean init) {
        switch (buffer.dataType()) {
            case INT:
                return createInt(buffer.length(), init);
            case FLOAT:
                return createFloat(buffer.length(), init);
            case DOUBLE:
                return createDouble(buffer.length(), init);
            case HALF:
                return createHalf(buffer.length(), init);
            default:
                throw new UnsupportedOperationException("Unknown dataType: " + buffer.dataType());
        }
    }

    /**
     * This method will create new DataBuffer of the same dataType & same length
     *
     * @param buffer
     * @param workspace
     * @return
     */
    @Override
    public DataBuffer createSame(DataBuffer buffer, boolean init, MemoryWorkspace workspace) {
        switch (buffer.dataType()) {
            case INT:
                return createInt(buffer.length(), init, workspace);
            case FLOAT:
                return createFloat(buffer.length(), init, workspace);
            case DOUBLE:
                return createDouble(buffer.length(), init, workspace);
            case HALF:
                return createHalf(buffer.length(), init, workspace);
            default:
                throw new UnsupportedOperationException("Unknown dataType: " + buffer.dataType());
        }
    }

    @Override
    public DataBuffer createDouble(int[] data) {
        return createDouble(data, true);
    }

    @Override
    public DataBuffer createFloat(int[] data) {
        return createFloat(data, true);
    }

    @Override
    public DataBuffer createInt(int[] data) {
        return createInt(data, true);
    }

    @Override
    public DataBuffer createInt(int[] data, MemoryWorkspace workspace) {
        return createInt(data, true, workspace);
    }

    @Override
    public DataBuffer createInt(int[] data, boolean copy, MemoryWorkspace workspace) {
        return new IntBuffer(data, copy, workspace);
    }

    @Override
    public DataBuffer createDouble(double[] data) {
        return createDouble(data, true);
    }

    @Override
    public DataBuffer createDouble(byte[] data, int length) {
        return new DoubleBuffer(ByteBuffer.wrap(data), length);
    }

    @Override
    public DataBuffer createFloat(byte[] data, int length) {
        return new FloatBuffer(ByteBuffer.wrap(data), length);
    }

    @Override
    public DataBuffer createFloat(double[] data) {
        return createFloat(data, true);
    }

    @Override
    public DataBuffer createInt(double[] data) {
        return createInt(data, true);
    }

    @Override
    public DataBuffer createDouble(float[] data) {
        return createDouble(data, true);
    }

    @Override
    public DataBuffer createFloat(float[] data) {
        return createFloat(data, true);
    }

    @Override
    public DataBuffer createFloat(float[] data, MemoryWorkspace workspace) {
        return createFloat(data, true, workspace);
    }

    @Override
    public DataBuffer createInt(float[] data) {
        return createInt(data, true);
    }

    @Override
    public DataBuffer createDouble(int[] data, boolean copy) {
        return new DoubleBuffer(ArrayUtil.toDoubles(data), copy);
    }

    @Override
    public DataBuffer createFloat(int[] data, boolean copy) {
        return new FloatBuffer(ArrayUtil.toFloats(data), copy);
    }

    @Override
    public DataBuffer createInt(int[] data, boolean copy) {
        return new IntBuffer(data, copy);
    }

    @Override
    public DataBuffer createLong(long[] data) {
        return createLong(data, true);
    }

    @Override
    public DataBuffer createLong(long[] data, boolean copy) {
        return new LongBuffer(data, copy);
    }

    @Override
    public DataBuffer createLong(long length) {
        return new LongBuffer(length);
    }

    @Override
    public DataBuffer createLong(long length, boolean initialize) {
        return new LongBuffer(length, initialize);
    }

    @Override
    public DataBuffer createLong(long length, boolean initialize, MemoryWorkspace workspace) {
        return new LongBuffer(length, initialize, workspace);
    }

    @Override
    public DataBuffer createDouble(double[] data, boolean copy) {
        return new DoubleBuffer(data, copy);
    }

    @Override
    public DataBuffer createDouble(double[] data, MemoryWorkspace workspace) {
        return createDouble(data, true, workspace);
    }

    @Override
    public DataBuffer createDouble(double[] data, boolean copy, MemoryWorkspace workspace) {
        return new DoubleBuffer(data, copy, workspace);
    }

    @Override
    public DataBuffer createDouble(long length, boolean initialize, MemoryWorkspace workspace) {
        return new DoubleBuffer(length, initialize, workspace);
    }

    @Override
    public DataBuffer createFloat(double[] data, boolean copy) {
        return new FloatBuffer(ArrayUtil.toFloats(data), copy);
    }

    @Override
    public DataBuffer createInt(double[] data, boolean copy) {
        return new IntBuffer(ArrayUtil.toInts(data), copy);
    }

    @Override
    public DataBuffer createDouble(float[] data, boolean copy) {
        return new DoubleBuffer(data, copy);
    }

    @Override
    public DataBuffer createFloat(float[] data, boolean copy) {
        return new FloatBuffer(data, copy);
    }

    @Override
    public DataBuffer createFloat(float[] data, boolean copy, MemoryWorkspace workspace) {
        return new FloatBuffer(data, copy, workspace);
    }

    @Override
    public DataBuffer createInt(float[] data, boolean copy) {
        return new IntBuffer(ArrayUtil.toInts(data), copy);
    }

    /**
     * Create a data buffer based on the
     * given pointer, data buffer opType,
     * and length of the buffer
     *
     * @param pointer the pointer to use
     * @param type    the opType of buffer
     * @param length  the length of the buffer
     * @param indexer the indexer for the pointer
     * @return the data buffer
     * backed by this pointer with the given
     * opType and length.
     */
    @Override
    public DataBuffer create(Pointer pointer, DataBuffer.Type type, long length, Indexer indexer) {
        switch (type) {
            case INT:
                return new IntBuffer(pointer, indexer, length);
            case DOUBLE:
                return new DoubleBuffer(pointer, indexer, length);
            case FLOAT:
                return new FloatBuffer(pointer, indexer, length);
            case LONG:
                return new LongBuffer(pointer, indexer, length);
        }
        throw new IllegalArgumentException("Invalid opType " + type);
    }

    /**
     * @param doublePointer
     * @param length
     * @return
     */
    @Override
    public DataBuffer create(DoublePointer doublePointer, long length) {
        doublePointer.capacity(length);
        doublePointer.limit(length);
        doublePointer.position(0);
        return new DoubleBuffer(doublePointer, DoubleIndexer.create(doublePointer), length);
    }

    /**
     * @param intPointer
     * @param length
     * @return
     */
    @Override
    public DataBuffer create(IntPointer intPointer, long length) {
        intPointer.capacity(length);
        intPointer.limit(length);
        intPointer.position(0);
        return new IntBuffer(intPointer, IntIndexer.create(intPointer), length);
    }

    /**
     * @param floatPointer
     * @param length
     * @return
     */
    @Override
    public DataBuffer create(FloatPointer floatPointer, long length) {
        floatPointer.capacity(length);
        floatPointer.limit(length);
        floatPointer.position(0);
        return new FloatBuffer(floatPointer, FloatIndexer.create(floatPointer), length);
    }


    @Override
    public DataBuffer createHalf(long length) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    @Override
    public DataBuffer createHalf(long length, boolean initialize) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
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
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
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
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
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
    public DataBuffer createHalf(long offset, double[] data, boolean copy) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
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
    public DataBuffer createHalf(long offset, float[] data, boolean copy) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
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
    public DataBuffer createHalf(long offset, int[] data, boolean copy) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(long offset, double[] data) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(long offset, float[] data) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    @Override
    public DataBuffer createHalf(long offset, float[] data, MemoryWorkspace workspace) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param data   the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(long offset, int[] data) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
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
    public DataBuffer createHalf(long offset, byte[] data, boolean copy) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
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
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(float[] data) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(double[] data) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param data the data to create the buffer from
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(int[] data) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
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
    public DataBuffer createHalf(long offset, byte[] data, int length) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    /**
     * Creates a half-precision data buffer
     *
     * @param offset
     * @param length
     * @return the new buffer
     */
    @Override
    public DataBuffer createHalf(long offset, int length) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
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
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
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
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    @Override
    public DataBuffer createHalf(long length, boolean initialize, MemoryWorkspace workspace) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    @Override
    public DataBuffer createHalf(float[] data, MemoryWorkspace workspace) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    @Override
    public DataBuffer createHalf(float[] data, boolean copy, MemoryWorkspace workspace) {
        throw new UnsupportedOperationException("FP16 isn't supported for CPU yet");
    }

    @Override
    public Class<? extends DataBuffer> intBufferClass() {
        return IntBuffer.class;
    }

    @Override
    public Class<? extends DataBuffer> longBufferClass() {
        return LongBuffer.class;
    }

    @Override
    public Class<? extends DataBuffer> halfBufferClass() {
        return null;    //Not yet supported
    }

    @Override
    public Class<? extends DataBuffer> floatBufferClass() {
        return FloatBuffer.class;
    }

    @Override
    public Class<? extends DataBuffer> doubleBufferClass() {
        return DoubleBuffer.class;
    }
}
