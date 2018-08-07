/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.jcublas.buffer.factory;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.LongBuffer;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.jcublas.buffer.*;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;

/**
 * Creates cuda buffers
 *
 * @author Adam Gibson
 */
@Slf4j
public class CudaDataBufferFactory implements DataBufferFactory {
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
            return new CudaDoubleDataBuffer(underlyingBuffer, length, offset);
        } else if (underlyingBuffer.dataType() == DataBuffer.Type.FLOAT) {
            return new CudaFloatDataBuffer(underlyingBuffer, length, offset);

        } else if (underlyingBuffer.dataType() == DataBuffer.Type.INT) {
            return new CudaIntDataBuffer(underlyingBuffer, length, offset);
        } else if (underlyingBuffer.dataType() == DataBuffer.Type.HALF) {
            return new CudaHalfDataBuffer(underlyingBuffer, length, offset);
        } else if (underlyingBuffer.dataType() == DataBuffer.Type.LONG) {
            return new CudaLongDataBuffer(underlyingBuffer, length, offset);
        }

        throw new ND4JIllegalStateException("Unknown data buffer type: " + underlyingBuffer.dataType().toString());
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
    public DataBuffer createFloat(float[] data, MemoryWorkspace workspace) {
        return createFloat(data, true, workspace);
    }

    @Override
    public DataBuffer createFloat(float[] data, boolean copy, MemoryWorkspace workspace) {
        return new CudaFloatDataBuffer(data, copy, workspace);
    }

    @Override
    public DataBuffer createInt(int[] data, MemoryWorkspace workspace) {
        return new CudaIntDataBuffer(data, workspace);
    }

    @Override
    public DataBuffer createInt(int[] data, boolean copy, MemoryWorkspace workspace) {
        return new CudaIntDataBuffer(data, copy, workspace);
    }

    @Override
    public DataBuffer createInt(long offset, ByteBuffer buffer, int length) {
        return new CudaIntDataBuffer(buffer, length, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, ByteBuffer buffer, int length) {
        return new CudaFloatDataBuffer(buffer, length, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, ByteBuffer buffer, int length) {
        return new CudaDoubleDataBuffer(buffer, length, offset);
    }


    @Override
    public DataBuffer createLong(ByteBuffer buffer, int length) {
        return new CudaLongDataBuffer(buffer, length);
    }

    @Override
    public DataBuffer createDouble(long offset, int length) {
        return new CudaDoubleDataBuffer(length, 8, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, int length) {
        return new CudaFloatDataBuffer(length, 4, length);
    }

    @Override
    public DataBuffer createInt(long offset, int length) {
        return new CudaIntDataBuffer(length, 4, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, int[] data) {
        return new CudaDoubleDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, int[] data) {
        return new CudaFloatDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createInt(long offset, int[] data) {
        return new CudaIntDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, double[] data) {
        return new CudaDoubleDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, double[] data, MemoryWorkspace workspace) {
        return new CudaDoubleDataBuffer(data, true, offset, workspace);
    }

    @Override
    public DataBuffer createDouble(long offset, byte[] data, int length) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDoubleArray(data), true, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, byte[] data, int length) {
        return new CudaFloatDataBuffer(ArrayUtil.toDoubleArray(data), true, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, double[] data) {
        return new CudaFloatDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createInt(long offset, double[] data) {
        return new CudaIntDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, float[] data) {
        return new CudaDoubleDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, float[] data) {
        return new CudaFloatDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, float[] data, MemoryWorkspace workspace) {
        return new CudaFloatDataBuffer(data, true, offset, workspace);
    }

    @Override
    public DataBuffer createInt(long offset, float[] data) {
        return new CudaIntDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, int[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, int[] data, boolean copy) {
        return new CudaFloatDataBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createInt(long offset, int[] data, boolean copy) {
        return new CudaIntDataBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, double[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, double[] data, boolean copy) {
        return new CudaFloatDataBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createInt(long offset, double[] data, boolean copy) {
        return new CudaIntDataBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createDouble(long offset, float[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createFloat(long offset, float[] data, boolean copy) {
        return new CudaFloatDataBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createInt(long offset, float[] data, boolean copy) {
        return new CudaIntDataBuffer(data, copy, offset);
    }

    @Override
    public DataBuffer createInt(ByteBuffer buffer, int length) {
        return new CudaIntDataBuffer(buffer, length);
    }

    @Override
    public DataBuffer createFloat(ByteBuffer buffer, int length) {
        return new CudaFloatDataBuffer(buffer, length);
    }

    @Override
    public DataBuffer createDouble(ByteBuffer buffer, int length) {
        return new CudaDoubleDataBuffer(buffer, length);
    }

    @Override
    public DataBuffer createDouble(long length) {
        return new CudaDoubleDataBuffer(length);
    }

    @Override
    public DataBuffer createDouble(long length, boolean initialize) {
        return new CudaDoubleDataBuffer(length, initialize);
    }

    @Override
    public DataBuffer createFloat(long length) {
        return new CudaFloatDataBuffer(length);
    }

    @Override
    public DataBuffer createFloat(long length, boolean initialize) {
        return new CudaFloatDataBuffer(length, initialize);
    }

    @Override
    public DataBuffer createFloat(long length, boolean initialize, MemoryWorkspace workspace) {
        return new CudaFloatDataBuffer(length, initialize, workspace);
    }

    @Override
    public DataBuffer createInt(long length) {
        return new CudaIntDataBuffer(length);
    }

    @Override
    public DataBuffer createInt(long length, boolean initialize) {
        return new CudaIntDataBuffer(length, initialize);
    }

    @Override
    public DataBuffer createInt(long length, boolean initialize, MemoryWorkspace workspace) {
        return new CudaIntDataBuffer(length, initialize, workspace);
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
    public DataBuffer createLong(int[] data, boolean copy) {
        return new CudaLongDataBuffer(data);
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

    @Override
    public DataBuffer createDouble(long[] data, boolean copy) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDoubles(data));
    }

    @Override
    public DataBuffer createFloat(long[] data, boolean copy) {
        return new CudaFloatDataBuffer(ArrayUtil.toFloats(data));
    }

    @Override
    public DataBuffer createInt(long[] data, boolean copy) {
        return new CudaIntDataBuffer(data);
    }

    /**
     * Create a data buffer based on the
     * given pointer, data buffer opType,
     * and length of the buffer
     *
     * @param pointer the pointer to use
     * @param type    the opType of buffer
     * @param length  the length of the buffer
     * @param indexer
     * @return the data buffer
     * backed by this pointer with the given
     * opType and length.
     */
    @Override
    public DataBuffer create(Pointer pointer, DataBuffer.Type type, long length, Indexer indexer) {
        switch (type) {
            case LONG:
                return new CudaLongDataBuffer(pointer, indexer, length);
            case INT:
                return new CudaIntDataBuffer(pointer, indexer, length);
            case DOUBLE:
                return new CudaDoubleDataBuffer(pointer, indexer, length);
            case FLOAT:
                return new CudaFloatDataBuffer(pointer, indexer, length);
            case HALF:
                return new CudaHalfDataBuffer(pointer, indexer, length);
        }

        throw new IllegalArgumentException("Illegal dtype " + type);
    }

    /**
     * @param doublePointer
     * @param length
     * @return
     */
    @Override
    public DataBuffer create(DoublePointer doublePointer, long length) {
        return new CudaDoubleDataBuffer(doublePointer,DoubleIndexer.create(doublePointer),length);
    }

    /**
     * @param intPointer
     * @param length
     * @return
     */
    @Override
    public DataBuffer create(IntPointer intPointer, long length) {
        return new CudaIntDataBuffer(intPointer, IntIndexer.create(intPointer),length);
    }

    /**
     * @param floatPointer
     * @param length
     * @return
     */
    @Override
    public DataBuffer create(FloatPointer floatPointer, long length) {
        return new CudaFloatDataBuffer(floatPointer, FloatIndexer.create(floatPointer),length);
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
    public DataBuffer createHalf(long offset, double[] data, boolean copy) {
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
    public DataBuffer createHalf(long offset, float[] data, boolean copy) {
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
    public DataBuffer createHalf(long offset, int[] data, boolean copy) {
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
    public DataBuffer createHalf(long offset, double[] data) {
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
    public DataBuffer createHalf(long offset, float[] data) {
        return new CudaHalfDataBuffer(data, true, offset);
    }

    @Override
    public DataBuffer createHalf(long offset, float[] data, MemoryWorkspace workspace) {
        return new CudaHalfDataBuffer(data, true, offset, workspace);
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
    public DataBuffer createHalf(long offset, byte[] data, boolean copy) {
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
    public DataBuffer createHalf(long offset, byte[] data, int length) {
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
    public DataBuffer createHalf(long offset, int length) {
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

    @Override
    public DataBuffer createDouble(long length, boolean initialize, MemoryWorkspace workspace) {
        return new CudaDoubleDataBuffer(length, initialize, workspace);
    }

    /**
     * Creates a double data buffer
     *
     * @param data      the data to create the buffer from
     * @param workspace
     * @return the new buffer
     */
    @Override
    public DataBuffer createDouble(double[] data, MemoryWorkspace workspace) {
        return createDouble(data, true, workspace);
    }

    /**
     * Creates a double data buffer
     *
     * @param data      the data to create the buffer from
     * @param copy
     * @param workspace @return the new buffer
     */
    @Override
    public DataBuffer createDouble(double[] data, boolean copy, MemoryWorkspace workspace) {
        return new CudaDoubleDataBuffer(data, copy, workspace);
    }

    @Override
    public DataBuffer createHalf(long length, boolean initialize, MemoryWorkspace workspace) {
        return new CudaHalfDataBuffer(length, initialize, workspace);
    }

    @Override
    public DataBuffer createHalf(float[] data, MemoryWorkspace workspace) {
        return createHalf(data, true, workspace);
    }

    @Override
    public DataBuffer createHalf(float[] data, boolean copy, MemoryWorkspace workspace) {
        return new CudaHalfDataBuffer(data, copy, workspace);
    }


    @Override
    public Class<? extends DataBuffer> intBufferClass() {
        return CudaIntDataBuffer.class;
    }

    @Override
    public Class<? extends DataBuffer> longBufferClass() {
        return CudaLongDataBuffer.class;
    }

    @Override
    public Class<? extends DataBuffer> halfBufferClass() {
        return CudaHalfDataBuffer.class;    //Not yet supported
    }

    @Override
    public Class<? extends DataBuffer> floatBufferClass() {
        return CudaFloatDataBuffer.class;
    }

    @Override
    public Class<? extends DataBuffer> doubleBufferClass() {
        return CudaDoubleDataBuffer.class;
    }



    @Override
    public DataBuffer createLong(long[] data) {
        return createLong(data, true);
    }

    @Override
    public DataBuffer createLong(long[] data, boolean copy) {
        return new CudaLongDataBuffer(data, copy);
    }

    @Override
    public DataBuffer createLong(long[] data, MemoryWorkspace workspace) {
        return new CudaLongDataBuffer(data, workspace);
    }

    @Override
    public DataBuffer createLong(long length) {
        return createLong(length, true);
    }

    @Override
    public DataBuffer createLong(long length, boolean initialize) {
        return new CudaLongDataBuffer(length, initialize);
    }

    @Override
    public DataBuffer createLong(long length, boolean initialize, MemoryWorkspace workspace) {
        return new CudaLongDataBuffer(length, initialize, workspace);
    }

}
