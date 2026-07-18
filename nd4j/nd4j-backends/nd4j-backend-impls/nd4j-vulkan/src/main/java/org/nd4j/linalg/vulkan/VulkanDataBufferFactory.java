/*
 * ******************************************************************************
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.nd4j.linalg.vulkan;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * Vulkan's buffer factory.
 *
 * <p>The factory implements the framework contract directly. It deliberately
 * does not inherit the native-CPU factory: all products are
 * {@link VulkanDataBuffer} instances with Vulkan primary/special ownership.</p>
 */
public class VulkanDataBufferFactory implements DataBufferFactory {
    private DataBuffer.AllocationMode allocationMode = DataBuffer.AllocationMode.MIXED_DATA_TYPES;

    @Override
    public void setAllocationMode(DataBuffer.AllocationMode allocationMode) {
        if (allocationMode == null) {
            throw new IllegalArgumentException("Allocation mode must not be null");
        }
        this.allocationMode = allocationMode;
    }

    @Override
    public DataBuffer.AllocationMode allocationMode() {
        return allocationMode;
    }

    @Override
    public DataBuffer createSame(DataBuffer buffer, boolean initialize) {
        return createSame(buffer, initialize, null);
    }

    @Override
    public DataBuffer createSame(DataBuffer buffer, boolean initialize, MemoryWorkspace workspace) {
        long length = buffer.length();
        if (buffer instanceof VulkanDataBuffer
                && ((VulkanDataBuffer) buffer).hasVariableLengthStringStorage()) {
            length = ((VulkanDataBuffer) buffer).stringElementCount();
        }
        return create(buffer.dataType(), length, initialize, workspace);
    }

    @Override
    public DataBuffer createBuffer(String[] data) {
        return createTypedBuffer(data, DataType.UTF8);
    }

    @Override
    public DataBuffer createTypedBuffer(String[] data, DataType dataType) {
        return VulkanDataBuffer.fromStrings(Arrays.asList(data), dataType);
    }

    private VulkanDataBuffer allocate(DataType type, long length, boolean initialize,
                                      MemoryWorkspace workspace) {
        return workspace == null
                ? new VulkanDataBuffer(type, length, initialize)
                : new VulkanDataBuffer(type, length, initialize, workspace);
    }

    private DataBuffer from(DataType type, double[] data, boolean copy, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = allocate(type, data.length, false, workspace);
        if (copy) {
            buffer.put(data);
        }
        return buffer;
    }

    private DataBuffer from(DataType type, float[] data, boolean copy, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = allocate(type, data.length, false, workspace);
        if (copy) {
            buffer.put(data);
        }
        return buffer;
    }

    private DataBuffer from(DataType type, int[] data, boolean copy, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = allocate(type, data.length, false, workspace);
        if (copy) {
            buffer.put(data);
        }
        return buffer;
    }

    private DataBuffer from(DataType type, long[] data, boolean copy, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = allocate(type, data.length, false, workspace);
        if (copy) {
            buffer.put(data);
        }
        return buffer;
    }

    @Override
    public DataBuffer create(DataType dataType, long length, boolean initialize) {
        return allocate(dataType, length, initialize, null);
    }

    @Override
    public DataBuffer create(DataType dataType, long length, boolean initialize,
                             MemoryWorkspace workspace) {
        return allocate(dataType, length, initialize, workspace);
    }

    @Override
    public DataBuffer createDouble(long length) {
        return createDouble(length, true);
    }

    @Override
    public DataBuffer createDouble(long length, boolean initialize) {
        return create(DataType.DOUBLE, length, initialize);
    }

    @Override
    public DataBuffer createDouble(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.DOUBLE, length, initialize, workspace);
    }

    @Override
    public DataBuffer createFloat(long length) {
        return createFloat(length, true);
    }

    @Override
    public DataBuffer createFloat(long length, boolean initialize) {
        return create(DataType.FLOAT, length, initialize);
    }

    @Override
    public DataBuffer createFloat(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.FLOAT, length, initialize, workspace);
    }

    @Override
    public DataBuffer createInt(long length) {
        return createInt(length, true);
    }

    @Override
    public DataBuffer createInt(long length, boolean initialize) {
        return create(DataType.INT, length, initialize);
    }

    @Override
    public DataBuffer createInt(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.INT, length, initialize, workspace);
    }

    @Override
    public DataBuffer createLong(long length) {
        return createLong(length, true);
    }

    @Override
    public DataBuffer createLong(long length, boolean initialize) {
        return create(DataType.LONG, length, initialize);
    }

    @Override
    public DataBuffer createLong(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.LONG, length, initialize, workspace);
    }

    @Override
    public DataBuffer createHalf(long length) {
        return createHalf(length, true);
    }

    @Override
    public DataBuffer createHalf(long length, boolean initialize) {
        return create(DataType.HALF, length, initialize);
    }

    @Override
    public DataBuffer createHalf(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.HALF, length, initialize, workspace);
    }

    @Override
    public DataBuffer createBFloat16(long length) {
        return createBFloat16(length, true);
    }

    @Override
    public DataBuffer createBFloat16(long length, boolean initialize) {
        return create(DataType.BFLOAT16, length, initialize);
    }

    @Override
    public DataBuffer createBFloat16(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.BFLOAT16, length, initialize, workspace);
    }

    @Override
    public DataBuffer createByte(long length) {
        return createByte(length, true);
    }

    @Override
    public DataBuffer createByte(long length, boolean initialize) {
        return create(DataType.BYTE, length, initialize);
    }

    @Override
    public DataBuffer createByte(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.BYTE, length, initialize, workspace);
    }

    @Override
    public DataBuffer createShort(long length) {
        return createShort(length, true);
    }

    @Override
    public DataBuffer createShort(long length, boolean initialize) {
        return create(DataType.SHORT, length, initialize);
    }

    @Override
    public DataBuffer createShort(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.SHORT, length, initialize, workspace);
    }

    @Override
    public DataBuffer createBool(long length) {
        return createBool(length, true);
    }

    @Override
    public DataBuffer createBool(long length, boolean initialize) {
        return create(DataType.BOOL, length, initialize);
    }

    @Override
    public DataBuffer createBool(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.BOOL, length, initialize, workspace);
    }

    @Override
    public DataBuffer createUShort(long length) {
        return createUShort(length, true);
    }

    @Override
    public DataBuffer createUShort(long length, boolean initialize) {
        return create(DataType.UINT16, length, initialize);
    }

    @Override
    public DataBuffer createUShort(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.UINT16, length, initialize, workspace);
    }

    @Override
    public DataBuffer createUInt(long length) {
        return createUInt(length, true);
    }

    @Override
    public DataBuffer createUInt(long length, boolean initialize) {
        return create(DataType.UINT32, length, initialize);
    }

    @Override
    public DataBuffer createUInt(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.UINT32, length, initialize, workspace);
    }

    @Override
    public DataBuffer createUByte(long length) {
        return createUByte(length, true);
    }

    @Override
    public DataBuffer createUByte(long length, boolean initialize) {
        return create(DataType.UBYTE, length, initialize);
    }

    @Override
    public DataBuffer createUByte(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.UBYTE, length, initialize, workspace);
    }

    @Override
    public DataBuffer createULong(long length) {
        return createULong(length, true);
    }

    @Override
    public DataBuffer createULong(long length, boolean initialize) {
        return create(DataType.UINT64, length, initialize);
    }

    @Override
    public DataBuffer createULong(long length, boolean initialize, MemoryWorkspace workspace) {
        return create(DataType.UINT64, length, initialize, workspace);
    }

    @Override
    public DataBuffer createLong(long[] data) {
        return createLong(data, true);
    }

    @Override
    public DataBuffer createLong(long[] data, boolean copy) {
        return from(DataType.LONG, data, copy, null);
    }

    @Override
    public DataBuffer createLong(long[] data, MemoryWorkspace workspace) {
        return from(DataType.LONG, data, true, workspace);
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
        return from(DataType.INT, data, copy, workspace);
    }

    @Override
    public DataBuffer createDouble(double[] data) {
        return createDouble(data, true);
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
        return from(DataType.DOUBLE, data, copy, null);
    }

    @Override
    public DataBuffer createFloat(int[] data, boolean copy) {
        return from(DataType.FLOAT, data, copy, null);
    }

    @Override
    public DataBuffer createInt(int[] data, boolean copy) {
        return from(DataType.INT, data, copy, null);
    }

    @Override
    public DataBuffer createLong(int[] data, boolean copy) {
        return from(DataType.LONG, data, copy, null);
    }

    @Override
    public DataBuffer createDouble(long[] data, boolean copy) {
        return from(DataType.DOUBLE, data, copy, null);
    }

    @Override
    public DataBuffer createFloat(long[] data, boolean copy) {
        return from(DataType.FLOAT, data, copy, null);
    }

    @Override
    public DataBuffer createInt(long[] data, boolean copy) {
        return from(DataType.INT, data, copy, null);
    }

    @Override
    public DataBuffer createDouble(double[] data, boolean copy) {
        return from(DataType.DOUBLE, data, copy, null);
    }

    @Override
    public DataBuffer createDouble(double[] data, MemoryWorkspace workspace) {
        return createDouble(data, true, workspace);
    }

    @Override
    public DataBuffer createDouble(double[] data, boolean copy, MemoryWorkspace workspace) {
        return from(DataType.DOUBLE, data, copy, workspace);
    }

    @Override
    public DataBuffer createFloat(double[] data, boolean copy) {
        return from(DataType.FLOAT, data, copy, null);
    }

    @Override
    public DataBuffer createInt(double[] data, boolean copy) {
        return from(DataType.INT, data, copy, null);
    }

    @Override
    public DataBuffer createDouble(float[] data, boolean copy) {
        return from(DataType.DOUBLE, data, copy, null);
    }

    @Override
    public DataBuffer createFloat(float[] data, boolean copy) {
        return from(DataType.FLOAT, data, copy, null);
    }

    @Override
    public DataBuffer createFloat(float[] data, boolean copy, MemoryWorkspace workspace) {
        return from(DataType.FLOAT, data, copy, workspace);
    }

    @Override
    public DataBuffer createInt(float[] data, boolean copy) {
        return from(DataType.INT, data, copy, null);
    }

    @Override
    public DataBuffer create(Pointer pointer, DataType type, long length, Indexer indexer) {
        return new VulkanDataBuffer(type, pointer, indexer, length);
    }

    @Override
    public DataBuffer create(Pointer pointer, Pointer specialPointer, DataType type,
                             long length, Indexer indexer) {
        return new VulkanDataBuffer(type, pointer, specialPointer, indexer, length);
    }

    @Override
    public DataBuffer create(DoublePointer pointer, long length) {
        return create(pointer, DataType.DOUBLE, length, DoubleIndexer.create(pointer));
    }

    @Override
    public DataBuffer create(IntPointer pointer, long length) {
        return create(pointer, DataType.INT, length, IntIndexer.create(pointer));
    }

    @Override
    public DataBuffer create(FloatPointer pointer, long length) {
        return create(pointer, DataType.FLOAT, length, FloatIndexer.create(pointer));
    }

    @Override
    public DataBuffer createBuffer(ByteBuffer underlyingBuffer, DataType dataType, long length) {
        return new VulkanDataBuffer(underlyingBuffer, dataType, length);
    }

    @Override
    public DataBuffer createHalf(float[] data, boolean copy) {
        return from(DataType.HALF, data, copy, null);
    }

    @Override
    public DataBuffer createHalf(float[] data, MemoryWorkspace workspace) {
        return createHalf(data, true, workspace);
    }

    @Override
    public DataBuffer createHalf(float[] data, boolean copy, MemoryWorkspace workspace) {
        return from(DataType.HALF, data, copy, workspace);
    }

    @Override
    public DataBuffer createHalf(double[] data, boolean copy) {
        return from(DataType.HALF, data, copy, null);
    }

    @Override
    public DataBuffer createHalf(long offset, float[] data) {
        return from(DataType.HALF, data, true, null);
    }

    @Override
    public DataBuffer createHalf(int[] data, boolean copy) {
        return from(DataType.HALF, data, copy, null);
    }

    @Override
    public DataBuffer createHalf(float[] data) {
        return createHalf(data, true);
    }

    @Override
    public DataBuffer createHalf(double[] data) {
        return createHalf(data, true);
    }

    @Override
    public DataBuffer createHalf(int[] data) {
        return createHalf(data, true);
    }

    @Override
    public Class<? extends DataBuffer> intBufferClass() {
        return VulkanDataBuffer.class;
    }

    @Override
    public Class<? extends DataBuffer> longBufferClass() {
        return VulkanDataBuffer.class;
    }

    @Override
    public Class<? extends DataBuffer> halfBufferClass() {
        return VulkanDataBuffer.class;
    }

    @Override
    public Class<? extends DataBuffer> floatBufferClass() {
        return VulkanDataBuffer.class;
    }

    @Override
    public Class<? extends DataBuffer> doubleBufferClass() {
        return VulkanDataBuffer.class;
    }

    @Override
    public DataBuffer createUtf8Buffer(byte[] data, long product) {
        return VulkanDataBuffer.fromEncodedStrings(data, product);
    }
}
