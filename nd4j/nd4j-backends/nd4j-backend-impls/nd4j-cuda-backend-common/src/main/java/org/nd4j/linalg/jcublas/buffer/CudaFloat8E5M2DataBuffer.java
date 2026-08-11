/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.jcublas.buffer;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

import java.nio.ByteBuffer;

/**
 * CUDA storage buffer for FP8 E5M2 values.
 *
 * <p>FP8 values are byte-addressed, but retain their floating-point data type so
 * native kernels perform FP8 conversion and arithmetic instead of integer math.</p>
 */
public class CudaFloat8E5M2DataBuffer extends BaseCudaDataBuffer {

    public CudaFloat8E5M2DataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);
    }

    public CudaFloat8E5M2DataBuffer(Pointer pointer, Pointer specialPointer, Indexer indexer, long length) {
        super(pointer, specialPointer, indexer, length);
    }

    public CudaFloat8E5M2DataBuffer(long length) {
        super(length, 1, true);
    }

    public CudaFloat8E5M2DataBuffer(long length, boolean initialize) {
        super(length, 1, initialize);
    }

    public CudaFloat8E5M2DataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        super(length, 1, initialize, workspace);
    }

    public CudaFloat8E5M2DataBuffer(ByteBuffer buffer, DataType dataType, long length) {
        super(buffer, dataType, length);
    }

    public CudaFloat8E5M2DataBuffer(ByteBuffer buffer, DataType dataType, long length, boolean cpuOnly) {
        super(buffer, dataType, length, cpuOnly);
    }

    @Override
    protected void initTypeAndSize() {
        elementSize = 1;
        type = DataType.FLOAT8_E5M2;
    }

    @Override
    protected DataBuffer create(long length) {
        return new CudaFloat8E5M2DataBuffer(length);
    }

    @Override
    public DataBuffer create(double[] data) {
        throw directCreationUnsupported();
    }

    @Override
    public DataBuffer create(float[] data) {
        throw directCreationUnsupported();
    }

    @Override
    public DataBuffer create(int[] data) {
        throw directCreationUnsupported();
    }

    @Override
    public void flush() {
        // No buffered Java-side state.
    }

    private UnsupportedOperationException directCreationUnsupported() {
        return new UnsupportedOperationException(
                "Create FP8 arrays by casting a floating-point INDArray to " + dataType());
    }
}
