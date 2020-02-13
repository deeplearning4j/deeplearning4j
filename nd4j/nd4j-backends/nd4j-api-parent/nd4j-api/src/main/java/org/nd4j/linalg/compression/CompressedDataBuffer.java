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

package org.nd4j.linalg.compression;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.val;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.MemcpyDirection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * @author raver119@gmail.com
 */
public class CompressedDataBuffer extends BaseDataBuffer {
    @Getter
    @Setter
    protected CompressionDescriptor compressionDescriptor;
    private static Logger logger = LoggerFactory.getLogger(CompressedDataBuffer.class);

    public CompressedDataBuffer(Pointer pointer, @NonNull CompressionDescriptor descriptor) {
        this.compressionDescriptor = descriptor;
        this.pointer = pointer;
        this.length = descriptor.getNumberOfElements();
        this.elementSize = (byte) descriptor.getOriginalElementSize();

        initTypeAndSize();
    }

    /**
     * Initialize the opType of this buffer
     */
    @Override
    protected void initTypeAndSize() {
        type = DataType.COMPRESSED;
        allocationMode = AllocationMode.MIXED_DATA_TYPES;
    }

    @Override
    public void write(DataOutputStream out) throws IOException {
        //        logger.info("Writing out CompressedDataBuffer");
        // here we should mimic to usual DataBuffer array
        out.writeUTF(allocationMode.name());
        out.writeLong(compressionDescriptor.getCompressedLength());
        out.writeUTF(DataType.COMPRESSED.name());
        // at this moment we don't care about mimics anymore
        //ByteIndexer indexer = ByteIndexer.create((BytePointer) pointer);
        out.writeUTF(compressionDescriptor.getCompressionAlgorithm());
        out.writeLong(compressionDescriptor.getCompressedLength());
        out.writeLong(compressionDescriptor.getOriginalLength());
        out.writeLong(compressionDescriptor.getNumberOfElements());
        out.writeInt(compressionDescriptor.getOriginalDataType().ordinal());
        //        out.write(((BytePointer) pointer).getStringBytes());
        for (int x = 0; x < pointer.capacity() * pointer.sizeof(); x++) {
            byte b = pointer.asByteBuffer().get(x);
            out.writeByte(b);
        }
    }

    @Override
    protected void setIndexer(Indexer indexer) {
        // no-op
    }

    @Override
    public Pointer addressPointer() {
        return pointer;
    }

    /**
     * Drop-in replacement wrapper for BaseDataBuffer.read() method, aware of CompressedDataBuffer
     * @param s
     * @return
     */
    public static DataBuffer readUnknown(DataInputStream s, AllocationMode allocMode, long length, DataType type) {
        // if buffer is uncompressed, it'll be valid buffer, so we'll just return it
        if (type != DataType.COMPRESSED) {
            DataBuffer buffer = Nd4j.createBuffer(type, length, false);
            buffer.read(s, allocMode, length, type);
            return buffer;
        } else {
            try {
                // if buffer is compressed one, we''ll restore it here
                String compressionAlgorithm = s.readUTF();
                long compressedLength = s.readLong();
                long originalLength = s.readLong();
                long numberOfElements = s.readLong();
                DataType originalType = DataType.values()[s.readInt()];

                byte[] temp = new byte[(int) compressedLength];
                for (int i = 0; i < compressedLength; i++) {
                    temp[i] = s.readByte();
                }

                Pointer pointer = new BytePointer(temp);
                CompressionDescriptor descriptor = new CompressionDescriptor();
                descriptor.setCompressedLength(compressedLength);
                descriptor.setCompressionAlgorithm(compressionAlgorithm);
                descriptor.setOriginalLength(originalLength);
                descriptor.setNumberOfElements(numberOfElements);
                descriptor.setOriginalDataType(originalType);
                return new CompressedDataBuffer(pointer, descriptor);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public DataBuffer dup() {
        Pointer nPtr = new BytePointer(compressionDescriptor.getCompressedLength());

        val perfD = PerformanceTracker.getInstance().helperStartTransaction();

        Pointer.memcpy(nPtr, pointer, compressionDescriptor.getCompressedLength());

        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfD, compressionDescriptor.getCompressedLength(), MemcpyDirection.HOST_TO_HOST);

        CompressionDescriptor nDesc = compressionDescriptor.clone();

        CompressedDataBuffer nBuf = new CompressedDataBuffer(nPtr, nDesc);
        return nBuf;
    }

    @Override
    public long length() {
        return compressionDescriptor.getNumberOfElements();
    }

    /**
     * Create with length
     *
     * @param length a databuffer of the same opType as
     *               this with the given length
     * @return a data buffer with the same length and datatype as this one
     */
    @Override
    protected DataBuffer create(long length) {
        throw new UnsupportedOperationException("This operation isn't supported for CompressedDataBuffer");
    }

    /**
     * Create the data buffer
     * with respect to the given byte buffer
     *
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    @Override
    public DataBuffer create(double[] data) {
        throw new UnsupportedOperationException("This operation isn't supported for CompressedDataBuffer");
    }

    /**
     * Create the data buffer
     * with respect to the given byte buffer
     *
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    @Override
    public DataBuffer create(float[] data) {
        throw new UnsupportedOperationException("This operation isn't supported for CompressedDataBuffer");
    }

    /**
     * Create the data buffer
     * with respect to the given byte buffer
     *
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    @Override
    public DataBuffer create(int[] data) {
        throw new UnsupportedOperationException("This method isn't supported by CompressedDataBuffer");
    }

    public void pointerIndexerByCurrentType(DataType currentType) {
        throw new UnsupportedOperationException("This method isn't supported by CompressedDataBuffer");
    }

    @Override
    public DataBuffer reallocate(long length) {
        throw new UnsupportedOperationException("This method isn't supported by CompressedDataBuffer");
    }

    @Override
    protected double getDoubleUnsynced(long index) {
        return super.getDouble(index);
    }

    @Override
    protected float getFloatUnsynced(long index) {
        return super.getFloat(index);
    }

    @Override
    protected long getLongUnsynced(long index) {
        return super.getLong(index);
    }

    @Override
    protected int getIntUnsynced(long index) {
        return super.getInt(index);
    }
}
