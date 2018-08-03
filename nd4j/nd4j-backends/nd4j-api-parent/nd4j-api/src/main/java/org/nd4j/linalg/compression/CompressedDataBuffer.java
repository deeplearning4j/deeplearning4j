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
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.MemcpyDirection;
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
        type = Type.COMPRESSED;
        allocationMode = AllocationMode.JAVACPP;
    }

    @Override
    public void write(DataOutputStream out) throws IOException {
        //        logger.info("Writing out CompressedDataBuffer");
        // here we should mimic to usual DataBuffer array
        out.writeUTF(allocationMode.name());
        out.writeInt((int) compressionDescriptor.getCompressedLength());
        out.writeUTF(Type.COMPRESSED.name());
        // at this moment we don't care about mimics anymore
        //ByteRawIndexer indexer = new ByteRawIndexer((BytePointer) pointer);
        out.writeUTF(compressionDescriptor.getCompressionAlgorithm());
        out.writeLong(compressionDescriptor.getCompressedLength());
        out.writeLong(compressionDescriptor.getOriginalLength());
        out.writeLong(compressionDescriptor.getNumberOfElements());
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

    /**
     * Drop-in replacement wrapper for BaseDataBuffer.read() method, aware of CompressedDataBuffer
     * @param s
     * @return
     */
    public static DataBuffer readUnknown(DataInputStream s, long length) {
        DataBuffer buffer = Nd4j.createBuffer(length);
        buffer.read(s);
        // if buffer is uncompressed, it'll be valid buffer, so we'll just return it
        if (buffer.dataType() != Type.COMPRESSED)
            return buffer;
        else {
            try {

                // if buffer is compressed one, we''ll restore it here
                String compressionAlgorithm = s.readUTF();
                long compressedLength = s.readLong();
                long originalLength = s.readLong();
                long numberOfElements = s.readLong();

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
        throw new UnsupportedOperationException("This operation isn't supported for CompressedDataBuffer");
    }
}
