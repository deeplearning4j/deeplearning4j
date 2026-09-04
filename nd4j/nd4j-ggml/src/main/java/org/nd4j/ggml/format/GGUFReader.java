/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.ggml.format;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.ggml.GGMLImportException;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Reader for GGUF (GGML Universal Format) files.
 * Supports GGUF versions 1, 2, and 3.
 */
@Slf4j
public class GGUFReader implements Closeable {

    /**
     * Metadata is consumed sequentially through a bounded window. Tensor payloads
     * are read positionally and never enter this buffer. Keeping this deliberately
     * small prevents a GGUF source file from becoming a second model-sized resident
     * allocation during conversion.
     */
    private static final int READ_BUFFER_BYTES = 1024 * 1024;

    // GGUF metadata value types
    private static final int GGUF_TYPE_UINT8 = 0;
    private static final int GGUF_TYPE_INT8 = 1;
    private static final int GGUF_TYPE_UINT16 = 2;
    private static final int GGUF_TYPE_INT16 = 3;
    private static final int GGUF_TYPE_UINT32 = 4;
    private static final int GGUF_TYPE_INT32 = 5;
    private static final int GGUF_TYPE_FLOAT32 = 6;
    private static final int GGUF_TYPE_BOOL = 7;
    private static final int GGUF_TYPE_STRING = 8;
    private static final int GGUF_TYPE_ARRAY = 9;
    private static final int GGUF_TYPE_UINT64 = 10;
    private static final int GGUF_TYPE_INT64 = 11;
    private static final int GGUF_TYPE_FLOAT64 = 12;

    private final File file;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final ByteBuffer buffer;
    private long bufferStart;
    private long readPosition;

    private GGUFHeader header;
    private List<GGMLTensorInfo> tensorInfos;
    private long dataOffset;
    private int alignment;

    /**
     * Create a new GGUF reader for the given file
     */
    public GGUFReader(File file) throws IOException {
        this.file = file;
        this.raf = new RandomAccessFile(file, "r");
        this.channel = raf.getChannel();
        this.buffer = ByteBuffer.allocate(READ_BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.buffer.limit(0);
        this.bufferStart = 0;
        this.readPosition = 0;
    }

    /**
     * Read and parse the GGUF header
     */
    public GGUFHeader readHeader() throws IOException, GGMLImportException {
        seek(0);

        // Read magic
        int magic = readInt();
        if (magic != GGMLFormatDetector.GGUF_MAGIC) {
            throw new GGMLImportException(String.format(
                    "Invalid GGUF magic: 0x%08X (expected 0x%08X)", magic, GGMLFormatDetector.GGUF_MAGIC));
        }

        // Read version
        int version = readInt();
        if (version < 1 || version > 3) {
            throw new GGMLImportException("Unsupported GGUF version: " + version);
        }

        // Read counts
        long tensorCount = readUInt64();
        long metadataKVCount = readUInt64();

        log.debug("GGUF version {}: {} tensors, {} metadata entries",
                version, tensorCount, metadataKVCount);

        // Read metadata
        Map<String, Object> metadata = new HashMap<>();
        for (long i = 0; i < metadataKVCount; i++) {
            String key = readString();
            int valueType = readInt();
            Object value = readMetadataValue(valueType);
            metadata.put(key, value);

            if (log.isTraceEnabled()) {
                log.trace("Metadata: {} = {} (type {})", key, value, valueType);
            }
        }

        header = GGUFHeader.builder()
                .magic(magic)
                .version(version)
                .tensorCount(tensorCount)
                .metadataKVCount(metadataKVCount)
                .metadata(metadata)
                .build();

        // Get alignment from metadata (default 32)
        alignment = header.getAlignment();

        return header;
    }

    /**
     * Read tensor information from the file
     */
    public List<GGMLTensorInfo> readTensorInfos() throws IOException, GGMLImportException {
        if (header == null) {
            readHeader();
        }

        tensorInfos = new ArrayList<>();

        for (long i = 0; i < header.getTensorCount(); i++) {
            String name = readString();
            int numDimensions = readInt();

            long[] shape = new long[numDimensions];
            for (int d = 0; d < numDimensions; d++) {
                shape[d] = readUInt64();
            }

            int typeId = readInt();
            GGMLDataType dataType;
            try {
                dataType = GGMLDataType.fromTypeId(typeId);
            } catch (IllegalArgumentException e) {
                throw new GGMLImportException("Unknown tensor type ID " + typeId + " for tensor " + name);
            }

            long offset = readUInt64();

            GGMLTensorInfo tensorInfo = GGMLTensorInfo.builder()
                    .name(name)
                    .numDimensions(numDimensions)
                    .shape(shape)
                    .dataType(dataType)
                    .dataOffset(offset)
                    .build();

            tensorInfos.add(tensorInfo);

            if (log.isDebugEnabled()) {
                log.debug("Tensor {}: {} {} at offset {}",
                        i, name, tensorInfo.getShapeString(), offset);
            }
        }

        // Calculate data section offset (aligned)
        long headerEnd = readPosition;
        dataOffset = alignOffset(headerEnd, alignment);

        log.debug("Data section starts at offset {} (aligned from {})", dataOffset, headerEnd);

        return tensorInfos;
    }

    /**
     * Read the raw data for a tensor
     */
    public byte[] readTensorData(GGMLTensorInfo tensorInfo) throws IOException {
        long absoluteOffset = dataOffset + tensorInfo.getDataOffset();
        long dataSize = tensorInfo.getDataSize();

        if (dataSize > Integer.MAX_VALUE) {
            throw new IOException("Tensor data too large to read into byte array: " + dataSize);
        }

        byte[] data = new byte[(int) dataSize];
        readFullyAt(ByteBuffer.wrap(data), absoluteOffset);

        return data;
    }

    /**
     * Read tensor data directly into a direct ByteBuffer, avoiding a heap byte[] copy.
     * The returned buffer is ready for reading (position=0, limit=dataSize) in little-endian order.
     * This is faster for non-quantized tensors where the data can be used directly.
     */
    public ByteBuffer readTensorDataDirect(GGMLTensorInfo tensorInfo) throws IOException {
        long absoluteOffset = dataOffset + tensorInfo.getDataOffset();
        long dataSize = tensorInfo.getDataSize();

        if (dataSize > Integer.MAX_VALUE) {
            throw new IOException("Tensor data too large: " + dataSize);
        }

        ByteBuffer direct = ByteBuffer.allocateDirect((int) dataSize).order(ByteOrder.LITTLE_ENDIAN);
        readFullyAt(direct, absoluteOffset);
        direct.flip();
        return direct;
    }

    /**
     * Read a bounded byte range from one tensor without materializing the complete tensor.
     *
     * @param tensorInfo tensor descriptor returned by this reader
     * @param tensorByteOffset byte offset relative to the start of the tensor payload
     * @param destination destination byte array
     * @param destinationOffset first destination index to fill
     * @param length number of bytes to read
     */
    public void readTensorDataRange(GGMLTensorInfo tensorInfo, long tensorByteOffset,
                                    byte[] destination, int destinationOffset, int length)
            throws IOException {
        if (tensorByteOffset < 0 || length < 0
                || tensorByteOffset > tensorInfo.getDataSize()
                || length > tensorInfo.getDataSize() - tensorByteOffset) {
            throw new IOException("Tensor byte range outside " + tensorInfo.getName()
                    + ": offset=" + tensorByteOffset + ", length=" + length
                    + ", tensorBytes=" + tensorInfo.getDataSize());
        }
        if (destinationOffset < 0 || destinationOffset > destination.length
                || length > destination.length - destinationOffset) {
            throw new IndexOutOfBoundsException("Destination byte range outside array: offset="
                    + destinationOffset + ", length=" + length
                    + ", capacity=" + destination.length);
        }
        long absoluteOffset = Math.addExact(
                Math.addExact(dataOffset, tensorInfo.getDataOffset()), tensorByteOffset);
        readFullyAt(ByteBuffer.wrap(destination, destinationOffset, length), absoluteOffset);
    }

    /**
     * Read a bounded byte range from one tensor directly into a {@link ByteBuffer}
     * without materializing the whole tensor. Direct buffers avoid JVM heap copies
     * for large streamed loads.
     *
     * @param tensorInfo tensor descriptor returned by this reader
     * @param tensorByteOffset byte offset relative to the start of the tensor payload
     * @param destination destination buffer; filled from position() to limit()
     */
    public void readTensorDataRange(GGMLTensorInfo tensorInfo, long tensorByteOffset,
                                    ByteBuffer destination)
            throws IOException {
        int length = destination.remaining();
        if (tensorByteOffset < 0 || length < 0
                || tensorByteOffset > tensorInfo.getDataSize()
                || length > tensorInfo.getDataSize() - tensorByteOffset) {
            throw new IOException("Tensor byte range outside " + tensorInfo.getName()
                    + ": offset=" + tensorByteOffset + ", length=" + length
                    + ", tensorBytes=" + tensorInfo.getDataSize());
        }
        long absoluteOffset = Math.addExact(
                Math.addExact(dataOffset, tensorInfo.getDataOffset()), tensorByteOffset);
        readFullyAt(destination, absoluteOffset);
    }

    /**
     * Get the metadata from the header
     */
    public GGMLMetadata getMetadata() throws IOException, GGMLImportException {
        if (header == null) {
            readHeader();
        }
        if (tensorInfos == null) {
            readTensorInfos();
        }
        return GGMLMetadata.fromGGUF(header, tensorInfos);
    }

    /**
     * Get the parsed header
     */
    public GGUFHeader getHeader() throws IOException, GGMLImportException {
        if (header == null) {
            readHeader();
        }
        return header;
    }

    /**
     * Get the list of tensor infos
     */
    public List<GGMLTensorInfo> getTensorInfos() throws IOException, GGMLImportException {
        if (tensorInfos == null) {
            readTensorInfos();
        }
        return tensorInfos;
    }

    /**
     * Get the offset where tensor data begins
     */
    public long getDataOffset() {
        return dataOffset;
    }

    // Private helper methods

    private String readString() throws IOException {
        long length = readUInt64();
        if (length > Integer.MAX_VALUE || length < 0) {
            throw new IllegalStateException("String length too large: " + length);
        }
        byte[] bytes = new byte[(int) length];
        readBytes(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private long readUInt64() throws IOException {
        return readLong();
    }

    private Object readMetadataValue(int valueType) throws IOException, GGMLImportException {
        switch (valueType) {
            case GGUF_TYPE_UINT8:
                return readByte() & 0xFF;
            case GGUF_TYPE_INT8:
                return readByte();
            case GGUF_TYPE_UINT16:
                return readShort() & 0xFFFF;
            case GGUF_TYPE_INT16:
                return readShort();
            case GGUF_TYPE_UINT32:
                return readInt() & 0xFFFFFFFFL;
            case GGUF_TYPE_INT32:
                return readInt();
            case GGUF_TYPE_FLOAT32:
                return readFloat();
            case GGUF_TYPE_BOOL:
                return readByte() != 0;
            case GGUF_TYPE_STRING:
                return readString();
            case GGUF_TYPE_ARRAY:
                return readArray();
            case GGUF_TYPE_UINT64:
                return readUInt64();
            case GGUF_TYPE_INT64:
                return readLong();
            case GGUF_TYPE_FLOAT64:
                return readDouble();
            default:
                throw new GGMLImportException("Unknown GGUF metadata value type: " + valueType);
        }
    }

    private Object readArray() throws IOException, GGMLImportException {
        int elementType = readInt();
        long length = readUInt64();

        if (length > Integer.MAX_VALUE) {
            throw new GGMLImportException("Array too large: " + length);
        }

        int len = (int) length;

        switch (elementType) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8: {
                byte[] arr = new byte[len];
                readBytes(arr);
                return arr;
            }
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16: {
                short[] arr = new short[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = readShort();
                }
                return arr;
            }
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32: {
                int[] arr = new int[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = readInt();
                }
                return arr;
            }
            case GGUF_TYPE_FLOAT32: {
                float[] arr = new float[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = readFloat();
                }
                return arr;
            }
            case GGUF_TYPE_BOOL: {
                boolean[] arr = new boolean[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = readByte() != 0;
                }
                return arr;
            }
            case GGUF_TYPE_STRING: {
                List<String> arr = new ArrayList<>(len);
                for (int i = 0; i < len; i++) {
                    arr.add(readString());
                }
                return arr;
            }
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64: {
                long[] arr = new long[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = readLong();
                }
                return arr;
            }
            case GGUF_TYPE_FLOAT64: {
                double[] arr = new double[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = readDouble();
                }
                return arr;
            }
            case GGUF_TYPE_ARRAY: {
                List<Object> arr = new ArrayList<>(len);
                for (int i = 0; i < len; i++) {
                    arr.add(readArray());
                }
                return arr;
            }
            default:
                throw new GGMLImportException("Unknown GGUF array element type: " + elementType);
        }
    }

    private void seek(long newPosition) throws IOException {
        if (newPosition < 0 || newPosition > channel.size()) {
            throw new IOException("GGUF seek outside file: " + newPosition);
        }

        long bufferedEnd = bufferStart + buffer.limit();
        if (newPosition >= bufferStart && newPosition <= bufferedEnd) {
            buffer.position((int) (newPosition - bufferStart));
        } else {
            buffer.clear();
            buffer.limit(0);
            bufferStart = newPosition;
        }
        readPosition = newPosition;
    }

    private void ensureAvailable(int requiredBytes) throws IOException {
        if (requiredBytes < 0 || requiredBytes > buffer.capacity()) {
            throw new IllegalArgumentException("Invalid buffered read size: " + requiredBytes);
        }
        if (buffer.remaining() >= requiredBytes) {
            return;
        }

        int preservedBytes = buffer.remaining();
        if (preservedBytes > 0) {
            buffer.compact();
        } else {
            buffer.clear();
        }

        bufferStart = readPosition;
        long filePosition = readPosition + preservedBytes;
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, filePosition);
            if (read < 0) {
                break;
            }
            if (read == 0) {
                break;
            }
            filePosition += read;
        }
        buffer.flip();

        if (buffer.remaining() < requiredBytes) {
            throw new IOException("Unexpected EOF reading GGUF metadata at offset " + readPosition
                    + " (needed " + requiredBytes + " bytes, found " + buffer.remaining() + ")");
        }
    }

    private byte readByte() throws IOException {
        ensureAvailable(Byte.BYTES);
        readPosition += Byte.BYTES;
        return buffer.get();
    }

    private short readShort() throws IOException {
        ensureAvailable(Short.BYTES);
        readPosition += Short.BYTES;
        return buffer.getShort();
    }

    private int readInt() throws IOException {
        ensureAvailable(Integer.BYTES);
        readPosition += Integer.BYTES;
        return buffer.getInt();
    }

    private long readLong() throws IOException {
        ensureAvailable(Long.BYTES);
        readPosition += Long.BYTES;
        return buffer.getLong();
    }

    private float readFloat() throws IOException {
        ensureAvailable(Float.BYTES);
        readPosition += Float.BYTES;
        return buffer.getFloat();
    }

    private double readDouble() throws IOException {
        ensureAvailable(Double.BYTES);
        readPosition += Double.BYTES;
        return buffer.getDouble();
    }

    private void readBytes(byte[] destination) throws IOException {
        int copied = 0;
        while (copied < destination.length) {
            if (!buffer.hasRemaining()) {
                ensureAvailable(1);
            }
            int chunk = Math.min(buffer.remaining(), destination.length - copied);
            buffer.get(destination, copied, chunk);
            copied += chunk;
            readPosition += chunk;
        }
    }

    private void readFullyAt(ByteBuffer destination, long absoluteOffset) throws IOException {
        long bytes = destination.remaining();
        long fileSize = channel.size();
        if (absoluteOffset < 0 || absoluteOffset > fileSize || bytes > fileSize - absoluteOffset) {
            throw new IOException("Tensor data range outside GGUF file: offset=" + absoluteOffset
                    + ", bytes=" + bytes + ", fileSize=" + fileSize);
        }

        long filePosition = absoluteOffset;
        while (destination.hasRemaining()) {
            int read = channel.read(destination, filePosition);
            if (read < 0) {
                throw new IOException("Unexpected EOF reading tensor data at offset " + filePosition);
            }
            if (read == 0) {
                throw new IOException("GGUF tensor read made no progress at offset " + filePosition);
            }
            filePosition += read;
        }
    }

    private static long alignOffset(long offset, int alignment) {
        return ((offset + alignment - 1) / alignment) * alignment;
    }

    @Override
    public void close() throws IOException {
        if (channel != null && channel.isOpen()) {
            channel.close();
        }
        if (raf != null) {
            raf.close();
        }
    }
}
