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
import org.nd4j.ggml.quantization.DequantizerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.LinkedHashSet;

/**
 * Reader for GGUF (GGML Universal Format) files.
 * Supports GGUF versions 1, 2, and 3.
 */
@Slf4j
public class GGUFReader implements Closeable {

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
    private MappedByteBuffer buffer;

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

        // Memory-map the file for efficient access
        this.buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, Math.min(channel.size(), Integer.MAX_VALUE));
        this.buffer.order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Read and parse the GGUF header
     */
    public GGUFHeader readHeader() throws IOException, GGMLImportException {
        buffer.position(0);

        // Read magic
        int magic = buffer.getInt();
        if (magic != GGMLFormatDetector.GGUF_MAGIC) {
            throw new GGMLImportException(String.format(
                    "Invalid GGUF magic: 0x%08X (expected 0x%08X)", magic, GGMLFormatDetector.GGUF_MAGIC));
        }

        // Read version
        int version = buffer.getInt();
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
            int valueType = buffer.getInt();
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
            int numDimensions = buffer.getInt();

            long[] shape = new long[numDimensions];
            for (int d = 0; d < numDimensions; d++) {
                shape[d] = readUInt64();
            }

            int typeId = buffer.getInt();
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
        int headerEnd = buffer.position();
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

        // Handle large files that may need multiple mapped regions
        if (absoluteOffset + dataSize > buffer.capacity()) {
            // Re-map to include this region
            channel.position(absoluteOffset);
            ByteBuffer tempBuffer = ByteBuffer.allocate((int) dataSize);
            channel.read(tempBuffer);
            tempBuffer.flip();
            tempBuffer.get(data);
        } else {
            buffer.position((int) absoluteOffset);
            buffer.get(data);
        }

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

        if (absoluteOffset + dataSize > buffer.capacity()) {
            // Beyond mmap window — read via channel
            channel.position(absoluteOffset);
            while (direct.hasRemaining()) {
                if (channel.read(direct) == -1) {
                    throw new IOException("Unexpected EOF reading tensor data at offset " + absoluteOffset);
                }
            }
        } else {
            // Within mmap window — bulk copy from mmap buffer
            ByteBuffer slice = buffer.duplicate();
            slice.position((int) absoluteOffset);
            slice.limit((int) (absoluteOffset + dataSize));
            direct.put(slice);
        }
        direct.flip();
        return direct;
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

    // ========== Static factory methods ==========

    /**
     * Open a GGUF reader for the given file.
     */
    public static GGUFReader open(File file) throws IOException {
        return new GGUFReader(file);
    }

    /**
     * Open a GGUF reader for the file at the given path.
     */
    public static GGUFReader open(String path) throws IOException {
        return new GGUFReader(new File(path));
    }

    // ========== High-level INDArray access ==========

    /**
     * Get the names of all tensors in the file. Parses header and tensor infos if not yet done.
     */
    public Set<String> getTensorNames() throws IOException, GGMLImportException {
        List<GGMLTensorInfo> infos = getTensorInfos();
        Set<String> names = new LinkedHashSet<>(infos.size());
        for (GGMLTensorInfo info : infos) {
            names.add(info.getName());
        }
        return names;
    }

    /**
     * Get the number of tensors in the file.
     */
    public int getTensorCount() throws IOException, GGMLImportException {
        return getTensorInfos().size();
    }

    /**
     * Get the tensor info for a tensor by name, or {@code null} if not found.
     */
    public GGMLTensorInfo getTensorInfo(String name) throws IOException, GGMLImportException {
        for (GGMLTensorInfo info : getTensorInfos()) {
            if (info.getName().equals(name)) {
                return info;
            }
        }
        return null;
    }

    /**
     * Read a tensor by name, dequantizing quantized types to float32.
     *
     * @param name tensor name
     * @return INDArray with the tensor data
     * @throws IllegalArgumentException if the tensor is not found
     */
    public INDArray readTensor(String name) throws IOException, GGMLImportException {
        return readTensor(name, true);
    }

    /**
     * Read a tensor by name.
     *
     * @param name       tensor name
     * @param dequantize if true, dequantize quantized tensors to float32; if false, return raw bytes
     * @return INDArray with the tensor data
     * @throws IllegalArgumentException if the tensor is not found
     */
    public INDArray readTensor(String name, boolean dequantize) throws IOException, GGMLImportException {
        GGMLTensorInfo info = getTensorInfo(name);
        if (info == null) {
            throw new IllegalArgumentException("Tensor not found: " + name);
        }
        return readTensorAsArray(info, dequantize);
    }

    /**
     * Read all tensors from the file, dequantizing quantized types to float32.
     * Tensors whose types are unsupported (e.g. exotic quantization variants) are skipped with a warning.
     *
     * @return ordered map of tensor name to INDArray
     */
    public Map<String, INDArray> readAllTensors() throws IOException, GGMLImportException {
        return readAllTensors(true);
    }

    /**
     * Read all tensors from the file.
     *
     * @param dequantize if true, dequantize quantized tensors to float32
     * @return ordered map of tensor name to INDArray
     */
    public Map<String, INDArray> readAllTensors(boolean dequantize) throws IOException, GGMLImportException {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (GGMLTensorInfo info : getTensorInfos()) {
            try {
                result.put(info.getName(), readTensorAsArray(info, dequantize));
            } catch (UnsupportedOperationException e) {
                log.warn("Skipping unsupported tensor {}: {}", info.getName(), e.getMessage());
            }
        }
        return result;
    }

    /**
     * Read a subset of tensors from the file.
     *
     * @param names      tensor names to read
     * @param dequantize if true, dequantize quantized tensors to float32
     * @return ordered map of tensor name to INDArray (only found tensors are included)
     */
    public Map<String, INDArray> readTensors(Collection<String> names, boolean dequantize)
            throws IOException, GGMLImportException {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (String name : names) {
            GGMLTensorInfo info = getTensorInfo(name);
            if (info == null) {
                log.warn("Tensor not found in file: {}", name);
                continue;
            }
            try {
                result.put(name, readTensorAsArray(info, dequantize));
            } catch (UnsupportedOperationException e) {
                log.warn("Skipping unsupported tensor {}: {}", name, e.getMessage());
            }
        }
        return result;
    }

    /**
     * Internal: convert a GGMLTensorInfo to an INDArray.
     */
    private INDArray readTensorAsArray(GGMLTensorInfo info, boolean dequantize)
            throws IOException, GGMLImportException {
        GGMLDataType dtype = info.getDataType();
        long[] shape = info.getShape();
        if (shape == null || shape.length == 0) {
            shape = new long[]{1};
        }
        long elementCount = info.getNumElements();

        if (dtype.isQuantized()) {
            byte[] raw = readTensorData(info);
            if (dequantize) {
                return DequantizerFactory.dequantizeToArray(raw, dtype, shape, DataType.FLOAT);
            } else {
                // Return raw quantized bytes as an INT8 row vector
                return Nd4j.createFromArray(raw).reshape(1, raw.length);
            }
        }

        // Non-quantized: read via direct ByteBuffer for efficiency
        ByteBuffer buf = readTensorDataDirect(info);
        DataType nd4jType = dtype.getNd4jType();
        if (nd4jType == null) {
            throw new UnsupportedOperationException("Unsupported GGML type for direct read: " + dtype);
        }

        switch (dtype) {
            case GGML_TYPE_F32: {
                float[] floats = new float[(int) elementCount];
                buf.asFloatBuffer().get(floats);
                return Nd4j.create(floats, shape, 'c');
            }
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16: {
                short[] halfs = new short[(int) elementCount];
                buf.asShortBuffer().get(halfs);
                org.nd4j.linalg.api.buffer.DataBuffer db = Nd4j.createTypedBuffer(halfs, nd4jType);
                return Nd4j.create(db, shape, Nd4j.getStrides(shape, 'c'), 0, 'c', nd4jType);
            }
            case GGML_TYPE_F64: {
                double[] doubles = new double[(int) elementCount];
                buf.asDoubleBuffer().get(doubles);
                return Nd4j.create(doubles, shape, 'c');
            }
            case GGML_TYPE_I32: {
                int[] ints = new int[(int) elementCount];
                buf.asIntBuffer().get(ints);
                return Nd4j.createFromArray(ints).reshape(shape);
            }
            case GGML_TYPE_I64: {
                long[] longs = new long[(int) elementCount];
                buf.asLongBuffer().get(longs);
                return Nd4j.createFromArray(longs).reshape(shape);
            }
            case GGML_TYPE_I8: {
                byte[] bytes = new byte[(int) elementCount];
                buf.get(bytes);
                return Nd4j.createFromArray(bytes).reshape(shape);
            }
            case GGML_TYPE_I16: {
                short[] shorts = new short[(int) elementCount];
                buf.asShortBuffer().get(shorts);
                return Nd4j.createFromArray(shorts).reshape(shape);
            }
            default:
                throw new UnsupportedOperationException("Unsupported GGML type: " + dtype);
        }
    }

    // Private helper methods

    private String readString() {
        long length = readUInt64();
        if (length > Integer.MAX_VALUE || length < 0) {
            throw new IllegalStateException("String length too large: " + length);
        }
        byte[] bytes = new byte[(int) length];
        buffer.get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private long readUInt64() {
        return buffer.getLong();
    }

    private Object readMetadataValue(int valueType) throws GGMLImportException {
        switch (valueType) {
            case GGUF_TYPE_UINT8:
                return buffer.get() & 0xFF;
            case GGUF_TYPE_INT8:
                return buffer.get();
            case GGUF_TYPE_UINT16:
                return buffer.getShort() & 0xFFFF;
            case GGUF_TYPE_INT16:
                return buffer.getShort();
            case GGUF_TYPE_UINT32:
                return buffer.getInt() & 0xFFFFFFFFL;
            case GGUF_TYPE_INT32:
                return buffer.getInt();
            case GGUF_TYPE_FLOAT32:
                return buffer.getFloat();
            case GGUF_TYPE_BOOL:
                return buffer.get() != 0;
            case GGUF_TYPE_STRING:
                return readString();
            case GGUF_TYPE_ARRAY:
                return readArray();
            case GGUF_TYPE_UINT64:
                return readUInt64();
            case GGUF_TYPE_INT64:
                return buffer.getLong();
            case GGUF_TYPE_FLOAT64:
                return buffer.getDouble();
            default:
                throw new GGMLImportException("Unknown GGUF metadata value type: " + valueType);
        }
    }

    private Object readArray() throws GGMLImportException {
        int elementType = buffer.getInt();
        long length = readUInt64();

        if (length > Integer.MAX_VALUE) {
            throw new GGMLImportException("Array too large: " + length);
        }

        int len = (int) length;

        switch (elementType) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8: {
                byte[] arr = new byte[len];
                buffer.get(arr);
                return arr;
            }
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16: {
                short[] arr = new short[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = buffer.getShort();
                }
                return arr;
            }
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32: {
                int[] arr = new int[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = buffer.getInt();
                }
                return arr;
            }
            case GGUF_TYPE_FLOAT32: {
                float[] arr = new float[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = buffer.getFloat();
                }
                return arr;
            }
            case GGUF_TYPE_BOOL: {
                boolean[] arr = new boolean[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = buffer.get() != 0;
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
                    arr[i] = buffer.getLong();
                }
                return arr;
            }
            case GGUF_TYPE_FLOAT64: {
                double[] arr = new double[len];
                for (int i = 0; i < len; i++) {
                    arr[i] = buffer.getDouble();
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

    private static long alignOffset(long offset, int alignment) {
        return ((offset + alignment - 1) / alignment) * alignment;
    }

    @Override
    public void close() throws IOException {
        if (buffer != null) {
            // MappedByteBuffer doesn't have an explicit close,
            // it will be garbage collected
            buffer = null;
        }
        if (channel != null && channel.isOpen()) {
            channel.close();
        }
        if (raf != null) {
            raf.close();
        }
    }
}
