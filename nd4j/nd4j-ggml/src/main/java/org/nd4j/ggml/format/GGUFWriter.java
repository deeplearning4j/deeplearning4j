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

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Writer for GGUF (GGML Universal Format) files.
 * This is the reverse operation of {@link GGUFReader}.
 *
 * <p>Usage:
 * <pre>{@code
 * try (GGUFWriter writer = new GGUFWriter(outputFile, 3)) {
 *     writer.setAlignment(32);
 *     writer.addMetadata("general.architecture", "llama");
 *     writer.addMetadata("llama.block_count", 32);
 *
 *     writer.registerTensor("token_embd.weight", new long[]{4096, 32000}, GGMLDataType.GGML_TYPE_F16);
 *
 *     writer.writeHeader();
 *
 *     writer.writeTensorData("token_embd.weight", tensorBytes);
 *
 *     writer.finalize();
 * }
 * }</pre>
 */
@Slf4j
public class GGUFWriter implements Closeable {
    private static final int INITIAL_HEADER_BYTES = 1024 * 1024;
    private static final int MAX_HEADER_BYTES = 256 * 1024 * 1024;

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

    private final File outputFile;
    private final RandomAccessFile raf;
    private final FileChannel channel;

    private final int ggufVersion;
    private int alignment = 32;
    private final Map<String, Object> metadata = new LinkedHashMap<>();
    private final List<TensorRegistration> tensorRegistrations = new ArrayList<>();
    private final Map<String, Long> tensorDataOffsets = new LinkedHashMap<>();

    private boolean headerWritten = false;
    private long dataStartOffset = 0;
    private long currentDataOffset = 0;
    private TensorRegistration activeTensor;
    private long activeTensorBytes;

    /**
     * Information about a registered tensor
     */
    private static class TensorRegistration {
        String name;
        long[] shape;
        GGMLDataType dataType;
        long expectedSize;

        TensorRegistration(String name, long[] shape, GGMLDataType dataType) {
            this.name = name;
            this.shape = shape;
            this.dataType = dataType;
            this.expectedSize = calculateTensorSize(shape, dataType);
        }

        private static long calculateTensorSize(long[] shape, GGMLDataType dataType) {
            long numElements = 1;
            for (long dim : shape) {
                numElements *= dim;
            }
            return dataType.calculateStorageBytes(numElements);
        }
    }

    /**
     * Create a new GGUF writer
     *
     * @param outputFile  the output file
     * @param ggufVersion GGUF version (1, 2, or 3)
     */
    public GGUFWriter(File outputFile, int ggufVersion) throws IOException {
        if (ggufVersion < 1 || ggufVersion > 3) {
            throw new IllegalArgumentException("GGUF version must be 1, 2, or 3");
        }

        this.outputFile = outputFile;
        this.ggufVersion = ggufVersion;
        this.raf = new RandomAccessFile(outputFile, "rw");
        this.channel = raf.getChannel();

        // Truncate file if it exists
        channel.truncate(0);
    }

    /**
     * Set the alignment for the data section
     */
    public void setAlignment(int alignment) {
        if (headerWritten) {
            throw new IllegalStateException("Cannot set alignment after header is written");
        }
        if (alignment <= 0) {
            throw new IllegalArgumentException("GGUF alignment must be positive, got " + alignment);
        }
        this.alignment = alignment;
    }

    /**
     * Add metadata with automatic type detection
     */
    public void addMetadata(String key, Object value) {
        if (headerWritten) {
            throw new IllegalStateException("Cannot add metadata after header is written");
        }
        metadata.put(key, value);
    }

    /**
     * Add string metadata
     */
    public void addMetadataString(String key, String value) {
        addMetadata(key, value);
    }

    /**
     * Add integer metadata
     */
    public void addMetadataInt(String key, int value) {
        addMetadata(key, value);
    }

    /**
     * Add long metadata
     */
    public void addMetadataLong(String key, long value) {
        addMetadata(key, value);
    }

    /**
     * Add float metadata
     */
    public void addMetadataFloat(String key, float value) {
        addMetadata(key, value);
    }

    /**
     * Add boolean metadata
     */
    public void addMetadataBool(String key, boolean value) {
        addMetadata(key, value);
    }

    /**
     * Register a tensor to be written
     */
    public void registerTensor(String name, long[] shape, GGMLDataType dataType) {
        if (headerWritten) {
            throw new IllegalStateException("Cannot register tensors after header is written");
        }
        tensorRegistrations.add(new TensorRegistration(name, shape, dataType));
    }

    /**
     * Write the GGUF header including metadata and tensor infos
     */
    public void writeHeader() throws IOException {
        if (headerWritten) {
            throw new IllegalStateException("Header already written");
        }

        // Calculate sizes first to get accurate header
        ByteBuffer headerBuffer = buildHeader();

        // Write header
        headerBuffer.flip();
        writeFully(headerBuffer);

        // Calculate data start (aligned)
        long headerEnd = channel.position();
        dataStartOffset = alignOffset(headerEnd, alignment);
        currentDataOffset = 0;

        // Pad to alignment
        if (dataStartOffset > headerEnd) {
            ByteBuffer padding = ByteBuffer.allocate((int) (dataStartOffset - headerEnd));
            channel.write(padding);
        }

        headerWritten = true;
        log.debug("Header written, data section starts at offset {}", dataStartOffset);
    }

    /**
     * Write tensor data
     *
     * @param name the tensor name (must match a registered tensor)
     * @param data the tensor data bytes
     */
    public void writeTensorData(String name, byte[] data) throws IOException {
        beginTensorData(name);
        writeTensorDataChunk(data, 0, data.length);
        endTensorData();
    }

    /**
     * Begin streaming the payload for one registered tensor.
     */
    public void beginTensorData(String name) {
        if (!headerWritten) {
            throw new IllegalStateException("Must write header before tensor data");
        }
        if (activeTensor != null) {
            throw new IllegalStateException("Tensor data is already active: " + activeTensor.name);
        }
        TensorRegistration registration = null;
        for (TensorRegistration candidate : tensorRegistrations) {
            if (candidate.name.equals(name)) {
                registration = candidate;
                break;
            }
        }
        if (registration == null) {
            throw new IllegalArgumentException("Tensor not registered: " + name);
        }
        if (tensorDataOffsets.containsKey(name)) {
            throw new IllegalStateException("Tensor was already written: " + name);
        }
        activeTensor = registration;
        activeTensorBytes = 0;
        tensorDataOffsets.put(name, currentDataOffset);
    }

    /**
     * Append one bounded chunk to the active tensor payload.
     */
    public void writeTensorDataChunk(byte[] data, int offset, int length) throws IOException {
        if (activeTensor == null) {
            throw new IllegalStateException("No tensor data stream is active");
        }
        if (offset < 0 || length < 0 || offset > data.length || length > data.length - offset) {
            throw new IndexOutOfBoundsException("Tensor chunk outside source array");
        }
        if (length > activeTensor.expectedSize - activeTensorBytes) {
            throw new IOException("Tensor " + activeTensor.name + " exceeds expected size "
                    + activeTensor.expectedSize + " bytes");
        }
        ByteBuffer buffer = ByteBuffer.wrap(data, offset, length);
        while (buffer.hasRemaining()) {
            int written = channel.write(buffer);
            if (written <= 0) {
                throw new IOException("GGUF tensor write made no progress for " + activeTensor.name);
            }
        }
        activeTensorBytes += length;
        currentDataOffset += length;
    }

    /**
     * Finish the active tensor and verify its exact encoded byte count.
     */
    public void endTensorData() throws IOException {
        if (activeTensor == null) {
            throw new IllegalStateException("No tensor data stream is active");
        }
        TensorRegistration completed = activeTensor;
        long completedBytes = activeTensorBytes;
        activeTensor = null;
        activeTensorBytes = 0;
        if (completedBytes != completed.expectedSize) {
            throw new IOException("Tensor " + completed.name + " size mismatch: expected "
                    + completed.expectedSize + " bytes, wrote " + completedBytes);
        }
        log.debug("Wrote tensor {} ({} bytes) at offset {}",
                completed.name, completedBytes, tensorDataOffsets.get(completed.name));
    }

    /**
     * Finalize the GGUF file.
     * This re-writes the header with correct tensor offsets.
     */
    public void finalizeFile() throws IOException {
        if (!headerWritten) {
            throw new IllegalStateException("Header not written");
        }
        if (activeTensor != null) {
            throw new IllegalStateException("Tensor data stream is still active: " + activeTensor.name);
        }

        // Check all tensors were written
        for (TensorRegistration reg : tensorRegistrations) {
            if (!tensorDataOffsets.containsKey(reg.name)) {
                throw new IllegalStateException("Tensor not written: " + reg.name);
            }
        }

        // Re-write header with correct offsets
        channel.position(0);
        ByteBuffer headerBuffer = buildHeader();
        headerBuffer.flip();
        writeFully(headerBuffer);

        log.info("Finalized GGUF file: {} ({} tensors, {} bytes)",
                outputFile.getName(), tensorRegistrations.size(), channel.size());
    }

    private ByteBuffer buildHeader() {
        int capacity = INITIAL_HEADER_BYTES;
        while (true) {
            try {
                return buildHeader(capacity);
            } catch (BufferOverflowException tooSmall) {
                if (capacity >= MAX_HEADER_BYTES) {
                    throw new IllegalStateException("GGUF header exceeds the supported "
                            + MAX_HEADER_BYTES + " byte limit", tooSmall);
                }
                capacity = Math.min(MAX_HEADER_BYTES, Math.multiplyExact(capacity, 2));
            }
        }
    }

    private ByteBuffer buildHeader(int capacity) {
        ByteBuffer buffer = ByteBuffer.allocate(capacity).order(ByteOrder.LITTLE_ENDIAN);

        // Magic: "GGUF"
        buffer.putInt(GGMLFormatDetector.GGUF_MAGIC);

        // Version
        buffer.putInt(ggufVersion);

        // Tensor count
        buffer.putLong(tensorRegistrations.size());

        // Metadata KV count
        buffer.putLong(metadata.size());

        // Write metadata
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            writeString(buffer, entry.getKey());
            writeMetadataValue(buffer, entry.getValue());
        }

        // Write tensor infos
        for (int i = 0; i < tensorRegistrations.size(); i++) {
            TensorRegistration reg = tensorRegistrations.get(i);

            // Name
            writeString(buffer, reg.name);

            // Number of dimensions
            buffer.putInt(reg.shape.length);

            // Shape
            for (long dim : reg.shape) {
                buffer.putLong(dim);
            }

            // Type
            buffer.putInt(reg.dataType.getTypeId());

            // Offset (from start of data section)
            Long offset = tensorDataOffsets.get(reg.name);
            buffer.putLong(offset != null ? offset : 0L);
        }

        return buffer;
    }

    private void writeFully(ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) {
            int written = channel.write(buffer);
            if (written <= 0) {
                throw new IOException("GGUF header write made no progress");
            }
        }
    }

    private void writeString(ByteBuffer buffer, String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        buffer.putLong(bytes.length);
        buffer.put(bytes);
    }

    private void writeMetadataValue(ByteBuffer buffer, Object value) {
        if (value instanceof String) {
            buffer.putInt(GGUF_TYPE_STRING);
            writeString(buffer, (String) value);
        } else if (value instanceof Integer) {
            buffer.putInt(GGUF_TYPE_INT32);
            buffer.putInt((Integer) value);
        } else if (value instanceof Long) {
            buffer.putInt(GGUF_TYPE_INT64);
            buffer.putLong((Long) value);
        } else if (value instanceof Float) {
            buffer.putInt(GGUF_TYPE_FLOAT32);
            buffer.putFloat((Float) value);
        } else if (value instanceof Double) {
            buffer.putInt(GGUF_TYPE_FLOAT64);
            buffer.putDouble((Double) value);
        } else if (value instanceof Boolean) {
            buffer.putInt(GGUF_TYPE_BOOL);
            buffer.put((byte) ((Boolean) value ? 1 : 0));
        } else if (value instanceof Byte) {
            buffer.putInt(GGUF_TYPE_INT8);
            buffer.put((Byte) value);
        } else if (value instanceof Short) {
            buffer.putInt(GGUF_TYPE_INT16);
            buffer.putShort((Short) value);
        } else if (value instanceof byte[]) {
            byte[] arr = (byte[]) value;
            buffer.putInt(GGUF_TYPE_ARRAY);
            buffer.putInt(GGUF_TYPE_INT8);
            buffer.putLong(arr.length);
            buffer.put(arr);
        } else if (value instanceof short[]) {
            short[] arr = (short[]) value;
            buffer.putInt(GGUF_TYPE_ARRAY);
            buffer.putInt(GGUF_TYPE_INT16);
            buffer.putLong(arr.length);
            for (short v : arr) {
                buffer.putShort(v);
            }
        } else if (value instanceof int[]) {
            int[] arr = (int[]) value;
            buffer.putInt(GGUF_TYPE_ARRAY);
            buffer.putInt(GGUF_TYPE_INT32);
            buffer.putLong(arr.length);
            for (int v : arr) {
                buffer.putInt(v);
            }
        } else if (value instanceof long[]) {
            long[] arr = (long[]) value;
            buffer.putInt(GGUF_TYPE_ARRAY);
            buffer.putInt(GGUF_TYPE_INT64);
            buffer.putLong(arr.length);
            for (long v : arr) {
                buffer.putLong(v);
            }
        } else if (value instanceof float[]) {
            float[] arr = (float[]) value;
            buffer.putInt(GGUF_TYPE_ARRAY);
            buffer.putInt(GGUF_TYPE_FLOAT32);
            buffer.putLong(arr.length);
            for (float v : arr) {
                buffer.putFloat(v);
            }
        } else if (value instanceof double[]) {
            double[] arr = (double[]) value;
            buffer.putInt(GGUF_TYPE_ARRAY);
            buffer.putInt(GGUF_TYPE_FLOAT64);
            buffer.putLong(arr.length);
            for (double v : arr) {
                buffer.putDouble(v);
            }
        } else if (value instanceof boolean[]) {
            boolean[] arr = (boolean[]) value;
            buffer.putInt(GGUF_TYPE_ARRAY);
            buffer.putInt(GGUF_TYPE_BOOL);
            buffer.putLong(arr.length);
            for (boolean v : arr) {
                buffer.put((byte) (v ? 1 : 0));
            }
        } else if (value instanceof String[]) {
            String[] arr = (String[]) value;
            buffer.putInt(GGUF_TYPE_ARRAY);
            buffer.putInt(GGUF_TYPE_STRING);
            buffer.putLong(arr.length);
            for (String s : arr) {
                writeString(buffer, s);
            }
        } else if (value instanceof List) {
            List<?> list = (List<?>) value;
            if (list.isEmpty()) {
                buffer.putInt(GGUF_TYPE_ARRAY);
                buffer.putInt(GGUF_TYPE_INT32);
                buffer.putLong(0);
            } else if (list.get(0) instanceof String) {
                buffer.putInt(GGUF_TYPE_ARRAY);
                buffer.putInt(GGUF_TYPE_STRING);
                buffer.putLong(list.size());
                for (Object item : list) {
                    writeString(buffer, (String) item);
                }
            } else if (list.get(0) instanceof Integer) {
                buffer.putInt(GGUF_TYPE_ARRAY);
                buffer.putInt(GGUF_TYPE_INT32);
                buffer.putLong(list.size());
                for (Object item : list) {
                    buffer.putInt((Integer) item);
                }
            } else if (list.get(0) instanceof Float) {
                buffer.putInt(GGUF_TYPE_ARRAY);
                buffer.putInt(GGUF_TYPE_FLOAT32);
                buffer.putLong(list.size());
                for (Object item : list) {
                    buffer.putFloat((Float) item);
                }
            } else if (list.get(0) instanceof Long) {
                buffer.putInt(GGUF_TYPE_ARRAY);
                buffer.putInt(GGUF_TYPE_INT64);
                buffer.putLong(list.size());
                for (Object item : list) {
                    buffer.putLong((Long) item);
                }
            } else if (list.get(0) instanceof Double) {
                buffer.putInt(GGUF_TYPE_ARRAY);
                buffer.putInt(GGUF_TYPE_FLOAT64);
                buffer.putLong(list.size());
                for (Object item : list) {
                    buffer.putDouble((Double) item);
                }
            } else if (list.get(0) instanceof Boolean) {
                buffer.putInt(GGUF_TYPE_ARRAY);
                buffer.putInt(GGUF_TYPE_BOOL);
                buffer.putLong(list.size());
                for (Object item : list) {
                    buffer.put((byte) ((Boolean) item ? 1 : 0));
                }
            } else {
                throw new IllegalArgumentException("Unsupported list element type: " + list.get(0).getClass());
            }
        } else {
            throw new IllegalArgumentException("Unsupported metadata value type: " + value.getClass());
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
