/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.ggml;

import lombok.Builder;
import lombok.Data;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * GGUF file header parser.
 * GGUF format: magic (4) + version (4) + tensor_count (8) + metadata_kv_count (8) + metadata + tensor_info + data
 */
@Data
@Builder
public class GGUFHeader {

    public static final int GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian
    public static final int GGUF_VERSION_MIN = 2;
    public static final int GGUF_VERSION_MAX = 3;

    private int version;
    private long tensorCount;
    private long metadataKvCount;
    private Map<String, Object> metadata;
    private Map<String, TensorInfo> tensors;
    private long dataOffset;

    @Data
    @Builder
    public static class TensorInfo {
        private String name;
        private int numDimensions;
        private long[] dimensions;
        private GGUFType type;
        private long offset;

        public long[] getShape() {
            if (dimensions == null) return new long[0];
            long[] shape = new long[dimensions.length];
            for (int i = 0; i < dimensions.length; i++) {
                shape[i] = dimensions[dimensions.length - 1 - i];
            }
            return shape;
        }

        public long getElementCount() {
            if (dimensions == null || dimensions.length == 0) return 0;
            long count = 1;
            for (long dim : dimensions) {
                count *= dim;
            }
            return count;
        }
    }

    public static GGUFHeader fromFile(File file) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            return fromRandomAccessFile(raf);
        }
    }

    public static GGUFHeader fromRandomAccessFile(RandomAccessFile raf) throws IOException {
        int magic = Integer.reverseBytes(raf.readInt());
        if (magic != GGUF_MAGIC) {
            throw new IOException("Invalid GGUF magic: " + Integer.toHexString(magic));
        }

        int version = Integer.reverseBytes(raf.readInt());
        if (version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX) {
            throw new IOException("Unsupported GGUF version: " + version);
        }

        long tensorCount = Long.reverseBytes(raf.readLong());
        long metadataKvCount = Long.reverseBytes(raf.readLong());

        Map<String, Object> metadata = readMetadata(raf, metadataKvCount);
        Map<String, TensorInfo> tensors = readTensorInfos(raf, tensorCount);

        long dataOffset = raf.getFilePointer();
        dataOffset = alignToOffset(dataOffset, 32);

        return GGUFHeader.builder()
                .version(version)
                .tensorCount(tensorCount)
                .metadataKvCount(metadataKvCount)
                .metadata(metadata)
                .tensors(tensors)
                .dataOffset(dataOffset)
                .build();
    }

    private static Map<String, Object> readMetadata(RandomAccessFile raf, long count) throws IOException {
        Map<String, Object> metadata = new LinkedHashMap<>();
        for (long i = 0; i < count; i++) {
            String key = readString(raf);
            Object value = readMetadataValue(raf);
            metadata.put(key, value);
        }
        return metadata;
    }

    private static Object readMetadataValue(RandomAccessFile raf) throws IOException {
        int typeId = Integer.reverseBytes(raf.readInt());
        GGUFMetadataType type = GGUFMetadataType.fromId(typeId);

        switch (type) {
            case UINT8:
                return raf.readByte() & 0xFF;
            case INT8:
                return raf.readByte();
            case UINT16:
                return Short.reverseBytes(raf.readShort()) & 0xFFFF;
            case INT16:
                return Short.reverseBytes(raf.readShort());
            case UINT32:
                return Integer.reverseBytes(raf.readInt()) & 0xFFFFFFFFL;
            case INT32:
                return Integer.reverseBytes(raf.readInt());
            case FLOAT32:
                return Float.intBitsToFloat(Integer.reverseBytes(raf.readInt()));
            case BOOL:
                return raf.readByte() != 0;
            case STRING:
                return readString(raf);
            case ARRAY:
                return readArray(raf);
            case UINT64:
            case INT64:
                return Long.reverseBytes(raf.readLong());
            case FLOAT64:
                return Double.longBitsToDouble(Long.reverseBytes(raf.readLong()));
            default:
                throw new IOException("Unknown metadata type: " + type);
        }
    }

    private static String readString(RandomAccessFile raf) throws IOException {
        long length = Long.reverseBytes(raf.readLong());
        if (length > 10_000_000) {
            throw new IOException("String too long: " + length);
        }
        byte[] bytes = new byte[(int) length];
        raf.readFully(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static List<Object> readArray(RandomAccessFile raf) throws IOException {
        int elementTypeId = Integer.reverseBytes(raf.readInt());
        long count = Long.reverseBytes(raf.readLong());
        GGUFMetadataType elementType = GGUFMetadataType.fromId(elementTypeId);

        List<Object> array = new ArrayList<>();
        for (long i = 0; i < count; i++) {
            array.add(readArrayElement(raf, elementType));
        }
        return array;
    }

    private static Object readArrayElement(RandomAccessFile raf, GGUFMetadataType type) throws IOException {
        switch (type) {
            case UINT8:
                return raf.readByte() & 0xFF;
            case INT8:
                return raf.readByte();
            case UINT16:
                return Short.reverseBytes(raf.readShort()) & 0xFFFF;
            case INT16:
                return Short.reverseBytes(raf.readShort());
            case UINT32:
                return Integer.reverseBytes(raf.readInt()) & 0xFFFFFFFFL;
            case INT32:
                return Integer.reverseBytes(raf.readInt());
            case FLOAT32:
                return Float.intBitsToFloat(Integer.reverseBytes(raf.readInt()));
            case BOOL:
                return raf.readByte() != 0;
            case STRING:
                return readString(raf);
            case UINT64:
            case INT64:
                return Long.reverseBytes(raf.readLong());
            case FLOAT64:
                return Double.longBitsToDouble(Long.reverseBytes(raf.readLong()));
            default:
                throw new IOException("Unknown array element type: " + type);
        }
    }

    private static Map<String, TensorInfo> readTensorInfos(RandomAccessFile raf, long count) throws IOException {
        Map<String, TensorInfo> tensors = new LinkedHashMap<>();
        for (long i = 0; i < count; i++) {
            String name = readString(raf);
            int numDimensions = Integer.reverseBytes(raf.readInt());

            long[] dimensions = new long[numDimensions];
            for (int d = 0; d < numDimensions; d++) {
                dimensions[d] = Long.reverseBytes(raf.readLong());
            }

            int typeId = Integer.reverseBytes(raf.readInt());
            GGUFType type = GGUFType.fromId(typeId);
            long offset = Long.reverseBytes(raf.readLong());

            tensors.put(name, TensorInfo.builder()
                    .name(name)
                    .numDimensions(numDimensions)
                    .dimensions(dimensions)
                    .type(type)
                    .offset(offset)
                    .build());
        }
        return tensors;
    }

    private static long alignToOffset(long offset, int alignment) {
        return ((offset + alignment - 1) / alignment) * alignment;
    }

    public Set<String> getTensorNames() {
        return tensors != null ? tensors.keySet() : Collections.emptySet();
    }

    public TensorInfo getTensorInfo(String name) {
        return tensors != null ? tensors.get(name) : null;
    }

    public int getTensorCount() {
        return tensors != null ? tensors.size() : 0;
    }

    @SuppressWarnings("unchecked")
    public <T> T getMetadata(String key) {
        return metadata != null ? (T) metadata.get(key) : null;
    }

    public String getArchitecture() {
        return getMetadata("general.architecture");
    }

    public String getModelName() {
        return getMetadata("general.name");
    }

    public Integer getContextLength() {
        // Try the actual architecture prefix first (e.g. "mistral.context_length", "phi.context_length")
        String arch = getArchitecture();
        if (arch != null) {
            Object val = getMetadata(arch + ".context_length");
            if (val != null) return ((Number) val).intValue();
        }
        // Fall back to known architecture prefixes
        Object val = getMetadata("llama.context_length");
        if (val == null) val = getMetadata("qwen2.context_length");
        if (val == null) val = getMetadata("phi3.context_length");
        return val != null ? ((Number) val).intValue() : null;
    }

    public Integer getEmbeddingLength() {
        // Try the actual architecture prefix first
        String arch = getArchitecture();
        if (arch != null) {
            Object val = getMetadata(arch + ".embedding_length");
            if (val != null) return ((Number) val).intValue();
        }
        // Fall back to known architecture prefixes
        Object val = getMetadata("llama.embedding_length");
        if (val == null) val = getMetadata("qwen2.embedding_length");
        if (val == null) val = getMetadata("phi3.embedding_length");
        return val != null ? ((Number) val).intValue() : null;
    }

    public Integer getBlockCount() {
        // Try the actual architecture prefix first
        String arch = getArchitecture();
        if (arch != null) {
            Object val = getMetadata(arch + ".block_count");
            if (val != null) return ((Number) val).intValue();
        }
        // Fall back to known architecture prefixes
        Object val = getMetadata("llama.block_count");
        if (val == null) val = getMetadata("qwen2.block_count");
        if (val == null) val = getMetadata("phi3.block_count");
        return val != null ? ((Number) val).intValue() : null;
    }
}
