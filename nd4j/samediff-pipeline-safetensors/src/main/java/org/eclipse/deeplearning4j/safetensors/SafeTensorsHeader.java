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

package org.eclipse.deeplearning4j.safetensors;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import lombok.Builder;
import lombok.Data;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Parses and represents the SafeTensors file header.
 * SafeTensors format: 8-byte header size (little-endian) + JSON header + tensor data
 */
@Data
@Builder
public class SafeTensorsHeader {

    private static final Gson GSON = new Gson();

    private long headerSize;
    private Map<String, TensorInfo> tensors;
    private Map<String, String> metadata;

    @Data
    @Builder
    public static class TensorInfo {
        private String name;
        private String dtype;
        private long[] shape;
        private long[] dataOffsets;

        public long getDataStart() {
            return dataOffsets != null && dataOffsets.length > 0 ? dataOffsets[0] : 0;
        }

        public long getDataEnd() {
            return dataOffsets != null && dataOffsets.length > 1 ? dataOffsets[1] : 0;
        }

        public long getDataLength() {
            return getDataEnd() - getDataStart();
        }

        public SafeTensorsDtype getSafeTensorsDtype() {
            return SafeTensorsDtype.fromString(dtype);
        }
    }

    public static SafeTensorsHeader fromFile(File file) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            return fromRandomAccessFile(raf);
        }
    }

    public static SafeTensorsHeader fromRandomAccessFile(RandomAccessFile raf) throws IOException {
        byte[] sizeBytes = new byte[8];
        raf.readFully(sizeBytes);
        long headerSize = ByteBuffer.wrap(sizeBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();

        if (headerSize <= 0 || headerSize > 100_000_000) {
            throw new IOException("Invalid SafeTensors header size: " + headerSize);
        }

        byte[] headerBytes = new byte[(int) headerSize];
        raf.readFully(headerBytes);
        String headerJson = new String(headerBytes, StandardCharsets.UTF_8);

        return parseHeader(headerSize, headerJson);
    }

    public static SafeTensorsHeader fromInputStream(InputStream is) throws IOException {
        DataInputStream dis = new DataInputStream(is);
        byte[] sizeBytes = new byte[8];
        dis.readFully(sizeBytes);
        long headerSize = ByteBuffer.wrap(sizeBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();

        if (headerSize <= 0 || headerSize > 100_000_000) {
            throw new IOException("Invalid SafeTensors header size: " + headerSize);
        }

        byte[] headerBytes = new byte[(int) headerSize];
        dis.readFully(headerBytes);
        String headerJson = new String(headerBytes, StandardCharsets.UTF_8);

        return parseHeader(headerSize, headerJson);
    }

    private static SafeTensorsHeader parseHeader(long headerSize, String headerJson) {
        JsonObject json = GSON.fromJson(headerJson, JsonObject.class);

        Map<String, TensorInfo> tensors = new LinkedHashMap<>();
        Map<String, String> metadata = new LinkedHashMap<>();

        for (Map.Entry<String, JsonElement> entry : json.entrySet()) {
            String key = entry.getKey();
            JsonElement value = entry.getValue();

            if ("__metadata__".equals(key)) {
                if (value.isJsonObject()) {
                    for (Map.Entry<String, JsonElement> me : value.getAsJsonObject().entrySet()) {
                        metadata.put(me.getKey(), me.getValue().getAsString());
                    }
                }
            } else {
                JsonObject tensorJson = value.getAsJsonObject();
                TensorInfo.TensorInfoBuilder tib = TensorInfo.builder().name(key);

                if (tensorJson.has("dtype")) {
                    tib.dtype(tensorJson.get("dtype").getAsString());
                }

                if (tensorJson.has("shape")) {
                    List<Long> shapeList = new ArrayList<>();
                    for (JsonElement se : tensorJson.getAsJsonArray("shape")) {
                        shapeList.add(se.getAsLong());
                    }
                    tib.shape(shapeList.stream().mapToLong(Long::longValue).toArray());
                }

                if (tensorJson.has("data_offsets")) {
                    List<Long> offsetList = new ArrayList<>();
                    for (JsonElement oe : tensorJson.getAsJsonArray("data_offsets")) {
                        offsetList.add(oe.getAsLong());
                    }
                    tib.dataOffsets(offsetList.stream().mapToLong(Long::longValue).toArray());
                }

                tensors.put(key, tib.build());
            }
        }

        return SafeTensorsHeader.builder()
                .headerSize(headerSize)
                .tensors(tensors)
                .metadata(metadata)
                .build();
    }

    public long getDataOffset() {
        return 8 + headerSize;
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
}
