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

package org.nd4j.torchscript.format;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Builder;
import lombok.Data;
import org.nd4j.torchscript.TorchScriptImportException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Parsed Safetensors JSON header.
 *
 * <p>Example header structure:
 * <pre>
 * {
 *   "__metadata__": {"format": "pt"},
 *   "layer1.weight": {"dtype": "F32", "shape": [64, 3, 7, 7], "data_offsets": [0, 37632]},
 *   "layer1.bias": {"dtype": "F32", "shape": [64], "data_offsets": [37632, 37888]}
 * }
 * </pre>
 */
@Data
@Builder
public class SafetensorsHeader {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final String METADATA_KEY = "__metadata__";

    /**
     * Optional metadata from the __metadata__ key
     */
    @Builder.Default
    private Map<String, Object> metadata = new HashMap<>();

    /**
     * Map of tensor names to their entries
     */
    @Builder.Default
    private Map<String, TensorEntry> tensors = new HashMap<>();

    /**
     * Represents a single tensor entry in the Safetensors header.
     */
    @Data
    @Builder
    public static class TensorEntry {
        /**
         * Data type string (e.g., "F32", "F16", "BF16")
         */
        private String dtype;

        /**
         * Tensor shape
         */
        private long[] shape;

        /**
         * Data offsets [start, end] relative to data section
         */
        private long[] dataOffsets;

        /**
         * Get the data size in bytes.
         */
        public long getDataSize() {
            if (dataOffsets == null || dataOffsets.length < 2) {
                return 0;
            }
            return dataOffsets[1] - dataOffsets[0];
        }

        /**
         * Convert to TorchScriptDataType.
         */
        public TorchScriptDataType getDataType() {
            return TorchScriptDataType.fromSafetensorsType(dtype);
        }
    }

    /**
     * Parse a Safetensors JSON header.
     *
     * @param json the JSON string
     * @return parsed header
     * @throws TorchScriptImportException if parsing fails
     */
    @SuppressWarnings("unchecked")
    public static SafetensorsHeader parse(String json) throws TorchScriptImportException {
        try {
            Map<String, Object> rawHeader = OBJECT_MAPPER.readValue(json,
                    new TypeReference<Map<String, Object>>() {});

            SafetensorsHeader.SafetensorsHeaderBuilder builder = SafetensorsHeader.builder();

            Map<String, Object> metadata = new HashMap<>();
            Map<String, TensorEntry> tensors = new HashMap<>();

            for (Map.Entry<String, Object> entry : rawHeader.entrySet()) {
                String key = entry.getKey();

                if (METADATA_KEY.equals(key)) {
                    // Parse metadata section
                    if (entry.getValue() instanceof Map) {
                        metadata = (Map<String, Object>) entry.getValue();
                    }
                } else {
                    // Parse tensor entry
                    if (entry.getValue() instanceof Map) {
                        Map<String, Object> tensorMap = (Map<String, Object>) entry.getValue();
                        TensorEntry tensorEntry = parseTensorEntry(tensorMap);
                        tensors.put(key, tensorEntry);
                    }
                }
            }

            return builder
                    .metadata(metadata)
                    .tensors(tensors)
                    .build();

        } catch (IOException e) {
            throw new TorchScriptImportException("Failed to parse Safetensors header: " + e.getMessage(), e);
        }
    }

    @SuppressWarnings("unchecked")
    private static TensorEntry parseTensorEntry(Map<String, Object> map) {
        String dtype = (String) map.get("dtype");

        // Parse shape
        List<Number> shapeList = (List<Number>) map.get("shape");
        long[] shape = new long[shapeList.size()];
        for (int i = 0; i < shapeList.size(); i++) {
            shape[i] = shapeList.get(i).longValue();
        }

        // Parse data_offsets
        List<Number> offsetList = (List<Number>) map.get("data_offsets");
        long[] offsets = new long[offsetList.size()];
        for (int i = 0; i < offsetList.size(); i++) {
            offsets[i] = offsetList.get(i).longValue();
        }

        return TensorEntry.builder()
                .dtype(dtype)
                .shape(shape)
                .dataOffsets(offsets)
                .build();
    }

    /**
     * Convert header tensor entries to TorchScriptTensorInfo list.
     *
     * @param dataOffset base offset of data section in file
     * @return list of tensor info objects
     */
    public List<TorchScriptTensorInfo> toTensorInfoList(long dataOffset) {
        List<TorchScriptTensorInfo> infos = new ArrayList<>();

        for (Map.Entry<String, TensorEntry> entry : tensors.entrySet()) {
            String name = entry.getKey();
            TensorEntry te = entry.getValue();

            TorchScriptTensorInfo info = TorchScriptTensorInfo.builder()
                    .name(name)
                    .numDimensions(te.getShape().length)
                    .shape(te.getShape())
                    .dataType(te.getDataType())
                    .dataOffset(dataOffset + te.getDataOffsets()[0])
                    .dataSize(te.getDataSize())
                    .requiresGrad(!name.contains("running_") && !name.contains("num_batches"))
                    .isBuffer(name.contains("running_") || name.contains("num_batches"))
                    .build();

            infos.add(info);
        }

        return infos;
    }

    /**
     * Get the total number of tensors.
     */
    public int getTensorCount() {
        return tensors.size();
    }

    /**
     * Get a specific tensor entry by name.
     */
    public TensorEntry getTensor(String name) {
        return tensors.get(name);
    }
}
