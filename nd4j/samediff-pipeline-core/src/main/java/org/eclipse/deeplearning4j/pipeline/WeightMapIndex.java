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

package org.eclipse.deeplearning4j.pipeline;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import lombok.Builder;
import lombok.Data;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Represents a model.safetensors.index.json file for sharded model loading.
 */
@Data
@Builder
public class WeightMapIndex {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final String SAFETENSORS_INDEX = "model.safetensors.index.json";
    public static final String PYTORCH_INDEX = "pytorch_model.bin.index.json";

    private Metadata metadata;
    private Map<String, String> weightMap;

    @Data
    @Builder
    public static class Metadata {
        private long totalSize;
    }

    public static WeightMapIndex fromFile(File file) throws IOException {
        try (Reader reader = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)) {
            JsonObject json = GSON.fromJson(reader, JsonObject.class);
            return fromJson(json);
        }
    }

    public static WeightMapIndex fromJson(JsonObject json) {
        WeightMapIndexBuilder builder = WeightMapIndex.builder();
        if (json.has("metadata")) {
            JsonObject meta = json.getAsJsonObject("metadata");
            Metadata.MetadataBuilder mb = Metadata.builder();
            if (meta.has("total_size")) mb.totalSize(meta.get("total_size").getAsLong());
            builder.metadata(mb.build());
        }
        if (json.has("weight_map")) {
            Map<String, String> wm = new LinkedHashMap<>();
            JsonObject wmJson = json.getAsJsonObject("weight_map");
            for (Map.Entry<String, com.google.gson.JsonElement> e : wmJson.entrySet()) {
                wm.put(e.getKey(), e.getValue().getAsString());
            }
            builder.weightMap(wm);
        }
        return builder.build();
    }

    public static File detectIndexFile(File dir) {
        File st = new File(dir, SAFETENSORS_INDEX);
        if (st.exists()) return st;
        File pt = new File(dir, PYTORCH_INDEX);
        if (pt.exists()) return pt;
        return null;
    }

    public static Optional<WeightMapIndex> loadIfExists(File dir) {
        File indexFile = detectIndexFile(dir);
        if (indexFile == null) return Optional.empty();
        try {
            return Optional.of(fromFile(indexFile));
        } catch (IOException e) {
            return Optional.empty();
        }
    }

    public Set<String> getShardFiles() {
        if (weightMap == null) return Collections.emptySet();
        return new LinkedHashSet<>(weightMap.values());
    }

    public int getShardCount() {
        return getShardFiles().size();
    }

    public long getTotalSize() {
        return metadata != null ? metadata.getTotalSize() : -1;
    }

    public String getShardForWeight(String weightName) {
        if (weightMap == null) return null;
        return weightMap.get(weightName);
    }
}
