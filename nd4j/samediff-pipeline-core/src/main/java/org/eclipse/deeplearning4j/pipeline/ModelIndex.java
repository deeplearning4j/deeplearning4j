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
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import lombok.Builder;
import lombok.Data;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Represents a model_index.json file following HuggingFace Diffusers convention.
 */
@Data
@Builder
public class ModelIndex {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final String MODEL_INDEX_FILE = "model_index.json";

    private String className;
    private String diffusersVersion;
    private Map<String, ComponentDefinition> components;

    @Data
    @Builder
    public static class ComponentDefinition {
        private String library;
        private String className;
        private String subfolder;

        public static ComponentDefinition fromJsonArray(List<String> arr, String componentName) {
            if (arr == null || arr.size() < 2) {
                return ComponentDefinition.builder().library("unknown").className("unknown").subfolder(componentName).build();
            }
            return ComponentDefinition.builder().library(arr.get(0)).className(arr.get(1)).subfolder(componentName).build();
        }
    }

    public static ModelIndex fromFile(File file) throws IOException {
        try (Reader reader = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)) {
            JsonObject json = GSON.fromJson(reader, JsonObject.class);
            return fromJson(json);
        }
    }

    public static ModelIndex fromJson(JsonObject json) {
        ModelIndexBuilder builder = ModelIndex.builder();
        if (json.has("_class_name")) builder.className(json.get("_class_name").getAsString());
        if (json.has("_diffusers_version")) builder.diffusersVersion(json.get("_diffusers_version").getAsString());

        Map<String, ComponentDefinition> components = new LinkedHashMap<>();
        for (Map.Entry<String, JsonElement> entry : json.entrySet()) {
            String key = entry.getKey();
            if (key.startsWith("_")) continue;
            JsonElement value = entry.getValue();
            if (value.isJsonArray()) {
                List<String> arr = new ArrayList<>();
                for (JsonElement el : value.getAsJsonArray()) arr.add(el.getAsString());
                components.put(key, ComponentDefinition.fromJsonArray(arr, key));
            } else if (value.isJsonNull()) {
                components.put(key, null);
            }
        }
        builder.components(components);
        return builder.build();
    }

    public static boolean exists(File dir) {
        return new File(dir, MODEL_INDEX_FILE).exists();
    }

    public static Optional<ModelIndex> loadIfExists(File dir) {
        File indexFile = new File(dir, MODEL_INDEX_FILE);
        if (!indexFile.exists()) return Optional.empty();
        try {
            return Optional.of(fromFile(indexFile));
        } catch (IOException e) {
            return Optional.empty();
        }
    }

    public Set<String> getComponentNames() {
        if (components == null) return Collections.emptySet();
        Set<String> names = new LinkedHashSet<>();
        for (Map.Entry<String, ComponentDefinition> e : components.entrySet()) {
            if (e.getValue() != null) names.add(e.getKey());
        }
        return names;
    }

    public boolean isPipeline() {
        return components != null && getComponentNames().size() > 1;
    }

    public void save(File file) throws IOException {
        JsonObject json = new JsonObject();
        if (className != null) json.addProperty("_class_name", className);
        if (diffusersVersion != null) json.addProperty("_diffusers_version", diffusersVersion);
        if (components != null) {
            for (Map.Entry<String, ComponentDefinition> e : components.entrySet()) {
                ComponentDefinition def = e.getValue();
                if (def == null) {
                    json.add(e.getKey(), null);
                } else {
                    com.google.gson.JsonArray arr = new com.google.gson.JsonArray();
                    arr.add(def.getLibrary());
                    arr.add(def.getClassName());
                    json.add(e.getKey(), arr);
                }
            }
        }
        try (Writer writer = new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8)) {
            GSON.toJson(json, writer);
        }
    }
}
