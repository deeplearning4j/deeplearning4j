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
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Represents the complete manifest of a model repository.
 */
@Data
@Builder
@Slf4j
public class ModelManifest {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private File modelDirectory;
    private ModelFormat format;
    private ModelType type;
    private ModelIndex modelIndex;
    private WeightMapIndex weightMapIndex;
    private JsonObject config;
    private List<File> weightFiles;
    private Map<String, ModelManifest> components;

    // Additional HuggingFace configs
    private GenerationConfig generationConfig;
    private TokenizerConfig tokenizerConfig;
    private SpecialTokensMap specialTokensMap;
    private PreprocessorConfig preprocessorConfig;
    private SchedulerConfig schedulerConfig;

    public enum ModelType {
        SINGLE, SHARDED, PIPELINE, UNKNOWN
    }

    public static ModelManifest fromDirectory(File modelDir) throws IOException {
        if (!modelDir.exists() || !modelDir.isDirectory()) {
            throw new IOException("Model directory does not exist: " + modelDir);
        }

        log.debug("Detecting model format in: {}", modelDir.getAbsolutePath());

        Optional<ModelIndex> modelIndex = ModelIndex.loadIfExists(modelDir);
        if (modelIndex.isPresent() && modelIndex.get().isPipeline()) {
            return detectPipeline(modelDir, modelIndex.get());
        }

        ModelFormat format = detectFormat(modelDir);
        ModelManifestBuilder builder = ModelManifest.builder()
                .modelDirectory(modelDir)
                .format(format);

        File configFile = new File(modelDir, "config.json");
        if (configFile.exists()) {
            try (Reader reader = new InputStreamReader(new FileInputStream(configFile), StandardCharsets.UTF_8)) {
                builder.config(GSON.fromJson(reader, JsonObject.class));
            }
        }

        Optional<WeightMapIndex> weightIndex = WeightMapIndex.loadIfExists(modelDir);
        if (weightIndex.isPresent()) {
            builder.type(ModelType.SHARDED);
            builder.weightMapIndex(weightIndex.get());
            builder.weightFiles(resolveShardFiles(modelDir, weightIndex.get()));
        } else {
            builder.type(ModelType.SINGLE);
            builder.weightFiles(findWeightFiles(modelDir, format));
        }

        // Load additional HuggingFace configs
        GenerationConfig.loadIfExists(modelDir).ifPresent(builder::generationConfig);
        TokenizerConfig.loadIfExists(modelDir).ifPresent(builder::tokenizerConfig);
        SpecialTokensMap.loadIfExists(modelDir).ifPresent(builder::specialTokensMap);
        PreprocessorConfig.loadIfExists(modelDir).ifPresent(builder::preprocessorConfig);
        SchedulerConfig.loadIfExists(modelDir).ifPresent(builder::schedulerConfig);

        return builder.build();
    }

    private static ModelManifest detectPipeline(File modelDir, ModelIndex index) throws IOException {
        Map<String, ModelManifest> components = new LinkedHashMap<>();
        for (String componentName : index.getComponentNames()) {
            File componentDir = new File(modelDir, componentName);
            if (componentDir.exists() && componentDir.isDirectory()) {
                try {
                    components.put(componentName, fromDirectory(componentDir));
                } catch (IOException e) {
                    log.warn("Failed to load component '{}': {}", componentName, e.getMessage());
                }
            }
        }

        return ModelManifest.builder()
                .modelDirectory(modelDir)
                .format(determineDominantFormat(components.values()))
                .type(ModelType.PIPELINE)
                .modelIndex(index)
                .components(components)
                .build();
    }

    private static ModelFormat detectFormat(File dir) {
        File[] files = dir.listFiles();
        if (files == null) return ModelFormat.UNKNOWN;

        for (File f : files) {
            String n = f.getName().toLowerCase();
            if (n.endsWith(".sdz")) return ModelFormat.SDZ;
        }
        for (File f : files) {
            String n = f.getName().toLowerCase();
            if (n.endsWith(".safetensors")) return ModelFormat.SAFETENSORS;
        }
        for (File f : files) {
            String n = f.getName().toLowerCase();
            if (n.endsWith(".gguf") || n.endsWith(".ggml")) return ModelFormat.GGUF;
        }
        for (File f : files) {
            String n = f.getName().toLowerCase();
            if (n.endsWith(".onnx")) return ModelFormat.ONNX;
        }
        for (File f : files) {
            String n = f.getName().toLowerCase();
            if (n.endsWith(".bin") || n.endsWith(".pt")) return ModelFormat.PYTORCH;
        }
        return ModelFormat.UNKNOWN;
    }

    private static List<File> resolveShardFiles(File dir, WeightMapIndex index) {
        List<File> files = new ArrayList<>();
        for (String shardName : index.getShardFiles()) {
            File shardFile = new File(dir, shardName);
            if (shardFile.exists()) files.add(shardFile);
        }
        files.sort(Comparator.comparing(File::getName));
        return files;
    }

    private static List<File> findWeightFiles(File dir, ModelFormat format) {
        List<File> files = new ArrayList<>();
        File[] allFiles = dir.listFiles();
        if (allFiles == null) return files;
        String ext = "." + format.getExtension();
        for (File f : allFiles) {
            if (f.getName().toLowerCase().endsWith(ext)) files.add(f);
        }
        files.sort(Comparator.comparing(File::getName));
        return files;
    }

    private static ModelFormat determineDominantFormat(Collection<ModelManifest> components) {
        Map<ModelFormat, Integer> counts = new EnumMap<>(ModelFormat.class);
        for (ModelManifest m : components) counts.merge(m.getFormat(), 1, Integer::sum);
        return counts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(ModelFormat.UNKNOWN);
    }

    public boolean isPipeline() { return type == ModelType.PIPELINE; }
    public boolean isSharded() { return type == ModelType.SHARDED; }

    public File getPrimaryWeightFile() {
        if (weightFiles == null || weightFiles.isEmpty()) return null;
        for (File f : weightFiles) {
            String n = f.getName().toLowerCase();
            if (n.startsWith("model.")) return f;
        }
        return weightFiles.get(0);
    }

    public ModelManifest getComponent(String name) {
        return components != null ? components.get(name) : null;
    }

    public Set<String> getComponentNames() {
        return components != null ? components.keySet() : Collections.emptySet();
    }

    public String getArchitecture() {
        if (config == null) return null;
        if (config.has("model_type")) return config.get("model_type").getAsString();
        if (config.has("architectures") && config.get("architectures").isJsonArray()) {
            var arr = config.getAsJsonArray("architectures");
            if (arr.size() > 0) return arr.get(0).getAsString();
        }
        return null;
    }

    /**
     * Check if this model has generation config.
     */
    public boolean hasGenerationConfig() {
        return generationConfig != null;
    }

    /**
     * Check if this model has tokenizer config.
     */
    public boolean hasTokenizerConfig() {
        return tokenizerConfig != null;
    }

    /**
     * Check if this model has preprocessor config (for vision models).
     */
    public boolean hasPreprocessorConfig() {
        return preprocessorConfig != null;
    }

    /**
     * Check if this model has a chat template.
     */
    public boolean hasChatTemplate() {
        return tokenizerConfig != null && tokenizerConfig.hasChatTemplate();
    }

    /**
     * Get the raw chat template string for this model if available.
     * Returns the Jinja2 template string from tokenizer_config.json, or null if not present.
     */
    public String getChatTemplateString() {
        if (tokenizerConfig == null) return null;
        return tokenizerConfig.getChatTemplate();
    }

    /**
     * Get the vocabulary size from config.json.
     */
    public Integer getVocabSize() {
        if (config == null) return null;
        if (config.has("vocab_size")) return config.get("vocab_size").getAsInt();
        return null;
    }

    /**
     * Get the hidden size from config.json.
     */
    public Integer getHiddenSize() {
        if (config == null) return null;
        if (config.has("hidden_size")) return config.get("hidden_size").getAsInt();
        if (config.has("d_model")) return config.get("d_model").getAsInt();
        return null;
    }

    /**
     * Get the number of attention heads from config.json.
     */
    public Integer getNumAttentionHeads() {
        if (config == null) return null;
        if (config.has("num_attention_heads")) return config.get("num_attention_heads").getAsInt();
        if (config.has("n_head")) return config.get("n_head").getAsInt();
        if (config.has("num_heads")) return config.get("num_heads").getAsInt();
        return null;
    }

    /**
     * Get the number of hidden layers from config.json.
     */
    public Integer getNumHiddenLayers() {
        if (config == null) return null;
        if (config.has("num_hidden_layers")) return config.get("num_hidden_layers").getAsInt();
        if (config.has("n_layer")) return config.get("n_layer").getAsInt();
        if (config.has("num_layers")) return config.get("num_layers").getAsInt();
        return null;
    }

    /**
     * Get the max position embeddings (context length) from config.json.
     */
    public Integer getMaxPositionEmbeddings() {
        if (config == null) return null;
        if (config.has("max_position_embeddings")) return config.get("max_position_embeddings").getAsInt();
        if (config.has("n_positions")) return config.get("n_positions").getAsInt();
        if (config.has("max_seq_len")) return config.get("max_seq_len").getAsInt();
        return null;
    }

    /**
     * Check if this is likely an LLM (has generation config or causal LM architecture).
     */
    public boolean isLikelyLLM() {
        if (generationConfig != null) return true;
        String arch = getArchitecture();
        if (arch == null) return false;
        return arch.toLowerCase().contains("causal") ||
                arch.toLowerCase().contains("gpt") ||
                arch.toLowerCase().contains("llama") ||
                arch.toLowerCase().contains("mistral") ||
                arch.toLowerCase().contains("phi") ||
                arch.toLowerCase().contains("qwen");
    }

    /**
     * Check if this is likely a vision model.
     */
    public boolean isLikelyVisionModel() {
        if (preprocessorConfig != null) return true;
        String arch = getArchitecture();
        if (arch == null) return false;
        return arch.toLowerCase().contains("vit") ||
                arch.toLowerCase().contains("clip") ||
                arch.toLowerCase().contains("vision") ||
                arch.toLowerCase().contains("image") ||
                arch.toLowerCase().contains("resnet") ||
                arch.toLowerCase().contains("convnext");
    }

    /**
     * Check if this is likely a diffusion model.
     */
    public boolean isLikelyDiffusionModel() {
        if (schedulerConfig != null) return true;
        if (modelIndex != null) {
            String className = modelIndex.getClassName();
            if (className != null) {
                return className.contains("Diffusion") ||
                        className.contains("StableDiffusion") ||
                        className.contains("SDXL");
            }
        }
        return false;
    }
}
