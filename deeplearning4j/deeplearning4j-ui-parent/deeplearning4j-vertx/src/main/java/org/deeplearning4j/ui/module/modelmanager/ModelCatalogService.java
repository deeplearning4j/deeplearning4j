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

package org.deeplearning4j.ui.module.modelmanager;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.omnihub.catalog.LLMCatalog;
import org.eclipse.deeplearning4j.omnihub.catalog.LLMEntry;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

import java.io.InputStream;
import java.util.*;

/**
 * Loads the pre-defined model catalog from model-sources.yml and merges
 * with the local registry to show installed status.
 *
 * Catalog entries include LLM-specific metadata (architecture, quantization,
 * context length, parameter count, chat template format) borrowed from
 * kompile-model-staging's CatalogModel pattern.
 */
@Slf4j
public class ModelCatalogService {

    private static final ObjectMapper YAML_MAPPER = new ObjectMapper(new YAMLFactory());

    private final ModelRegistry registry;
    private List<CatalogEntry> catalogEntries;

    public ModelCatalogService(ModelRegistry registry) {
        this.registry = registry;
        loadCatalog();
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class CatalogEntry {
        private String id;
        private String category;
        private String source;
        private String repo;
        private String format;
        private String filePath;
        private String description;
        private boolean installed;
        private String localPath;

        // LLM metadata fields
        private String architecture;
        private String quantizationType;
        private int contextLength;
        private long parameterCount;
        private String chatTemplateFormat;
        private String modelType;
        private List<String> tokenizerFiles;
    }

    @SuppressWarnings("unchecked")
    private void loadCatalog() {
        catalogEntries = new ArrayList<>();
        try (InputStream is = getClass().getResourceAsStream("/model-sources.yml")) {
            if (is == null) {
                log.warn("model-sources.yml not found on classpath");
                return;
            }

            Map<String, Object> root = YAML_MAPPER.readValue(is, Map.class);
            Map<String, Object> catalog = (Map<String, Object>) root.get("model_catalog");
            if (catalog == null) return;

            for (Map.Entry<String, Object> categoryEntry : catalog.entrySet()) {
                String category = categoryEntry.getKey();
                List<Map<String, Object>> models = (List<Map<String, Object>>) categoryEntry.getValue();
                if (models == null) continue;

                for (Map<String, Object> model : models) {
                    CatalogEntry entry = new CatalogEntry();
                    entry.setId((String) model.get("id"));
                    entry.setCategory(category);
                    entry.setSource((String) model.get("source"));
                    entry.setRepo((String) model.get("repo"));
                    entry.setFormat((String) model.get("format"));
                    entry.setFilePath((String) model.get("file_path"));
                    entry.setDescription((String) model.get("description"));

                    // LLM metadata
                    entry.setArchitecture((String) model.get("architecture"));
                    entry.setQuantizationType((String) model.get("quantization"));
                    entry.setChatTemplateFormat((String) model.get("chat_template_format"));
                    entry.setModelType((String) model.get("model_type"));
                    if (model.containsKey("context_length")) {
                        entry.setContextLength(((Number) model.get("context_length")).intValue());
                    }
                    if (model.containsKey("parameter_count")) {
                        entry.setParameterCount(((Number) model.get("parameter_count")).longValue());
                    }
                    if (model.containsKey("tokenizer_files")) {
                        entry.setTokenizerFiles((List<String>) model.get("tokenizer_files"));
                    }

                    catalogEntries.add(entry);
                }
            }
        } catch (Exception e) {
            log.error("Failed to load model catalog", e);
        }

        // Add OmniHub zoo models
        addOmniHubZooModels();
        // Add OmniHub LLM catalog models
        addOmniHubLLMModels();
    }

    private void addOmniHubZooModels() {
        catalogEntries.add(createEntry(
                "age-googlenet", "omnihub_zoo", "omnihub", "omnihub-zoo",
                "fb", "age_googlenet.fb",
                "Age GoogLeNet - Age detection (converted from ONNX model zoo)",
                null, null, 0, 0, null, "OTHER", null
        ));
        catalogEntries.add(createEntry(
                "resnet18", "omnihub_zoo", "omnihub", "omnihub-zoo",
                "fb", "resnet18.fb",
                "ResNet-18 - Image classification (converted from PyTorch torchvision)",
                null, null, 0, 0, null, "OTHER", null
        ));
        catalogEntries.add(createEntry(
                "vgg19-no-top", "omnihub_zoo", "omnihub", "omnihub-zoo",
                "zip", "vgg19_weights_tf_dim_ordering_tf_kernels_notop.zip",
                "VGG-19 (no top) - Feature extraction (converted from Keras applications)",
                null, null, 0, 0, null, "OTHER", null
        ));
    }

    private void addOmniHubLLMModels() {
        try {
            Map<String, LLMEntry> llmModels = LLMCatalog.getModels();
            for (Map.Entry<String, LLMEntry> entry : llmModels.entrySet()) {
                LLMEntry llm = entry.getValue();
                catalogEntries.add(createEntry(
                        "omnihub-" + llm.getName(), "omnihub_llm", "omnihub",
                        llm.getHuggingFaceId(), "safetensors", null,
                        llm.getDescription() + " (via OmniHub pipeline)",
                        null, null, 0, 0, null, "LLM_SAMEDIFF",
                        Arrays.asList("tokenizer.json", "tokenizer_config.json")
                ));
            }
        } catch (Exception e) {
            log.warn("Failed to load OmniHub LLM catalog: {}", e.getMessage());
        }
    }

    private CatalogEntry createEntry(String id, String category, String source, String repo,
                                     String format, String filePath, String description,
                                     String architecture, String quantization,
                                     int contextLength, long parameterCount,
                                     String chatTemplateFormat, String modelType,
                                     List<String> tokenizerFiles) {
        CatalogEntry entry = new CatalogEntry();
        entry.setId(id);
        entry.setCategory(category);
        entry.setSource(source);
        entry.setRepo(repo);
        entry.setFormat(format);
        entry.setFilePath(filePath);
        entry.setDescription(description);
        entry.setArchitecture(architecture);
        entry.setQuantizationType(quantization);
        entry.setContextLength(contextLength);
        entry.setParameterCount(parameterCount);
        entry.setChatTemplateFormat(chatTemplateFormat);
        entry.setModelType(modelType);
        entry.setTokenizerFiles(tokenizerFiles);
        return entry;
    }

    /**
     * Get the full catalog with installed status merged from registry.
     */
    public List<CatalogEntry> getCatalog() {
        List<ModelRegistryEntry> installed = registry.listEntries();
        Set<String> installedIds = new HashSet<>();
        Map<String, String> installedPaths = new HashMap<>();
        for (ModelRegistryEntry e : installed) {
            if (e.getStatus() == ModelRegistryEntry.ModelStatus.READY) {
                installedIds.add(e.getId());
                installedPaths.put(e.getId(), e.getLocalPath());
            }
        }

        List<CatalogEntry> result = new ArrayList<>();
        for (CatalogEntry entry : catalogEntries) {
            CatalogEntry copy = new CatalogEntry(
                    entry.getId(), entry.getCategory(), entry.getSource(),
                    entry.getRepo(), entry.getFormat(), entry.getFilePath(),
                    entry.getDescription(), installedIds.contains(entry.getId()),
                    installedPaths.get(entry.getId()),
                    entry.getArchitecture(), entry.getQuantizationType(),
                    entry.getContextLength(), entry.getParameterCount(),
                    entry.getChatTemplateFormat(), entry.getModelType(),
                    entry.getTokenizerFiles()
            );
            result.add(copy);
        }

        // Add registry entries not in catalog
        for (ModelRegistryEntry re : installed) {
            boolean inCatalog = catalogEntries.stream().anyMatch(c -> c.getId().equals(re.getId()));
            if (!inCatalog && re.getStatus() == ModelRegistryEntry.ModelStatus.READY) {
                CatalogEntry extra = new CatalogEntry();
                extra.setId(re.getId());
                extra.setCategory("custom");
                extra.setSource(re.getSource());
                extra.setRepo(re.getRepo());
                extra.setFormat(re.getFormat());
                extra.setDescription(re.getDescription());
                extra.setInstalled(true);
                extra.setLocalPath(re.getLocalPath());
                if (re.getMetadata() != null) {
                    extra.setArchitecture(re.getMetadata().getArchitecture());
                    extra.setQuantizationType(re.getMetadata().getQuantizationType());
                    extra.setContextLength(re.getMetadata().getContextLength());
                    extra.setParameterCount(re.getMetadata().getParameterCount());
                }
                if (re.getTokenizer() != null) {
                    extra.setChatTemplateFormat(re.getTokenizer().getChatTemplateFormat());
                }
                if (re.getModelType() != null) {
                    extra.setModelType(re.getModelType().name());
                }
                result.add(extra);
            }
        }

        return result;
    }

    public Optional<CatalogEntry> getEntry(String id) {
        return getCatalog().stream().filter(e -> e.getId().equals(id)).findFirst();
    }

    public void reload() {
        loadCatalog();
    }
}
