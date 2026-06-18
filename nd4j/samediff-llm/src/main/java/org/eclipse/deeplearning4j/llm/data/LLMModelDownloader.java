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

package org.eclipse.deeplearning4j.llm.data;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.model.download.ModelDownloader;

import java.io.*;
import java.nio.file.Files;

/**
 * Utility class for downloading LLM models from HuggingFace and other sources.
 *
 * Supports downloading GGUF format models for text-only language model pipelines.
 * Provides a registry of pre-defined Qwen3.5 models in various sizes and quantizations.
 */
@Slf4j
public class LLMModelDownloader {

    public static final String DEFAULT_CACHE_DIR = System.getProperty("user.home") + "/.cache/dl4j-llm-models";
    public static final String CACHE_DIR_PROPERTY = "llm.model.cache.dir";

    private static final String UNSLOTH_BASE = "https://huggingface.co/unsloth/";

    // ==================== Quantization Types ====================

    public enum QuantType {
        Q2_K("Q2_K"),
        Q3_K_M("Q3_K_M"),
        Q4_K_M("Q4_K_M"),
        Q5_K_M("Q5_K_M"),
        Q6_K("Q6_K"),
        Q8_0("Q8_0"),
        BF16("BF16"),
        F16("F16"),
        UD_Q2_K("UD-Q2_K"),
        UD_Q4_K_M("UD-Q4_K_M"),
        UD_Q6_K("UD-Q6_K"),
        UD_Q8_0("UD-Q8_0"),
        UD_IQ1_S("UD-IQ1_S"),
        UD_IQ2_M("UD-IQ2_M");

        private final String suffix;

        QuantType(String suffix) {
            this.suffix = suffix;
        }

        public String getSuffix() { return suffix; }
    }

    // ==================== Model Family ====================

    public enum ModelFamily {
        QWEN35("qwen3.5"),
        GEMMA3("gemma3"),
        GEMMA4("gemma4"),
        NEMOTRON("nemotron"),
        LFM2("lfm2"),
        OLMO("olmo"),
        OPENELM("openelm"),
        GPT_OSS("gpt-oss"),
        PHI("phi"),
        MISTRAL("mistral");

        private final String familyId;

        ModelFamily(String familyId) {
            this.familyId = familyId;
        }

        public String getFamilyId() { return familyId; }
    }

    // ==================== Model Definitions ====================

    public enum LLMModel {
        // Qwen3.5 Dense models
        QWEN35_0_8B("Qwen3.5-0.8B", "0.8B", false, ModelFamily.QWEN35,
                "Qwen3.5 0.8B - Ultra-lightweight dense model"),
        QWEN35_2B("Qwen3.5-2B", "2B", false, ModelFamily.QWEN35,
                "Qwen3.5 2B - Small dense model"),
        QWEN35_4B("Qwen3.5-4B", "4B", false, ModelFamily.QWEN35,
                "Qwen3.5 4B - Medium dense model"),
        QWEN35_9B("Qwen3.5-9B", "9B", false, ModelFamily.QWEN35,
                "Qwen3.5 9B - Large dense model"),
        QWEN35_27B("Qwen3.5-27B", "27B", false, ModelFamily.QWEN35,
                "Qwen3.5 27B - Extra large dense model"),

        // Qwen3.5 MoE models
        QWEN35_35B_A3B("Qwen3.5-35B-A3B", "35B-A3B", true, ModelFamily.QWEN35,
                "Qwen3.5 35B-A3B - Sparse MoE, 3B active params"),
        QWEN35_122B_A10B("Qwen3.5-122B-A10B", "122B-A10B", true, ModelFamily.QWEN35,
                "Qwen3.5 122B-A10B - Large sparse MoE, 10B active params"),
        QWEN35_397B_A17B("Qwen3.5-397B-A17B", "397B-A17B", true, ModelFamily.QWEN35,
                "Qwen3.5 397B-A17B - Largest sparse MoE, 17B active params"),

        // Gemma 3 models
        GEMMA3_1B("gemma-3-1b-it", "1B", false, ModelFamily.GEMMA3,
                "Gemma 3 1B-it - Smallest Gemma 3 instruction-tuned model"),
        GEMMA3_4B("gemma-3-4b-it", "4B", false, ModelFamily.GEMMA3,
                "Gemma 3 4B-it - Small Gemma 3 instruction-tuned model"),

        // Gemma 4 models
        GEMMA4_E2B("gemma-4-E2B-it", "E2B", false, ModelFamily.GEMMA4,
                "Gemma 4 E2B-it - Efficient 2B multimodal model"),
        GEMMA4_E4B("gemma-4-E4B-it", "E4B", false, ModelFamily.GEMMA4,
                "Gemma 4 E4B-it - Efficient 4B multimodal model"),

        // Nemotron models (NVIDIA hybrid Mamba-2 + Transformer)
        NEMOTRON_NANO_4B("Nemotron-3-Nano-4B", "4B", false, ModelFamily.NEMOTRON,
                "Nemotron Nano 4B - Hybrid Mamba-2 + Transformer dense model"),
        NEMOTRON_NANO_30B_A3B("Nemotron-3-Nano-30B-A3B", "30B-A3B", true, ModelFamily.NEMOTRON,
                "Nemotron Nano 30B-A3B - Hybrid Mamba-2 + Transformer sparse MoE"),

        // LFM2 models (Liquid Foundation Models)
        LFM2_1_2B("LFM2-1.2B", "1.2B", false, ModelFamily.LFM2,
                "LFM2 1.2B - Liquid Foundation Model with gated short convolutions"),
        LFM2_2_6B("LFM2-2.6B", "2.6B", false, ModelFamily.LFM2,
                "LFM2 2.6B - Liquid Foundation Model with gated short convolutions"),
        LFM2_350M_EXTRACT("LFM2-350M-Extract", "350M", false, ModelFamily.LFM2,
                "LFM2 350M Extract - Structured extraction (JSON/XML/YAML) from unstructured text",
                "https://huggingface.co/LiquidAI/"),

        // OLMo models (Allen AI)
        OLMO2_7B("OLMo-2-1124-7B-Instruct", "7B", false, ModelFamily.OLMO,
                "OLMo 2 7B Instruct - Allen AI open language model with QK norms"),

        // GPT-OSS models (OpenAI sparse MoE)
        GPT_OSS_20B("gpt-oss-20b", "20B", true, ModelFamily.GPT_OSS,
                "GPT-OSS 20B - Sparse MoE with 32 experts, top-4 routing"),

        // Phi models (Microsoft)
        PHI3_MINI_4K("Phi-3-mini-4k-instruct", "3.8B", false, ModelFamily.PHI,
                "Phi-3 Mini 4K Instruct - 3.8B dense model with SuRoPE"),
        PHI3_5_MINI("Phi-3.5-mini-instruct", "3.8B-3.5", false, ModelFamily.PHI,
                "Phi-3.5 Mini Instruct - 3.8B dense model with long context"),
        PHI4("Phi-4", "14B", false, ModelFamily.PHI,
                "Phi-4 - 14B dense model with SuRoPE and SwiGLU"),

        // Mistral models (Mistral AI)
        MISTRAL_7B("Mistral-7B-Instruct-v0.3", "7B", false, ModelFamily.MISTRAL,
                "Mistral 7B Instruct v0.3 - GQA with sliding window attention"),
        MIXTRAL_8X7B("Mixtral-8x7B-Instruct-v0.1", "8x7B", true, ModelFamily.MISTRAL,
                "Mixtral 8x7B Instruct - Sparse MoE with 8 experts, top-2 routing"),
        MISTRAL_NEMO_12B("Mistral-Nemo-Instruct-2407", "12B", false, ModelFamily.MISTRAL,
                "Mistral Nemo 12B - Dense model with extended context");

        private final String name;
        private final String sizeLabel;
        private final boolean moe;
        private final ModelFamily family;
        private final String description;
        private final String urlBase;

        LLMModel(String name, String sizeLabel, boolean moe, ModelFamily family, String description) {
            this(name, sizeLabel, moe, family, description, null);
        }

        LLMModel(String name, String sizeLabel, boolean moe, ModelFamily family, String description, String urlBase) {
            this.name = name;
            this.sizeLabel = sizeLabel;
            this.moe = moe;
            this.family = family;
            this.description = description;
            this.urlBase = urlBase;
        }

        public String getName() { return name; }
        public String getSizeLabel() { return sizeLabel; }
        public boolean isMoe() { return moe; }
        public ModelFamily getFamily() { return family; }
        public String getDescription() { return description; }

        public String getUrl(QuantType quant) {
            String base = urlBase != null ? urlBase : UNSLOTH_BASE;
            return base + name + "-GGUF/resolve/main/" + name + "-" + quant.getSuffix() + ".gguf";
        }

        public String getFileName(QuantType quant) {
            return name + "-" + quant.getSuffix() + ".gguf";
        }

        public static LLMModel fromSizeLabel(String label) {
            for (LLMModel m : values()) {
                if (m.sizeLabel.equalsIgnoreCase(label)) return m;
            }
            throw new IllegalArgumentException("Unknown model size: " + label +
                    ". Valid sizes: 350M, 0.8B, 1B, 1.2B, 2B, 2.6B, 3.8B, 4B, 7B, 8x7B, 9B, 12B, 14B, 20B, 27B, 30B-A3B, 35B-A3B, 122B-A10B, 397B-A17B, E2B, E4B");
        }

        public static LLMModel fromFamilyAndSize(ModelFamily family, String sizeLabel) {
            for (LLMModel m : values()) {
                if (m.family == family && m.sizeLabel.equalsIgnoreCase(sizeLabel)) return m;
            }
            throw new IllegalArgumentException("Unknown model: " + family.getFamilyId() + " " + sizeLabel);
        }
    }

    @Data
    @Builder
    public static class DownloadResult {
        private final File modelFile;
        private final LLMModel model;
        private final QuantType quantType;
        private final boolean downloadedNow;
        private final long fileSizeBytes;
    }

    // ==================== Download Methods ====================

    public static DownloadResult download(LLMModel model) throws IOException {
        return download(model, QuantType.Q4_K_M);
    }

    public static DownloadResult download(LLMModel model, QuantType quant) throws IOException {
        return download(model, quant, getCacheDir());
    }

    public static DownloadResult download(LLMModel model, QuantType quant, File cacheDir) throws IOException {
        String fileName = model.getFileName(quant);
        String url = model.getUrl(quant);

        ModelDownloader.DownloadResult result = ModelDownloader.download(url, fileName, cacheDir);

        return DownloadResult.builder()
                .modelFile(result.getModelFile())
                .model(model)
                .quantType(quant)
                .downloadedNow(result.isDownloadedNow())
                .fileSizeBytes(result.getFileSizeBytes())
                .build();
    }

    public static File downloadCustom(String url, String fileName) throws IOException {
        ModelDownloader.DownloadResult result = ModelDownloader.download(url, fileName, getCacheDir());
        return result.getModelFile();
    }

    public static boolean isCached(LLMModel model, QuantType quant) {
        return ModelDownloader.isCached(model.getFileName(quant), getCacheDir());
    }

    public static File getCacheDir() {
        return ModelDownloader.getCacheDir(CACHE_DIR_PROPERTY, DEFAULT_CACHE_DIR);
    }

    public static void clearCache() throws IOException {
        ModelDownloader.clearCache(getCacheDir());
    }

    public static File[] listCachedModels() {
        return ModelDownloader.listCachedFiles(getCacheDir(), "gguf");
    }
}
