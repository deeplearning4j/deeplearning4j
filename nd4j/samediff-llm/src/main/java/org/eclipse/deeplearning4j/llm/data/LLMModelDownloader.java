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

    private static final String UNSLOTH_BASE = "https://huggingface.co/unsloth/Qwen3.5-";

    // ==================== Quantization Types ====================

    public enum QuantType {
        Q2_K("Q2_K"),
        Q3_K_M("Q3_K_M"),
        Q4_K_M("Q4_K_M"),
        Q5_K_M("Q5_K_M"),
        Q6_K("Q6_K"),
        Q8_0("Q8_0"),
        BF16("BF16"),
        UD_Q2_K("UD-Q2_K"),
        UD_Q4_K_M("UD-Q4_K_M"),
        UD_Q6_K("UD-Q6_K"),
        UD_Q8_0("UD-Q8_0");

        private final String suffix;

        QuantType(String suffix) {
            this.suffix = suffix;
        }

        public String getSuffix() { return suffix; }
    }

    // ==================== Model Definitions ====================

    public enum LLMModel {
        // Dense models
        QWEN35_0_8B("Qwen3.5-0.8B", "0.8B", false,
                "Qwen3.5 0.8B - Ultra-lightweight dense model"),
        QWEN35_2B("Qwen3.5-2B", "2B", false,
                "Qwen3.5 2B - Small dense model"),
        QWEN35_4B("Qwen3.5-4B", "4B", false,
                "Qwen3.5 4B - Medium dense model"),
        QWEN35_9B("Qwen3.5-9B", "9B", false,
                "Qwen3.5 9B - Large dense model"),
        QWEN35_27B("Qwen3.5-27B", "27B", false,
                "Qwen3.5 27B - Extra large dense model"),

        // MoE models
        QWEN35_35B_A3B("Qwen3.5-35B-A3B", "35B-A3B", true,
                "Qwen3.5 35B-A3B - Sparse MoE, 3B active params"),
        QWEN35_122B_A10B("Qwen3.5-122B-A10B", "122B-A10B", true,
                "Qwen3.5 122B-A10B - Large sparse MoE, 10B active params"),
        QWEN35_397B_A17B("Qwen3.5-397B-A17B", "397B-A17B", true,
                "Qwen3.5 397B-A17B - Largest sparse MoE, 17B active params");

        private final String name;
        private final String sizeLabel;
        private final boolean moe;
        private final String description;

        LLMModel(String name, String sizeLabel, boolean moe, String description) {
            this.name = name;
            this.sizeLabel = sizeLabel;
            this.moe = moe;
            this.description = description;
        }

        public String getName() { return name; }
        public String getSizeLabel() { return sizeLabel; }
        public boolean isMoe() { return moe; }
        public String getDescription() { return description; }

        public String getUrl(QuantType quant) {
            return UNSLOTH_BASE + sizeLabel + "-GGUF/resolve/main/" + name + "-" + quant.getSuffix() + ".gguf";
        }

        public String getFileName(QuantType quant) {
            return name + "-" + quant.getSuffix() + ".gguf";
        }

        public static LLMModel fromSizeLabel(String label) {
            for (LLMModel m : values()) {
                if (m.sizeLabel.equalsIgnoreCase(label)) return m;
            }
            throw new IllegalArgumentException("Unknown model size: " + label +
                    ". Valid sizes: 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B");
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
