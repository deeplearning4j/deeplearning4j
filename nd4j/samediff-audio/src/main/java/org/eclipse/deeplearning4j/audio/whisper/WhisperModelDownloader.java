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

package org.eclipse.deeplearning4j.audio.whisper;

import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.model.download.ModelDownloader;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Downloads Whisper models from HuggingFace with local caching.
 * Supports ONNX format (encoder + decoder + tokenizer) and GGML format.
 */
@Slf4j
public class WhisperModelDownloader {

    private static final String DEFAULT_CACHE_DIR = System.getProperty("user.home")
            + File.separator + ".cache" + File.separator + "dl4j-whisper-models";

    /**
     * Whisper model sizes.
     */
    public enum WhisperModelSize {
        TINY("tiny"),
        BASE("base"),
        SMALL("small"),
        MEDIUM("medium"),
        LARGE_V2("large-v2"),
        LARGE_V3("large-v3"),
        TURBO("turbo");

        @Getter
        private final String huggingFaceName;

        WhisperModelSize(String huggingFaceName) {
            this.huggingFaceName = huggingFaceName;
        }

        /**
         * Get the WhisperConfig for this model size.
         */
        public WhisperConfig getConfig() {
            switch (this) {
                case TINY: return WhisperConfig.tiny();
                case BASE: return WhisperConfig.base();
                case SMALL: return WhisperConfig.small();
                case MEDIUM: return WhisperConfig.medium();
                case LARGE_V2: return WhisperConfig.largeV2();
                case LARGE_V3: return WhisperConfig.largeV3();
                case TURBO: return WhisperConfig.turbo();
                default: throw new IllegalStateException("Unknown model size: " + this);
            }
        }
    }

    /**
     * Model format to download.
     */
    public enum WhisperModelFormat {
        ONNX,
        GGML
    }

    /**
     * Result of a model download operation.
     */
    @Data
    @Builder
    public static class DownloadResult {
        private File modelDir;
        private WhisperModelSize modelSize;
        private WhisperModelFormat format;
        private WhisperConfig config;
    }

    private final File cacheDir;

    public WhisperModelDownloader() {
        this(new File(DEFAULT_CACHE_DIR));
    }

    public WhisperModelDownloader(File cacheDir) {
        this.cacheDir = cacheDir;
        if (!cacheDir.exists()) {
            cacheDir.mkdirs();
        }
    }

    /**
     * Download a Whisper model in ONNX format from HuggingFace.
     * Downloads encoder_model.onnx, decoder_model.onnx, decoder_with_past_model.onnx, and tokenizer.json.
     *
     * @param size Model size to download
     * @return Download result with model directory path
     * @throws IOException if download fails
     */
    public DownloadResult downloadOnnx(WhisperModelSize size) throws IOException {
        String modelName = "whisper-" + size.getHuggingFaceName();
        File modelDir = new File(cacheDir, modelName + "-onnx");

        String baseUrl = "https://huggingface.co/onnx-community/" + modelName
                + "/resolve/main/onnx/";

        String[] files = {
                "encoder_model.onnx",
                // The merged decoder (past_key_values inputs + use_cache branch) is the
                // one WhisperModel decodes with — the plain decoder_model.onnx has no
                // past inputs and cannot carry history across decode steps.
                "decoder_model_merged.onnx",
                "decoder_model.onnx",
                "decoder_with_past_model.onnx"
        };

        for (String file : files) {
            File target = new File(modelDir, file);
            if (!target.exists()) {
                log.info("Downloading {}...", file);
                ModelDownloader.download(baseUrl + file, file, modelDir);
            } else {
                log.info("Using cached {}", file);
            }
        }

        // Tokenizer is at the repo root, not in onnx/
        String tokenizerUrl = "https://huggingface.co/onnx-community/" + modelName
                + "/resolve/main/tokenizer.json";
        File tokenizerFile = new File(modelDir, "tokenizer.json");
        if (!tokenizerFile.exists()) {
            log.info("Downloading tokenizer.json...");
            ModelDownloader.download(tokenizerUrl, "tokenizer.json", modelDir);
        }

        return DownloadResult.builder()
                .modelDir(modelDir)
                .modelSize(size)
                .format(WhisperModelFormat.ONNX)
                .config(size.getConfig())
                .build();
    }

    /**
     * Download a Whisper model in GGML format (whisper.cpp compatible).
     *
     * @param size Model size to download
     * @return Download result with model directory path
     * @throws IOException if download fails
     */
    public DownloadResult downloadGgml(WhisperModelSize size) throws IOException {
        String fileName = "ggml-" + size.getHuggingFaceName() + ".bin";
        File modelDir = new File(cacheDir, "ggml");
        String url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/" + fileName;

        File target = new File(modelDir, fileName);
        if (!target.exists()) {
            log.info("Downloading {}...", fileName);
            ModelDownloader.download(url, fileName, modelDir);
        } else {
            log.info("Using cached {}", fileName);
        }

        return DownloadResult.builder()
                .modelDir(modelDir)
                .modelSize(size)
                .format(WhisperModelFormat.GGML)
                .config(size.getConfig())
                .build();
    }

    /**
     * Download a Whisper model in the specified format.
     */
    public DownloadResult download(WhisperModelSize size, WhisperModelFormat format) throws IOException {
        switch (format) {
            case ONNX: return downloadOnnx(size);
            case GGML: return downloadGgml(size);
            default: throw new IllegalArgumentException("Unknown format: " + format);
        }
    }

    /**
     * Check if a model is already cached locally.
     */
    public boolean isCached(WhisperModelSize size, WhisperModelFormat format) {
        if (format == WhisperModelFormat.ONNX) {
            String modelName = "whisper-" + size.getHuggingFaceName();
            File modelDir = new File(cacheDir, modelName + "-onnx");
            return new File(modelDir, "encoder_model.onnx").exists()
                    && new File(modelDir, "decoder_model.onnx").exists()
                    && new File(modelDir, "tokenizer.json").exists();
        } else {
            String fileName = "ggml-" + size.getHuggingFaceName() + ".bin";
            return new File(new File(cacheDir, "ggml"), fileName).exists();
        }
    }

    /**
     * Get the cache directory path.
     */
    public File getCacheDir() {
        return cacheDir;
    }
}
