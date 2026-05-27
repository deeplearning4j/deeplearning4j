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

package org.eclipse.deeplearning4j.vlm.model;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.config.ModelConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/**
 * Loader for multi-part VLM models in SDZ (SameDiff ZIP) format.
 *
 * SDZ is the native format for SameDiff models, providing:
 * - Efficient sharded storage for large models
 * - Native SameDiff operations without conversion overhead
 * - Optimized inference paths
 *
 * <p>To use ONNX models, first import them using the ONNX importer:</p>
 * <pre>{@code
 * // Import ONNX and save as SDZ
 * OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
 * SameDiff model = importer.runImport("model.onnx", Collections.emptyMap(), false, false);
 * SDZSerializer.save(model, new File("model.sdz"), false, null);
 * }</pre>
 *
 * <p>Expected directory structure:</p>
 * <pre>
 * model_dir/
 * ├── config.json
 * ├── tokenizer.json
 * ├── tokenizer_config.json
 * ├── preprocessor_config.json
 * ├── vision_encoder.sdz
 * ├── embed_tokens.sdz (optional)
 * └── decoder.sdz
 * </pre>
 *
 * Adam Gibson
 */
@Slf4j
public class MultiPartModelLoader {

    private static final String SDZ_EXTENSION = ".sdz";

    // Common model component names
    private static final String[] VISION_ENCODER_NAMES = {
        "vision_encoder", "encoder", "image_encoder", "visual_encoder"
    };
    private static final String[] EMBED_TOKENS_NAMES = {
        "embed_tokens", "embeddings", "token_embeddings"
    };
    private static final String[] DECODER_NAMES = {
        "decoder", "decoder_model", "decoder_model_merged", "model", "language_model"
    };

    /**
     * Load a multi-part model from a directory containing SDZ files.
     *
     * @param modelDir the model directory
     * @return the loaded model components
     * @throws IOException if loading fails
     */
    public static LoadedModel load(File modelDir) throws IOException {
        log.info("Loading multi-part model from: {}", modelDir.getAbsolutePath());

        if (!modelDir.exists() || !modelDir.isDirectory()) {
            throw new IOException("Model directory does not exist: " + modelDir.getAbsolutePath());
        }

        // Resolve which SDZ files exist (fast filesystem check, no I/O)
        File visionEncoderFile = findSdzFile(modelDir, VISION_ENCODER_NAMES);
        File embedTokensFile = findSdzFile(modelDir, EMBED_TOKENS_NAMES);
        File decoderFile = findSdzFile(modelDir, DECODER_NAMES);

        if (decoderFile == null) {
            throw new IOException("No decoder model found in: " + modelDir.getAbsolutePath() +
                ". Expected one of: " + String.join(", ", DECODER_NAMES) + SDZ_EXTENSION);
        }

        // Load SDZ components in parallel — each file is independent
        long loadStart = System.currentTimeMillis();
        CompletableFuture<SameDiff> visionFuture = visionEncoderFile != null
                ? CompletableFuture.supplyAsync(() -> loadSdzQuiet(visionEncoderFile))
                : CompletableFuture.completedFuture(null);
        CompletableFuture<SameDiff> embedFuture = embedTokensFile != null
                ? CompletableFuture.supplyAsync(() -> loadSdzQuiet(embedTokensFile))
                : CompletableFuture.completedFuture(null);
        CompletableFuture<SameDiff> decoderFuture =
                CompletableFuture.supplyAsync(() -> loadSdzQuiet(decoderFile));

        // Wait for all loads to complete
        SameDiff visionEncoder;
        SameDiff embedTokens;
        SameDiff decoder;
        try {
            CompletableFuture.allOf(visionFuture, embedFuture, decoderFuture).join();
            visionEncoder = visionFuture.get();
            embedTokens = embedFuture.get();
            decoder = decoderFuture.get();
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            throw cause instanceof IOException ? (IOException) cause
                    : new IOException("Parallel model loading failed", cause);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Model loading interrupted", e);
        }
        log.info("Parallel model loading completed in {}ms", System.currentTimeMillis() - loadStart);

        if (decoder == null) {
            throw new IOException("Failed to load decoder from: " + decoderFile.getAbsolutePath());
        }

        // Load configurations (lightweight, no need to parallelize)
        ModelConfig config = loadConfig(modelDir);
        Tokenizer tokenizer = loadTokenizer(modelDir);
        VLMImagePreprocessor preprocessor = loadPreprocessor(modelDir);

        // Collect metadata
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("model_dir", modelDir.getAbsolutePath());
        metadata.put("has_vision_encoder", visionEncoder != null);
        metadata.put("has_embed_tokens", embedTokens != null);

        log.info("Model loaded - vision_encoder: {}, embed_tokens: {}, decoder: {}",
            visionEncoder != null, embedTokens != null, decoder != null);

        return LoadedModel.builder()
                .visionEncoder(visionEncoder)
                .embedTokens(embedTokens)
                .decoder(decoder)
                .tokenizer(tokenizer)
                .imagePreprocessor(preprocessor)
                .config(config)
                .metadata(metadata)
                .build();
    }

    /**
     * Load a single SDZ model file.
     *
     * @param sdzFile the SDZ file to load
     * @return the loaded SameDiff model
     * @throws IOException if loading fails
     */
    public static SameDiff loadSdz(File sdzFile) throws IOException {
        if (!sdzFile.exists()) {
            throw new IOException("SDZ file not found: " + sdzFile.getAbsolutePath());
        }
        if (!isSdzFile(sdzFile)) {
            throw new IOException("File is not a valid SDZ file: " + sdzFile.getAbsolutePath());
        }
        log.info("Loading SDZ file: {}", sdzFile.getName());
        return SDZSerializer.load(sdzFile, false);
    }

    /**
     * Load a SmolDocling model.
     *
     * @param modelDir the SmolDocling model directory
     * @return the loaded model
     * @throws IOException if loading fails
     */
    public static LoadedModel loadSmolDocling(File modelDir) throws IOException {
        log.info("Loading SmolDocling model from: {}", modelDir.getAbsolutePath());

        LoadedModel loaded = load(modelDir);

        if (loaded.getVisionEncoder() == null) {
            log.warn("SmolDocling model missing vision_encoder - document understanding may be limited");
        }

        return loaded;
    }

    /**
     * Load a LLaVA model.
     *
     * @param modelDir the LLaVA model directory
     * @return the loaded model
     * @throws IOException if loading fails
     */
    public static LoadedModel loadLlava(File modelDir) throws IOException {
        log.info("Loading LLaVA model from: {}", modelDir.getAbsolutePath());
        return load(modelDir);
    }

    /**
     * Check if a file is a valid SDZ (ZIP) file.
     */
    private static boolean isSdzFile(File file) {
        if (file == null || !file.exists() || !file.isFile() || file.length() < 4) {
            return false;
        }
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] magic = new byte[4];
            if (fis.read(magic) != 4) return false;
            // ZIP magic bytes: PK\x03\x04
            return magic[0] == 0x50 && magic[1] == 0x4b && magic[2] == 0x03 && magic[3] == 0x04;
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Find the first existing SDZ file matching any of the given names.
     * Does not load the file — just resolves the path.
     */
    private static File findSdzFile(File dir, String[] names) {
        for (String name : names) {
            File file = new File(dir, name + SDZ_EXTENSION);
            if (file.exists() && isSdzFile(file)) {
                return file;
            }
        }
        return null;
    }

    /**
     * Load an SDZ file, wrapping checked exceptions for use in CompletableFuture.
     */
    private static SameDiff loadSdzQuiet(File file) {
        try {
            log.info("Loading SDZ: {}", file.getName());
            return SDZSerializer.load(file, false);
        } catch (Exception e) {
            log.warn("Failed to load {}: {}", file.getName(), e.getMessage());
            throw new RuntimeException("Failed to load " + file.getName(), e);
        }
    }

    private static SameDiff loadSdzIfExists(File dir, String[] names) {
        File file = findSdzFile(dir, names);
        if (file != null) {
            return loadSdzQuiet(file);
        }
        return null;
    }

    private static ModelConfig loadConfig(File modelDir) {
        File configFile = new File(modelDir, "config.json");
        if (configFile.exists()) {
            try {
                return ModelConfig.fromFile(configFile);
            } catch (Exception e) {
                log.warn("Failed to load config.json: {}", e.getMessage());
            }
        }
        return null;
    }

    private static Tokenizer loadTokenizer(File modelDir) {
        File tokenizerFile = new File(modelDir, "tokenizer.json");
        if (tokenizerFile.exists()) {
            try {
                return HuggingFaceTokenizer.fromFile(tokenizerFile);
            } catch (Exception e) {
                log.warn("Failed to load tokenizer.json: {}", e.getMessage());
            }
        }
        return null;
    }

    private static VLMImagePreprocessor loadPreprocessor(File modelDir) {
        File preprocessorFile = new File(modelDir, "preprocessor_config.json");
        if (preprocessorFile.exists()) {
            try {
                return VLMImagePreprocessor.fromConfig(preprocessorFile);
            } catch (Exception e) {
                log.warn("Failed to load preprocessor_config.json: {}", e.getMessage());
            }
        }
        return VLMImagePreprocessor.defaultPreprocessor();
    }

    /**
     * Container for loaded model components.
     */
    @Data
    @Builder
    public static class LoadedModel {
        private final SameDiff visionEncoder;
        private final SameDiff embedTokens;
        private final SameDiff decoder;
        private final Tokenizer tokenizer;
        private final VLMImagePreprocessor imagePreprocessor;
        private final ModelConfig config;
        private final Map<String, Object> metadata;

        /**
         * Convert to a VisionLanguageModel.
         *
         * @return the VLM wrapper
         */
        public VisionLanguageModel toVisionLanguageModel() {
            return toVisionLanguageModel(null);
        }

        /**
         * Convert to a VisionLanguageModel with an optional externally-provided IOConfig.
         *
         * @param ioConfig external vision encoder IO config, or null to auto-discover
         * @return the VLM wrapper
         */
        public VisionLanguageModel toVisionLanguageModel(VisionEncoderIOConfig ioConfig) {
            VisionLanguageModel.VisionLanguageModelBuilder builder = VisionLanguageModel.builder()
                    .visionEncoder(visionEncoder)
                    .embedTokens(embedTokens)
                    .decoder(decoder)
                    .tokenizer(tokenizer)
                    .imagePreprocessor(imagePreprocessor)
                    .config(config);
            if (ioConfig != null) {
                builder.visionEncoderIOConfig(ioConfig);
            }
            return builder.build();
        }

        /**
         * Check if all required components are present.
         *
         * @return true if the model is complete
         */
        public boolean isComplete() {
            return decoder != null && tokenizer != null;
        }

        /**
         * Check if vision components are present.
         *
         * @return true if vision encoder is loaded
         */
        public boolean hasVision() {
            return visionEncoder != null;
        }

        /**
         * Save the model components to SDZ format.
         *
         * @param outputDir the output directory
         * @throws IOException if saving fails
         */
        public void save(File outputDir) throws IOException {
            if (!outputDir.exists() && !outputDir.mkdirs()) {
                throw new IOException("Could not create output directory: " + outputDir.getAbsolutePath());
            }

            if (visionEncoder != null) {
                File visionFile = new File(outputDir, "vision_encoder.sdz");
                log.info("Saving vision encoder to: {}", visionFile.getName());
                SDZSerializer.save(visionEncoder, visionFile, false, null);
            }

            if (embedTokens != null) {
                File embedFile = new File(outputDir, "embed_tokens.sdz");
                log.info("Saving embed tokens to: {}", embedFile.getName());
                SDZSerializer.save(embedTokens, embedFile, false, null);
            }

            if (decoder != null) {
                File decoderFile = new File(outputDir, "decoder.sdz");
                log.info("Saving decoder to: {}", decoderFile.getName());
                SDZSerializer.save(decoder, decoderFile, false, null);
            }

            log.info("Model saved to: {}", outputDir.getAbsolutePath());
        }
    }
}
