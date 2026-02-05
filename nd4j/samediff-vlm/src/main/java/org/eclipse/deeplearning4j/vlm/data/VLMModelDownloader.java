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

package org.eclipse.deeplearning4j.vlm.data;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Utility class for downloading VLM models from HuggingFace and other sources.
 *
 * Supports downloading ONNX and GGUF format models for vision-language pipelines.
 */
@Slf4j
public class VLMModelDownloader {

    /**
     * Default directory for cached models.
     */
    public static final String DEFAULT_CACHE_DIR = System.getProperty("user.home") + "/.cache/dl4j-vlm-models";

    /**
     * System property to override cache directory.
     */
    public static final String CACHE_DIR_PROPERTY = "vlm.model.cache.dir";

    // ==================== Model Definitions ====================

    /**
     * Available pre-defined models for VLM testing.
     */
    public enum VLMModel {
        // VLM ONNX Models - SigLIP (smaller, 355MB)
        SIGLIP_VISION(
                "siglip-base-patch16-224-vision",
                "https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/onnx/vision_model.onnx",
                ModelFormat.ONNX,
                224, 224,
                "SigLIP Base Vision Encoder - Google's improved CLIP (~355MB)"
        ),

        // VLM ONNX Models - CLIP (full model, 577MB)
        CLIP_VIT_BASE_PATCH32(
                "clip-vit-base-patch32",
                "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/model.onnx",
                ModelFormat.ONNX,
                224, 224,
                "OpenAI CLIP ViT-B/32 - vision-language model (~577MB)"
        ),

        // VLM ONNX Models - MobileViT (lightweight, ~22MB)
        MOBILEVIT_SMALL(
                "mobilevit-small",
                "https://huggingface.co/Xenova/mobilevit-small/resolve/main/onnx/model.onnx",
                ModelFormat.ONNX,
                256, 256,
                "Apple MobileViT Small - lightweight mobile vision transformer (~22MB)"
        ),

        // VLM ONNX Models - DeiT Base 384 (larger input resolution)
        VIT_BASE_PATCH16_384(
                "deit-base-distilled-patch16-384",
                "https://huggingface.co/Xenova/deit-base-distilled-patch16-384/resolve/main/onnx/model.onnx",
                ModelFormat.ONNX,
                384, 384,
                "Facebook DeiT Base Distilled with 384x384 input - higher resolution vision transformer"
        ),

        // GGUF Models - CLIP Vision
        CLIP_VIT_B32_VISION_GGUF(
                "clip-vit-b32-vision",
                "https://huggingface.co/mys/ggml_CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16.gguf",
                ModelFormat.GGUF,
                224, 224,
                "CLIP ViT-B/32 Vision Encoder in GGUF format (~290MB)"
        ),

        // GGUF Models - LLaVA Multimodal Projector
        LLAVA_MMPROJ_F16_GGUF(
                "llava-v1.5-7b-mmproj-f16",
                "https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf",
                ModelFormat.GGUF,
                336, 336,
                "LLaVA v1.5 Multimodal Projector in GGUF format (~600MB)"
        ),

        // Docling Models - Document Understanding
        SMOLDOCLING_VISION_ENCODER(
                "smoldocling-vision-encoder",
                "https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/onnx/vision_encoder.onnx",
                ModelFormat.ONNX,
                512, 512,  // Model expects 512x512 with 5D input [batch, frames, 3, H, W]
                "SmolDocling Vision Encoder - IBM document understanding VLM"
        ),

        SMOLDOCLING_DECODER(
                "smoldocling-decoder",
                "https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/onnx/decoder_model_merged.onnx",
                ModelFormat.ONNX,
                512, 512,
                "SmolDocling Decoder - generates DocTags text from encoded images"
        ),

        SMOLDOCLING_EMBED_TOKENS(
                "smoldocling-embed-tokens",
                "https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/onnx/embed_tokens.onnx",
                ModelFormat.ONNX,
                0, 0,
                "SmolDocling Text Embeddings - converts token IDs to embeddings for decoder"
        ),

        // SmolDocling Tokenizer files (needed for text generation)
        SMOLDOCLING_TOKENIZER(
                "smoldocling-tokenizer",
                "https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/tokenizer.json",
                ModelFormat.JSON,
                0, 0,  // Not an image model
                "SmolDocling Tokenizer - HuggingFace tokenizer.json for text decoding"
        ),

        SMOLDOCLING_TOKENIZER_CONFIG(
                "smoldocling-tokenizer-config",
                "https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/tokenizer_config.json",
                ModelFormat.JSON,
                0, 0,
                "SmolDocling Tokenizer Config - configuration for the tokenizer"
        ),

        // Docling TableFormer - Table Structure Recognition
        DOCLING_TABLEFORMER_ACCURATE(
                "docling-tableformer-accurate",
                "https://huggingface.co/asmud/ds4sd-docling-models-onnx/resolve/main/ds4sd_docling_models_tableformer_accurate_jpqd.onnx",
                ModelFormat.ONNX,
                448, 448,
                "Docling TableFormer Accurate - high accuracy table structure recognition"
        ),

        DOCLING_TABLEFORMER_FAST(
                "docling-tableformer-fast",
                "https://huggingface.co/asmud/ds4sd-docling-models-onnx/resolve/main/ds4sd_docling_models_tableformer_fast_jpqd.onnx",
                ModelFormat.ONNX,
                448, 448,
                "Docling TableFormer Fast - fast table structure recognition"
        );

        private final String name;
        private final String url;
        private final ModelFormat format;
        private final int inputWidth;
        private final int inputHeight;
        private final String description;

        VLMModel(String name, String url, ModelFormat format, int inputWidth, int inputHeight, String description) {
            this.name = name;
            this.url = url;
            this.format = format;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.description = description;
        }

        public String getName() { return name; }
        public String getUrl() { return url; }
        public ModelFormat getFormat() { return format; }
        public int getInputWidth() { return inputWidth; }
        public int getInputHeight() { return inputHeight; }
        public String getDescription() { return description; }

        public String getFileName() {
            return name + "." + format.getExtension();
        }
    }

    public enum ModelFormat {
        ONNX("onnx"),
        GGUF("gguf"),
        GGML("ggml"),
        JSON("json");

        private final String extension;

        ModelFormat(String extension) {
            this.extension = extension;
        }

        public String getExtension() { return extension; }
    }

    @Data
    @Builder
    public static class DownloadResult {
        private final File modelFile;
        private final VLMModel model;
        private final boolean downloadedNow;
        private final long fileSizeBytes;
    }

    // ==================== Download Methods ====================

    /**
     * Download a pre-defined VLM model.
     *
     * @param model the model to download
     * @return download result with file path
     * @throws IOException if download fails
     */
    public static DownloadResult download(VLMModel model) throws IOException {
        return download(model, getCacheDir());
    }

    /**
     * Download a pre-defined VLM model to a specific directory.
     *
     * @param model the model to download
     * @param cacheDir the directory to store the model
     * @return download result with file path
     * @throws IOException if download fails
     */
    public static DownloadResult download(VLMModel model, File cacheDir) throws IOException {
        File modelFile = new File(cacheDir, model.getFileName());

        boolean downloadedNow = false;
        if (!modelFile.exists()) {
            log.info("Downloading {} from {}", model.getName(), model.getUrl());
            downloadFile(model.getUrl(), modelFile);
            downloadedNow = true;
            log.info("Downloaded {} to {}", model.getName(), modelFile.getAbsolutePath());
        } else {
            log.info("Using cached model: {}", modelFile.getAbsolutePath());
        }

        return DownloadResult.builder()
                .modelFile(modelFile)
                .model(model)
                .downloadedNow(downloadedNow)
                .fileSizeBytes(modelFile.length())
                .build();
    }

    /**
     * Download a model from a custom URL.
     *
     * @param url the URL to download from
     * @param fileName the file name to save as
     * @param format the model format
     * @return the downloaded file
     * @throws IOException if download fails
     */
    public static File downloadCustom(String url, String fileName, ModelFormat format) throws IOException {
        File cacheDir = getCacheDir();
        File modelFile = new File(cacheDir, fileName);

        if (!modelFile.exists()) {
            log.info("Downloading from {}", url);
            downloadFile(url, modelFile);
            log.info("Downloaded to {}", modelFile.getAbsolutePath());
        } else {
            log.info("Using cached file: {}", modelFile.getAbsolutePath());
        }

        return modelFile;
    }

    /**
     * Check if a model is already cached.
     *
     * @param model the model to check
     * @return true if cached
     */
    public static boolean isCached(VLMModel model) {
        File modelFile = new File(getCacheDir(), model.getFileName());
        return modelFile.exists();
    }

    /**
     * Get the cache directory.
     *
     * @return the cache directory
     */
    public static File getCacheDir() {
        String cacheDir = System.getProperty(CACHE_DIR_PROPERTY, DEFAULT_CACHE_DIR);
        File dir = new File(cacheDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        return dir;
    }

    /**
     * Clear all cached models.
     *
     * @throws IOException if deletion fails
     */
    public static void clearCache() throws IOException {
        File cacheDir = getCacheDir();
        if (cacheDir.exists()) {
            for (File file : cacheDir.listFiles()) {
                if (file.isFile()) {
                    Files.delete(file.toPath());
                    log.info("Deleted cached model: {}", file.getName());
                }
            }
        }
    }

    /**
     * List all cached models.
     *
     * @return array of cached model files
     */
    public static File[] listCachedModels() {
        File cacheDir = getCacheDir();
        if (cacheDir.exists()) {
            return cacheDir.listFiles((dir, name) ->
                    name.endsWith(".onnx") || name.endsWith(".gguf") || name.endsWith(".ggml") || name.endsWith(".json"));
        }
        return new File[0];
    }

    // ==================== Internal Methods ====================

    private static void downloadFile(String urlString, File outputFile) throws IOException {
        // Ensure parent directory exists
        outputFile.getParentFile().mkdirs();

        URL url = new URL(urlString);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");
        connection.setConnectTimeout(30000);
        connection.setReadTimeout(60000);
        connection.setRequestProperty("User-Agent", "DL4J-VLM-ModelDownloader/1.0");

        // Handle redirects
        int responseCode = connection.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_MOVED_TEMP ||
                responseCode == HttpURLConnection.HTTP_MOVED_PERM ||
                responseCode == HttpURLConnection.HTTP_SEE_OTHER ||
                responseCode == 307 || responseCode == 308) {
            String newUrl = connection.getHeaderField("Location");
            log.debug("Redirecting to: {}", newUrl);
            connection.disconnect();
            downloadFile(newUrl, outputFile);
            return;
        }

        if (responseCode != HttpURLConnection.HTTP_OK) {
            throw new IOException("Failed to download: HTTP " + responseCode + " from " + urlString);
        }

        long contentLength = connection.getContentLengthLong();
        String sizeStr = contentLength > 0 ? formatBytes(contentLength) : "unknown size";
        log.info("Downloading {} ...", sizeStr);

        // Download to temp file first, then move
        Path tempFile = Files.createTempFile("dl4j-download-", ".tmp");
        try (InputStream in = new BufferedInputStream(connection.getInputStream());
             OutputStream out = new BufferedOutputStream(Files.newOutputStream(tempFile))) {

            byte[] buffer = new byte[8192];
            long totalRead = 0;
            int bytesRead;
            long lastUpdateTime = System.currentTimeMillis();
            int lastPercent = -1;

            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
                totalRead += bytesRead;

                // Update progress bar
                long now = System.currentTimeMillis();
                if (now - lastUpdateTime > 100) { // Update every 100ms max
                    int percent = contentLength > 0 ? (int) ((totalRead * 100) / contentLength) : -1;
                    if (percent != lastPercent) {
                        printProgressBar(totalRead, contentLength, outputFile.getName());
                        lastPercent = percent;
                    }
                    lastUpdateTime = now;
                }
            }

            // Final progress update
            printProgressBar(totalRead, contentLength, outputFile.getName());
            System.out.println(); // New line after progress bar
        }

        // Move temp file to final location
        Files.move(tempFile, outputFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        connection.disconnect();
        log.info("Download complete: {}", outputFile.getName());
    }

    /**
     * Print a progress bar to the console.
     */
    private static void printProgressBar(long current, long total, String fileName) {
        int barWidth = 40;
        String downloadedStr = formatBytes(current);

        if (total > 0) {
            int percent = (int) ((current * 100) / total);
            int filled = (int) ((current * barWidth) / total);
            int empty = barWidth - filled;

            StringBuilder bar = new StringBuilder();
            bar.append("\r");
            bar.append(String.format("%-30s ", truncateFileName(fileName, 30)));
            bar.append("[");
            for (int i = 0; i < filled; i++) bar.append("=");
            if (filled < barWidth) bar.append(">");
            for (int i = 0; i < empty - 1; i++) bar.append(" ");
            bar.append("] ");
            bar.append(String.format("%3d%% ", percent));
            bar.append(String.format("%s / %s", downloadedStr, formatBytes(total)));

            System.out.print(bar);
            System.out.flush();
        } else {
            // Unknown total size - show spinner
            char[] spinner = {'|', '/', '-', '\\'};
            int spinIdx = (int) ((current / 10000) % 4);

            StringBuilder bar = new StringBuilder();
            bar.append("\r");
            bar.append(String.format("%-30s ", truncateFileName(fileName, 30)));
            bar.append("[");
            bar.append(spinner[spinIdx]);
            bar.append("] ");
            bar.append(downloadedStr);

            System.out.print(bar);
            System.out.flush();
        }
    }

    /**
     * Format bytes into human-readable string.
     */
    private static String formatBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        } else if (bytes < 1024 * 1024) {
            return String.format("%.1f KB", bytes / 1024.0);
        } else if (bytes < 1024 * 1024 * 1024) {
            return String.format("%.1f MB", bytes / (1024.0 * 1024));
        } else {
            return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
        }
    }

    /**
     * Truncate file name to fit in display width.
     */
    private static String truncateFileName(String fileName, int maxLen) {
        if (fileName.length() <= maxLen) {
            return fileName;
        }
        return "..." + fileName.substring(fileName.length() - maxLen + 3);
    }
}
