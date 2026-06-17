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
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.model.download.ModelDownloader;

import java.io.*;
import java.nio.file.Files;

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
    @Getter
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
        ),

        // SmolLM2-135M - Draft model for speculative decoding
        SMOLLM2_135M_DECODER(
                "smollm2-135m-decoder",
                "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/onnx/model.onnx",
                ModelFormat.ONNX,
                0, 0,
                "SmolLM2-135M-Instruct Decoder - tiny draft model for speculative decoding (~540MB)"
        ),

        SMOLLM2_135M_TOKENIZER(
                "smollm2-135m-tokenizer",
                "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json",
                ModelFormat.JSON,
                0, 0,
                "SmolLM2-135M-Instruct Tokenizer - HuggingFace tokenizer.json"
        ),

        SMOLLM2_135M_CONFIG(
                "smollm2-135m-config",
                "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json",
                ModelFormat.JSON,
                0, 0,
                "SmolLM2-135M-Instruct Config - model configuration"
        ),

        SMOLDOCLING_PREPROCESSOR_CONFIG(
                "smoldocling-preprocessor-config",
                "https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/preprocessor_config.json",
                ModelFormat.JSON,
                0, 0,
                "SmolDocling Preprocessor Config - image preprocessing parameters"
        ),

        // Video VLM Models - SmolVLM2 (Apache 2.0, smallest video VLM)
        SMOLVLM2_256M_VIDEO_GGUF(
                "smolvlm2-256m-video-instruct",
                "https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                ModelFormat.GGUF,
                384, 384,
                "SmolVLM2-256M Video Instruct Q8_0 - smallest video VLM (~175MB)"
        ),

        SMOLVLM2_2_2B_VIDEO_GGUF(
                "smolvlm2-2.2b-video-instruct",
                "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
                ModelFormat.GGUF,
                384, 384,
                "SmolVLM2-2.2B Video Instruct Q4_K_M - mid-size video VLM (~1.4GB)"
        ),

        // Video VLM Models - Qwen3-VL (Apache 2.0, best video quality)
        QWEN3_VL_2B_GGUF(
                "qwen3-vl-2b-instruct",
                "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/qwen3-vl-2b-instruct-q4_k_m.gguf",
                ModelFormat.GGUF,
                0, 0,
                "Qwen3-VL-2B LLM decoder Q4_K_M - dynamic resolution video VLM (~1.5GB)"
        ),

        QWEN3_VL_2B_MMPROJ_GGUF(
                "qwen3-vl-2b-mmproj",
                "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/qwen3-vl-2b-instruct-vision-q8_0.gguf",
                ModelFormat.GGUF,
                0, 0,
                "Qwen3-VL-2B vision encoder mmproj Q8_0 - paired with LLM GGUF"
        ),

        // Video VLM Models - MiniCPM-V 4.5 (Apache 2.0, best video token efficiency)
        MINICPM_V_4_5_GGUF(
                "minicpm-v-4.5",
                "https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf/resolve/main/ggml-model-Q4_K_M.gguf",
                ModelFormat.GGUF,
                0, 0,
                "MiniCPM-V 4.5 Q4_K_M - 3D-Resampler 96x video compression (~5GB)"
        ),

        MINICPM_V_4_5_MMPROJ_GGUF(
                "minicpm-v-4.5-mmproj",
                "https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf/resolve/main/mmproj-model-f16.gguf",
                ModelFormat.GGUF,
                0, 0,
                "MiniCPM-V 4.5 vision mmproj F16 - paired with LLM GGUF"
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

        public String getFileName() {
            return name + "." + format.getExtension();
        }
    }

    @Getter
    public enum ModelFormat {
        ONNX("onnx"),
        GGUF("gguf"),
        GGML("ggml"),
        JSON("json");

        private final String extension;

        ModelFormat(String extension) {
            this.extension = extension;
        }
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
        ModelDownloader.DownloadResult result = ModelDownloader.download(
                model.getUrl(), model.getFileName(), cacheDir);

        return DownloadResult.builder()
                .modelFile(result.getModelFile())
                .model(model)
                .downloadedNow(result.isDownloadedNow())
                .fileSizeBytes(result.getFileSizeBytes())
                .build();
    }

    public static File downloadCustom(String url, String fileName, ModelFormat format) throws IOException {
        ModelDownloader.DownloadResult result = ModelDownloader.download(url, fileName, getCacheDir());
        return result.getModelFile();
    }

    public static boolean isCached(VLMModel model) {
        return ModelDownloader.isCached(model.getFileName(), getCacheDir());
    }

    public static File getCacheDir() {
        return ModelDownloader.getCacheDir(CACHE_DIR_PROPERTY, DEFAULT_CACHE_DIR);
    }

    public static void clearCache() throws IOException {
        ModelDownloader.clearCache(getCacheDir());
    }

    public static File[] listCachedModels() {
        return ModelDownloader.listCachedFiles(getCacheDir(), "onnx", "gguf", "ggml", "json");
    }
}
