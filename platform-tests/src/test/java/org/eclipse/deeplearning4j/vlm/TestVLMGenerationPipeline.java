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

package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.llm.generation.*;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.encoder.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * VLM GenerationPipeline test using SmolDocling, modeled after run-benchmark.sh.
 *
 * <p>Tests the full VLM pipeline: download models, preprocess image, vision encode,
 * merge embeddings, and decode with GenerationPipeline.</p>
 *
 * <p>Models are loaded once in {@code @BeforeAll} and shared across all tests to
 * avoid GPU OOM from repeated loading (~21GB per load).</p>
 *
 * <p>System properties:</p>
 * <ul>
 *   <li>{@code -Dvlm.test.pdf.path=pathfinder-mythic.pdf} - PDF file path</li>
 *   <li>{@code -Dvlm.test.pdf.page=10} - PDF page to render (0-indexed)</li>
 *   <li>{@code -Dvlm.test.maxTokens=20} - max tokens to generate (default: 20)</li>
 * </ul>
 *
 * <p>Run with:</p>
 * <pre>
 * cd platform-tests && mvn test \
 *   -Dtest=TestVLMGenerationPipeline \
 *   -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestVLMGenerationPipeline {

    // Shared models loaded once — reused across all tests
    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static SameDiff visionEncoder;
    private static Tokenizer tokenizer;
    private static long downloadMs;
    private static long importMs;
    private static boolean modelsLoaded = false;

    @BeforeAll
    public static void setup() throws Exception {
        if (System.getProperty("nd4j.optimizer.enabled") == null) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }

        // Download SmolDocling models
        log.info("Downloading SmolDocling models...");
        long t0 = System.currentTimeMillis();
        File decoderDir = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER).getModelFile();
        File embedDir = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS).getModelFile();
        File visionDir = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
        File tokenizerDir = VLMModelDownloader.download(
                VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER).getModelFile();
        downloadMs = System.currentTimeMillis() - t0;
        log.info("Download: {}ms", downloadMs);

        // Import ONNX -> SameDiff with SDZ caching
        long t1 = System.currentTimeMillis();
        decoder = OnnxModelCache.importWithCache(decoderDir.getAbsolutePath());
        embedTokens = OnnxModelCache.importWithCache(embedDir.getAbsolutePath());
        visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
        importMs = System.currentTimeMillis() - t1;
        log.info("Import: {}ms (decoder={} ops, embed={} ops, vision={} ops)",
                importMs, decoder.ops().length, embedTokens.ops().length,
                visionEncoder.ops().length);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerDir);
        modelsLoaded = true;
    }

    @AfterAll
    public static void teardown() {
        if (!modelsLoaded) {
            return;
        }
        try {
            if (visionEncoder != null) {
                SameDiffMemoryUtils.freeVisionEncoder(visionEncoder);
                visionEncoder = null;
            }
            if (decoder != null) {
                SameDiffMemoryUtils.freeModelArrays(decoder);
                decoder.close();
                decoder = null;
            }
            if (embedTokens != null) {
                SameDiffMemoryUtils.freeModelArrays(embedTokens);
                embedTokens.close();
                embedTokens = null;
            }
            if (tokenizer != null) {
                tokenizer.close();
                tokenizer = null;
            }
            log.info("All models freed in @AfterAll");
        } catch (Exception e) {
            log.warn("Error during teardown: {}", e.getMessage());
        }
    }

    // ─── Image processing ───────────────────────────────────────────────

    private INDArray processImageThroughVision(BufferedImage image) throws Exception {
        PreprocessorConfig ppConfig = PreprocessorConfig.fromFile(VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_PREPROCESSOR_CONFIG).getModelFile());
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        int targetSize = ppConfig.getTargetHeight();

        // Tile the image
        ImageTiler.SplitImageResult tileResult = ImageTiler.splitImageForVLM(image, targetSize);
        List<INDArray> frames = new ArrayList<>();
        List<INDArray> masks = new ArrayList<>();
        for (int i = 0; i < tileResult.frames.size(); i++) {
            INDArray tensor = preprocessor.preprocess(tileResult.frames.get(i));
            // Reshape to 5D [1, 1, 3, H, W] as expected by the vision encoder
            frames.add(tensor.reshape(1, 1, 3, targetSize, targetSize));
            ImageTiler.ContentRegion region = tileResult.contentRegions.get(i);
            masks.add(ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize));
        }
        log.info("Preprocessed {} frames", frames.size());

        // Check if vision encoder has pixel_attention_mask input
        boolean hasMaskInput = visionEncoder.getVariable("pixel_attention_mask") != null;

        // Vision encode each frame
        String visionOutput = visionEncoder.outputs().get(0);
        List<INDArray> encodedFrames = new ArrayList<>();
        for (int i = 0; i < frames.size(); i++) {
            Map<String, INDArray> inputs = new java.util.HashMap<>();
            inputs.put("pixel_values", frames.get(i));
            if (hasMaskInput) {
                inputs.put("pixel_attention_mask", masks.get(i));
            }
            Map<String, INDArray> visionResult = visionEncoder.output(inputs, visionOutput);
            // Dup to detach from SameDiff session
            encodedFrames.add(visionResult.get(visionOutput).dup());
        }

        // Close frame and mask arrays after encoding
        for (INDArray frame : frames) {
            SameDiffMemoryUtils.safeClose(frame);
        }
        for (INDArray mask : masks) {
            SameDiffMemoryUtils.safeClose(mask);
        }

        INDArray visionEmbeddings = Nd4j.concat(1, encodedFrames.toArray(new INDArray[0]));
        // Close individual encoded frames after concat
        for (INDArray encoded : encodedFrames) {
            SameDiffMemoryUtils.safeClose(encoded);
        }
        log.info("Vision embeddings shape: {}", java.util.Arrays.toString(visionEmbeddings.shape()));

        return visionEmbeddings;
    }

    private MergedEmbeddings mergeEmbeddings(BufferedImage image) throws Exception {
        PreprocessorConfig ppConfig = PreprocessorConfig.fromFile(VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_PREPROCESSOR_CONFIG).getModelFile());
        ImageTiler.SplitImageResult tileResult = ImageTiler.splitImageForVLM(image, ppConfig.getTargetHeight());

        INDArray visionEmbeddings = processImageThroughVision(image);

        // Build prompt and text embeddings
        String prompt = ImagePromptBuilder.buildImagePromptString(
                tileResult.numRows, tileResult.numCols, (int) visionEmbeddings.shape()[1]);
        int[] promptTokenIds = tokenizer.encode(prompt, false).getIds();

        INDArray inputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.INT64);
        String embedInput = embedTokens.inputs().get(0);
        String embedOutput = embedTokens.outputs().get(0);
        Map<String, INDArray> embedResult = embedTokens.output(
                Map.of(embedInput, inputIds), embedOutput);
        // Dup to detach from SameDiff session
        INDArray textEmbeddings = embedResult.get(embedOutput).dup();
        SameDiffMemoryUtils.safeClose(inputIds);

        // Merge vision + text
        int imageTokenId = tokenizer.getTokenId("<image>");
        INDArray merged = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
        log.info("Merged embeddings shape: {}", java.util.Arrays.toString(merged.shape()));

        // Close intermediates after merge
        SameDiffMemoryUtils.safeClose(textEmbeddings);
        SameDiffMemoryUtils.safeClose(visionEmbeddings);

        return new MergedEmbeddings(merged, promptTokenIds, merged.shape()[2]);
    }

    private static class MergedEmbeddings {
        final INDArray embeddings;
        final int[] promptTokenIds;
        final long hiddenSize;

        MergedEmbeddings(INDArray embeddings, int[] promptTokenIds, long hiddenSize) {
            this.embeddings = embeddings;
            this.promptTokenIds = promptTokenIds;
            this.hiddenSize = hiddenSize;
        }
    }

    // ─── Test: VLM pipeline config validation ───────────────────────────

    @Test
    @Order(1)
    @DisplayName("VLM GenerationPipeline: config validation")
    public void testVLMConfigValidation() {
        // Missing tokenizer should throw
        assertThrows(IllegalArgumentException.class, () ->
                GenerationPipeline.create(GenerationPipelineConfig.builder()
                        .decoder(SameDiff.create())
                        .embedTokens(SameDiff.create())
                        .tokenizer(null)
                        .build()));

        // Missing decoder should throw
        assertThrows(IllegalArgumentException.class, () ->
                GenerationPipeline.create(GenerationPipelineConfig.builder()
                        .embedTokens(SameDiff.create())
                        .tokenizer(null)
                        .build()));
    }

    // ─── Test: VLM with synthetic image ─────────────────────────────────

    @Test
    @Order(2)
    @DisplayName("VLM GenerationPipeline: SmolDocling with synthetic image")
    public void testVLMPipelineWithSyntheticImage() throws Exception {
        int maxTokens = Integer.parseInt(getPropertyOrDefault("vlm.test.maxTokens", "20"));

        BufferedImage testImage = createTestImage();

        long t0 = System.currentTimeMillis();
        MergedEmbeddings merged = mergeEmbeddings(testImage);
        long embedMs = System.currentTimeMillis() - t0;
        log.info("Embedding merge: {}ms", embedMs);

        // Free vision encoder after encoding — reclaims ~5GB GPU memory
        SameDiffMemoryUtils.freeVisionEncoder(visionEncoder);
        visionEncoder = null;

        // Build GenerationPipeline with preprocessor config
        PreprocessorConfig ppConfig = PreprocessorConfig.fromFile(VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_PREPROCESSOR_CONFIG).getModelFile());
        GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .maxPrefillLength(3072)
                .maxKvCacheLength(4608)
                .benchmarkConfig(org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig.optimal())
                .hiddenSize(merged.hiddenSize)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .preprocessorConfig(ppConfig)
                .build();

        GenerationResult result;
        try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
            assertNotNull(pipeline.getIoConfig(), "ModelIOConfig should be auto-discovered");
            assertNotNull(pipeline.getDecoder(), "Decoder should be set");
            assertNotNull(pipeline.getEmbedTokens(), "EmbedTokens should be set");
            assertNotNull(pipelineConfig.getPreprocessorConfig(), "PreprocessorConfig should be set");

            log.info("Generating from synthetic image...");
            t0 = System.currentTimeMillis();
            result = pipeline.generate(merged.embeddings, merged.promptTokenIds);
            long genMs = System.currentTimeMillis() - t0;
            log.info("Generation: {}ms", genMs);
        }

        SameDiffMemoryUtils.safeClose(merged.embeddings);

        // Validate
        validateResult(result, maxTokens);

        log.info("=== VLM Synthetic Image Result ===");
        logResult(result);
    }

    // ─── Test: VLM with PDF (like run-benchmark.sh) ─────────────────────

    @Test
    @Order(3)
    @DisplayName("VLM GenerationPipeline: SmolDocling with PDF (benchmark-style)")
    public void testVLMPipelineWithPDF() throws Exception {
        String pdfPath = getPropertyOrDefault("vlm.test.pdf.path", null);
        int pdfPage = Integer.parseInt(getPropertyOrDefault("vlm.test.pdf.page", "10"));
        int maxTokens = Integer.parseInt(getPropertyOrDefault("vlm.test.maxTokens", "50"));

        // Load the image
        BufferedImage image;
        if (pdfPath != null && !pdfPath.isEmpty()) {
            File pdfFile = new File(pdfPath);
            if (!pdfFile.isAbsolute()) {
                pdfFile = new File(System.getProperty("user.dir"), pdfPath);
            }
            if (!pdfFile.exists()) {
                log.warn("PDF file not found: {}. Falling back to synthetic image.", pdfFile);
                image = createTestImage();
            } else {
                log.info("Rendering PDF page {} from: {}", pdfPage, pdfFile);
                image = renderPdfPage(pdfFile, pdfPage);
            }
        } else {
            log.info("No PDF path specified (-Dvlm.test.pdf.path=...). Using synthetic image.");
            image = createTestImage();
        }

        // If vision encoder was freed by a previous test, reload it
        if (visionEncoder == null) {
            log.info("Reloading vision encoder for PDF test...");
            File visionDir = VLMModelDownloader.download(
                    VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
            visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
        }

        long t0 = System.currentTimeMillis();
        MergedEmbeddings merged = mergeEmbeddings(image);
        long embedMs = System.currentTimeMillis() - t0;
        log.info("Embedding merge: {}ms", embedMs);

        // Free vision encoder after encoding
        SameDiffMemoryUtils.freeVisionEncoder(visionEncoder);
        visionEncoder = null;

        // Build GenerationPipeline
        GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .maxPrefillLength(3072)
                .maxKvCacheLength(4608)
                .benchmarkConfig(org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig.optimal())
                .hiddenSize(merged.hiddenSize)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .build();

        GenerationResult result;
        try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
            log.info("Generating from PDF page...");
            t0 = System.currentTimeMillis();
            result = pipeline.generate(merged.embeddings, merged.promptTokenIds);
            long genMs = System.currentTimeMillis() - t0;
            log.info("Generation: {}ms", genMs);
        }

        SameDiffMemoryUtils.safeClose(merged.embeddings);

        // Validate
        validateResult(result, maxTokens);

        log.info("=== VLM PDF Result ===");
        logResult(result);

        // For PDF with sufficient tokens, check for coherent output
        if (maxTokens >= 50 && result.getText() != null) {
            assertFalse(result.getText().trim().isEmpty(),
                    "Generated text should not be empty for " + maxTokens + " token limit");
        }
    }

    // ─── Test: VLM pipeline with streaming ──────────────────────────────

    @Test
    @Order(4)
    @DisplayName("VLM GenerationPipeline: streaming generation")
    public void testVLMPipelineStreaming() throws Exception {
        int maxTokens = Integer.parseInt(getPropertyOrDefault("vlm.test.maxTokens", "10"));

        // If vision encoder was freed by a previous test, reload it
        if (visionEncoder == null) {
            log.info("Reloading vision encoder for streaming test...");
            File visionDir = VLMModelDownloader.download(
                    VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
            visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
        }

        BufferedImage testImage = createTestImage();
        MergedEmbeddings merged = mergeEmbeddings(testImage);

        // Free vision encoder after encoding
        SameDiffMemoryUtils.freeVisionEncoder(visionEncoder);
        visionEncoder = null;

        GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(merged.hiddenSize)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .build();

        List<String> streamedTokens = new ArrayList<>();
        try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
            GenerationResult result = pipeline.generate(merged.embeddings, merged.promptTokenIds);
            assertNotNull(result, "Result must not be null");

            if (result.getTokenIds() != null) {
                for (int tokenId : result.getTokenIds()) {
                    String tokenText = tokenizer.decode(new int[]{tokenId}, true);
                    streamedTokens.add(tokenText);
                }
            }
        }

        SameDiffMemoryUtils.safeClose(merged.embeddings);

        log.info("Streamed {} tokens: {}", streamedTokens.size(), String.join("", streamedTokens));
        assertTrue(streamedTokens.size() > 0, "Should produce at least 1 token");
    }

    // ─── Helpers ────────────────────────────────────────────────────────

    private static String getPropertyOrDefault(String key, String defaultValue) {
        String value = System.getProperty(key);
        return (value == null || value.isEmpty()) ? defaultValue : value;
    }

    private void validateResult(GenerationResult result, int maxTokens) {
        assertNotNull(result, "GenerationResult must not be null");
        assertNotNull(result.getText(), "Generated text must not be null");
        assertTrue(result.getGeneratedTokenCount() > 0,
                "Should generate at least 1 token, got: " + result.getGeneratedTokenCount());
        assertNotNull(result.getTokenIds(), "Token IDs must not be null");
        assertEquals(result.getGeneratedTokenCount(), result.getTokenIds().length,
                "Token count must match tokenIds length");
        assertTrue(result.getGenerationTimeMs() > 0, "Generation time must be positive");
        assertTrue(result.getTokensPerSecond() > 0, "Tokens/sec must be positive");
        assertNotNull(result.getFinishReason(), "Finish reason must not be null");
    }

    private void logResult(GenerationResult result) {
        log.info("Text: {}", result.getText().substring(0, Math.min(200, result.getText().length())));
        log.info("Tokens: {}, Time: {}ms, Speed: {} tok/s",
                result.getGeneratedTokenCount(), result.getGenerationTimeMs(),
                String.format("%.1f", result.getTokensPerSecond()));
        log.info("First token latency: {}ms", result.getFirstTokenLatencyMs());
        log.info("Finish reason: {}", result.getFinishReason());
        log.info("Pipeline setup: download={}ms import={}ms", downloadMs, importMs);
    }

    private static BufferedImage createTestImage() {
        BufferedImage img = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 800, 600);
        g.setColor(Color.BLACK);
        g.setFont(new Font("Serif", Font.PLAIN, 24));
        g.drawString("Test Document", 50, 50);
        g.drawString("This is a test page for the VLM GenerationPipeline.", 50, 100);
        g.drawString("Section 1: Introduction", 50, 160);
        g.drawString("Lorem ipsum dolor sit amet, consectetur adipiscing elit.", 50, 200);
        g.drawString("Section 2: Methods", 50, 260);
        g.drawString("The method involves processing images through a vision encoder.", 50, 300);
        g.dispose();
        return img;
    }

    private static BufferedImage renderPdfPage(File pdfFile, int pageIndex) throws Exception {
        try (PDDocument doc = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(doc);
            int renderDpi = 150;
            return renderer.renderImageWithDPI(pageIndex, renderDpi, ImageType.RGB);
        }
    }
}
