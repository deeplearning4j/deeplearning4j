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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
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
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive stress test for GenerationPipeline.
 *
 * <p>Exercises the pipeline under many conditions:</p>
 * <ul>
 *   <li>Multiple token sizes on the SAME pipeline instance (replay stress)</li>
 *   <li>Multiple sampling configs (greedy, temperature, topK, topP, combined)</li>
 *   <li>Fresh pipeline after full model reset (state isolation)</li>
 *   <li>Repeatability across fresh GenerationPipeline instances</li>
 *   <li>Structural tag validation at all token sizes</li>
 * </ul>
 */
@Slf4j
public class TestGenerationPipelineComprehensive {

    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static Tokenizer tokenizer;
    private static INDArray inputsEmbeds;
    private static int[] promptTokenIds;
    private static long hiddenSize;
    private static boolean modelsLoaded = false;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    private static synchronized void ensureModelsLoaded() throws Exception {
        if (modelsLoaded) return;

        log.info("Loading SmolDocling models for comprehensive stress test...");

        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        decoder = models[1];
        embedTokens = models[2];

        int targetSize = 512;
        String pdfPath = System.getProperty("vlm.test.pdf.path",
                "platform-tests/pathfinder-mythic.pdf");
        File pdfFile = new File(pdfPath);
        BufferedImage image;
        if (pdfFile.exists()) {
            log.info("Using mythic PDF: {}", pdfFile.getAbsolutePath());
            try (org.apache.pdfbox.pdmodel.PDDocument doc = org.apache.pdfbox.pdmodel.PDDocument.load(pdfFile)) {
                org.apache.pdfbox.rendering.PDFRenderer renderer = new org.apache.pdfbox.rendering.PDFRenderer(doc);
                image = renderer.renderImageWithDPI(0, 150, org.apache.pdfbox.rendering.ImageType.RGB);
            }
        } else {
            log.info("PDF not found, generating synthetic document image");
            image = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_3BYTE_BGR);
            java.awt.Graphics2D g = image.createGraphics();
            g.setColor(java.awt.Color.WHITE);
            g.fillRect(0, 0, targetSize, targetSize);
            g.setColor(java.awt.Color.BLACK);
            g.setFont(new java.awt.Font("SansSerif", java.awt.Font.PLAIN, 24));
            g.drawString("Test Document", 50, 100);
            g.drawString("Line 2: Mythic Regression", 50, 150);
            g.dispose();
        }

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(image, targetSize, 9);

        PreprocessorConfig ppConfig = new PreprocessorConfig();
        ppConfig.setSize(new PreprocessorConfig.ImageSize(targetSize, targetSize));
        ppConfig.setDoRescale(true);
        ppConfig.setRescaleFactor(1.0 / 255.0);
        ppConfig.setDoNormalize(true);
        ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
        ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, targetSize);
        preprocessor.shutdown();

        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);

        List<INDArray> frameEmbeddings = new ArrayList<>();
        for (int frameIdx = 0; frameIdx < splitResult.getTotalFrames(); frameIdx++) {
            INDArray frameSlice = imageInput.get(
                    NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray singleFrame = frameSlice.reshape(1, 1, 3, targetSize, targetSize).dup();

            Map<String, INDArray> visionInputMap = new HashMap<>();
            for (String inputName : visionInputNames) {
                if (inputName.equals("pixel_values")) {
                    visionInputMap.put(inputName, singleFrame);
                } else if (inputName.equals("pixel_attention_mask")) {
                    ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                    visionInputMap.put(inputName,
                            ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize));
                }
            }

            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            frameEmbeddings.add(selected.tensor.dup());
            for (var entry : visionOutputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
            }
            singleFrame.close();
        }

        visionEncoder.clearPlaceholders(false);
        visionEncoder.clearOpInputs();
        visionEncoder.resetSession();
        Nd4j.getExecutioner().commit();

        INDArray visionEmbeddings = frameEmbeddings.size() == 1
                ? frameEmbeddings.get(0).dup()
                : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0]));

        hiddenSize = visionEmbeddings.size(-1);

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.shape()[1] / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] encoded = tokenizer.encode(chatPrompt, false).getIds();
        promptTokenIds = encoded;

        INDArray tokenIds = Nd4j.createFromArray(encoded).reshape(1, encoded.length).castTo(DataType.INT64);
        Map<String, INDArray> embedInputs = new HashMap<>();
        for (String inputName : embedTokens.inputs()) {
            embedInputs.put(inputName, tokenIds);
        }
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs,
                embedTokens.outputs().toArray(new String[0]));
        INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();
        tokenIds.close();

        inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, encoded, imageTokenId);
        assertEquals(promptTokenIds.length, inputsEmbeds.size(1),
                "Merged embedding sequence length must match prompt token length");

        log.info("Models loaded: decoder={} ops, embed={} ops, hiddenSize={}, promptTokens={}",
                decoder.getOps().size(), embedTokens.getOps().size(), hiddenSize, promptTokenIds.length);
        modelsLoaded = true;
    }

    /**
     * Fully reset decoder and embedTokens state to avoid PLAN_CACHE_BUG when reusing models.
     */
    private static void hardResetModels() {
        if (decoder != null) {
            decoder.clearPlaceholders(true);
            decoder.clearOpInputs();
            decoder.resetSession();
        }
        if (embedTokens != null) {
            embedTokens.clearPlaceholders(true);
            embedTokens.clearOpInputs();
            embedTokens.resetSession();
        }
        Nd4j.getExecutioner().commit();
        System.gc();
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO A: Same pipeline, multiple token sizes (replay stress test)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("SCENARIO A: Same pipeline instance, multiple token sizes")
    public void testScenarioA_ReplayStressMultipleTokenSizes() throws Exception {
        ensureModelsLoaded();
        hardResetModels();

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(250)
                .hiddenSize(hiddenSize)
                .build());

        int[] tokenSizes = {5, 10, 15, 25, 50};

        for (int maxTokens : tokenSizes) {
            log.info("=== SCENARIO A: generate() with maxTokens={} ===", maxTokens);

            GenerationResult result = pipeline.generate(inputsEmbeds.dup(), promptTokenIds, maxTokens);
            String text = result.getText();
            int[] tokens = result.getTokenIds();

            log.info("  Result: {} tokens generated, text='{}'", tokens.length, text);

            assertNotNull(text, "Text must not be null for maxTokens=" + maxTokens);
            assertTrue(tokens.length > 0,
                    "Must generate at least 1 token for maxTokens=" + maxTokens);
            assertTrue(tokens.length <= maxTokens,
                    "Generated " + tokens.length + " tokens but maxTokens was " + maxTokens);

            // Structural tag check — even small outputs should start with structural tags
            if (maxTokens >= 5) {
                boolean hasTags = text.contains("<") && text.contains(">");
                log.info("  Structural tags present: {}", hasTags);
                assertTrue(hasTags,
                        "Output should contain structural tags for maxTokens=" + maxTokens
                                + ", text=" + text);
            }
        }

        pipeline.close();
        log.info("SCENARIO A PASSED: All token sizes produced valid output on same pipeline");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO B: Same pipeline, different sampling configs
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("SCENARIO B: Same pipeline instance, different sampling configs")
    public void testScenarioB_DifferentSamplingConfigs() throws Exception {
        ensureModelsLoaded();
        hardResetModels();

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(20)
                .hiddenSize(hiddenSize)
                .build());

        SamplingConfig[] configs = {
                SamplingConfig.greedy(),
                SamplingConfig.builder().temperature(0.5).doSample(true).build(),
                SamplingConfig.builder().temperature(0.8).topK(40).doSample(true).build(),
                SamplingConfig.builder().temperature(0.7).topP(0.9).doSample(true).build(),
                SamplingConfig.builder().temperature(0.8).topK(50).topP(0.95).doSample(true).build(),
        };

        for (SamplingConfig config : configs) {
            log.info("=== SCENARIO B: sampling={} ===", config);

            // Create a new pipeline with this sampling config, but reuse the models
            GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                    .decoder(decoder)
                    .embedTokens(embedTokens)
                    .tokenizer(tokenizer)
                    .ioConfig(ioConfig)
                    .samplingConfig(config)
                    .maxNewTokens(20)
                    .hiddenSize(hiddenSize)
                    .build();

            // We can't change sampling on existing pipeline, so create new one
            // But models are reused — this tests model state stability
            GenerationPipeline subPipeline = GenerationPipeline.create(pipelineConfig);

            GenerationResult result = subPipeline.generate(inputsEmbeds.dup(), promptTokenIds, 20);
            String text = result.getText();
            int[] tokens = result.getTokenIds();

            log.info("  Result: {} tokens, text='{}'", tokens.length, text);

            assertNotNull(text);
            assertTrue(tokens.length > 0);
            assertTrue(text.contains("<") && text.contains(">"),
                    "Should have structural tags with sampling=" + config);

            subPipeline.close();
        }

        pipeline.close();
        log.info("SCENARIO B PASSED: All sampling configs produced valid output");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO C: Fresh pipeline after full reset, repeatability across token sizes
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("SCENARIO C: Fresh GenerationPipeline is repeatable at multiple token sizes")
    public void testScenarioC_FreshPipelineRepeatability() throws Exception {
        ensureModelsLoaded();

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        int[] tokenSizes = {10, 20, 30};

        for (int maxTokens : tokenSizes) {
            log.info("=== SCENARIO C: maxTokens={} ===", maxTokens);

            hardResetModels();
            GenerationResult first;
            GenerationPipeline firstPipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                    .decoder(decoder)
                    .embedTokens(embedTokens)
                    .tokenizer(tokenizer)
                    .ioConfig(ioConfig)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(maxTokens)
                    .hiddenSize(hiddenSize)
                    .build());
            try {
                first = firstPipeline.generate(inputsEmbeds.dup(), promptTokenIds, maxTokens);
            } finally {
                firstPipeline.close();
            }

            hardResetModels();
            GenerationResult second;
            GenerationPipeline secondPipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                    .decoder(decoder)
                    .embedTokens(embedTokens)
                    .tokenizer(tokenizer)
                    .ioConfig(ioConfig)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(maxTokens)
                    .hiddenSize(hiddenSize)
                    .build());
            try {
                second = secondPipeline.generate(inputsEmbeds.dup(), promptTokenIds, maxTokens);
            } finally {
                secondPipeline.close();
            }

            log.info("  FIRST: {} tokens, text='{}'", first.getTokenIds().length, first.getText());
            log.info("  SECOND: {} tokens, text='{}'", second.getTokenIds().length, second.getText());
            assertArrayEquals(first.getTokenIds(), second.getTokenIds(),
                    "Fresh GenerationPipeline runs must be token-repeatable for maxTokens=" + maxTokens);
            assertEquals(first.getText(), second.getText(),
                    "Fresh GenerationPipeline text must be repeatable for maxTokens=" + maxTokens);
        }

        log.info("SCENARIO C PASSED: Fresh GenerationPipeline is repeatable at all token sizes");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO D: Large token count (benchmark scale)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("SCENARIO D: Large token count (benchmark scale 250)")
    public void testScenarioD_LargeTokenCount() throws Exception {
        ensureModelsLoaded();
        hardResetModels();

        int maxTokens = Integer.getInteger("vlm.test.maxNewTokens", 250);
        log.info("=== SCENARIO D: Large token count maxTokens={} ===", maxTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build());

        long startMs = System.currentTimeMillis();
        GenerationResult result = pipeline.generate(inputsEmbeds.dup(), promptTokenIds, maxTokens);
        long elapsedMs = System.currentTimeMillis() - startMs;

        String text = result.getText();
        int[] tokens = result.getTokenIds();

        log.info("  Generated {} tokens in {}ms ({} tok/s)",
                tokens.length, elapsedMs,
                elapsedMs > 0 ? String.format("%.2f", tokens.length * 1000.0 / elapsedMs) : "N/A");
        log.info("  Text prefix (first 200 chars): {}",
                text.length() > 200 ? text.substring(0, 200) : text);

        assertNotNull(text);
        assertTrue(tokens.length > 0, "Must generate tokens");

        // Must contain structural tags
        boolean hasStructuralTags = text.contains("<") && text.contains(">");
        assertTrue(hasStructuralTags,
                "Large-token generation must contain structural tags. Text: " + text);

        // Must contain mythic-related content or docling structure
        boolean hasContent = text.contains("<") || text.contains("doctag") || text.contains("page")
                || text.contains("section") || text.contains("loc_");
        assertTrue(hasContent,
                "Large-token generation must have recognizable content. Text: " + text);

        pipeline.close();
        log.info("SCENARIO D PASSED: Large token generation produced valid output");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO E: Rapid successive calls (replay stability)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("SCENARIO E: Rapid successive calls on same pipeline")
    public void testScenarioE_RapidSuccessiveCalls() throws Exception {
        ensureModelsLoaded();
        hardResetModels();

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(15)
                .hiddenSize(hiddenSize)
                .build());

        int iterations = 3;
        String[] texts = new String[iterations];
        int[][] tokenArrays = new int[iterations][];

        for (int i = 0; i < iterations; i++) {
            log.info("=== SCENARIO E: iteration {}/{} ===", i + 1, iterations);
            GenerationResult result = pipeline.generate(inputsEmbeds.dup(), promptTokenIds, 15);
            texts[i] = result.getText();
            tokenArrays[i] = result.getTokenIds();
            log.info("  Iteration {}: {} tokens, text='{}'", i + 1, tokenArrays[i].length, texts[i]);
        }

        // All iterations should produce identical output (greedy decoding)
        for (int i = 1; i < iterations; i++) {
            assertArrayEquals(tokenArrays[0], tokenArrays[i],
                    "Iteration " + (i + 1) + " diverged from iteration 1.\n"
                            + "Iteration 1: " + texts[0] + "\n"
                            + "Iteration " + (i + 1) + ": " + texts[i]);
        }

        pipeline.close();
        log.info("SCENARIO E PASSED: {} rapid calls all produced identical output", iterations);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO F: Text-only prompt (no image embeddings)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("SCENARIO F: Text-only generation")
    public void testScenarioF_TextOnlyGeneration() throws Exception {
        ensureModelsLoaded();
        hardResetModels();

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        // Build text-only embeddings
        INDArray tokenIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.INT64);
        Map<String, INDArray> embedInputs = new HashMap<>();
        for (String inputName : embedTokens.inputs()) {
            embedInputs.put(inputName, tokenIds);
        }
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs,
                embedTokens.outputs().toArray(new String[0]));
        INDArray textOnlyEmbeds = embedOutputs.values().iterator().next().dup();
        tokenIds.close();

        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(15)
                .hiddenSize(hiddenSize)
                .build());

        GenerationResult result = pipeline.generate(textOnlyEmbeds, promptTokenIds, 15);
        String text = result.getText();
        int[] tokens = result.getTokenIds();

        log.info("Text-only generation: {} tokens, text='{}'", tokens.length, text);

        assertNotNull(text);
        assertTrue(tokens.length > 0);

        pipeline.close();
        log.info("SCENARIO F PASSED: Text-only generation works");
    }
}
