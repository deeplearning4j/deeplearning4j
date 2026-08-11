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
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test demonstrating the simplified {@link GenerationPipeline} API
 * for VLM text generation with SmolDocling.
 *
 * <p>Complements {@code TestSmolDoclingOptimizedPipeline} which tests the full
 * benchmark configuration matrix. This test validates the high-level pipeline
 * abstraction works correctly with minimal boilerplate.</p>
 *
 * <p>Run with:</p>
 * <pre>
 * cd platform-tests && mvn test \
 *   -Dtest=TestGenerationPipeline#testVLMGenerationPipeline \
 *   -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
public class TestGenerationPipeline {

    @Test
    @DisplayName("TokenizerConfig accepts Hugging Face unlimited-length sentinel")
    public void testTokenizerConfigAcceptsHuggingFaceUnlimitedLengthSentinel() throws Exception {
        org.eclipse.deeplearning4j.llm.config.TokenizerConfig config =
                org.eclipse.deeplearning4j.llm.config.TokenizerConfig.fromJson(
                        "{\"model_max_length\":1000000000000000019884624838656,"
                                + "\"chat_template\":\"{{ messages[0]['content'] }}\"}");

        assertEquals("1000000000000000019884624838656", config.getModelMaxLength().toString());
        assertTrue(config.hasChatTemplate());
    }

    @Test
    @DisplayName("In-graph native handoff consumes the sampled token at the advanced position")
    public void testInGraphNativeDecodeHandoffAdvancesTokenAndPosition() {
        INDArray inputIds = Nd4j.createFromArray(10L).reshape(1, 1);
        INDArray positionOffset = Nd4j.scalar(DataType.INT64, 531L);
        INDArray cachePosition = Nd4j.scalar(DataType.INT64, 531L);
        try {
            GenerationPipeline.prepareInGraphNativeDecodeHandoff(
                    inputIds, positionOffset, cachePosition, 568, 532L);

            assertEquals(568L, inputIds.getLong(0, 0),
                    "native decode must consume the token sampled by Java warmup");
            assertEquals(532L, positionOffset.getLong(0),
                    "RoPE position must advance past the warmup write");
            assertEquals(532L, cachePosition.getLong(0),
                    "KV write position must advance past the warmup write");
        } finally {
            inputIds.close();
            positionOffset.close();
            cachePosition.close();
        }
    }

    @Test
    @DisplayName("Fixed-buffer prefill mask keeps padded FP16 softmax rows finite")
    public void testPaddedPrefillMaskKeepsFp16SoftmaxFinite() {
        int actualLength = 2;
        int paddedLength = 4;
        int maxKvLength = 6;
        INDArray mask = GenerationPipeline.buildPaddedPrefillCausalMask(
                actualLength, paddedLength, maxKvLength, DataType.HALF);
        INDArray paddedScores = null;
        INDArray probabilities = null;
        try {
            assertArrayEquals(new long[]{1, 1, paddedLength, maxKvLength}, mask.shape());
            for (int query = 0; query < actualLength; query++) {
                for (int key = 0; key <= query; key++) {
                    assertEquals(0.0f, mask.getFloat(0, 0, query, key), 0.0f);
                }
                for (int key = query + 1; key < maxKvLength; key++) {
                    assertTrue(mask.getFloat(0, 0, query, key) < 0.0f);
                }
            }
            for (int query = actualLength; query < paddedLength; query++) {
                assertEquals(0.0f, mask.getFloat(0, 0, query, 0), 0.0f,
                        "Padding queries need one finite attention target");
                for (int key = 1; key < maxKvLength; key++) {
                    assertTrue(mask.getFloat(0, 0, query, key) < 0.0f);
                }
            }

            paddedScores = mask.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(actualLength),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
            probabilities = Transforms.softmax(paddedScores, true);
            assertFalse(probabilities.isNaN().any(),
                    "FP16 softmax over a padded query row must remain finite");
            assertEquals(1.0, probabilities.sumNumber().doubleValue(), 1.0e-3);
        } finally {
            if (probabilities != null && !probabilities.wasClosed()) probabilities.close();
            if (paddedScores != null && !paddedScores.wasClosed()) paddedScores.close();
            if (!mask.wasClosed()) mask.close();
        }
    }

    @Test
    @DisplayName("GenerationPipeline: VLM generation with SmolDocling")
    public void testVLMGenerationPipeline() throws Exception {
        // --- 1. Download and load models ---
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
        long downloadMs = System.currentTimeMillis() - t0;
        log.info("Download complete: {}ms", downloadMs);

        // Import ONNX models
        long t1 = System.currentTimeMillis();
        SameDiff decoder = OnnxModelCache.importWithCache(decoderDir.getAbsolutePath());
        SameDiff embedTokens = OnnxModelCache.importWithCache(embedDir.getAbsolutePath());
        SameDiff visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
        long importMs = System.currentTimeMillis() - t1;
        log.info("Import complete: {}ms (decoder={} ops, embed={} ops, vision={} ops)",
                importMs, decoder.ops().length, embedTokens.ops().length, visionEncoder.ops().length);

        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerDir);

        // --- 2. Process a synthetic test image ---
        long t2 = System.currentTimeMillis();
        BufferedImage testImage = createTestImage();
        PreprocessorConfig ppConfig = PreprocessorConfig.fromFile(VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_PREPROCESSOR_CONFIG).getModelFile());
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);

        int targetSize = ppConfig.getTargetHeight();
        ImageTiler.SplitImageResult tileResult = ImageTiler.splitImageForVLM(testImage, targetSize);
        List<INDArray> frames = new ArrayList<>();
        List<INDArray> masks = new ArrayList<>();
        for (int i = 0; i < tileResult.frames.size(); i++) {
            INDArray tensor = preprocessor.preprocess(tileResult.frames.get(i));
            // Reshape to 5D [1, 1, 3, H, W] as expected by the vision encoder
            frames.add(tensor.reshape(1, 1, 3, targetSize, targetSize));
            ImageTiler.ContentRegion region = tileResult.contentRegions.get(i);
            masks.add(ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize));
        }
        log.info("Preprocessed {} frames from test image", frames.size());

        // Vision encode - run each frame through the vision encoder
        // Use hardcoded placeholder names and check for optional pixel_attention_mask.
        boolean hasMaskInput = visionEncoder.getVariable("pixel_attention_mask") != null;
        String visionOutput = visionEncoder.outputs().get(0);
        List<INDArray> encodedFrames = new ArrayList<>();
        for (int i = 0; i < frames.size(); i++) {
            Map<String, INDArray> visionInputMap = new HashMap<>();
            visionInputMap.put("pixel_values", frames.get(i));
            if (hasMaskInput) {
                visionInputMap.put("pixel_attention_mask", masks.get(i));
            }
            Map<String, INDArray> visionResult = visionEncoder.output(visionInputMap, visionOutput);
            encodedFrames.add(visionResult.get(visionOutput).dup());
        }
        INDArray visionEmbeddings = Nd4j.concat(1, encodedFrames.toArray(new INDArray[0]));
        long visionMs = System.currentTimeMillis() - t2;
        log.info("Vision encoding: {}ms, embeddings shape={}", visionMs,
                java.util.Arrays.toString(visionEmbeddings.shape()));

        // --- 3. Build merged embeddings ---
        long t3 = System.currentTimeMillis();
        String prompt = ImagePromptBuilder.buildImagePromptString(
                tileResult.numRows, tileResult.numCols, (int) visionEmbeddings.shape()[1]);
        int[] promptTokenIds = tokenizer.encode(prompt, false).getIds();

        // Text embedding lookup
        INDArray inputIds = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.INT64);
        String embedInput = embedTokens.inputs().get(0);
        String embedOutput = embedTokens.outputs().get(0);
        Map<String, INDArray> embedResult = embedTokens.output(
                Map.of(embedInput, inputIds), embedOutput);
        INDArray textEmbeddings = embedResult.get(embedOutput);

        // Merge vision + text
        int imageTokenId = tokenizer.getTokenId("<image>");
        INDArray mergedEmbeddings = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
        long embedMs = System.currentTimeMillis() - t3;
        log.info("Embedding merge: {}ms, merged shape={}", embedMs,
                java.util.Arrays.toString(mergedEmbeddings.shape()));

        long hiddenSize = mergedEmbeddings.shape()[2];

        // --- 4. Use GenerationPipeline ---
        GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(20)
                .hiddenSize(hiddenSize)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .build();

        GenerationResult result;
        try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
            assertNotNull(pipeline.getIoConfig(), "ModelIOConfig should be auto-discovered");
            log.info("Pipeline created with auto-discovered I/O config");

            result = pipeline.generate(mergedEmbeddings, promptTokenIds);
        }

        // --- 5. Validate result ---
        assertNotNull(result, "GenerationResult must not be null");
        assertNotNull(result.getText(), "Generated text must not be null");
        assertTrue(result.getGeneratedTokenCount() > 0,
                "Should have generated at least one token, got: " + result.getGeneratedTokenCount());
        assertNotNull(result.getTokenIds(), "Token IDs must not be null");
        assertEquals(result.getGeneratedTokenCount(), result.getTokenIds().length,
                "Token count must match tokenIds length");
        assertTrue(result.getGenerationTimeMs() > 0, "Generation time must be positive");
        assertTrue(result.getTokensPerSecond() > 0, "Tokens/sec must be positive");
        assertNotNull(result.getFinishReason(), "Finish reason must not be null");

        log.info("=== GenerationPipeline Result ===");
        log.info("Text: {}", result.getText().substring(0, Math.min(200, result.getText().length())));
        log.info("Tokens: {}, Time: {}ms, Speed: {} tok/s",
                result.getGeneratedTokenCount(), result.getGenerationTimeMs(),
                String.format("%.1f", result.getTokensPerSecond()));
        log.info("Finish reason: {}", result.getFinishReason());

        // Cleanup
        tokenizer.close();
    }

    @Test
    @DisplayName("GenerationPipeline: config validation")
    public void testConfigValidation() {
        // Missing decoder should throw
        assertThrows(IllegalArgumentException.class, () ->
                GenerationPipeline.create(GenerationPipelineConfig.builder()
                        .embedTokens(SameDiff.create())
                        .tokenizer(null)
                        .build()));
    }

    private static BufferedImage createTestImage() {
        BufferedImage img = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 800, 600);
        g.setColor(Color.BLACK);
        g.setFont(new Font("Serif", Font.PLAIN, 24));
        g.drawString("Test Document", 50, 50);
        g.drawString("This is a test page for the GenerationPipeline.", 50, 100);
        g.drawString("Section 1: Introduction", 50, 160);
        g.drawString("Lorem ipsum dolor sit amet.", 50, 200);
        g.dispose();
        return img;
    }
}
