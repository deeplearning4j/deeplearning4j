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
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
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

        Tokenizer tokenizer = HuggingFaceTokenizer.fromDirectory(tokenizerDir);

        // --- 2. Process a synthetic test image ---
        long t2 = System.currentTimeMillis();
        BufferedImage testImage = createTestImage();
        PreprocessorConfig ppConfig = PreprocessorConfig.forSmolDocling();
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);

        ImageTiler.SplitImageResult tileResult = ImageTiler.splitImageForVLM(testImage, ppConfig.getTargetHeight());
        List<INDArray> frames = new ArrayList<>();
        for (BufferedImage frame : tileResult.frames) {
            frames.add(preprocessor.preprocess(frame));
        }
        log.info("Preprocessed {} frames from test image", frames.size());

        // Vision encode - run each frame through the vision encoder
        String visionInput = visionEncoder.inputs().get(0);
        String visionOutput = visionEncoder.outputs().get(0);
        List<INDArray> encodedFrames = new ArrayList<>();
        for (INDArray frame : frames) {
            Map<String, INDArray> visionResult = visionEncoder.output(
                    Map.of(visionInput, frame), visionOutput);
            encodedFrames.add(visionResult.get(visionOutput));
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
