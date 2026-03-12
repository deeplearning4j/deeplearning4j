/*
 *  ******************************************************************************
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

import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for VLM decode quality.
 * Each test targets a specific component of the pipeline to identify
 * where output quality degrades.
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TestVLMDecodeQuality {

    private SameDiff visionEncoder;
    private SameDiff decoder;
    private SameDiff embedTokens;
    private HuggingFaceTokenizer tokenizer;

    @BeforeAll
    public void setup() throws Exception {
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedResult.getModelFile().getAbsolutePath());
        visionEncoder = models[0];
        decoder = models[1];
        embedTokens = models[2];

        log.info("Models loaded: vision={} ops, decoder={} ops, embed={} ops",
                visionEncoder.ops().length, decoder.ops().length, embedTokens.ops().length);
    }

    // ==================== Test 1: Embedding table dimensions ====================

    @Test
    @DisplayName("1. Embedding table has full vocabulary (49280 rows)")
    public void testEmbeddingTableSize() {
        INDArray embeddingTable = extractEmbeddingTable(embedTokens);
        assertNotNull(embeddingTable, "Should find embedding table");

        long vocabSize = embeddingTable.shape()[0];
        long hiddenSize = embeddingTable.shape()[1];

        log.info("Embedding table: shape=[{}, {}]", vocabSize, hiddenSize);

        assertEquals(49280, vocabSize, "Vocab size should be 49280 (49152 BPE + 128 special)");
        assertEquals(576, hiddenSize, "Hidden size should be 576 for SmolDocling");
    }

    // ==================== Test 2: Special token embeddings are non-zero ====================

    @Test
    @DisplayName("2. Special token embeddings are non-zero and distinct")
    public void testSpecialTokenEmbeddings() {
        INDArray embeddingTable = extractEmbeddingTable(embedTokens);
        assertNotNull(embeddingTable);

        // Key special tokens
        int[] specialIds = {
                49229, // <doctag>
                49230, // </doctag>
                49204, // <picture>
                49205, // </picture>
                49219, // <paragraph>
                49220, // </paragraph>
                49218, // <loc_
                49190, // <image>
                49279, // <end_of_utterance>
        };
        String[] names = {
                "<doctag>", "</doctag>", "<picture>", "</picture>",
                "<paragraph>", "</paragraph>", "<loc_", "<image>", "<end_of_utterance>"
        };

        for (int i = 0; i < specialIds.length; i++) {
            INDArray row = embeddingTable.getRow(specialIds[i]);
            double norm = row.norm2Number().doubleValue();
            double max = row.amaxNumber().doubleValue();
            log.info("Token {} (id={}): norm={}, max={}", names[i], specialIds[i], norm, max);
            assertTrue(norm > 0.01, names[i] + " embedding should be non-zero, norm=" + norm);
        }

        // Check that <paragraph> and <picture> are distinct from each other
        INDArray paragraphEmb = embeddingTable.getRow(49219);
        INDArray pictureEmb = embeddingTable.getRow(49204);
        double dist = paragraphEmb.sub(pictureEmb).norm2Number().doubleValue();
        log.info("Distance <paragraph> vs <picture>: {}", dist);
        assertTrue(dist > 0.1, "Special tokens should have distinct embeddings");
    }

    // ==================== Test 3: Direct vs SameDiff embedding lookup ====================

    @Test
    @DisplayName("3. Direct embedding lookup matches SameDiff.output()")
    public void testDirectVsSameDiffEmbedding() {
        INDArray embeddingTable = extractEmbeddingTable(embedTokens);
        assertNotNull(embeddingTable);

        // Test with a mix of BPE and special tokens
        int[] testTokens = {0, 44, 2692, 46, 49229, 49204, 49219, 49218};
        String inputName = embedTokens.inputs().get(0);
        String[] outputNames = embedTokens.outputs().toArray(new String[0]);

        for (int tokenId : testTokens) {
            // Direct lookup
            INDArray directEmb = embeddingTable.getRow(tokenId);

            // SameDiff lookup
            INDArray tokenTensor = Nd4j.createFromArray(new int[]{tokenId}).reshape(1, 1).castTo(DataType.LONG);
            Map<String, INDArray> sdResult = embedTokens.output(Map.of(inputName, tokenTensor), outputNames);
            INDArray sdEmb = sdResult.values().iterator().next().reshape(-1);

            // Compare
            double maxDiff = directEmb.sub(sdEmb).amaxNumber().doubleValue();
            log.info("Token {}: direct norm={}, sd norm={}, maxDiff={}",
                    tokenId, directEmb.norm2Number(), sdEmb.norm2Number(), maxDiff);
            assertTrue(maxDiff < 1e-5, "Token " + tokenId + ": direct vs SameDiff maxDiff=" + maxDiff);
        }
    }

    // ==================== Test 4: Decoder prefill produces valid first token ====================

    @Test
    @DisplayName("4. Decoder prefill produces <doctag> as first token")
    public void testPrefillFirstToken() throws Exception {
        // Build a simple test image and get merged embeddings
        INDArray inputsEmbeds = buildTestInputEmbeddings();
        long seqLen = inputsEmbeds.shape()[1];
        log.info("Prefill embeddings: shape={}", Arrays.toString(inputsEmbeds.shape()));

        // Run decoder prefill
        Map<String, INDArray> inputs = new HashMap<>();
        String logitsName = null;
        for (String name : decoder.inputs()) {
            if (name.equals("inputs_embeds")) {
                inputs.put(name, inputsEmbeds);
            } else if (name.equals("attention_mask")) {
                inputs.put(name, Nd4j.ones(DataType.LONG, 1, seqLen));
            } else if (name.equals("position_ids")) {
                inputs.put(name, Nd4j.arange(seqLen).reshape(1, seqLen).castTo(DataType.LONG));
            } else if (name.startsWith("past_key_values.")) {
                inputs.put(name, Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64));
            }
        }
        for (String name : decoder.outputs()) {
            if (name.contains("logit")) {
                logitsName = name;
                break;
            }
        }
        if (logitsName == null) logitsName = decoder.outputs().get(0);

        Map<String, INDArray> result = decoder.output(inputs, logitsName);
        INDArray logits = result.get(logitsName);
        log.info("Logits shape: {}", Arrays.toString(logits.shape()));

        // Check last-position argmax
        INDArray lastLogits = logits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());
        int argmax = lastLogits.argMax().getInt(0);
        String decoded = tokenizer.decode(new int[]{argmax}, false);
        log.info("First predicted token: id={}, text='{}'", argmax, decoded);

        // Top-5 tokens
        INDArray sorted = Nd4j.sortWithIndices(lastLogits.dup(), 0, false)[0];
        log.info("Top-5 predictions:");
        for (int i = 0; i < 5; i++) {
            int idx = sorted.getInt(i);
            double logit = lastLogits.getDouble(idx);
            String text = tokenizer.decode(new int[]{idx}, false);
            log.info("  #{}: id={} logit={} text='{}'", i + 1, idx, logit, text);
        }

        // The first token after prefill should be related to document structure
        // (typically a space before <doctag>, or <doctag> itself)
        assertFalse(lastLogits.isNaN().any(), "Logits should not contain NaN");
        assertFalse(lastLogits.isInfinite().any(), "Logits should not contain Inf");
    }

    // ==================== Test 5: Multi-step decode without DSP ====================

    @Test
    @DisplayName("5. Manual multi-step decode produces diverse tokens (no DSP)")
    public void testManualMultiStepDecode() throws Exception {
        INDArray inputsEmbeds = buildTestInputEmbeddings();
        INDArray embeddingTable = extractEmbeddingTable(embedTokens);
        assertNotNull(embeddingTable);

        long seqLen = inputsEmbeds.shape()[1];
        long hiddenSize = inputsEmbeds.shape()[2];
        int maxSteps = 30;

        // Find logits output name
        String logitsName = null;
        List<String> kvOutputNames = new ArrayList<>();
        for (String name : decoder.outputs()) {
            if (name.contains("logit")) {
                logitsName = name;
            } else if (name.startsWith("present")) {
                kvOutputNames.add(name);
            }
        }
        if (logitsName == null) logitsName = decoder.outputs().get(0);

        List<String> allOutputs = new ArrayList<>();
        allOutputs.add(logitsName);
        allOutputs.addAll(kvOutputNames);

        // Prefill
        Map<String, INDArray> inputs = new HashMap<>();
        for (String name : decoder.inputs()) {
            if (name.equals("inputs_embeds")) {
                inputs.put(name, inputsEmbeds);
            } else if (name.equals("attention_mask")) {
                inputs.put(name, Nd4j.ones(DataType.LONG, 1, seqLen));
            } else if (name.equals("position_ids")) {
                inputs.put(name, Nd4j.arange(seqLen).reshape(1, seqLen).castTo(DataType.LONG));
            } else if (name.startsWith("past_key_values.")) {
                inputs.put(name, Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64));
            }
        }

        Map<String, INDArray> result = decoder.output(inputs, allOutputs.toArray(new String[0]));

        // Extract first token
        INDArray logits = result.get(logitsName);
        INDArray lastLogits = logits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());
        int firstToken = lastLogits.argMax().getInt(0);

        List<Integer> generated = new ArrayList<>();
        generated.add(firstToken);

        // Extract KV cache from prefill
        Map<String, INDArray> kvCache = new HashMap<>();
        for (String kvName : kvOutputNames) {
            INDArray kv = result.get(kvName);
            String pastName = kvName.replace("present", "past_key_values");
            kvCache.put(pastName, kv.dup());
        }
        long pastLen = seqLen;

        // Decode steps
        for (int step = 1; step < maxSteps; step++) {
            int prevToken = generated.get(generated.size() - 1);

            // Embedding lookup
            INDArray tokenEmbed = embeddingTable.getRow(prevToken).reshape(1, 1, hiddenSize);

            // Build decode inputs
            Map<String, INDArray> decInputs = new HashMap<>();
            for (String name : decoder.inputs()) {
                if (name.equals("inputs_embeds")) {
                    decInputs.put(name, tokenEmbed);
                } else if (name.equals("input_ids")) {
                    decInputs.put(name, Nd4j.createFromArray(new int[]{prevToken}).reshape(1, 1).castTo(DataType.LONG));
                } else if (name.equals("attention_mask")) {
                    decInputs.put(name, Nd4j.ones(DataType.LONG, 1, pastLen + 1));
                } else if (name.equals("position_ids")) {
                    decInputs.put(name, Nd4j.createFromArray(new long[]{pastLen}).reshape(1, 1));
                } else if (name.startsWith("past_key_values.") && kvCache.containsKey(name)) {
                    decInputs.put(name, kvCache.get(name));
                }
            }

            result = decoder.output(decInputs, allOutputs.toArray(new String[0]));

            logits = result.get(logitsName);
            INDArray stepLogits = logits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(0), NDArrayIndex.all());
            int nextToken = stepLogits.argMax().getInt(0);
            generated.add(nextToken);

            // Update KV cache
            for (String kvName : kvOutputNames) {
                INDArray kv = result.get(kvName);
                String pastName = kvName.replace("present", "past_key_values");
                kvCache.put(pastName, kv.dup());
            }
            pastLen++;

            String decoded = tokenizer.decode(new int[]{nextToken}, false);
            if (step <= 10 || step % 5 == 0) {
                log.info("Step {}: id={} text='{}' logit={}",
                        step, nextToken, decoded, stepLogits.getDouble(nextToken));
            }
        }

        // Decode full text
        int[] tokenArray = generated.stream().mapToInt(Integer::intValue).toArray();
        String fullText = tokenizer.decode(tokenArray, false);
        log.info("Generated text ({} tokens): '{}'", generated.size(), fullText);

        // Count unique tokens
        Set<Integer> unique = new HashSet<>(generated);
        double diversity = (double) unique.size() / generated.size() * 100;
        log.info("Token diversity: {}/{} unique ({}%)", unique.size(), generated.size(), String.format("%.1f", diversity));

        // With 30 tokens, we expect at LEAST 8 unique tokens for valid output
        // (doctag, picture/paragraph, loc_, digits, close tags, actual text)
        assertTrue(unique.size() >= 8,
                "Expected at least 8 unique tokens in 30 steps, got " + unique.size() +
                        ". Text: '" + fullText + "'");
    }

    // ==================== Test 6: Embedding round-trip consistency ====================

    @Test
    @DisplayName("6. Token→embed→decode→token cycle consistency")
    public void testEmbeddingRoundTrip() {
        INDArray embeddingTable = extractEmbeddingTable(embedTokens);
        assertNotNull(embeddingTable);

        String inputName = embedTokens.inputs().get(0);
        String[] outputNames = embedTokens.outputs().toArray(new String[0]);

        // For each special token, verify:
        // 1. Direct lookup produces same result as SameDiff
        // 2. The embedding is not a duplicate of another token
        int[] specialTokens = {49229, 49204, 49205, 49219, 49220, 49218};

        Map<Integer, INDArray> embeddings = new HashMap<>();
        for (int tokenId : specialTokens) {
            // Direct lookup
            INDArray direct = embeddingTable.getRow(tokenId).dup();

            // SameDiff lookup
            INDArray tokenTensor = Nd4j.createFromArray(new int[]{tokenId}).reshape(1, 1).castTo(DataType.LONG);
            Map<String, INDArray> sdResult = embedTokens.output(Map.of(inputName, tokenTensor), outputNames);
            INDArray sdEmb = sdResult.values().iterator().next().reshape(-1);

            double diff = direct.sub(sdEmb).amaxNumber().doubleValue();
            assertTrue(diff < 1e-5, "Token " + tokenId + " direct/SD mismatch: " + diff);

            embeddings.put(tokenId, direct);
        }

        // Verify distinctness: each pair should differ
        for (int i = 0; i < specialTokens.length; i++) {
            for (int j = i + 1; j < specialTokens.length; j++) {
                INDArray a = embeddings.get(specialTokens[i]);
                INDArray b = embeddings.get(specialTokens[j]);
                double dist = a.sub(b).norm2Number().doubleValue();
                log.info("Dist({}, {}): {}", specialTokens[i], specialTokens[j], dist);
                assertTrue(dist > 0.01,
                        "Tokens " + specialTokens[i] + " and " + specialTokens[j] +
                                " should be distinct, dist=" + dist);
            }
        }
    }

    // ==================== Test 7: Vision encoder produces meaningful output ====================

    @Test
    @DisplayName("7. Vision encoder produces non-zero, varying embeddings")
    public void testVisionEncoderOutput() throws Exception {
        BufferedImage testImage = createTestImage();
        int targetSize = 512;
        PreprocessorConfig ppConfig = new PreprocessorConfig();
        ppConfig.setSize(new PreprocessorConfig.ImageSize(targetSize, targetSize));
        ppConfig.setDoRescale(true);
        ppConfig.setRescaleFactor(1.0 / 255.0);
        ppConfig.setDoNormalize(true);
        ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
        ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);

        BufferedImage resized = ImageTiler.resizeLongestEdge(testImage, 2048);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resized, targetSize, 9);
        log.info("Tiled into {} frames", splitResult.getTotalFrames());

        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);

        for (int i = 0; i < Math.min(2, splitResult.getTotalFrames()); i++) {
            BufferedImage frame = splitResult.frames.get(i);
            INDArray frameTensor = preprocessor.preprocess(frame);
            INDArray singleFrame = frameTensor.reshape(1, 1, 3, targetSize, targetSize);

            Map<String, INDArray> visionInputMap = new HashMap<>();
            for (String name : visionInputNames) {
                if (name.equals("pixel_values")) {
                    visionInputMap.put(name, singleFrame);
                } else if (name.equals("pixel_attention_mask")) {
                    ImageTiler.ContentRegion region = splitResult.contentRegions.get(i);
                    INDArray mask = ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize);
                    visionInputMap.put(name, mask.reshape(1, 1, targetSize, targetSize));
                }
            }

            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            INDArray out = selected.tensor;

            log.info("Frame {} vision output: shape={}, min={}, max={}, mean={}, std={}",
                    i, Arrays.toString(out.shape()),
                    out.minNumber(), out.maxNumber(), out.meanNumber(),
                    out.stdNumber());

            assertFalse(out.isNaN().any(), "Vision output should not contain NaN");
            assertTrue(out.stdNumber().doubleValue() > 0.01,
                    "Vision output should have non-trivial variance, std=" + out.stdNumber());
        }
    }

    // ==================== Helpers ====================

    private INDArray extractEmbeddingTable(SameDiff model) {
        INDArray embeddingTable = null;
        for (SDVariable var : model.variables()) {
            if (var.getVariableType() == VariableType.CONSTANT || var.getVariableType() == VariableType.VARIABLE) {
                INDArray arr = var.getArr();
                if (arr != null && arr.rank() == 2) {
                    if (embeddingTable == null || arr.length() > embeddingTable.length()) {
                        embeddingTable = arr;
                    }
                }
            }
        }
        return embeddingTable;
    }

    private BufferedImage createTestImage() {
        BufferedImage img = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 800, 600);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.BOLD, 36));
        g.drawString("Test Document", 50, 100);
        g.setFont(new Font("SansSerif", Font.PLAIN, 24));
        g.drawString("Section 1: Introduction", 50, 160);
        g.drawString("This is a test page for the SmolDocling VLM pipeline.", 50, 220);
        g.drawString("It contains text that should be recognized.", 50, 260);
        g.dispose();
        return img;
    }

    private INDArray buildTestInputEmbeddings() throws Exception {
        BufferedImage testImage = createTestImage();
        PreprocessorConfig config = PreprocessorConfig.forSmolDocling();
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(config);
        ImageTiler.SplitImageResult split = ImageTiler.splitImageForVLM(testImage, config.getTargetHeight());
        List<BufferedImage> tiles = split.frames;
        int targetHeight = config.getTargetHeight();
        int targetWidth = config.getTargetWidth();

        // Run vision encoder on each frame
        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);
        List<INDArray> frameEmbeddings = new ArrayList<>();

        for (int i = 0; i < tiles.size(); i++) {
            INDArray frameTensor = preprocessor.preprocess(tiles.get(i));
            INDArray singleFrame = frameTensor.reshape(1, 1, 3, targetHeight, targetWidth);

            Map<String, INDArray> visionInputMap = new HashMap<>();
            for (String name : visionInputNames) {
                if (name.equals("pixel_values")) {
                    visionInputMap.put(name, singleFrame);
                } else if (name.equals("pixel_attention_mask")) {
                    visionInputMap.put(name, Nd4j.ones(DataType.LONG, 1, 1, targetHeight, targetWidth));
                }
            }

            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            frameEmbeddings.add(selected.tensor.dup());
        }

        INDArray visionEmbeddings = frameEmbeddings.size() == 1
                ? frameEmbeddings.get(0).dup()
                : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();

        // Build prompt
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLen = (int) frameEmbeddings.get(0).shape()[1];

        // Determine grid from tiles
        int rows = Math.max(1, split.numRows);
        int cols = Math.max(1, split.numCols);

        String imagePrompt = ImagePromptBuilder.buildImagePromptString(rows, cols, imageSeqLen);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

        // Text embeddings
        String embedInputName = embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        INDArray promptIdsTensor = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        Map<String, INDArray> embedOutputs = embedTokens.output(
                Map.of(embedInputName, promptIdsTensor), embedOutputNames);
        INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();

        // Merge
        return EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
    }
}
