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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.StaticKvCacheDecodeLoop;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests whether the vision embedding merge or causal mask construction causes
 * graph replay divergence in the SmolDocling VLM pipeline.
 *
 * The SmolDocling pipeline merges vision encoder outputs with text embeddings
 * before feeding to the decoder. The attention mask also has special handling
 * for image token positions. This test isolates whether the embedding merge or
 * mask construction is the trigger for the graph replay bug.
 *
 * Test strategy:
 *   1. Load all three SmolDocling models (decoder, vision encoder, embed_tokens)
 *   2. Run the full prefill -> decode recompilation -> frozen decode flow
 *   3. Compare graph-capture ON vs OFF (OPTIMAL vs SLOT_BY_SLOT)
 *   4. Also test with RANDOM embeddings instead of vision output -- if random
 *      embeddings pass but vision output fails, the issue is in the vision
 *      output values (NaN, Inf, extreme magnitudes)
 *
 * Run with:
 *   cd platform-tests && mvn test -Dtest=VisionEmbedGraphReplayTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class VisionEmbedGraphReplayTest extends BaseNd4jTestWithBackends {

    // Number of decode tokens to generate for comparison (keep small for test speed)
    private static final int MAX_DECODE_TOKENS = 10;

    // Shared pipeline state loaded once
    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static SameDiff visionEncoder;
    private static Tokenizer tokenizer;

    // Prepared from the vision pipeline
    private static INDArray visionEmbedInputsEmbeds;
    private static INDArray randomEmbedInputsEmbeds;
    private static int[] promptTokenIds;
    private static long hiddenSize;

    @BeforeAll
    public static void setup() throws Exception {
        // Enable graph optimizer for FP16 pre-cast etc.
        String optEnabled = System.getProperty("nd4j.optimizer.enabled");
        if (optEnabled == null || optEnabled.isEmpty()) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        String fp16Prop = System.getProperty("nd4j.optimizer.fp16");
        if (fp16Prop == null || fp16Prop.isEmpty()) {
            System.setProperty("nd4j.optimizer.fp16", "true");
        }

        // ---- Download models ----
        log.info("Downloading SmolDocling models...");
        VLMModelDownloader.DownloadResult visionResult =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        VLMModelDownloader.DownloadResult decoderResult =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        VLMModelDownloader.DownloadResult embedResult =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        VLMModelDownloader.DownloadResult tokenizerResult =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        log.info("Download complete.");

        // ---- Load tokenizer ----
        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        assertNotNull(tokenizer, "Tokenizer failed to load");
        log.info("Tokenizer loaded: vocab_size={}", tokenizer.getVocabSize());

        // ---- Import ONNX models ----
        log.info("Importing ONNX models...");
        long importStart = System.currentTimeMillis();
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedResult.getModelFile().getAbsolutePath()
        );
        visionEncoder = models[0];
        decoder = models[1];
        embedTokens = models[2];
        log.info("ONNX import done [{}ms]: vision={} ops, decoder={} ops, embed={} ops",
                System.currentTimeMillis() - importStart,
                visionEncoder.getOps().size(), decoder.getOps().size(), embedTokens.getOps().size());

        // ---- Generate test image ----
        int targetSize = 512;
        BufferedImage testImage = createTestImage();

        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(testImage, 2048);
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resizedForTiling, targetSize, 9);
        int visionFrames = splitResult.getTotalFrames();
        log.info("Test image: {}x{}, {} frames", testImage.getWidth(), testImage.getHeight(), visionFrames);

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

        // ---- Run vision encoder ----
        log.info("Running vision encoder on {} frames...", visionFrames);
        long visionStart = System.currentTimeMillis();
        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);

        List<INDArray> frameEmbeddings = new ArrayList<>();
        for (int frameIdx = 0; frameIdx < visionFrames; frameIdx++) {
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
            INDArray out = selected.tensor.dup();
            frameEmbeddings.add(out);

            for (var entry : visionOutputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
            }
            singleFrame.close();
        }

        // Clean up vision encoder
        visionEncoder.clearPlaceholders(false);
        visionEncoder.clearOpInputs();
        visionEncoder.resetSession();
        Nd4j.getExecutioner().commit();

        INDArray visionEmbeddings = frameEmbeddings.size() == 1
                ? frameEmbeddings.get(0).dup()
                : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();
        for (INDArray fe : frameEmbeddings) {
            if (fe != null && fe.closeable() && !fe.wasClosed()) fe.close();
        }
        imageInput.close();
        log.info("Vision encoder done [{}ms]: shape={}", System.currentTimeMillis() - visionStart,
                Arrays.toString(visionEmbeddings.shape()));

        // ---- Build prompt + embeddings ----
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / visionFrames;

        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt +
                "Convert this page to docling.<end_of_utterance>\nAssistant:";
        promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
        log.info("Prompt: {} tokens", promptTokenIds.length);

        INDArray promptIdsTensor = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        String embedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        Map<String, INDArray> embedOutputs = embedTokens.output(
                Map.of(embedInputName, promptIdsTensor), embedOutputNames);
        INDArray textEmbeddings = null;
        for (var entry : embedOutputs.entrySet()) {
            textEmbeddings = entry.getValue().dup();
        }
        assertNotNull(textEmbeddings, "embed_tokens produced no output");

        hiddenSize = visionEmbeddings.shape()[2];
        assertEquals(hiddenSize, textEmbeddings.shape()[2],
                "Hidden size mismatch: vision=" + hiddenSize + " text=" + textEmbeddings.shape()[2]);
        log.info("Hidden size: {}", hiddenSize);

        // ---- Merge REAL vision embeddings with text ----
        visionEmbedInputsEmbeds = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
        assertNotNull(visionEmbedInputsEmbeds, "Merged vision embeddings are null");
        log.info("Vision-merged embeddings: shape={}", Arrays.toString(visionEmbedInputsEmbeds.shape()));

        // Diagnose vision embedding values
        logEmbeddingStats("vision-merged", visionEmbedInputsEmbeds);

        // ---- Build RANDOM embeddings with same shape ----
        // Random normal embeddings in the same range as the real ones
        long seqLen = visionEmbedInputsEmbeds.shape()[1];
        INDArray randomVisionEmbeddings = Nd4j.randn(DataType.FLOAT, 1, visionEmbeddings.shape()[1], hiddenSize)
                .muli(0.1);  // Scale to reasonable range
        INDArray randomMerged = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, randomVisionEmbeddings, promptTokenIds, imageTokenId);
        randomEmbedInputsEmbeds = randomMerged;
        log.info("Random-merged embeddings: shape={}", Arrays.toString(randomEmbedInputsEmbeds.shape()));

        logEmbeddingStats("random-merged", randomEmbedInputsEmbeds);

        // Cleanup
        if (textEmbeddings.closeable() && !textEmbeddings.wasClosed()) textEmbeddings.close();
        randomVisionEmbeddings.close();

        // Free vision encoder constants to reclaim GPU memory
        freeModelConstants(visionEncoder, "vision encoder");

        log.info("Setup complete: visionEmbeds={}, randomEmbeds={}, promptTokens={}",
                Arrays.toString(visionEmbedInputsEmbeds.shape()),
                Arrays.toString(randomEmbedInputsEmbeds.shape()),
                promptTokenIds.length);
    }

    // ---- Test 1: SLOT_BY_SLOT baseline with vision embeddings ----
    // This establishes the correct output for comparison.

    @Test
    @DisplayName("SLOT_BY_SLOT baseline with vision embeddings")
    public void testSlotBySlotVisionEmbeddings() throws Exception {
        GenerationResult result = runDecode("SLOT_BY_SLOT_VISION", visionEmbedInputsEmbeds,
                GraphExecutionMode.SLOT_BY_SLOT, false);
        assertNotNull(result, "SLOT_BY_SLOT vision result is null");
        assertTrue(result.getGeneratedTokenCount() > 0,
                "SLOT_BY_SLOT vision should generate at least 1 token, got 0");
        log.info("SLOT_BY_SLOT vision: {} tokens, text='{}', finish={}",
                result.getGeneratedTokenCount(),
                result.getText().substring(0, Math.min(80, result.getText().length())),
                result.getFinishReason());
    }

    // ---- Test 2: SLOT_BY_SLOT baseline with random embeddings ----

    @Test
    @DisplayName("SLOT_BY_SLOT baseline with random embeddings")
    public void testSlotBySlotRandomEmbeddings() throws Exception {
        GenerationResult result = runDecode("SLOT_BY_SLOT_RANDOM", randomEmbedInputsEmbeds,
                GraphExecutionMode.SLOT_BY_SLOT, false);
        assertNotNull(result, "SLOT_BY_SLOT random result is null");
        assertTrue(result.getGeneratedTokenCount() > 0,
                "SLOT_BY_SLOT random should generate at least 1 token, got 0");
        log.info("SLOT_BY_SLOT random: {} tokens, text='{}', finish={}",
                result.getGeneratedTokenCount(),
                result.getText().substring(0, Math.min(80, result.getText().length())),
                result.getFinishReason());
    }

    // ---- Test 3: Graph capture with vision embeddings vs SLOT_BY_SLOT ----
    // This is the key divergence test. If graph replay produces different output
    // from SLOT_BY_SLOT with the SAME embeddings, the bug is in graph replay.

    @Test
    @DisplayName("Graph capture vs SLOT_BY_SLOT with vision embeddings")
    public void testGraphCaptureVsSlotBySlotVisionEmbeddings() throws Exception {
        boolean tritonAvailable = Nd4j.getNativeOps().isTritonAvailable();
        if (!tritonAvailable) {
            log.info("Triton not available, skipping graph capture test");
            return;
        }

        // Run SLOT_BY_SLOT first (ground truth)
        GenerationResult slotResult = runDecode("SBS_VISION_COMPARE", visionEmbedInputsEmbeds,
                GraphExecutionMode.SLOT_BY_SLOT, false);
        assertNotNull(slotResult, "SLOT_BY_SLOT vision result is null");

        // Run OPTIMAL (graph capture ON)
        GenerationResult gcResult = runDecode("GC_VISION_COMPARE", visionEmbedInputsEmbeds,
                null, true);  // null executionMode = Triton/OPTIMAL
        assertNotNull(gcResult, "Graph-capture vision result is null");

        // Compare outputs
        compareResults("Vision embeddings", slotResult, gcResult);
    }

    // ---- Test 4: Graph capture with random embeddings vs SLOT_BY_SLOT ----
    // If random embeddings pass but vision fails, the issue is in the vision
    // output values (NaN, Inf, extreme magnitudes).

    @Test
    @DisplayName("Graph capture vs SLOT_BY_SLOT with random embeddings")
    public void testGraphCaptureVsSlotBySlotRandomEmbeddings() throws Exception {
        boolean tritonAvailable = Nd4j.getNativeOps().isTritonAvailable();
        if (!tritonAvailable) {
            log.info("Triton not available, skipping graph capture test");
            return;
        }

        // Run SLOT_BY_SLOT first (ground truth)
        GenerationResult slotResult = runDecode("SBS_RANDOM_COMPARE", randomEmbedInputsEmbeds,
                GraphExecutionMode.SLOT_BY_SLOT, false);
        assertNotNull(slotResult, "SLOT_BY_SLOT random result is null");

        // Run OPTIMAL (graph capture ON)
        GenerationResult gcResult = runDecode("GC_RANDOM_COMPARE", randomEmbedInputsEmbeds,
                null, true);  // null executionMode = Triton/OPTIMAL
        assertNotNull(gcResult, "Graph-capture random result is null");

        // Compare outputs
        compareResults("Random embeddings", slotResult, gcResult);
    }

    // ---- Test 5: Embedding value diagnostics ----
    // Check for NaN, Inf, extreme magnitudes in vision vs random embeddings.
    // This runs without decode -- just validates the embeddings themselves.

    @Test
    @DisplayName("Vision embedding value diagnostics (NaN, Inf, extremes)")
    public void testEmbeddingValueDiagnostics() {
        // Vision embeddings
        INDArray vFlat = visionEmbedInputsEmbeds.reshape(visionEmbedInputsEmbeds.length());
        float[] visionData = vFlat.dup().data().asFloat();

        int vNanCount = 0, vInfCount = 0;
        float vMin = Float.MAX_VALUE, vMax = -Float.MAX_VALUE;
        double vSum = 0, vSumSq = 0;
        for (float v : visionData) {
            if (Float.isNaN(v)) vNanCount++;
            else if (Float.isInfinite(v)) vInfCount++;
            else {
                vMin = Math.min(vMin, v);
                vMax = Math.max(vMax, v);
                vSum += v;
                vSumSq += (double) v * v;
            }
        }
        double vMean = vSum / visionData.length;
        double vStd = Math.sqrt(vSumSq / visionData.length - vMean * vMean);

        log.info("Vision embeddings: len={}, NaN={}, Inf={}, min={}, max={}, mean={}, std={}",
                visionData.length, vNanCount, vInfCount, vMin, vMax, vMean, vStd);

        // Random embeddings
        INDArray rFlat = randomEmbedInputsEmbeds.reshape(randomEmbedInputsEmbeds.length());
        float[] randomData = rFlat.dup().data().asFloat();

        int rNanCount = 0, rInfCount = 0;
        float rMin = Float.MAX_VALUE, rMax = -Float.MAX_VALUE;
        double rSum = 0, rSumSq = 0;
        for (float v : randomData) {
            if (Float.isNaN(v)) rNanCount++;
            else if (Float.isInfinite(v)) rInfCount++;
            else {
                rMin = Math.min(rMin, v);
                rMax = Math.max(rMax, v);
                rSum += v;
                rSumSq += (double) v * v;
            }
        }
        double rMean = rSum / randomData.length;
        double rStd = Math.sqrt(rSumSq / randomData.length - rMean * rMean);

        log.info("Random embeddings: len={}, NaN={}, Inf={}, min={}, max={}, mean={}, std={}",
                randomData.length, rNanCount, rInfCount, rMin, rMax, rMean, rStd);

        // Assert no NaN/Inf in vision embeddings
        assertEquals(0, vNanCount, "Vision embeddings contain NaN values");
        assertEquals(0, vInfCount, "Vision embeddings contain Inf values");

        // Assert no NaN/Inf in random embeddings
        assertEquals(0, rNanCount, "Random embeddings contain NaN values");
        assertEquals(0, rInfCount, "Random embeddings contain Inf values");

        // Check for extreme magnitudes in vision embeddings
        // If vision embeddings have much larger magnitudes than random, that could
        // trigger different code paths or numerical issues in graph replay
        log.info("Magnitude comparison: vision [min={}, max={}] vs random [min={}, max={}]",
                vMin, vMax, rMin, rMax);
        if (Math.abs(vMax) > 1000 || Math.abs(vMin) > 1000) {
            log.warn("WARNING: Vision embeddings have extreme magnitudes (>1000). " +
                    "This may cause numerical issues in graph replay.");
        }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /**
     * Run a decode with the specified config, returning the generation result.
     * Handles model reset, config application, compilation, and decode.
     */
    private GenerationResult runDecode(String label, INDArray inputsEmbeds,
                                       GraphExecutionMode executionMode,
                                       boolean useOptimal) {
        log.info("---- {} ----", label);

        // Reset state
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        Nd4j.getExecutioner().commit();

        // Build config
        BenchmarkConfig config;
        if (useOptimal) {
            config = BenchmarkConfig.optimal().maxTokens(MAX_DECODE_TOKENS).minDiversityPct(0);
        } else {
            config = BenchmarkConfig.create(label)
                    .executionMode(executionMode != null ? executionMode : GraphExecutionMode.SLOT_BY_SLOT)
                    .maxTokens(MAX_DECODE_TOKENS)
                    .minDiversityPct(0);
        }

        // Apply config + compile
        BenchmarkConfigApplier.apply(config);
        BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);

        // Build decode loop
        ModelIOConfig decoderIOConfig = ModelIOConfig.discover(decoder);
        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(decoderIOConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(config.getMaxTokens())
                .hiddenSize(hiddenSize)
                .build();

        GenerationResult result = loop.decode(inputsEmbeds, promptTokenIds);
        log.info("{}: {} tokens in {}ms, {} tok/s, finish={}, text='{}'",
                label, result.getGeneratedTokenCount(), result.getGenerationTimeMs(),
                String.format("%.1f", result.getTokensPerSecond()), result.getFinishReason(),
                result.getText().substring(0, Math.min(80, result.getText().length())));

        return result;
    }

    /**
     * Compare results from two decode runs, logging detailed differences.
     * Checks both token sequence and text output.
     */
    private void compareResults(String label, GenerationResult slotResult, GenerationResult gcResult) {
        log.info("=== Comparing {} ===", label);
        log.info("  SLOT_BY_SLOT: {} tokens, text='{}'",
                slotResult.getGeneratedTokenCount(),
                slotResult.getText().substring(0, Math.min(100, slotResult.getText().length())));
        log.info("  GRAPH_CAPTURE: {} tokens, text='{}'",
                gcResult.getGeneratedTokenCount(),
                gcResult.getText().substring(0, Math.min(100, gcResult.getText().length())));

        // Compare token IDs
        int[] slotTokens = slotResult.getTokenIds();
        int[] gcTokens = gcResult.getTokenIds();
        int minLen = Math.min(slotTokens.length, gcTokens.length);

        int firstDivergence = -1;
        for (int i = 0; i < minLen; i++) {
            if (slotTokens[i] != gcTokens[i]) {
                firstDivergence = i;
                break;
            }
        }
        if (firstDivergence < 0 && slotTokens.length != gcTokens.length) {
            firstDivergence = minLen;
        }

        if (firstDivergence >= 0) {
            log.warn("TOKEN DIVERGENCE at step {}: SLOT_BY_SLOT={} vs GRAPH_CAPTURE={}",
                    firstDivergence,
                    firstDivergence < slotTokens.length ? slotTokens[firstDivergence] : "<end>",
                    firstDivergence < gcTokens.length ? gcTokens[firstDivergence] : "<end>");
            log.warn("  SLOT_BY_SLOT tokens: {}", Arrays.toString(slotTokens));
            log.warn("  GRAPH_CAPTURE tokens: {}", Arrays.toString(gcTokens));

            // Decode individual tokens for human readability
            for (int i = 0; i < minLen; i++) {
                String marker = (slotTokens[i] != gcTokens[i]) ? " <<< DIVERGED" : "";
                log.info("  step {}: SLOT={} ({}) vs GC={} ({}){}",
                        i,
                        slotTokens[i], safeDecodeToken(slotTokens[i]),
                        gcTokens[i], safeDecodeToken(gcTokens[i]),
                        marker);
            }
        } else {
            log.info("All {} tokens MATCH between SLOT_BY_SLOT and GRAPH_CAPTURE", minLen);
        }

        // Compare text
        boolean textMatch = slotResult.getText().equals(gcResult.getText());
        log.info("Text match: {}", textMatch);

        // Log the result -- do NOT assert token-level match because the known bug
        // is that graph replay diverges. The purpose of this test is to CHARACTERIZE
        // the divergence, not to require it doesn't exist.
        // We DO log enough detail to diagnose where and why divergence starts.
        if (firstDivergence >= 0) {
            log.warn("DIVERGENCE DETECTED in [{}]: first mismatch at token step {}",
                    label, firstDivergence);
            log.warn("  This means graph replay produces DIFFERENT output from slot-by-slot " +
                    "execution with {} embeddings.", label.toLowerCase());

            // This is the key diagnostic: if BOTH vision and random diverge at the same
            // step, the issue is in graph replay infrastructure (not embedding values).
            // If only vision diverges, the issue is in how vision embedding values interact
            // with the graph replay path.
        }
    }

    /**
     * Safely decode a single token ID to its string representation.
     */
    private String safeDecodeToken(int tokenId) {
        try {
            return tokenizer.decode(new int[]{tokenId});
        } catch (Exception e) {
            return "?";
        }
    }

    /**
     * Log statistics for an embedding tensor.
     */
    private static void logEmbeddingStats(String label, INDArray embeddings) {
        // Use dup() + bulk data transfer for CUDA safety
        float[] data = embeddings.dup().data().asFloat();
        int nanCount = 0, infCount = 0;
        float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
        for (float v : data) {
            if (Float.isNaN(v)) nanCount++;
            else if (Float.isInfinite(v)) infCount++;
            else {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        }
        log.info("{}: shape={}, NaN={}, Inf={}, min={}, max={}",
                label, Arrays.toString(embeddings.shape()), nanCount, infCount, min, max);
    }

    /**
     * Generate a simple test image for vision encoding.
     */
    private static BufferedImage createTestImage() {
        BufferedImage img = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 512, 512);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 24));
        g.drawString("Test Document", 50, 100);
        g.drawString("Section 1: Introduction", 50, 160);
        g.drawString("This is a test page for the", 50, 220);
        g.drawString("SmolDocling VLM pipeline.", 50, 260);
        g.drawString("Section 2: Content", 50, 340);
        g.drawString("Lorem ipsum dolor sit amet,", 50, 400);
        g.drawString("consectetur adipiscing elit.", 50, 440);
        g.dispose();
        return img;
    }

    /**
     * Free model constants to reclaim GPU memory.
     */
    private static void freeModelConstants(SameDiff model, String label) {
        int closedArrays = 0;
        long closedBytes = 0;
        for (org.nd4j.autodiff.samediff.ArrayHolder holder :
                new org.nd4j.autodiff.samediff.ArrayHolder[]{model.getConstantArrays(), model.getVariablesArrays()}) {
            for (String name : new ArrayList<>(holder.arrayNames())) {
                INDArray arr = holder.removeArray(name);
                if (arr != null && !arr.wasClosed()) {
                    closedBytes += arr.length() * arr.dataType().width();
                    arr.data().setConstant(false);
                    arr.close();
                    closedArrays++;
                }
            }
        }
        Nd4j.getExecutioner().commit();
        log.info("Freed {} {} arrays ({}MB)", closedArrays, label, closedBytes / (1024 * 1024));
    }
}
