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
import org.eclipse.deeplearning4j.model.benchmark.*;
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
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.deallocation.DeallocatableReference;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DSP Validation Test Framework.
 *
 * Runs the SmolDocling decoder under different DSP execution modes and asserts that each
 * reproduces the SLOT_BY_SLOT reference (generated tokens, or decoder outputs for a single
 * forward pass), that plan state seen through DspHandle and DspPlanAssertions holds its
 * invariants, and that repeated decodes do not keep memory.
 *
 * Run with:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestDspValidation \
 *     -Dbackend.artifactId=nd4j-cuda-13.1
 *
 * System properties:
 *   -Dvlm.validation.tokens=N       Override max decode tokens (default: 5 for accuracy, 10 for decode)
 *   -Dvlm.validation.configs=LIST   Comma-separated configs for outputAccuracy (SLOT_BY_SLOT,TRITON_NO_GC,OPTIMAL)
 *   -Dvlm.validation.tolerance=NAME Tolerance preset: standard, strict, tf32 (default: standard)
 *   -Dvlm.validation.matchRate=N    Minimum token match rate percent (default: 90)
 *   -Dvlm.validation.verbose=true   Enable verbose per-step logging
 */
@Slf4j
public class TestDspValidation {

    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static Tokenizer tokenizer;
    private static INDArray inputsEmbeds;
    private static int[] promptTokenIds;
    private static long hiddenSize;
    private static boolean modelsLoaded = false;

    // Configurable properties
    private static int configuredTokens = -1;        // -1 = use per-test defaults
    private static double configuredMatchRate = 90.0; // percent
    private static boolean verbose = false;
    private static String tolerancePreset = "standard";
    private static String configFilter = null;        // null = all configs

    @BeforeAll
    public static void setup() {
        String optEnabled = System.getProperty("nd4j.optimizer.enabled");
        if (optEnabled == null || optEnabled.isEmpty()) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        String fp16Prop = System.getProperty("nd4j.optimizer.fp16");
        if (fp16Prop == null || fp16Prop.isEmpty()) {
            System.setProperty("nd4j.optimizer.fp16", "true");
        }

        // Read validation properties
        String tokensProp = System.getProperty("vlm.validation.tokens");
        if (tokensProp != null && !tokensProp.isEmpty()) {
            configuredTokens = Integer.parseInt(tokensProp);
        }
        String matchRateProp = System.getProperty("vlm.validation.matchRate");
        if (matchRateProp != null && !matchRateProp.isEmpty()) {
            configuredMatchRate = Double.parseDouble(matchRateProp);
        }
        verbose = "true".equalsIgnoreCase(System.getProperty("vlm.validation.verbose"));
        String tolProp = System.getProperty("vlm.validation.tolerance");
        if (tolProp != null && !tolProp.isEmpty()) {
            tolerancePreset = tolProp;
        }
        configFilter = System.getProperty("vlm.validation.configs");

        // Debug+verbose only when explicitly requested via system property
        if (Boolean.getBoolean("nd4j.debug")) {
            Nd4j.getEnvironment().setDebug(true);
            Nd4j.getEnvironment().setVerbose(true);
        }

        // NaN panic disabled — was used for one-time diagnosis, not needed in steady state
    }

    private static int getTokens(int defaultTokens) {
        return configuredTokens > 0 ? configuredTokens : defaultTokens;
    }

    /**
     * Fraction of positions at which two decodes produced the same token, over the longer
     * decode: a decode that stops early or runs longer than the other counts as diverging.
     */
    private static double tokenMatchRate(int[] test, int[] ref) {
        int maxLen = Math.max(test.length, ref.length);
        if (maxLen == 0) return 1.0;
        int matches = 0;
        for (int i = 0; i < Math.min(test.length, ref.length); i++) {
            if (test[i] == ref[i]) matches++;
        }
        return (double) matches / maxLen;
    }

    /** The configured token match rate, as a fraction. */
    private static double requiredMatchRate() {
        return configuredMatchRate / 100.0;
    }

    /**
     * The rate a config must reach against a decode without TF32: the configured rate, or at
     * most 15% when the config enables TF32, whose rounding can diverge an autoregressive decode
     * within a few steps (the same policy as testOutputAccuracy).
     */
    private static double requiredMatchRateAgainstFp32(BenchmarkConfig config) {
        boolean tf32 = config.isCublasTf32() || config.isTritonTf32();
        return tf32 ? Math.min(requiredMatchRate(), 0.15) : requiredMatchRate();
    }

    private static String matchRateMessage(String comparison, double rate, double required) {
        return String.format("%s: token match rate %.1f%% (required %.1f%%)",
                comparison, rate * 100, required * 100);
    }

    private static ValidationConfig getValidationConfig() {
        switch (tolerancePreset.toLowerCase()) {
            case "strict": return ValidationConfig.strict();
            case "tf32":   return ValidationConfig.tf32Tolerant();
            default:       return ValidationConfig.standard();
        }
    }

    private static synchronized void ensureModelsLoaded() throws Exception {
        if (modelsLoaded) return;

        log.info("Loading SmolDocling models for DSP validation...");

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

        // Generate a simple test image
        int targetSize = 512;
        BufferedImage testImage = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g = testImage.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, targetSize, targetSize);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 24));
        g.drawString("Test Document", 50, 100);
        g.drawString("Line 2: DSP Validation", 50, 150);
        g.dispose();

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(testImage, targetSize, 9);

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

        // Run vision encoder
        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);

        List<INDArray> frameEmbeddings = new ArrayList<>();
        for (int frameIdx = 0; frameIdx < splitResult.getTotalFrames(); frameIdx++) {
            INDArray frameSlice = imageInput.get(
                    NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            // SmolDocling's vision encoder contract is rank-4 NCHW per frame.
            // A rank-5 [1, 1, C, H, W] test-only input changes ONNX reshape
            // semantics and can enqueue an invalid asynchronous CUDA access.
            INDArray singleFrame = frameSlice.reshape(1, 3, targetSize, targetSize).dup();

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
            // The duplicate is submitted asynchronously. Match VisionEncoder's
            // production lifetime boundary before closing the source outputs.
            Nd4j.getExecutioner().commit();
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

        // Match the real SmolDocling benchmark prompt path: expand the image prompt
        // so the merged embedding sequence actually contains the vision tokens.
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.shape()[1] / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] encoded = tokenizer.encode(chatPrompt, false).getIds();
        promptTokenIds = encoded;

        INDArray tokenIds = Nd4j.createFromArray(new int[][]{encoded}).castTo(DataType.INT64);
        Map<String, INDArray> embedInputs = new HashMap<>();
        for (String inputName : embedTokens.inputs()) {
            embedInputs.put(inputName, tokenIds);
        }
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs,
                embedTokens.outputs().toArray(new String[0]));
        INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();
        tokenIds.close();

        long expectedImageSlots = visionEmbeddings.shape()[1];
        long actualImageSlots = ImagePromptBuilder.countOccurrences(encoded, imageTokenId);
        assertEquals(expectedImageSlots, actualImageSlots,
                "Benchmark-style prompt must expose one <image> slot per vision token. "
                        + "visionSeqLen=" + expectedImageSlots + " imageSlots=" + actualImageSlots);

        inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, encoded, imageTokenId);
        assertEquals(promptTokenIds.length, inputsEmbeds.size(1),
                "Merged embedding sequence length must match prompt token length");

        log.info("Models loaded: decoder={} ops, embed={} ops, hiddenSize={}, promptTokens={}, imageSlots={}",
                decoder.getOps().size(), embedTokens.getOps().size(), hiddenSize,
                promptTokenIds.length, actualImageSlots);
        modelsLoaded = true;
    }

    // ─── Test configs ──────────────────────────────────────────────────────

    static Stream<BenchmarkConfig> outputAccuracyConfigs() {
        int tokens = getTokens(5);
        List<BenchmarkConfig> allConfigs = new ArrayList<>();
        allConfigs.add(BenchmarkConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(tokens));

        if (Nd4j.getNativeOps().isTritonAvailable()) {
            allConfigs.add(BenchmarkConfig.create("TRITON_NO_GC")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .maxTokens(tokens));

            allConfigs.add(BenchmarkConfig.create("BISECT_argTable")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .maxTokens(tokens));

            allConfigs.add(BenchmarkConfig.create("BISECT_batchedGemm")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .dspBatchedGemm(true)
                    .maxTokens(tokens));

            allConfigs.add(BenchmarkConfig.create("BISECT_tf32")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .cublasTf32(true).tritonTf32(true)
                    .maxTokens(tokens));

            allConfigs.add(BenchmarkConfig.create("BISECT_graphCapture_only")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(false)
                    .maxTokens(tokens));

            allConfigs.add(BenchmarkConfig.create("BISECT_graphCapture_allSettings")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonGraphCapture(true).tritonAllowFallbackCapture(false)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true).tritonTf32(true)
                    .dspBatchedGemm(true)
                    .maxTokens(tokens));

            allConfigs.add(BenchmarkConfig.create("BISECT_noGC_allSettings")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                    .tritonSectionFusion(true).tritonCompileAll(true)
                    .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                    .tritonFusionScoring(false)
                    .tritonNumWarps(4).tritonNumStages(1)
                    .cublasTf32(true).tritonTf32(true)
                    .dspBatchedGemm(true)
                    .maxTokens(tokens));

            allConfigs.add(BenchmarkConfig.optimal().maxTokens(tokens));
        }

        // Filter by vlm.validation.configs if specified
        if (configFilter != null && !configFilter.isEmpty()) {
            Set<String> allowed = new LinkedHashSet<>();
            for (String s : configFilter.split(",")) {
                allowed.add(s.trim().toUpperCase());
            }
            allConfigs.removeIf(c -> !allowed.contains(c.getName().toUpperCase()));
            log.info("Filtered to {} configs via vlm.validation.configs={}", allConfigs.size(), configFilter);
        }

        return allConfigs.stream();
    }

    // ─── Test: Output accuracy across configs ──────────────────────────────

    @ParameterizedTest(name = "outputAccuracy[{0}]")
    @MethodSource("outputAccuracyConfigs")
    @DisplayName("DSP output accuracy vs SLOT_BY_SLOT baseline")
    public void testOutputAccuracy(BenchmarkConfig config) throws Exception {
        ensureModelsLoaded();
        log.info("Testing output accuracy for config: {}", config.getName());

        int maxTokens = config.getMaxTokens();

        // Reference: SLOT_BY_SLOT decode
        GenerationResult refResult = runDecode(
                BenchmarkConfig.create("REF_SLOT_BY_SLOT")
                        .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                        .maxTokens(maxTokens),
                maxTokens);

        // Test: config under test
        GenerationResult testResult = runDecode(config, maxTokens);

        // Compare generated tokens
        int[] refTokens = refResult.getTokenIds();
        int[] testTokens = testResult.getTokenIds();
        int minLen = Math.min(refTokens.length, testTokens.length);

        int matches = 0;
        int firstDivergent = -1;
        for (int i = 0; i < minLen; i++) {
            if (refTokens[i] == testTokens[i]) {
                matches++;
            } else if (firstDivergent < 0) {
                firstDivergent = i;
            }
        }

        double matchRate = minLen > 0 ? (double) matches / minLen : 1.0;
        log.info("[{}] Token match rate: {}/{} ({}%)", config.getName(),
                matches, minLen, String.format("%.1f", matchRate * 100));
        log.info("[{}] Reference text: {}", config.getName(), refResult.getText());
        log.info("[{}] Test text:      {}", config.getName(), testResult.getText());
        if (firstDivergent >= 0) {
            log.info("[{}] First divergent token at step {}: ref={} test={}",
                    config.getName(), firstDivergent,
                    refTokens[firstDivergent], testTokens[firstDivergent]);
        }
        if (verbose) {
            for (int i = 0; i < minLen; i++) {
                String match = refTokens[i] == testTokens[i] ? "OK" : "DIVERGE";
                log.info("[{}] Step {}: ref={} test={} [{}]", config.getName(),
                        i, refTokens[i], testTokens[i], match);
            }
        }

        double requiredRate;
        if (config.getExecutionMode() == GraphExecutionMode.SLOT_BY_SLOT) {
            requiredRate = 1.0;
        } else if (config.isCublasTf32() || config.isTritonTf32()) {
            // TF32 reduces mantissa from 23 to 10 bits. For autoregressive decode,
            // small numerical differences accumulate across steps, causing token
            // divergence within 2-3 steps. Token-level match is not meaningful —
            // verify that the model produces non-degenerate output instead.
            requiredRate = Math.min(configuredMatchRate / 100.0, 0.15);
        } else {
            requiredRate = configuredMatchRate / 100.0;
        }
        assertTrue(matchRate >= requiredRate,
                config.getName() + ": token match rate too low: "
                        + String.format("%.1f%% (required %.1f%%)",
                        matchRate * 100, requiredRate * 100));

        // Degenerate output check: if ALL reference tokens are the same (e.g., all token-0),
        // the model is broken — matching rate is meaningless when both paths produce garbage.
        if (minLen >= 3) {
            boolean allSame = true;
            for (int i = 1; i < refTokens.length; i++) {
                if (refTokens[i] != refTokens[0]) {
                    allSame = false;
                    break;
                }
            }
            assertFalse(allSame,
                    config.getName() + ": DEGENERATE OUTPUT — all " + refTokens.length
                            + " reference tokens are the same (token " + refTokens[0]
                            + "). The model is producing garbage.");
        }

        // Degenerate content check: model should produce content tags, not just
        // picture structure. Premature EOS after <picture>...</picture></doctag> is
        // a sign of broken vision encoder or attention computation.
        String refTextStr = refResult.getText();
        if (refTextStr != null && refTokens.length > 5) {
            boolean hasPictureOnly = refTextStr.contains("<picture>") && refTextStr.contains("</doctag>")
                    && !refTextStr.contains("<text>") && !refTextStr.contains("<section_header>")
                    && !refTextStr.contains("<otsl>") && !refTextStr.contains("<table>");
            if (hasPictureOnly && refTokens.length < 30) {
                log.warn("{}: DEGENERATE CONTENT — only {} tokens with picture-only output: {}",
                        config.getName(), refTokens.length, refTextStr);
            }
        }
    }

    // ─── Test: Single forward pass comparison ─────────────────────────────

    @Test
    @DisplayName("Single forward pass: SLOT_BY_SLOT vs Triton on decoder")
    public void testSingleForwardPassComparison() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available — skipping");
            return;
        }
        ensureModelsLoaded();

        Map<String, INDArray> placeholders = buildDecoderStep0Inputs();
        List<String> outputs = new ArrayList<>(decoder.outputs());
        log.info("Single forward pass comparison: {} placeholders, {} outputs",
                placeholders.size(), outputs.size());

        // Apply Triton environment settings (include types, fusion, etc.)
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfig tritonConfig = BenchmarkConfig.create("COMPARE_TRITON")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true);
        BenchmarkConfigApplier.apply(tritonConfig);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        // Compare standard (non-DSP) vs SLOT_BY_SLOT DSP (informational — not asserted)
        log.info("=== Compare: standard vs SLOT_BY_SLOT ===");
        Map<String, Double> stdVsSlot = decoder.compareExecutionPaths(
                placeholders, outputs, 1e-3, GraphExecutionMode.SLOT_BY_SLOT);
        log.info("standard vs SLOT_BY_SLOT: {} divergent outputs of {}", stdVsSlot.size(), outputs.size());
        for (Map.Entry<String, Double> e : stdVsSlot.entrySet()) {
            log.info("  std vs slot divergence: {} maxDiff={}", e.getKey(), e.getValue());
        }

        // Rebuild placeholders — compareExecutionPaths may consume originals
        Map<String, INDArray> freshPlaceholders = buildDecoderStep0Inputs();

        // Compare SLOT_BY_SLOT vs TRITON (Triton-compiled) — the key comparison
        log.info("=== Compare: SLOT_BY_SLOT vs TRITON ===");
        Map<String, Double> slotVsTriton = decoder.compareDspModes(
                freshPlaceholders, outputs, 1e-3,
                GraphExecutionMode.SLOT_BY_SLOT,
                GraphExecutionMode.TRITON);
        log.info("SLOT_BY_SLOT vs TRITON: {} divergent outputs of {}", slotVsTriton.size(), outputs.size());
        for (Map.Entry<String, Double> e : slotVsTriton.entrySet()) {
            log.info("  slot vs triton divergence: {} maxDiff={}", e.getKey(), e.getValue());
        }

        // Cleanup
        for (INDArray arr : placeholders.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
        for (INDArray arr : freshPlaceholders.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }

        // Assert SLOT_BY_SLOT vs TRITON match — this isolates Triton orchestration issues
        assertTrue(slotVsTriton.isEmpty(),
                "Single forward pass diverges between SLOT_BY_SLOT and Triton: " + slotVsTriton);
    }

    // ─── Test: Decode-shape single forward pass comparison ─────────────────

    @Test
    @DisplayName("Decode-shape forward pass: SLOT_BY_SLOT vs Triton (seqLen=1, populated KV)")
    public void testDecodeShapeForwardPassComparison() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available — skipping");
            return;
        }
        ensureModelsLoaded();

        // Decode-shape inputs: one token at cache position kvSeqLen, over caches whose first
        // kvSeqLen positions hold random data in place of a prefill's keys and values
        int maxKvLen = 2048;
        int kvSeqLen = 18;  // 17 prefill + 1 decode
        Map<String, INDArray> kvBuffers = InGraphKvDecoderInputs.kvBuffers(decoder, maxKvLen);
        for (INDArray kv : kvBuffers.values()) {
            INDArray cached = kv.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, kvSeqLen), NDArrayIndex.all());
            cached.assign(Nd4j.randn(DataType.FLOAT, cached.shape()).muli(0.1f));
        }
        // inputs_embeds: [1, 1, hidden] — single token embedding
        Map<String, INDArray> decodePlaceholders = InGraphKvDecoderInputs.stepInputs(decoder, hiddenSize,
                Nd4j.randn(DataType.FLOAT, 1, 1, hiddenSize).muli(0.02f), kvSeqLen, kvBuffers);

        List<String> outputs = new ArrayList<>(decoder.outputs());
        log.info("Decode-shape comparison: {} placeholders, {} outputs, kvSeqLen={}",
                decodePlaceholders.size(), outputs.size(), kvSeqLen);

        // Apply Triton config
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfig tritonConfig = BenchmarkConfig.create("COMPARE_TRITON")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true);
        BenchmarkConfigApplier.apply(tritonConfig);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        // Compare SLOT_BY_SLOT vs TRITON at decode shape
        log.info("=== Compare: SLOT_BY_SLOT vs TRITON (decode shape) ===");
        Map<String, Double> slotVsTriton = decoder.compareDspModes(
                decodePlaceholders, outputs, 1e-3,
                GraphExecutionMode.SLOT_BY_SLOT,
                GraphExecutionMode.TRITON);
        log.info("SLOT_BY_SLOT vs TRITON (decode shape): {} divergent of {}",
                slotVsTriton.size(), outputs.size());
        for (Map.Entry<String, Double> e : slotVsTriton.entrySet()) {
            log.info("  divergent: {} maxDiff={}", e.getKey(), e.getValue());
        }

        // Cleanup
        for (INDArray arr : decodePlaceholders.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }

        assertTrue(slotVsTriton.isEmpty(),
                "Decode-shape forward pass diverges between SLOT_BY_SLOT and Triton: " + slotVsTriton);
    }

    // ─── Test: Multi-step decode mode comparison ──────────────────────────

    @Test
    @DisplayName("Multi-step decode: Triton skip, verify and no-capture decodes reproduce SLOT_BY_SLOT")
    public void testMultiStepDecodeComparison() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available — skipping");
            return;
        }
        ensureModelsLoaded();

        int maxTokens = getTokens(5);
        log.info("=== Multi-step decode comparison: {} tokens ===", maxTokens);

        // Run SLOT_BY_SLOT (reference — known 100% accurate)
        BenchmarkConfig slotConfig = BenchmarkConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(maxTokens);
        GenerationResult slotResult = runDecode(slotConfig, maxTokens);
        int[] slotTokens = slotResult.getTokenIds();
        log.info("SLOT_BY_SLOT: {} tokens, text='{}'", slotTokens.length, slotResult.getText());

        // TRITON_SKIP_KERNELS: the Triton backend with every compiled kernel skipped, so each
        // segment runs on the native ordered-range executor: DSP orchestration without Triton code.
        BenchmarkConfig skipConfig = BenchmarkConfig.create("TRITON_SKIP_KERNELS")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonSkipKernels(true)
                .maxTokens(maxTokens);
        GenerationResult skipResult = runDecode(skipConfig, maxTokens);
        int[] skipTokens = skipResult.getTokenIds();
        log.info("TRITON_SKIP_KERNELS: {} tokens, text='{}'", skipTokens.length, skipResult.getText());

        // TRITON_VERIFY: runs each Triton kernel and its native counterpart and compares them;
        // a HASH_MISMATCH line names a kernel whose output differs. tritonVerifyFullSnapshot
        // records every slot before and after.
        BenchmarkConfig verifyConfig = BenchmarkConfig.create("TRITON_VERIFY")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonVerifyKernels(true)
                .tritonVerifyFullSnapshot(true)
                .maxTokens(maxTokens);
        GenerationResult verifyResult = runDecode(verifyConfig, maxTokens);
        int[] verifyTokens = verifyResult.getTokenIds();
        log.info("TRITON_VERIFY: {} tokens, text='{}'", verifyTokens.length, verifyResult.getText());

        // TRITON_NO_GC: Triton kernels without CUDA graph capture
        BenchmarkConfig tritonConfig = BenchmarkConfig.create("TRITON_NO_GC")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .maxTokens(maxTokens);
        GenerationResult tritonResult = runDecode(tritonConfig, maxTokens);
        int[] tritonTokens = tritonResult.getTokenIds();
        log.info("TRITON_NO_GC: {} tokens, text='{}'", tritonTokens.length, tritonResult.getText());

        // Each must reproduce the SLOT_BY_SLOT decode: SKIP_KERNELS checks the orchestration,
        // VERIFY and NO_GC the compiled kernels
        double required = requiredMatchRate();
        double skipRate = tokenMatchRate(skipTokens, slotTokens);
        double verifyRate = tokenMatchRate(verifyTokens, slotTokens);
        double tritonRate = tokenMatchRate(tritonTokens, slotTokens);
        log.info("Match vs SLOT_BY_SLOT: SKIP_KERNELS={}% VERIFY={}% NO_GC={}%",
                String.format("%.1f", skipRate * 100), String.format("%.1f", verifyRate * 100),
                String.format("%.1f", tritonRate * 100));
        assertAll(
                () -> assertTrue(skipRate >= required,
                        matchRateMessage("TRITON_SKIP_KERNELS vs SLOT_BY_SLOT", skipRate, required)),
                () -> assertTrue(verifyRate >= required,
                        matchRateMessage("TRITON_VERIFY vs SLOT_BY_SLOT", verifyRate, required)),
                () -> assertTrue(tritonRate >= required,
                        matchRateMessage("TRITON_NO_GC vs SLOT_BY_SLOT", tritonRate, required)));
    }

    @Test
    @DisplayName("Per-slot fingerprint comparison: SLOT_BY_SLOT vs TRITON_NO_GC")
    public void testPerSlotFingerprint() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available — skipping");
            return;
        }
        ensureModelsLoaded();

        int maxTokens = 5;
        log.info("=== Per-slot fingerprint comparison: {} tokens ===", maxTokens);

        // Both runs with debug+verbose, which print per-slot fingerprints
        BenchmarkConfig slotConfig = BenchmarkConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(maxTokens);
        BenchmarkConfig tritonConfig = BenchmarkConfig.create("TRITON_NO_GC")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .maxTokens(maxTokens);
        GenerationResult slotResult;
        GenerationResult tritonResult;
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);
        try {
            log.info(">>> SLOT_BY_SLOT run with debug+verbose <<<");
            slotResult = runDecode(slotConfig, maxTokens);
            log.info(">>> TRITON_NO_GC run with debug+verbose <<<");
            tritonResult = runDecode(tritonConfig, maxTokens);
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
        }
        log.info("SLOT_BY_SLOT tokens: {}", Arrays.toString(slotResult.getTokenIds()));
        log.info("TRITON_NO_GC tokens: {}", Arrays.toString(tritonResult.getTokenIds()));

        // The per-slot DSP_FINGERPRINT lines of both runs are in the output for locating the
        // first divergent slot; the decodes themselves must agree
        int[] slotTokens = slotResult.getTokenIds();
        int[] tritonTokens = tritonResult.getTokenIds();
        int minLen = Math.min(slotTokens.length, tritonTokens.length);
        for (int i = 0; i < minLen; i++) {
            log.info("Step {}: SLOT={} TRITON={} {}",
                    i, slotTokens[i], tritonTokens[i],
                    slotTokens[i] == tritonTokens[i] ? "MATCH" : "DIVERGE");
        }
        double rate = tokenMatchRate(tritonTokens, slotTokens);
        assertTrue(rate >= requiredMatchRate(),
                matchRateMessage("TRITON_NO_GC vs SLOT_BY_SLOT", rate, requiredMatchRate()));
    }

    /**
     * Build decoder placeholders for step 0 (the prompt) — reusable for any single-pass test.
     * The caches leave room for a short decode after the prompt.
     */
    private Map<String, INDArray> buildDecoderStep0Inputs() {
        long seqLen = inputsEmbeds.size(1);
        Map<String, INDArray> placeholders = InGraphKvDecoderInputs.stepInputs(decoder, hiddenSize,
                inputsEmbeds.dup(), 0, InGraphKvDecoderInputs.kvBuffers(decoder, seqLen + getTokens(5)));
        log.info("buildDecoderStep0Inputs: {} total placeholders", placeholders.size());
        return placeholders;
    }

    // ─── Test: Per-op slot validation ──────────────────────────────────────


    // ─── Test: Decode step validation ──────────────────────────────────────

    @Test
    @DisplayName("Decode step comparison: fresh Triton execution vs OPTIMAL (graph replay)")
    public void testDecodeStepValidation() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping decode step validation");
            return;
        }
        ensureModelsLoaded();
        log.info("Running decode step validation...");

        int maxTokens = getTokens(10);

        // Reference: OPTIMAL config — already validated correct by testOutputAccuracy.
        // We use OPTIMAL as the ground truth and verify that FORCE_RECAPTURE
        // (capture+replay each step) produces the same tokens. This validates
        // that CUDA graph capture/replay doesn't corrupt output.
        BenchmarkConfig optimalCfg = BenchmarkConfig.optimal();

        // Run OPTIMAL as the reference (known-good from testOutputAccuracy)
        GenerationResult refResult = runDecode(
                optimalCfg.maxTokens(maxTokens),
                maxTokens);

        // Run FORCE-RECAPTURE: capture+replay each step (re-captures after every replay).
        // If this matches OPTIMAL, capture+replay is correct.
        // If this diverges, the bug is in capture+replay itself.
        BenchmarkConfig forceRecapCfg = BenchmarkConfig.create("FORCE_RECAPTURE")
                .tritonIncludeTypes(optimalCfg.getTritonIncludeTypes())
                .tritonSectionFusion(optimalCfg.isTritonSectionFusion())
                .tritonCompileAll(optimalCfg.isTritonCompileAll())
                .tritonGraphCapture(true)
                .tritonConsolidatedArgTable(optimalCfg.isTritonConsolidatedArgTable())
                .tritonArgDirtyTracking(optimalCfg.isTritonArgDirtyTracking())
                .tritonFusionScoring(optimalCfg.isTritonFusionScoring())
                .tritonNumWarps(optimalCfg.getTritonNumWarps())
                .tritonNumStages(optimalCfg.getTritonNumStages())
                .tritonForceRecapture(true)
                .cublasTf32(optimalCfg.isCublasTf32())
                .tritonTf32(optimalCfg.isTritonTf32())
                .dspBatchedGemm(optimalCfg.isDspBatchedGemm())
                .maxTokens(maxTokens);
        GenerationResult forceRecapResult = runDecode(forceRecapCfg, maxTokens);

        // No-capture: the same Triton kernels launched directly, without CUDA graphs and without
        // the consolidated argument table
        BenchmarkConfig noCaptureConfig = BenchmarkConfig.create("REF_TRITON_NO_CAPTURE")
                .tritonIncludeTypes(optimalCfg.getTritonIncludeTypes())
                .tritonSectionFusion(optimalCfg.isTritonSectionFusion())
                .tritonCompileAll(optimalCfg.isTritonCompileAll())
                .tritonGraphCapture(false)
                .tritonConsolidatedArgTable(false)
                .tritonArgDirtyTracking(false)
                .tritonFusionScoring(optimalCfg.isTritonFusionScoring())
                .tritonNumWarps(optimalCfg.getTritonNumWarps())
                .tritonNumStages(optimalCfg.getTritonNumStages())
                .cublasTf32(optimalCfg.isCublasTf32())
                .tritonTf32(optimalCfg.isTritonTf32())
                .dspBatchedGemm(optimalCfg.isDspBatchedGemm())
                .maxTokens(maxTokens);
        GenerationResult noCaptureResult = runDecode(noCaptureConfig, maxTokens);

        // Compare generated token IDs
        int[] refTokens = refResult.getTokenIds();
        int[] forceRecapTokens = forceRecapResult.getTokenIds();
        int[] noCaptureTokens = noCaptureResult.getTokenIds();

        // --- Force-recapture vs OPTIMAL ---
        int forceRecapMinLen = Math.min(refTokens.length, forceRecapTokens.length);
        int forceRecapMatches = 0;
        int forceRecapFirstDiv = -1;
        for (int i = 0; i < forceRecapMinLen; i++) {
            if (refTokens[i] == forceRecapTokens[i]) {
                forceRecapMatches++;
            } else if (forceRecapFirstDiv < 0) {
                forceRecapFirstDiv = i;
            }
        }
        double forceRecapMatchRate = tokenMatchRate(forceRecapTokens, refTokens);
        log.info("=== FORCE-RECAPTURE vs OPTIMAL: {}/{} ({}%) ===",
                forceRecapMatches, Math.max(refTokens.length, forceRecapTokens.length),
                String.format("%.1f", forceRecapMatchRate * 100));
        log.info("  OPTIMAL text:          {}", refResult.getText());
        log.info("  Force-recapture text:  {}", forceRecapResult.getText());
        if (forceRecapFirstDiv >= 0) {
            log.info("  First divergent at step {}: optimal={} forceRecap={}",
                    forceRecapFirstDiv, refTokens[forceRecapFirstDiv], forceRecapTokens[forceRecapFirstDiv]);
        }

        // --- No-capture vs OPTIMAL ---
        double noCaptureMatchRate = tokenMatchRate(noCaptureTokens, refTokens);
        log.info("=== NO-CAPTURE vs OPTIMAL: {}% ===", String.format("%.1f", noCaptureMatchRate * 100));
        log.info("  No-capture text: {}", noCaptureResult.getText());

        if (verbose) {
            for (int i = 0; i < forceRecapMinLen; i++) {
                String matchStr = refTokens[i] == forceRecapTokens[i] ? "OK" : "DIVERGE";
                log.info("Step {}: optimal={} forceRecap={} noCapture={} [{}]",
                        i, refTokens[i], forceRecapTokens[i],
                        i < noCaptureTokens.length ? noCaptureTokens[i] : -1,
                        matchStr);
            }
        }

        // Both must match OPTIMAL at the configured rate: recapturing every step checks capture
        // and replay, launching without graphs checks the kernels and their arguments alone.
        double requiredRate = requiredMatchRate();
        assertAll(
                () -> assertTrue(forceRecapMatchRate >= requiredRate,
                        matchRateMessage("FORCE_RECAPTURE vs OPTIMAL", forceRecapMatchRate, requiredRate)),
                () -> assertTrue(noCaptureMatchRate >= requiredRate,
                        matchRateMessage("REF_TRITON_NO_CAPTURE vs OPTIMAL", noCaptureMatchRate, requiredRate)));
    }

    // ─── Test: executeSteadyState fast path isolation ─────────────────────

    /**
     * The executeSteadyState fast path and graph replay against the paths they shortcut.
     *
     * Configs:
     *   1. SLOT_BY_SLOT: baseline (no graph capture, no fast path)
     *   2. OPTIMAL: full fast path (executeSteadyState -> platformTryFrozenFastPath)
     *   3. OPTIMAL + tritonVerifyKernels=true: forces the execute() path (no fast path)
     *   4. OPTIMAL + tritonForceRecapture=true: re-captures every step (no replay reuse)
     *
     * Each must reproduce the baseline as closely as TF32 allows, OPTIMAL must agree with (3)
     * and (4), and no slot of the replayed plan may hold NaN. A divergence between (2) and (3)
     * points at the fast path; one between (2) and (4) at graph replay.
     */
    @Test
    @DisplayName("Isolate executeSteadyState fast path vs execute() path divergence")
    public void testSteadyStateFastPathIsolation() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping fast path isolation test");
            return;
        }
        ensureModelsLoaded();
        int maxTokens = getTokens(10);
        log.info("=== STEADY STATE FAST PATH ISOLATION (tokens={}) ===", maxTokens);

        // 1. SLOT_BY_SLOT baseline (no graph capture, no fast path)
        GenerationResult baselineResult = runDecode(
                BenchmarkConfig.create("ISO_BASELINE_SBS")
                        .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                        .maxTokens(maxTokens),
                maxTokens);
        int[] baselineTokens = baselineResult.getTokenIds();
        log.info("[ISO_BASELINE_SBS] tokens={} text='{}'",
                baselineTokens.length, baselineResult.getText());

        // 2. OPTIMAL (executeSteadyState fast path active). The DspHandle inspects the
        // post-decode slot state while the decode's plan is still alive.
        BenchmarkConfig optimalCfg = BenchmarkConfig.optimal().maxTokens(maxTokens);
        GenerationResult optimalResult = runDecode(optimalCfg, maxTokens, decoded -> {
            DspHandle h = decoder.dsp();
            assertTrue(h.isCompiled(), "an OPTIMAL decode must leave a compiled decoder plan to inspect");
            int nanSlot = h.firstNaNSlot();
            log.info("[ISO_OPTIMAL] DspHandle: totalSlots={} firstNaNSlot={}",
                    h.totalSlots(), nanSlot);
            if (nanSlot >= 0) {
                Map<Integer, String> snapshot = h.snapshotAllSlots();
                int count = 0;
                for (Map.Entry<Integer, String> e : snapshot.entrySet()) {
                    if (e.getValue().contains("NaN") && count++ < 5) {
                        log.info("  NaN: {}", e.getValue());
                    }
                }
            }
            assertEquals(-1, nanSlot, "a slot of the replayed decode plan holds NaN");
        });
        int[] optimalTokens = optimalResult.getTokenIds();
        log.info("[ISO_OPTIMAL] tokens={} text='{}'",
                optimalTokens.length, optimalResult.getText());

        // 3. OPTIMAL + tritonVerifyKernels=true (forces execute() path, bypasses fast path)
        // The C++ precondition is: if (tritonVerifyKernels()) return execute(...)
        // This is the KEY isolation: same config, but never enters executeSteadyState fast path.
        BenchmarkConfig noFastPathCfg = BenchmarkConfig.create("ISO_NO_FAST_PATH")
                .tritonIncludeTypes(optimalCfg.getTritonIncludeTypes())
                .tritonSectionFusion(optimalCfg.isTritonSectionFusion())
                .tritonCompileAll(optimalCfg.isTritonCompileAll())
                .tritonGraphCapture(optimalCfg.isTritonGraphCapture())
                .tritonAllowFallbackCapture(false)
                .tritonConsolidatedArgTable(optimalCfg.isTritonConsolidatedArgTable())
                .tritonArgDirtyTracking(optimalCfg.isTritonArgDirtyTracking())
                .tritonFusionScoring(optimalCfg.isTritonFusionScoring())
                .tritonMergedCaptureThroughViews(optimalCfg.isTritonMergedCaptureThroughViews())
                .tritonNumWarps(optimalCfg.getTritonNumWarps())
                .tritonNumStages(optimalCfg.getTritonNumStages())
                .cublasTf32(optimalCfg.isCublasTf32())
                .tritonTf32(optimalCfg.isTritonTf32())
                .dspBatchedGemm(optimalCfg.isDspBatchedGemm())
                .dspFreezeMergeSegments(optimalCfg.isDspFreezeMergeSegments())
                .tritonVerifyKernels(true)     // Forces execute() path
                .maxTokens(maxTokens);
        GenerationResult noFastResult = runDecode(noFastPathCfg, maxTokens);
        int[] noFastTokens = noFastResult.getTokenIds();
        log.info("[ISO_NO_FAST_PATH] tokens={} text='{}'",
                noFastTokens.length, noFastResult.getText());

        // 4. OPTIMAL + tritonForceRecapture=true (re-captures every step, no replay reuse)
        BenchmarkConfig recapCfg = BenchmarkConfig.create("ISO_FORCE_RECAPTURE")
                .tritonIncludeTypes(optimalCfg.getTritonIncludeTypes())
                .tritonSectionFusion(optimalCfg.isTritonSectionFusion())
                .tritonCompileAll(optimalCfg.isTritonCompileAll())
                .tritonGraphCapture(true)
                .tritonAllowFallbackCapture(false)
                .tritonConsolidatedArgTable(optimalCfg.isTritonConsolidatedArgTable())
                .tritonArgDirtyTracking(optimalCfg.isTritonArgDirtyTracking())
                .tritonFusionScoring(optimalCfg.isTritonFusionScoring())
                .tritonMergedCaptureThroughViews(optimalCfg.isTritonMergedCaptureThroughViews())
                .tritonNumWarps(optimalCfg.getTritonNumWarps())
                .tritonNumStages(optimalCfg.getTritonNumStages())
                .cublasTf32(optimalCfg.isCublasTf32())
                .tritonTf32(optimalCfg.isTritonTf32())
                .dspBatchedGemm(optimalCfg.isDspBatchedGemm())
                .dspFreezeMergeSegments(optimalCfg.isDspFreezeMergeSegments())
                .tritonForceRecapture(true)   // Re-capture every step
                .maxTokens(maxTokens);
        GenerationResult recapResult = runDecode(recapCfg, maxTokens);
        int[] recapTokens = recapResult.getTokenIds();
        log.info("[ISO_FORCE_RECAPTURE] tokens={} text='{}'",
                recapTokens.length, recapResult.getText());

        // ─── Analysis ───
        log.info("=== FAST PATH ISOLATION ANALYSIS ===");
        double optVsBase = logTokenComparison("OPTIMAL vs BASELINE",
                optimalTokens, baselineTokens);
        double noFastVsBase = logTokenComparison("NO_FAST_PATH vs BASELINE",
                noFastTokens, baselineTokens);
        double recapVsBase = logTokenComparison("FORCE_RECAPTURE vs BASELINE",
                recapTokens, baselineTokens);
        double optVsNoFast = logTokenComparison("OPTIMAL vs NO_FAST_PATH",
                optimalTokens, noFastTokens);
        double optVsRecap = logTokenComparison("OPTIMAL vs FORCE_RECAPTURE",
                optimalTokens, recapTokens);

        // Each path must reproduce the fp32 baseline as closely as its TF32 setting allows, and
        // the three TF32 paths must agree: OPTIMAL differs from NO_FAST_PATH only in the
        // executeSteadyState fast path, and from FORCE_RECAPTURE only in replaying captured graphs.
        double vsBaseline = requiredMatchRateAgainstFp32(optimalCfg);
        double samePath = requiredMatchRate();
        assertAll(
                () -> assertTrue(optVsBase >= vsBaseline,
                        matchRateMessage("OPTIMAL vs SLOT_BY_SLOT", optVsBase, vsBaseline)),
                () -> assertTrue(noFastVsBase >= vsBaseline,
                        matchRateMessage("NO_FAST_PATH vs SLOT_BY_SLOT", noFastVsBase, vsBaseline)),
                () -> assertTrue(recapVsBase >= vsBaseline,
                        matchRateMessage("FORCE_RECAPTURE vs SLOT_BY_SLOT", recapVsBase, vsBaseline)),
                () -> assertTrue(optVsNoFast >= samePath, matchRateMessage(
                        "OPTIMAL vs NO_FAST_PATH (the executeSteadyState fast path)", optVsNoFast, samePath)),
                () -> assertTrue(optVsRecap >= samePath, matchRateMessage(
                        "OPTIMAL vs FORCE_RECAPTURE (graph replay)", optVsRecap, samePath)));
    }

    /**
     * Compare two token sequences, logging the first divergent step; returns the match rate.
     */
    private double logTokenComparison(String label, int[] test, int[] ref) {
        int minLen = Math.min(test.length, ref.length);
        int matches = 0;
        int firstDiv = -1;
        for (int i = 0; i < minLen; i++) {
            if (test[i] == ref[i]) {
                matches++;
            } else if (firstDiv < 0) {
                firstDiv = i;
            }
        }
        double rate = tokenMatchRate(test, ref);
        log.info("[{}] match={}/{} ({}%) firstDivStep={}{}",
                label, matches, Math.max(test.length, ref.length), String.format("%.1f", rate * 100),
                firstDiv,
                firstDiv >= 0 ? String.format(" (ref=%d test=%d)", ref[firstDiv], test[firstDiv]) : "");
        if (verbose && firstDiv >= 0) {
            for (int i = 0; i < minLen; i++) {
                String m = test[i] == ref[i] ? "OK" : "DIVERGE";
                log.info("  [{}] step {}: ref={} test={} [{}]", label, i, ref[i], test[i], m);
            }
        }
        return rate;
    }

    // ─── Test: Staging buffer D2D introspection during CUDA graph replay ──

    @Test
    @DisplayName("Staging buffer introspection: verify D2D copies during graph replay")
    public void testStagingBufferReplayIntrospection() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping staging introspection test");
            return;
        }
        ensureModelsLoaded();
        int maxTokens = getTokens(8);
        log.info("=== STAGING BUFFER REPLAY INTROSPECTION (tokens={}) ===", maxTokens);

        // Run OPTIMAL config to exercise CUDA graph capture + replay; the plan state is
        // inspected before the pipeline closes
        runDecode(BenchmarkConfig.optimal().maxTokens(maxTokens), maxTokens, this::inspectStagingState);
    }

    private void inspectStagingState(GenerationResult result) {
        int[] tokens = result.getTokenIds();
        log.info("[STAGING] generated {} tokens: '{}'", tokens.length, result.getText());

        // Now inspect the plan state via DspHandle
        DspHandle h = decoder.dsp();
        assertTrue(h.isCompiled(), "an OPTIMAL decode must leave a compiled decoder plan to inspect");

        int execCount = h.executeCount();
        int numStaging = h.numStagingBuffers();
        int numCachedVar = h.numCachedVariableExtIndices();
        int numExt = h.numExternalInputs();
        int totalSlots = h.totalSlots();

        log.info("[STAGING] Plan state: executeCount={} numExt={} numStaging={} numCachedVar={} totalSlots={}",
                execCount, numExt, numStaging, numCachedVar, totalSlots);

        // Log all cached variable ext indices
        List<Integer> varIndices = h.cachedVariableExtIndices();
        log.info("[STAGING] Cached variable ext indices ({}): {}", varIndices.size(), varIndices);

        // Log staging buffer state for each variable ext input
        Map<Integer, String> stagingState = h.snapshotStagingState();
        log.info("[STAGING] Staging state ({} entries):", stagingState.size());
        for (Map.Entry<Integer, String> e : stagingState.entrySet()) {
            log.info("  {}", e.getValue());
        }

        // Verify staging buffers exist for variable inputs
        assertTrue(numStaging > 0,
                "Expected staging buffers for variable ext inputs, got 0");
        assertTrue(numCachedVar > 0,
                "Expected cached variable ext indices, got 0");

        // Verify effective addresses match staging addresses for variable inputs
        int addressMismatches = 0;
        for (int extIdx : varIndices) {
            long stagingAddr = h.stagingBufferAddress(extIdx);
            long effectiveAddr = h.effectiveExternalAddress(extIdx);
            if (stagingAddr != 0 && effectiveAddr != 0 && stagingAddr != effectiveAddr) {
                log.warn("[STAGING] ADDRESS MISMATCH ext[{}]: staging=0x{} effective=0x{} — " +
                         "CUDA graph reads from effective but staging was D2D-copied to",
                        extIdx, Long.toHexString(stagingAddr), Long.toHexString(effectiveAddr));
                addressMismatches++;
            }
        }
        log.info("[STAGING] Address mismatches: {}/{}", addressMismatches, varIndices.size());

        // Check for stuck tokens (repeating pattern starting at step 4)
        boolean hasStuckTokens = false;
        if (tokens.length >= 6) {
            int tok4 = tokens[3]; // step 4 (0-indexed: 3)
            boolean allSame = true;
            for (int i = 4; i < Math.min(tokens.length, 8); i++) {
                if (tokens[i] != tok4) { allSame = false; break; }
            }
            if (allSame) {
                hasStuckTokens = true;
                log.warn("[STAGING] STUCK TOKEN DETECTED: token {} repeats from step 4 onward", tok4);
            }
        }

        // Log per-token output for diagnosis
        for (int i = 0; i < tokens.length; i++) {
            String tokenText = "";
            try { tokenText = tokenizer.decode(new int[]{tokens[i]}); } catch (Exception e) { /* ignore */ }
            log.info("[STAGING] step={} token={} text='{}'", i, tokens[i], tokenText);
        }

        // Check for variable inputs NOT in cached list (marked after cache was built)
        NativeOps nOps = Nd4j.getNativeOps();
        Pointer handle = h.getNativePlanHandle();
        int numVariable = nOps.getPlanNumVariableExternalInputs(handle);
        log.info("[STAGING] Total variable ext inputs (from externalInputIsVariable_): {}", numVariable);
        log.info("[STAGING] Cached in fast path: {} — DELTA (uncached variable): {}",
                numCachedVar, numVariable - numCachedVar);

        // Scan for variable inputs NOT in cached list
        List<Integer> uncachedVariable = new ArrayList<>();
        for (int i = 0; i < numExt && i < 1400; i++) {
            if (nOps.getPlanIsExternalInputVariable(handle, i)) {
                if (!varIndices.contains(i)) {
                    uncachedVariable.add(i);
                }
            }
        }
        if (!uncachedVariable.isEmpty()) {
            log.warn("[STAGING] UNCACHED VARIABLE EXT INPUTS (marked variable but NOT in D2D fast path): {}",
                    uncachedVariable);
            // These inputs are changing per step but NOT getting D2D copied to staging!
            // This is likely the root cause of the replay bug.
            for (int idx : uncachedVariable) {
                long stagingAddr = nOps.getPlanStagingBufferAddress(handle, idx);
                boolean isPlaceholder = nOps.getPlanIsExternalInputPlaceholder(handle, idx);
                log.warn("[STAGING]   ext[{}] isPlaceholder={} stagingAddr=0x{} — " +
                         "{}",
                        idx, isPlaceholder, Long.toHexString(stagingAddr),
                        stagingAddr == 0 ? "NO STAGING BUFFER ALLOCATED" : "has staging");
            }
        }

        // Lookup critical per-step ext input indices by name
        String[] criticalNames = {"inputs_embeds", "attention_mask", "position_ids",
                "input_ids", "causal_mask", "cache_position"};
        log.info("[STAGING] Critical per-step ext input index lookup:");
        for (String name : criticalNames) {
            int idx = h.extInputIndex(name);
            if (idx >= 0) {
                boolean isVar = nOps.getPlanIsExternalInputVariable(handle, idx);
                long stagingAddr = nOps.getPlanStagingBufferAddress(handle, idx);
                log.info("[STAGING]   '{}' -> ext[{}] isVariable={} stagingAddr=0x{}",
                        name, idx, isVar, Long.toHexString(stagingAddr));
            } else {
                log.info("[STAGING]   '{}' -> NOT FOUND (-1)", name);
            }
        }

        log.info("=== END STAGING BUFFER REPLAY INTROSPECTION ===");

        // A replay reads each per-step input from its staging buffer, so the effective address
        // must be the staging address, every variable input must be on the fast path's copy
        // list, and a stale input shows up as one token repeating from step 4
        final int mismatches = addressMismatches;
        final boolean stuck = hasStuckTokens;
        assertAll(
                () -> assertEquals(0, mismatches,
                        "variable inputs whose effective address is not their staging buffer"),
                () -> assertTrue(uncachedVariable.isEmpty(),
                        "variable inputs missing from the fast path's copy list: " + uncachedVariable),
                () -> assertFalse(stuck, "replayed decode repeats one token from step 4: "
                        + Arrays.toString(tokens)));
    }

    // ─── Test: TF32 impact isolation ──────────────────────────────────────

    @Test
    @DisplayName("TF32 impact isolation: same config with/without TF32")
    public void testTf32ImpactIsolation() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping TF32 isolation test");
            return;
        }
        ensureModelsLoaded();
        log.info("Running TF32 impact isolation...");

        int maxTokens = getTokens(5);

        // Without TF32
        GenerationResult noTf32Result = runDecode(
                BenchmarkConfig.create("NO_TF32")
                        .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                        .tritonSectionFusion(true).tritonCompileAll(true)
                        .cublasTf32(false)
                        .dspBatchedGemm(true)
                        .maxTokens(maxTokens),
                maxTokens);

        // With TF32
        BenchmarkConfig tf32Config = BenchmarkConfig.create("WITH_TF32")
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true)
                .cublasTf32(true)
                .dspBatchedGemm(true)
                .maxTokens(maxTokens);
        GenerationResult tf32Result = runDecode(tf32Config, maxTokens);

        // Compare tokens
        int[] noTf32Tokens = noTf32Result.getTokenIds();
        int[] tf32Tokens = tf32Result.getTokenIds();
        int minLen = Math.min(noTf32Tokens.length, tf32Tokens.length);
        int matches = 0;
        for (int i = 0; i < minLen; i++) {
            if (noTf32Tokens[i] == tf32Tokens[i]) matches++;
        }
        double matchRate = tokenMatchRate(tf32Tokens, noTf32Tokens);

        log.info("TF32 token match rate: {}/{} ({}%)", matches,
                Math.max(noTf32Tokens.length, tf32Tokens.length), String.format("%.1f", matchRate * 100));
        log.info("NO_TF32 text: {}", noTf32Result.getText());
        log.info("TF32 text:    {}", tf32Result.getText());

        // TF32 may move the decode only as far as the class's TF32 policy allows
        double required = requiredMatchRateAgainstFp32(tf32Config);
        assertTrue(matchRate >= required, matchRateMessage("WITH_TF32 vs NO_TF32", matchRate, required));
    }

    @Test
    @DisplayName("Decoder plan introspection: slots 348-358 boundary")
    public void testDecoderPlanBoundary348To358() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping boundary introspection");
            return;
        }
        ensureModelsLoaded();

        int maxTokens = getTokens(5);
        runDecode(BenchmarkConfig.optimal().maxTokens(maxTokens), maxTokens, result -> {
            assertNotNull(result, "Decode result should exist");

            InferenceSession session = decoder.getOrCreateSession();
            DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
            assertNotNull(executor, "DSP executor must exist");
            assertNotNull(executor.getCurrentPlan(), "Current plan must exist");

            var plan = executor.getCurrentPlan();
            assertTrue(plan.getSlots().length > 358, "Expected decoder plan to include slot 358");

            log.info("=== DECODER PLAN BOUNDARY: slots 348-358 ===");
            for (int slotIdx = 348; slotIdx <= 358; slotIdx++) {
                log.info(PlanIntrospection.formatSlot(plan, slotIdx));
            }

            String[] auxVarNames = {
                    "/model/layers.0/attn/v_proj/repeat_kv/Unsqueeze_2/output_0",
                    "/model/layers.0/attn/v_proj/repeat_kv/Mul_1/output_0",
                    "/model/layers.0/attn/v_proj/repeat_kv/Unsqueeze_4/output_0"
            };
            log.info("=== DECODER PLAN AUXILIARY ARRAYS (348-358) ===");
            for (String varName : auxVarNames) {
                SDVariable var = decoder.getVariable(varName);
                INDArray arr = decoder.getArrForVarName(varName);
                String creator = (var != null && var.getCreator() != null) ? var.getCreator().getOwnName() : "null";
                String type = (var != null) ? String.valueOf(var.getVariableType()) : "null";
                log.info("  {} -> type={} creator={} shape={} dtype={} values={}",
                        varName, type, creator,
                        arr != null ? Arrays.toString(arr.shape()) : "null",
                        arr != null ? arr.dataType() : "null",
                        (arr != null && arr.length() <= 16) ? arr.toStringFull() : "<len=" + (arr != null ? arr.length() : -1) + ">");
            }
        });
    }

    @Test
    @DisplayName("Decoder plan introspection: slots 400-430 boundary")
    public void testDecoderPlanBoundary400To430() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping boundary introspection");
            return;
        }
        ensureModelsLoaded();

        int maxTokens = getTokens(5);
        runDecode(BenchmarkConfig.optimal().maxTokens(maxTokens), maxTokens, result -> {
            assertNotNull(result, "Decode result should exist");

            InferenceSession session = decoder.getOrCreateSession();
            DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
            assertNotNull(executor, "DSP executor must exist");
            assertNotNull(executor.getCurrentPlan(), "Current plan must exist");

            var plan = executor.getCurrentPlan();
            assertTrue(plan.getSlots().length > 430, "Expected decoder plan to include slot 430");

            log.info("=== DECODER PLAN BOUNDARY: slots 400-430 ===");
            for (int slotIdx = 400; slotIdx <= 430; slotIdx++) {
                log.info(PlanIntrospection.formatSlot(plan, slotIdx));
            }

            int[] auxSlots = {399, 420, 421, 430, 431, 432};
            log.info("=== DECODER PLAN AUXILIARY SLOTS (400-430) ===");
            for (int slotIdx : auxSlots) {
                log.info(PlanIntrospection.formatSlot(plan, slotIdx));
            }

            String[] auxVarNames = {
                    "/model/layers.0/input_layernorm/output_0",
                    "/model/layers.0/attn/v_proj/MatMul/output_0",
                    "model.layers.0.attn.v_proj.MatMul.weight",
                    "model.layers.0.input_layernorm.weight"
            };
            log.info("=== DECODER PLAN AUXILIARY ARRAYS (400-430) ===");
            for (String varName : auxVarNames) {
                INDArray arr = decoder.getArrForVarName(varName);
                if (arr == null) {
                    log.info("  {} -> null", varName);
                    continue;
                }
                log.info("  {} -> shape={} dtype={} values={}",
                        varName, Arrays.toString(arr.shape()), arr.dataType(),
                        arr.length() <= 16 ? arr.toStringFull() : "<len=" + arr.length() + ">");
            }
        });
    }

    @Test
    @DisplayName("Decoder plan introspection: slots 431-453 boundary")
    public void testDecoderPlanBoundary431To453() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping boundary introspection");
            return;
        }
        ensureModelsLoaded();

        int maxTokens = getTokens(5);
        runDecode(BenchmarkConfig.optimal().maxTokens(maxTokens), maxTokens, result -> {
            assertNotNull(result, "Decode result should exist");

            InferenceSession session = decoder.getOrCreateSession();
            DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
            assertNotNull(executor, "DSP executor must exist");
            assertNotNull(executor.getCurrentPlan(), "Current plan must exist");

            var plan = executor.getCurrentPlan();
            assertTrue(plan.getSlots().length > 453, "Expected decoder plan to include slot 453");

            log.info("=== DECODER PLAN BOUNDARY: slots 431-453 ===");
            for (int slotIdx = 431; slotIdx <= 453; slotIdx++) {
                log.info(PlanIntrospection.formatSlot(plan, slotIdx));
            }

            int[] auxSlots = {263, 265, 276, 277, 278, 430};
            log.info("=== DECODER PLAN AUXILIARY SLOTS ===");
            for (int slotIdx : auxSlots) {
                log.info(PlanIntrospection.formatSlot(plan, slotIdx));
            }

            String[] auxVarNames = {
                    "/model/layers.0/attn/v_proj/repeat_kv/Mul_1/output_0",
                    "/model/layers.0/attn/v_proj/repeat_kv/Unsqueeze_4/output_0",
                    "/model/layers.0/attn/v_proj/repeat_kv/Unsqueeze_2/output_0",
                    "sd_var_21",
                    "sd_var_22",
                    "sd_var_23",
                    "sd_var_24",
                    "sd_var_25",
                    "sd_var_26",
                    "sd_var_27",
                    "sd_var_28",
                    "sd_var_29"
            };
            log.info("=== DECODER PLAN AUXILIARY ARRAYS ===");
            for (String varName : auxVarNames) {
                INDArray arr = decoder.getArrForVarName(varName);
                if (arr == null) {
                    log.info("  {} -> null", varName);
                    continue;
                }
                log.info("  {} -> shape={} dtype={} values={}",
                        varName, Arrays.toString(arr.shape()), arr.dataType(),
                        arr.length() <= 16 ? arr.toStringFull() : "<len=" + arr.length() + ">");
            }
        });
    }

    @Test
    @DisplayName("Decoder plan introspection: slots 455-523 boundary")
    public void testDecoderPlanBoundary455To523() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping boundary introspection");
            return;
        }
        ensureModelsLoaded();

        int maxTokens = getTokens(5);
        runDecode(BenchmarkConfig.optimal().maxTokens(maxTokens), maxTokens, result -> {
            assertNotNull(result, "Decode result should exist");

            InferenceSession session = decoder.getOrCreateSession();
            DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
            assertNotNull(executor, "DSP executor must exist");
            assertNotNull(executor.getCurrentPlan(), "Current plan must exist");

            var plan = executor.getCurrentPlan();
            assertTrue(plan.getSlots().length > 523, "Expected decoder plan to include slot 523");

            log.info("=== DECODER PLAN BOUNDARY: slots 455-523 ===");
            for (int slotIdx = 455; slotIdx <= 523; slotIdx++) {
                log.info(PlanIntrospection.formatSlot(plan, slotIdx));
            }

            int[] auxSlots = {455, 467, 489, 503, 523, 524};
            log.info("=== DECODER PLAN AUXILIARY SLOTS (455-523) ===");
            for (int slotIdx : auxSlots) {
                log.info(PlanIntrospection.formatSlot(plan, slotIdx));
            }
        });
    }

    @Test
    @DisplayName("Decoder plan introspection: slots 793-846 boundary")
    public void testDecoderPlanBoundary793To846() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping boundary introspection");
            return;
        }
        ensureModelsLoaded();

        int maxTokens = getTokens(5);
        runDecode(BenchmarkConfig.optimal().maxTokens(maxTokens), maxTokens, result -> {
            assertNotNull(result, "Decode result should exist");

            InferenceSession session = decoder.getOrCreateSession();
            DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
            assertNotNull(executor, "DSP executor must exist");
            assertNotNull(executor.getCurrentPlan(), "Current plan must exist");

            // This range was picked on the unoptimized 2742-op decoder. OnnxModelCache now imports
            // the optimized graph, whose plan ends inside the range: log the slots that exist.
            var plan = executor.getCurrentPlan();
            int numSlots = plan.getSlots().length;
            assertTrue(numSlots > 793, "Expected decoder plan to include slot 793");

            log.info("=== DECODER PLAN BOUNDARY: slots 793-846 (plan has {} slots) ===", numSlots);
            for (int slotIdx = 793; slotIdx <= Math.min(846, numSlots - 1); slotIdx++) {
                log.info(PlanIntrospection.formatSlot(plan, slotIdx));
            }

            int[] auxSlots = {792, 793, 819, 820, 821, 822, 823, 846, 847};
            log.info("=== DECODER PLAN AUXILIARY SLOTS (793-846) ===");
            for (int slotIdx : auxSlots) {
                if (slotIdx < numSlots) {
                    log.info(PlanIntrospection.formatSlot(plan, slotIdx));
                }
            }
        });
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private GenerationResult runDecode(BenchmarkConfig config, int maxTokens) throws Exception {
        return runDecode(config, maxTokens, null);
    }

    /**
     * Decodes the prompt with {@code config} on {@link #decoder}, then closes the pipeline.
     *
     * <p>{@link OnnxModelCache} returns the models already optimized, so the pipeline is told not
     * to optimize again. A second pass decodes a private copy of the decoder: the plan state this
     * class reads through {@code decoder} would belong to a model that never ran, and each call
     * would keep one more copy of the weights and its frozen plan alive until the pipeline is
     * closed. The config goes to the pipeline, which applies it to the models it runs; without it
     * the pipeline applies its default config over this one.</p>
     *
     * <p>Closing the pipeline resets the decoder's session and clears its plan cache, so a test
     * that reads the decode's plan state does so in {@code inspect}, before the close.</p>
     */
    private GenerationResult runDecode(BenchmarkConfig config, int maxTokens,
                                       DecodeInspection inspect) throws Exception {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        GenerationResult result;
        // The prefill copy is this method's: closed after the pipeline, which may retain it as a
        // stable prefill input until close
        try (INDArray prefill = inputsEmbeds.dup();
             GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig(config, maxTokens))) {
            result = pipeline.generate(prefill, promptTokenIds);
            if (inspect != null) {
                inspect.inspect(result);
            }
        }
        log.info("[runDecode] {}: {} tokens, physicalBytes={}MB after close",
                config.getName(), result.getTokenIds().length, mb(Pointer.physicalBytes()));
        return result;
    }

    private GenerationPipelineConfig pipelineConfig(BenchmarkConfig config, int maxTokens) {
        return GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ModelIOConfig.discover(decoder))
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .graphOptimizerEnabled(false)
                .benchmarkConfig(config)
                .build();
    }

    /** Reads the decode's plan state; runs after generate, before the pipeline closes. */
    @FunctionalInterface
    private interface DecodeInspection {
        void inspect(GenerationResult result) throws Exception;
    }

    // ─── Pipeline lifecycle retained memory ─────────────────────────────────

    /** Native memory a steady-state pipeline lifecycle may keep: measurement noise. */
    private static final long LIFECYCLE_RETENTION_PER_CYCLE_BYTES = 32L * 1024 * 1024;

    static Stream<String> lifecycleConfigs() {
        return Stream.of("SLOT_BY_SLOT", "OPTIMAL");
    }

    /**
     * Repeated pipeline lifecycles on the same decoder must not keep native memory. A closed
     * pipeline frees what it allocated itself instead of leaving it to garbage collection, which
     * a mostly idle Java heap may not run for a long time (a FLOAT copy of an FP16 embedding
     * table, 108MB for SmolDocling, used to pile up once per pipeline this way). Process RSS
     * minus the committed Java heap is measured after each lifecycle without forcing a
     * collection; from the third lifecycle on (plans compiled, caches and pools warm) it must
     * stay flat. A final collection reports what the lifecycles still left to it.
     */
    @ParameterizedTest(name = "lifecycles[{0}]")
    @MethodSource("lifecycleConfigs")
    public void testPipelineLifecycleRetainedMemory(String configName) throws Exception {
        ensureModelsLoaded();
        int cycles = Math.max(4, Integer.getInteger("vlm.validation.lifecycles", 6));
        int tokens = 3;
        BenchmarkConfig config = "OPTIMAL".equals(configName)
                ? BenchmarkConfig.optimal().maxTokens(tokens)
                : BenchmarkConfig.create("LIFECYCLE_" + configName)
                        .executionMode(GraphExecutionMode.SLOT_BY_SLOT).maxTokens(tokens);
        Runtime runtime = Runtime.getRuntime();
        long[] nativeRss = new long[cycles];
        for (int cycle = 0; cycle < cycles; cycle++) {
            runDecode(config, tokens);
            Nd4j.getExecutioner().commit();
            long rss = Pointer.physicalBytes();
            nativeRss[cycle] = rss - runtime.totalMemory();
            log.info("[LIFECYCLE] config={} cycle={} rss={}MB heapCommitted={}MB native={}MB",
                    configName, cycle, mb(rss), mb(runtime.totalMemory()), mb(nativeRss[cycle]));
        }

        // What the lifecycles left to garbage collection, by deallocator and bytes still held
        Map<Long, DeallocatableReference> references = Nd4j.getDeallocatorService().getReferenceMap();
        Map<Long, String> live = new HashMap<>();
        for (Map.Entry<Long, DeallocatableReference> entry : references.entrySet()) {
            Deallocator deallocator = entry.getValue().getDeallocator();
            live.put(entry.getKey(), (deallocator == null ? "none" : deallocator.getClass().getSimpleName())
                    + ":" + entry.getValue().getBytes());
        }
        for (int i = 0; i < 3; i++) {
            System.gc();
        }
        Nd4j.getDeallocatorService().forceFlushAll();
        Map<String, Integer> reclaimed = new TreeMap<>();
        for (Map.Entry<Long, String> entry : live.entrySet()) {
            if (!references.containsKey(entry.getKey())) reclaimed.merge(entry.getValue(), 1, Integer::sum);
        }
        log.info("[LIFECYCLE] config={} left to collection over {} lifecycles "
                + "(deallocator:bytes=count): {}", configName, cycles, reclaimed);

        long perCycle = (nativeRss[cycles - 1] - nativeRss[2]) / (cycles - 3);
        assertTrue(perCycle <= LIFECYCLE_RETENTION_PER_CYCLE_BYTES,
                configName + ": each pipeline lifecycle kept "
                        + mb(perCycle) + "MB of native memory after close");
    }

    // ─── Per-phase memory tracking: output() vs outputDirect() ─────────────

    /** Device memory the steady-state steps of a decode may keep in total: measurement noise. */
    private static final long STEADY_RETENTION_TOLERANCE_BYTES = 64L * 1024 * 1024;

    /**
     * Per-phase device memory of decode steps through output() and outputDirect(), over the
     * decoder's fixed in-graph caches. For each step it measures device free memory before the
     * call, after it returns (consumed), after its outputs close (released) and after a commit
     * and pool trim (released). The first steps compile, capture and seal the plan; from
     * {@code steadyFrom} on the plan replays, and those steps together must keep nothing beyond
     * measurement noise.
     */
    @Test
    @DisplayName("Per-phase memory tracking: output() vs outputDirect() decode")
    public void testPerPhaseMemoryTracking() throws Exception {
        ensureModelsLoaded();
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int device = Nd4j.getAffinityManager().getDeviceForCurrentThread().intValue();
        final int steps = 8;
        final int steadyFrom = 4;

        log.info("=== PER-PHASE MEMORY TRACKING ===");
        long[][] outputPhases = runPerPhaseDecodeSteps(nativeOps, device, steps, false);
        long[][] directPhases = runPerPhaseDecodeSteps(nativeOps, device, steps, true);

        log.info("");
        log.info("══════════════════════════════════════════════════════════════════════════════════");
        log.info("  SUMMARY: output() vs outputDirect() per-step memory (MB)");
        log.info("══════════════════════════════════════════════════════════════════════════════════");
        log.info(String.format("%-8s %12s %12s %12s | %12s %12s %12s",
                "Step",
                "out_delta", "out_recvCls", "out_recvTrm",
                "dir_delta", "dir_recvCls", "dir_recvTrm"));
        for (int i = 0; i < steps; i++) {
            log.info(String.format("step%-4d %12d %12d %12d | %12d %12d %12d",
                    i + 1,
                    mb(outputPhases[i][0]), mb(outputPhases[i][1]), mb(outputPhases[i][2]),
                    mb(directPhases[i][0]), mb(directPhases[i][1]), mb(directPhases[i][2])));
        }
        long outputRetained = retainedFrom(outputPhases, steadyFrom);
        long directRetained = retainedFrom(directPhases, steadyFrom);
        log.info("Retained over steps {}-{} (delta - recovered): output()={}MB outputDirect()={}MB",
                steadyFrom + 1, steps, mb(outputRetained), mb(directRetained));
        log.info("══════════════════════════════════════════════════════════════════════════════════");

        assertAll(
                () -> assertTrue(outputRetained <= STEADY_RETENTION_TOLERANCE_BYTES,
                        "output() decode steps " + (steadyFrom + 1) + "-" + steps + " kept "
                                + mb(outputRetained) + "MB"),
                () -> assertTrue(directRetained <= STEADY_RETENTION_TOLERANCE_BYTES,
                        "outputDirect() decode steps " + (steadyFrom + 1) + "-" + steps + " kept "
                                + mb(directRetained) + "MB"));
    }

    /** Net bytes the steps from index {@code from} on kept: consumed minus released by close and trim. */
    private static long retainedFrom(long[][] phases, int from) {
        long retained = 0;
        for (int i = from; i < phases.length; i++) {
            retained += phases[i][0] - phases[i][1] - phases[i][2];
        }
        return retained;
    }

    /**
     * Run {@code steps} OPTIMAL decode steps over fixed in-graph caches, measuring device free
     * memory at 4 phases per step. Returns long[steps][3] where each row is
     * [deltaOutput, recoveredClose, recoveredTrim]:
     *   deltaOutput    = beforeFree - afterOutputFree  (positive = consumed)
     *   recoveredClose = afterCloseFree - afterOutputFree  (positive = freed)
     *   recoveredTrim  = afterTrimFree - afterCloseFree  (positive = freed)
     */
    private long[][] runPerPhaseDecodeSteps(NativeOps nativeOps, int device, int steps,
                                            boolean useDirect) throws Exception {
        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(steps);
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.apply(config);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        // The in-graph caches are written in place, so logits is the only output
        String[] outputNames = {ModelIOConfig.findLogitsOutputName(decoder)};
        BenchmarkConfigApplier.compileModel(decoder, "decoder", Arrays.asList(outputNames), config);

        final long maxKvLen = 32;
        INDArray stepEmbeds = Nd4j.zeros(DataType.FLOAT, 1, 1, hiddenSize);
        Map<String, INDArray> caches = InGraphKvDecoderInputs.kvBuffers(decoder, maxKvLen);
        Set<INDArray> reused = Collections.newSetFromMap(new IdentityHashMap<>());
        reused.add(stepEmbeds);
        reused.addAll(caches.values());
        String modeName = useDirect ? "outputDirect" : "output";
        long[][] phases = new long[steps][3];

        log.info("");
        log.info("--- {} decode steps (mode={}) ---", steps, modeName);
        Nd4j.getExecutioner().commit();
        nativeOps.trimMemoryPool(device);
        try {
            for (int step = 0; step < steps; step++) {
                // ── Phase 1: device free memory BEFORE the step ──
                long beforeFree = nativeOps.getDeviceFreeMemoryDefault();

                // ── Phase 2: the step, one token at cache position step + 1 ──
                Map<String, INDArray> inputs = InGraphKvDecoderInputs.stepInputs(decoder, hiddenSize,
                        stepEmbeds, step + 1, caches);
                Map<String, INDArray> outputs = useDirect
                        ? decoder.outputDirect(inputs, outputNames)
                        : decoder.output(inputs, outputNames);
                long afterOutputFree = nativeOps.getDeviceFreeMemoryDefault();

                // ── Phase 3: close the returned outputs ──
                for (INDArray arr : outputs.values()) {
                    if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
                }
                long afterCloseFree = nativeOps.getDeviceFreeMemoryDefault();

                // ── Phase 4: commit and trim the pool ──
                Nd4j.getExecutioner().commit();
                nativeOps.trimMemoryPool(device);
                long afterTrimFree = nativeOps.getDeviceFreeMemoryDefault();

                // The step's own inputs; the embeddings and caches carry over
                for (INDArray arr : inputs.values()) {
                    if (arr != null && !reused.contains(arr) && arr.closeable() && !arr.wasClosed()) {
                        arr.close();
                    }
                }

                phases[step] = new long[]{beforeFree - afterOutputFree,
                        afterCloseFree - afterOutputFree, afterTrimFree - afterCloseFree};
                log.info("step={} before={}MB after_output={}MB delta_output={}MB "
                                + "after_close={}MB recovered_close={}MB after_trim={}MB recovered_trim={}MB [{}]",
                        step + 1,
                        mb(beforeFree), mb(afterOutputFree), mb(phases[step][0]),
                        mb(afterCloseFree), mb(phases[step][1]),
                        mb(afterTrimFree), mb(phases[step][2]),
                        modeName.toUpperCase());
            }
        } finally {
            // Retire the plan that holds the caches before closing them
            BenchmarkConfigApplier.resetModelState(decoder);
            stepEmbeds.close();
            for (INDArray cache : caches.values()) {
                if (cache.closeable() && !cache.wasClosed()) cache.close();
            }
        }
        return phases;
    }

    private static long mb(long bytes) {
        return bytes / (1024 * 1024);
    }

    // ─── Exact decode loop memory isolation ──────────────────────────────────

    /**
     * Whether any decode-loop operation grows memory or plans per step.
     *
     * After 3 warmup steps with identical inputs, runs 5 variants over the decoder's fixed
     * caches, each changing ONE thing:
     *   A. New HashMap each step (same arrays)
     *   B. clearPlaceholders(false) between steps
     *   C. New position_ids and cache_position arrays each step, in the input builder's layout
     *   D. New causal_mask and caches each step (same shapes)
     *   E. Full DecoderInputBuilder.buildDecoderInputMap path each step
     *
     * No variant changes an input's layout, so all of them must run on the plan warmup compiled,
     * and after its first step none may keep device memory beyond measurement noise.
     */
    @Test
    @DisplayName("Exact decode loop memory isolation: no operation grows memory or plans per step")
    public void testExactDecodeLoopMemoryIsolation() throws Exception {
        ensureModelsLoaded();
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int device = Nd4j.getAffinityManager().getDeviceForCurrentThread().intValue();
        final int maxKvLen = 20;
        final long cachePos = 1;

        // ── Setup: reset decoder, enable DSP, compile ──
        BenchmarkConfigApplier.resetModelState(decoder);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        // The in-graph caches are written in place, so logits is the only output
        String[] fullOutputArray = {ModelIOConfig.findLogitsOutputName(decoder)};
        decoder.compileNativeDynamicShapePlan(Arrays.asList(fullOutputArray), GraphExecutionMode.SLOT_BY_SLOT, true);
        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        String causalName = ioConfig.getCausalMaskName();
        DataType maskType = decoder.getVariable(causalName).dataType();
        DataType cachePositionType = decoder.getVariable(ioConfig.getCachePositionName()).dataType();

        // ── Build FIXED inputs (reused across warmup and variants A/B) ──
        INDArray stepEmbeds = Nd4j.zeros(DataType.FLOAT, 1, 1, hiddenSize);
        Map<String, INDArray> staticKvBuffers = InGraphKvDecoderInputs.kvBuffers(decoder, maxKvLen);
        Map<String, INDArray> fixedInputMap = InGraphKvDecoderInputs.stepInputs(decoder, hiddenSize,
                stepEmbeds, cachePos, staticKvBuffers);
        Set<INDArray> reused = Collections.newSetFromMap(new IdentityHashMap<>());
        reused.add(stepEmbeds);
        reused.addAll(staticKvBuffers.values());

        log.info("=== EXACT DECODE LOOP MEMORY ISOLATION ===");
        log.info("Device={}, maxKvLen={}, inputs={}, outputs={}",
                device, maxKvLen, fixedInputMap.size(), fullOutputArray.length);

        Map<String, Long> growth = new LinkedHashMap<>();
        int plansAfterWarmup;
        int plansAfterVariants;
        try {
            // ── Warmup: 3 steps with FIXED identical inputs ──
            log.info("--- WARMUP: 3 steps with identical inputs ---");
            Nd4j.getExecutioner().commit();
            nativeOps.trimMemoryPool(device);

            for (int step = 0; step < 3; step++) {
                long before = nativeOps.getDeviceFreeMemoryDefault();
                Map<String, INDArray> out = decoder.output(fixedInputMap, fullOutputArray);
                for (INDArray arr : out.values()) {
                    if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
                }
                Nd4j.getExecutioner().commit();
                nativeOps.trimMemoryPool(device);
                long after = nativeOps.getDeviceFreeMemoryDefault();
                log.info("[WARMUP] step={} gpuFree={}MB delta={}MB", step, mb(after), mb(before - after));
            }
            DynamicShapePlanExecutor executor = decoder.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor, "the warmup decode must run on a DSP plan");
            plansAfterWarmup = executor.getPinnedPlanCount();

            // ── VARIANT A: New HashMap each step, same arrays ──
            growth.put("A_NEW_HASHMAP", runVariant("A_NEW_HASHMAP", nativeOps, device, 5, step -> {
                Map<String, INDArray> newMap = new HashMap<>(fixedInputMap);
                Map<String, INDArray> out = decoder.output(newMap, fullOutputArray);
                for (INDArray arr : out.values()) {
                    if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
                }
            }));

            // ── VARIANT B: clearPlaceholders(false) between steps ──
            growth.put("B_CLEAR_PLACEHOLDERS", runVariant("B_CLEAR_PLACEHOLDERS", nativeOps, device, 5, step -> {
                decoder.clearPlaceholders(false);
                Map<String, INDArray> out = decoder.output(fixedInputMap, fullOutputArray);
                for (INDArray arr : out.values()) {
                    if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
                }
            }));

            // ── VARIANT C: new position_ids and cache_position arrays each step ──
            // The input builder's layout: position_ids [1, 1] LONG, cache_position a scalar of the
            // variable's dtype. A rank-1 cache_position is a different layout and its own plan.
            growth.put("C_NEW_POSITION_ARRAYS", runVariant("C_NEW_POSITION_ARRAYS", nativeOps, device, 5, step -> {
                INDArray positions = Nd4j.createFromArray(new long[][]{{cachePos}});
                INDArray cachePosition = Nd4j.scalar(cachePositionType, cachePos);
                Map<String, INDArray> variantMap = new LinkedHashMap<>(fixedInputMap);
                variantMap.put(ioConfig.getPositionIdsName(), positions);
                variantMap.put(ioConfig.getCachePositionName(), cachePosition);
                Map<String, INDArray> out = decoder.output(variantMap, fullOutputArray);
                for (INDArray arr : out.values()) {
                    if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
                }
                positions.close();
                cachePosition.close();
            }));

            // ── VARIANT D: new causal_mask and caches each step, same shapes ──
            growth.put("D_NEW_MASK_AND_CACHES", runVariant("D_NEW_MASK_AND_CACHES", nativeOps, device, 5, step -> {
                INDArray mask = InGraphKvDecoderInputs.decodeCausalMask(cachePos, maxKvLen, maskType);
                Map<String, INDArray> caches = InGraphKvDecoderInputs.kvBuffers(decoder, maxKvLen);
                Map<String, INDArray> variantMap = new LinkedHashMap<>(fixedInputMap);
                variantMap.put(causalName, mask);
                variantMap.putAll(caches);
                Map<String, INDArray> out = decoder.output(variantMap, fullOutputArray);
                for (INDArray arr : out.values()) {
                    if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
                }
                mask.close();
                for (INDArray cache : caches.values()) cache.close();
            }));

            // ── VARIANT E: Full DecoderInputBuilder.buildDecoderInputMap each step ──
            growth.put("E_FULL_BUILD_INPUT_MAP", runVariant("E_FULL_BUILD_INPUT_MAP", nativeOps, device, 5, step -> {
                Map<String, INDArray> builtMap = InGraphKvDecoderInputs.stepInputs(decoder, hiddenSize,
                        stepEmbeds, cachePos + step, staticKvBuffers);
                Map<String, INDArray> out = decoder.output(builtMap, fullOutputArray);
                for (INDArray arr : out.values()) {
                    if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
                }
                // Close the arrays this step built; the embeddings and caches are reused
                for (INDArray arr : builtMap.values()) {
                    if (arr != null && !reused.contains(arr) && arr.closeable() && !arr.wasClosed()) {
                        arr.close();
                    }
                }
            }));
            plansAfterVariants = executor.getPinnedPlanCount();
        } finally {
            // Retire the plan that holds the fixed inputs before closing them
            BenchmarkConfigApplier.resetModelState(decoder);
            for (INDArray arr : fixedInputMap.values()) {
                if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
            }
        }

        log.info("Plans: after warmup={} after variants={}; growth after each variant's first step (MB): {}",
                plansAfterWarmup, plansAfterVariants, growth.entrySet().stream()
                        .map(e -> e.getKey() + "=" + mb(e.getValue())).collect(Collectors.joining(", ")));
        List<Executable> checks = new ArrayList<>();
        checks.add(() -> assertEquals(plansAfterWarmup, plansAfterVariants,
                "no variant changes an input layout, so none may compile another plan"));
        for (Map.Entry<String, Long> e : growth.entrySet()) {
            checks.add(() -> assertTrue(e.getValue() <= STEADY_RETENTION_TOLERANCE_BYTES,
                    e.getKey() + " kept " + mb(e.getValue()) + "MB after its first step"));
        }
        assertAll(checks);
    }

    @FunctionalInterface
    private interface VariantStep {
        void run(int step) throws Exception;
    }

    /**
     * Runs {@code steps} steps of a variant, logging device free memory after each. Returns the
     * bytes the steps after the first kept; the first may allocate what the later ones reuse.
     */
    private long runVariant(String name, NativeOps nativeOps,
                            int device, int steps, VariantStep action) throws Exception {
        log.info("--- VARIANT {} ({} steps) ---", name, steps);
        Nd4j.getExecutioner().commit();
        nativeOps.trimMemoryPool(device);
        long baselineFree = nativeOps.getDeviceFreeMemoryDefault();
        long afterFirstStepFree = baselineFree;

        for (int step = 0; step < steps; step++) {
            long before = nativeOps.getDeviceFreeMemoryDefault();
            action.run(step);
            Nd4j.getExecutioner().commit();
            nativeOps.trimMemoryPool(device);
            long after = nativeOps.getDeviceFreeMemoryDefault();
            if (step == 0) afterFirstStepFree = after;
            long delta = before - after;
            log.info("[VARIANT] name={} step={} gpuFree={}MB delta={}MB", name, step, mb(after), mb(delta));
        }

        long finalFree = nativeOps.getDeviceFreeMemoryDefault();
        long totalLeak = baselineFree - finalFree;
        log.info("[VARIANT] name={} TOTAL: baseline={}MB final={}MB totalLeak={}MB avgPerStep={}MB",
                name, mb(baselineFree), mb(finalFree), mb(totalLeak), mb(totalLeak) / steps);
        return afterFirstStepFree - finalFree;
    }

    // ─── Flag bisection: isolate which OPTIMAL flag combination causes divergence ──

    /**
     * Builds TRITON_NO_GC base config (known correct) then adds one flag at a time
     * from the OPTIMAL config to find the exact flag or combination that breaks output.
     *
     * Run:
     *   cd platform-tests && mvn test \
     *     -Dtest=TestDspValidation#testOptimalFlagBisection \
     *     -Dbackend.artifactId=nd4j-cuda-13.1
     */
    static Stream<BenchmarkConfig> bisectionConfigs() {
        int tokens = getTokens(10);
        List<BenchmarkConfig> configs = new ArrayList<>();

        // Base: TRITON_NO_GC (known 100% correct)
        configs.add(tritonNoGcBase("TRITON_NO_GC").maxTokens(tokens));

        // Single-flag additions on top of TRITON_NO_GC
        configs.add(tritonNoGcBase("BISECT_GC")
                .tritonGraphCapture(true).maxTokens(tokens));
        // Force recapture: capture+launch every step, ZERO replays.
        // If this passes → replay is the bug. If this fails → capture itself is the bug.
        configs.add(tritonNoGcBase("BISECT_GC_FORCE_RECAPTURE")
                .tritonGraphCapture(true).tritonForceRecapture(true).maxTokens(tokens));
        configs.add(tritonNoGcBase("BISECT_TF32")
                .cublasTf32(true).tritonTf32(true).maxTokens(tokens));
        configs.add(tritonNoGcBase("BISECT_BATCHED_GEMM")
                .dspBatchedGemm(true).maxTokens(tokens));
        configs.add(tritonNoGcBase("BISECT_ARG_TABLE")
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true).maxTokens(tokens));
        configs.add(tritonNoGcBase("BISECT_FUSION_SCORING_OFF")
                .tritonFusionScoring(false).maxTokens(tokens));

        // Two-flag combos (most likely interaction pairs)
        configs.add(tritonNoGcBase("BISECT_GC+TF32")
                .tritonGraphCapture(true).cublasTf32(true).tritonTf32(true).maxTokens(tokens));
        configs.add(tritonNoGcBase("BISECT_GC+BATCHED_GEMM")
                .tritonGraphCapture(true).dspBatchedGemm(true).maxTokens(tokens));
        configs.add(tritonNoGcBase("BISECT_GC+ARG_TABLE")
                .tritonGraphCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true).maxTokens(tokens));
        configs.add(tritonNoGcBase("BISECT_GC+FUSION_OFF")
                .tritonGraphCapture(true).tritonFusionScoring(false).maxTokens(tokens));

        // Three-flag: GC + ARG_TABLE + BATCHED_GEMM
        configs.add(tritonNoGcBase("BISECT_GC+ARG+GEMM")
                .tritonGraphCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .dspBatchedGemm(true).maxTokens(tokens));

        // Full OPTIMAL minus TF32 (isolate TF32 as contributor)
        configs.add(tritonNoGcBase("BISECT_ALL_MINUS_TF32")
                .tritonGraphCapture(true)
                .tritonConsolidatedArgTable(true).tritonArgDirtyTracking(true)
                .tritonFusionScoring(false)
                .dspBatchedGemm(true).maxTokens(tokens));

        // Full OPTIMAL
        configs.add(BenchmarkConfig.optimal().maxTokens(tokens));

        return configs.stream();
    }

    private static BenchmarkConfig tritonNoGcBase(String name) {
        return BenchmarkConfig.create(name)
                .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                .tritonSectionFusion(true).tritonCompileAll(true);
    }

    @ParameterizedTest(name = "flagBisection[{0}]")
    @MethodSource("bisectionConfigs")
    @DisplayName("Flag bisection: TRITON_NO_GC → OPTIMAL")
    public void testOptimalFlagBisection(BenchmarkConfig config) throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available — skipping bisection test");
            return;
        }
        ensureModelsLoaded();
        int maxTokens = config.getMaxTokens();

        // Reference: SLOT_BY_SLOT decode
        GenerationResult refResult = runDecode(
                BenchmarkConfig.create("BISECT_REF")
                        .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                        .maxTokens(maxTokens),
                maxTokens);

        // Test: bisection config
        GenerationResult testResult = runDecode(config, maxTokens);

        int[] refTokens = refResult.getTokenIds();
        int[] testTokens = testResult.getTokenIds();
        int minLen = Math.min(refTokens.length, testTokens.length);

        int matches = 0;
        int firstDivergent = -1;
        for (int i = 0; i < minLen; i++) {
            if (refTokens[i] == testTokens[i]) {
                matches++;
            } else if (firstDivergent < 0) {
                firstDivergent = i;
            }
        }

        double matchRate = minLen > 0 ? (double) matches / minLen : 0;
        String refText = tokenizer.decode(refTokens);
        String testText = tokenizer.decode(testTokens);

        // Log result with clear PASS/FAIL for easy grep
        boolean hasTf32 = config.isCublasTf32() || config.isTritonTf32();
        double threshold = hasTf32 ? 0.15 : 0.90;
        boolean passed = matchRate >= threshold;

        log.info("[BISECT] {} — match={}/{} ({}%) {} (threshold={}%)",
                config.getName(), matches, minLen,
                String.format("%.1f", matchRate * 100),
                passed ? "PASS" : "FAIL",
                String.format("%.0f", threshold * 100));
        log.info("[BISECT] {} — ref: {}", config.getName(), refText);
        log.info("[BISECT] {} — got: {}", config.getName(), testText);
        if (firstDivergent >= 0) {
            log.info("[BISECT] {} — first divergence at step {}: ref={} test={}",
                    config.getName(), firstDivergent, refTokens[firstDivergent], testTokens[firstDivergent]);
        }

        // Assert correctness — this is NOT lenient. Every config should match the
        // baseline except TF32 configs which have precision reduction.
        assertTrue(passed,
                String.format("[BISECT] %s FAILED: match=%d/%d (%.1f%%), threshold=%.0f%%. " +
                              "Ref: %s | Got: %s",
                        config.getName(), matches, minLen, matchRate * 100,
                        threshold * 100, refText, testText));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DSP Pipeline Introspection Tests
    //
    // These tests use the DspHandle and DspPlanAssertions APIs to
    // programmatically verify pipeline state instead of log-parsing.
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Verify D2D copy integrity across decode steps.
     * After decode, captures a StepSnapshot and asserts:
     * - All D2D copies fired for variable ext inputs
     * - No address drift in any segment
     * - No pointer drift
     *
     * Answers Q1 (D2D fired?), Q2 (addresses match?), Q10 (staging==original bug?)
     */
    @Test
    @DisplayName("D2D Copy Integrity — assert D2D copies fire and pointers are stable")
    void testD2DCopyIntegrity() throws Exception {
        ensureModelsLoaded();
        int maxTokens = getTokens(5);
        log.info("=== testD2DCopyIntegrity: maxTokens={} ===", maxTokens);

        runDecode(BenchmarkConfig.optimal().maxTokens(maxTokens), maxTokens, result -> {
            int[] tokens = result.getTokenIds();
            assertNotNull(tokens, "decode returned null");
            assertTrue(tokens.length > 0, "decode returned 0 tokens");

            DspHandle h = decoder.dsp();
            assertTrue(h.isCompiled(), "plan should be compiled after decode");

            // Capture snapshot and verify D2D status
            DspHandle.StepSnapshot snap = h.captureStepSnapshot();
            log.info("StepSnapshot: {}", snap);

            // Assert all D2D copies fired
            if (!snap.d2dStatusByExtIdx.isEmpty()) {
                DspPlanAssertions.assertAllD2DCopiesFired(decoder, "post-decode");
                log.info("D2D: {}/{} copies fired",
                        snap.d2dStatusByExtIdx.values().stream()
                                .filter(s -> s.fired).count(),
                        snap.d2dStatusByExtIdx.size());
            }

            // Assert no address drift
            DspPlanAssertions.assertNoStagingAddressDrift(decoder, "post-decode");
            DspPlanAssertions.assertNoAddressDrift(decoder, "post-decode");

            // Log D2D status for each variable ext input
            for (Map.Entry<Integer, DspHandle.D2DStatus> e : snap.d2dStatusByExtIdx.entrySet()) {
                DspHandle.D2DStatus s = e.getValue();
                log.info("  {}", s);
            }

            // Log pointer drift status
            Map<Integer, Boolean> ptrMatch = h.allSegmentsPointersMatch();
            for (Map.Entry<Integer, Boolean> e : ptrMatch.entrySet()) {
                if (!e.getValue()) {
                    log.error("POINTER DRIFT seg[{}]: {}", e.getKey(),
                            h.segmentTrackedPointersJson(e.getKey()));
                }
            }
        });

        log.info("=== testD2DCopyIntegrity PASSED ===");
    }

    /**
     * Verify capture completeness after reaching replay phase.
     * Asserts:
     * - No permanent capture failures
     * - Capture stats are healthy
     *
     * Answers Q8 (why capture failed?), Q9 (which ops escaped?)
     */
    @Test
    @DisplayName("Capture Completeness — assert no perm failures, report host-only ops")
    void testCaptureCompleteness() throws Exception {
        ensureModelsLoaded();
        int maxTokens = getTokens(5);
        log.info("=== testCaptureCompleteness: maxTokens={} ===", maxTokens);

        runDecode(BenchmarkConfig.optimal().maxTokens(maxTokens), maxTokens, result -> {
            assertNotNull(result.getTokenIds(), "decode returned null");

            DspHandle h = decoder.dsp();
            assertTrue(h.isCompiled(), "plan should be compiled after decode");

            // Log capture stats
            DspHandle.CaptureStats cs = h.parsedCaptureStats();
            log.info("Capture stats: {}", cs);

            // Assert no permanent failures
            DspPlanAssertions.assertZeroPermCaptureFailures(decoder, "post-decode");

            // Check host-only ops
            int hostOps = h.numHostOnlyOps();
            log.info("Host-only ops: {}", hostOps);
            if (hostOps > 0) {
                log.warn("Ops that escaped capture: {}", h.hostOnlyOpNames());
            }

            // Log segment details
            int numSegs = h.numSegments();
            log.info("Segments: {}", numSegs);
            for (int s = 0; s < numSegs; s++) {
                log.info("  seg[{}]: backend={} phase={} replayCount={} capturable={} failed={}",
                        s, h.segmentBackendName(s), h.segmentExecutionPhase(s),
                        h.segmentReplayCount(s), h.isSegmentCapturable(s),
                        h.isSegmentCaptureFailed(s));
            }

            // Log plan lifecycle state
            log.info("Plan: phase={} ptrsStable={} frozenExec={} sealed={} replays={}",
                    h.planPhase(), h.pointersStable(), h.frozenExecCount(),
                    h.isCompilationSealed(), h.totalGraphReplays());
        });

        log.info("=== testCaptureCompleteness PASSED ===");
    }

    /**
     * Verify output integrity across decode steps.
     * Uses validateOutputs() to check for NaN/Inf/null/all-zero outputs,
     * and checks for stuck tokens (stale output symptom).
     *
     * Answers Q6 (output stale?), Q7 (output corrupt?)
     */
    @Test
    @DisplayName("Output Staleness Detection — assert outputs are fresh and valid per step")
    void testOutputStalenessDetection() throws Exception {
        ensureModelsLoaded();
        int maxTokens = getTokens(5);
        log.info("=== testOutputStalenessDetection: maxTokens={} ===", maxTokens);

        runDecode(BenchmarkConfig.optimal().maxTokens(maxTokens), maxTokens, result -> {
            int[] tokens = result.getTokenIds();
            assertNotNull(tokens, "decode returned null");
            assertTrue(tokens.length > 0, "decode returned 0 tokens");

            DspHandle h = decoder.dsp();
            assertTrue(h.isCompiled(), "plan should be compiled after decode");

            // Validate outputs
            int[] flags = h.validateOutputs();
            boolean anyIssues = false;
            for (int i = 0; i < flags.length; i++) {
                if (flags[i] != 0) {
                    log.error("Output[{}] has issues: flags=0x{}", i, Integer.toHexString(flags[i]));
                    anyIssues = true;
                }
            }
            assertFalse(anyIssues, "outputs should be valid (no NaN/Inf/null/all-zero)");

            // Log full state snapshot for diagnosis
            String fullState = DspPlanAssertions.snapshotFullState(decoder);
            log.info("Full plan state:\n{}", fullState);

            // Check for stuck tokens (stale output symptom)
            if (tokens.length >= 4) {
                boolean allSame = true;
                for (int i = 2; i < tokens.length; i++) {
                    if (tokens[i] != tokens[1]) {
                        allSame = false;
                        break;
                    }
                }
                assertFalse(allSame,
                        "all tokens after step 1 are identical (" + tokens[1] + ") — " +
                        "stale output suspected. Full state:\n" + fullState);
            }

            log.info("Generated {} tokens: {}", tokens.length, Arrays.toString(tokens));
        });
        log.info("=== testOutputStalenessDetection PASSED ===");
    }
}
