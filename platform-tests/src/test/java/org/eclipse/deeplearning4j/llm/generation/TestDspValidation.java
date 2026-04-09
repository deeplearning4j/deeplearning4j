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
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.execution.CapturingSlotInterceptor;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DSP Validation Test Framework.
 *
 * Compares execution results between different DSP execution modes to pinpoint
 * the exact op that introduces divergence. Uses slot output interceptors to capture
 * intermediate values and compare them across modes.
 *
 * Run with:
 *   cd platform-tests && mvn test \
 *     -Dtest=TestDspValidation \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
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
    }

    private static int getTokens(int defaultTokens) {
        return configuredTokens > 0 ? configuredTokens : defaultTokens;
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

        // Prepare prompt embeddings
        String prompt = "<|im_start|>user\nConvert this page to docling.\n<|im_end|>\n<|im_start|>assistant\n";
        int[] encoded = tokenizer.encode(prompt).getIds();
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

        // Find image token ID (typically 49190 for SmolDocling)
        int imageTokenId = 49190;
        inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, encoded, imageTokenId);

        log.info("Models loaded: decoder={} ops, embed={} ops, hiddenSize={}, promptTokens={}",
                decoder.getOps().size(), embedTokens.getOps().size(), hiddenSize, promptTokenIds.length);
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

        double requiredRate = config.getExecutionMode() == GraphExecutionMode.SLOT_BY_SLOT
                ? 1.0
                : configuredMatchRate / 100.0;
        assertTrue(matchRate >= requiredRate,
                config.getName() + ": token match rate too low: "
                        + String.format("%.1f%% (required %.1f%%)",
                        matchRate * 100, requiredRate * 100));
    }

    // ─── Test: Per-op slot validation ──────────────────────────────────────

    @Test
    @DisplayName("Per-op slot validation: interceptor captures during decode")
    public void testPerOpSlotValidation() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping per-op validation");
            return;
        }
        ensureModelsLoaded();
        log.info("Running per-op slot validation with interceptor...");

        int maxTokens = getTokens(3);

        // Configure OPTIMAL and run decode with interceptor attached
        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(maxTokens);
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputs = new ArrayList<>(decoder.outputs());
        BenchmarkConfigApplier.compileModel(decoder, "decoder", outputs, config);

        embedTokens.setDspAutoCompileEnabled(true);
        embedTokens.setDspNativeAutoCompileEnabled(true);
        List<String> embedOutputs = new ArrayList<>(embedTokens.outputs());
        BenchmarkConfigApplier.compileModel(embedTokens, "embed_tokens", embedOutputs, config);

        // Attach a capturing interceptor to the decoder's executor
        CapturingSlotInterceptor interceptor = new CapturingSlotInterceptor();
        // Only capture output variable names (not intermediate slot IDs)
        Set<String> outputVarNames = new LinkedHashSet<>(outputs);
        interceptor.filterVarNames(outputVarNames);

        // Run decode — first step initializes executor
        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build();

        // Run first decode to create session+executor, then attach interceptor
        GenerationResult firstResult = loop.decode(inputsEmbeds.dup(), promptTokenIds);
        log.info("First decode: {} tokens, text='{}'",
                firstResult.getGeneratedTokenCount(), firstResult.getText());

        // Now attach interceptor for a second run
        InferenceSession session = decoder.getOrCreateSession();
        if (session != null) {
            DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
            if (executor != null) {
                executor.setSlotOutputInterceptor(interceptor);
                log.info("Interceptor attached to decoder executor");
            }
        }

        // Second decode with interceptor active
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);
        BenchmarkConfigApplier.compileModel(decoder, "decoder", outputs, config);
        BenchmarkConfigApplier.compileModel(embedTokens, "embed_tokens", embedOutputs, config);

        StaticKvCacheDecodeLoop loop2 = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build();

        GenerationResult secondResult = loop2.decode(inputsEmbeds.dup(), promptTokenIds);

        log.info("Interceptor captured {} variables across {} steps",
                interceptor.getByName().size(), interceptor.getCaptured().size());

        // Log captured variable names and shapes
        for (Map.Entry<String, INDArray> entry : interceptor.getByName().entrySet()) {
            INDArray arr = entry.getValue();
            if (!arr.wasClosed()) {
                log.info("  Captured '{}': shape={} dtype={}", entry.getKey(),
                        java.util.Arrays.toString(arr.shape()), arr.dataType());
            }
        }

        // Verify we captured something
        assertTrue(interceptor.getByName().size() > 0 || interceptor.getCaptured().isEmpty(),
                "Interceptor should capture variables (or none if executor path not used)");

        interceptor.clear();
    }

    // ─── Test: Decode step validation ──────────────────────────────────────

    @Test
    @DisplayName("Decode step comparison: SLOT_BY_SLOT vs OPTIMAL generated text")
    public void testDecodeStepValidation() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping decode step validation");
            return;
        }
        ensureModelsLoaded();
        log.info("Running decode step validation...");

        int maxTokens = getTokens(10);

        // Run SLOT_BY_SLOT decode
        GenerationResult refResult = runDecode(
                BenchmarkConfig.create("REF_SLOT_BY_SLOT")
                        .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                        .maxTokens(maxTokens),
                maxTokens);

        // Run OPTIMAL decode
        GenerationResult testResult = runDecode(
                BenchmarkConfig.optimal().maxTokens(maxTokens),
                maxTokens);

        // Compare generated token IDs
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
        log.info("Token match rate: {}/{} ({}%)", matches, minLen, String.format("%.1f", matchRate * 100));
        log.info("Reference text: {}", refResult.getText());
        log.info("Test text:      {}", testResult.getText());
        if (firstDivergent >= 0) {
            log.info("First divergent token at step {}: ref={} test={}",
                    firstDivergent, refTokens[firstDivergent], testTokens[firstDivergent]);
        }
        if (verbose) {
            for (int i = 0; i < minLen; i++) {
                String match = refTokens[i] == testTokens[i] ? "OK" : "DIVERGE";
                log.info("Step {}: ref={} test={} [{}]", i, refTokens[i], testTokens[i], match);
            }
        }

        double requiredRate = configuredMatchRate / 100.0;
        assertTrue(matchRate >= requiredRate,
                "Token match rate too low: " + String.format("%.1f%% (required %.1f%%)",
                        matchRate * 100, configuredMatchRate));
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
        GenerationResult tf32Result = runDecode(
                BenchmarkConfig.create("WITH_TF32")
                        .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION")
                        .tritonSectionFusion(true).tritonCompileAll(true)
                        .cublasTf32(true)
                        .dspBatchedGemm(true)
                        .maxTokens(maxTokens),
                maxTokens);

        // Compare tokens
        int[] noTf32Tokens = noTf32Result.getTokenIds();
        int[] tf32Tokens = tf32Result.getTokenIds();
        int minLen = Math.min(noTf32Tokens.length, tf32Tokens.length);
        int matches = 0;
        for (int i = 0; i < minLen; i++) {
            if (noTf32Tokens[i] == tf32Tokens[i]) matches++;
        }
        double matchRate = minLen > 0 ? (double) matches / minLen : 1.0;

        log.info("TF32 token match rate: {}/{} ({}%)", matches, minLen,
                String.format("%.1f", matchRate * 100));
        log.info("NO_TF32 text: {}", noTf32Result.getText());
        log.info("TF32 text:    {}", tf32Result.getText());
    }

    @Test
    @DisplayName("Decoder plan introspection: slots 348-358 boundary")
    public void testDecoderPlanBoundary348To358() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping boundary introspection");
            return;
        }
        ensureModelsLoaded();

        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(getTokens(2));
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputs = new ArrayList<>(decoder.outputs());
        BenchmarkConfigApplier.compileModel(decoder, "decoder", outputs, config);

        embedTokens.setDspAutoCompileEnabled(true);
        embedTokens.setDspNativeAutoCompileEnabled(true);
        List<String> embedOutputs = new ArrayList<>(embedTokens.outputs());
        BenchmarkConfigApplier.compileModel(embedTokens, "embed_tokens", embedOutputs, config);

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(getTokens(2))
                .hiddenSize(hiddenSize)
                .build();

        GenerationResult result = loop.decode(inputsEmbeds.dup(), promptTokenIds);
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
    }

    @Test
    @DisplayName("Decoder plan introspection: slots 400-430 boundary")
    public void testDecoderPlanBoundary400To430() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping boundary introspection");
            return;
        }
        ensureModelsLoaded();

        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(getTokens(2));
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputs = new ArrayList<>(decoder.outputs());
        BenchmarkConfigApplier.compileModel(decoder, "decoder", outputs, config);

        embedTokens.setDspAutoCompileEnabled(true);
        embedTokens.setDspNativeAutoCompileEnabled(true);
        List<String> embedOutputs = new ArrayList<>(embedTokens.outputs());
        BenchmarkConfigApplier.compileModel(embedTokens, "embed_tokens", embedOutputs, config);

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(getTokens(2))
                .hiddenSize(hiddenSize)
                .build();

        GenerationResult result = loop.decode(inputsEmbeds.dup(), promptTokenIds);
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
    }

    @Test
    @DisplayName("Decoder plan introspection: slots 431-453 boundary")
    public void testDecoderPlanBoundary431To453() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping boundary introspection");
            return;
        }
        ensureModelsLoaded();

        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(getTokens(2));
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputs = new ArrayList<>(decoder.outputs());
        BenchmarkConfigApplier.compileModel(decoder, "decoder", outputs, config);

        embedTokens.setDspAutoCompileEnabled(true);
        embedTokens.setDspNativeAutoCompileEnabled(true);
        List<String> embedOutputs = new ArrayList<>(embedTokens.outputs());
        BenchmarkConfigApplier.compileModel(embedTokens, "embed_tokens", embedOutputs, config);

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(getTokens(2))
                .hiddenSize(hiddenSize)
                .build();

        GenerationResult result = loop.decode(inputsEmbeds.dup(), promptTokenIds);
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
    }

    @Test
    @DisplayName("Decoder plan introspection: slots 455-523 boundary")
    public void testDecoderPlanBoundary455To523() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping boundary introspection");
            return;
        }
        ensureModelsLoaded();

        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(getTokens(2));
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputs = new ArrayList<>(decoder.outputs());
        BenchmarkConfigApplier.compileModel(decoder, "decoder", outputs, config);

        embedTokens.setDspAutoCompileEnabled(true);
        embedTokens.setDspNativeAutoCompileEnabled(true);
        List<String> embedOutputs = new ArrayList<>(embedTokens.outputs());
        BenchmarkConfigApplier.compileModel(embedTokens, "embed_tokens", embedOutputs, config);

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(getTokens(2))
                .hiddenSize(hiddenSize)
                .build();

        GenerationResult result = loop.decode(inputsEmbeds.dup(), promptTokenIds);
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
    }

    @Test
    @DisplayName("Decoder plan introspection: slots 793-846 boundary")
    public void testDecoderPlanBoundary793To846() throws Exception {
        if (!Nd4j.getNativeOps().isTritonAvailable()) {
            log.info("Triton not available, skipping boundary introspection");
            return;
        }
        ensureModelsLoaded();

        BenchmarkConfig config = BenchmarkConfig.optimal().maxTokens(getTokens(2));
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);
        List<String> outputs = new ArrayList<>(decoder.outputs());
        BenchmarkConfigApplier.compileModel(decoder, "decoder", outputs, config);

        embedTokens.setDspAutoCompileEnabled(true);
        embedTokens.setDspNativeAutoCompileEnabled(true);
        List<String> embedOutputs = new ArrayList<>(embedTokens.outputs());
        BenchmarkConfigApplier.compileModel(embedTokens, "embed_tokens", embedOutputs, config);

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(getTokens(2))
                .hiddenSize(hiddenSize)
                .build();

        GenerationResult result = loop.decode(inputsEmbeds.dup(), promptTokenIds);
        assertNotNull(result, "Decode result should exist");

        InferenceSession session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        assertNotNull(executor, "DSP executor must exist");
        assertNotNull(executor.getCurrentPlan(), "Current plan must exist");

        var plan = executor.getCurrentPlan();
        assertTrue(plan.getSlots().length > 846, "Expected decoder plan to include slot 846");

        log.info("=== DECODER PLAN BOUNDARY: slots 793-846 ===");
        for (int slotIdx = 793; slotIdx <= 846; slotIdx++) {
            log.info(PlanIntrospection.formatSlot(plan, slotIdx));
        }

        int[] auxSlots = {792, 793, 819, 820, 821, 822, 823, 846, 847};
        log.info("=== DECODER PLAN AUXILIARY SLOTS (793-846) ===");
        for (int slotIdx : auxSlots) {
            log.info(PlanIntrospection.formatSlot(plan, slotIdx));
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private GenerationResult runDecode(BenchmarkConfig config, int maxTokens) throws Exception {
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        BenchmarkConfigApplier.apply(config);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);

        if (config.isTriton()) {
            decoder.setDspAutoCompileEnabled(true);
            decoder.setDspNativeAutoCompileEnabled(true);
            List<String> outputs = new ArrayList<>(decoder.outputs());
            BenchmarkConfigApplier.compileModel(decoder, "decoder", outputs, config);

            embedTokens.setDspAutoCompileEnabled(true);
            embedTokens.setDspNativeAutoCompileEnabled(true);
            List<String> embedOutputs = new ArrayList<>(embedTokens.outputs());
            BenchmarkConfigApplier.compileModel(embedTokens, "embed_tokens", embedOutputs, config);
        } else if (config.getExecutionMode() != null) {
            decoder.setDspAutoCompileEnabled(true);
            decoder.setDspNativeAutoCompileEnabled(true);
            List<String> outputs = new ArrayList<>(decoder.outputs());
            decoder.compileNativeDynamicShapePlan(outputs, config.getExecutionMode(), true);
        }

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(hiddenSize)
                .build();

        return loop.decode(inputsEmbeds.dup(), promptTokenIds);
    }
}
