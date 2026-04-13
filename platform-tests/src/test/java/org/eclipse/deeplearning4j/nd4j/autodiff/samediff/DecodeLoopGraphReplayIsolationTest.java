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
import org.eclipse.deeplearning4j.llm.generation.DecoderUtils;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Progressive isolation test for graph replay divergence in the decode pipeline.
 *
 * <p>Tests that pass with direct {@code decoder.outputDirect()} calls fail when
 * routed through {@link org.eclipse.deeplearning4j.llm.generation.StaticKvCacheDecodeLoop}.
 * This test progressively adds StaticKvCacheDecodeLoop features to the basic
 * (passing) decode flow to identify which feature triggers the divergence.</p>
 *
 * <h3>Features isolated (each test adds ONE to the baseline):</h3>
 * <ol>
 *   <li>Logits-only recompilation (recompile DSP with only logits as output names)</li>
 *   <li>C++ KV scatter via {@code configureKvCacheRetention()}</li>
 *   <li>{@code outputDirect()} instead of {@code output()}</li>
 *   <li>FP16 weight pre-casting via GraphOptimizer</li>
 *   <li>Logits-only + C++ KV scatter combined</li>
 *   <li>Full pipeline (all features combined)</li>
 * </ol>
 *
 * <p>For each test, we run 5 decode steps with graph capture OFF (baseline)
 * then 5 decode steps with graph capture ON (treatment), and compare tokens.</p>
 *
 * <p>Run:</p>
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=DecodeLoopGraphReplayIsolationTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 \
 *     2>&1 | tee /tmp/decode-isolation.log
 * </pre>
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class DecodeLoopGraphReplayIsolationTest {

    /**
     * 17-token prefill matching the VLM benchmark validation path.
     */
    private static final int[] PREFILL_TOKENS = {
            49229, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };

    private static final int NUM_DECODE_STEPS = 5;
    private static final long MAX_KV_LEN = 2048;

    private SameDiff decoder;
    private SameDiff embedTokens;
    private INDArray embeddingTable;
    private String logitsName;
    private ModelIOConfig ioConfig;
    private DecoderUtils.KVCacheNames kvNames;
    private boolean modelsLoaded = false;

    /** Summary of results for the final report. */
    private final Map<String, TestResult> testResults = new LinkedHashMap<>();

    // ========== Setup / Teardown ==========

    @BeforeAll
    public void loadModel() throws Exception {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        try {
            var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
            var embedResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);

            decoder = OnnxModelCache.importWithCache(decoderResult.getModelFile().getAbsolutePath());
            decoder.setDspAutoCompileEnabled(true);
            decoder.setDspNativeAutoCompileEnabled(true);

            embedTokens = OnnxModelCache.importWithCache(embedResult.getModelFile().getAbsolutePath());

            // Find embedding table (largest 2D constant)
            embeddingTable = null;
            long bestRows = 0;
            for (SDVariable var : embedTokens.variables()) {
                INDArray arr = embedTokens.getArrForVarName(var.name());
                if (arr != null && arr.rank() == 2 && arr.size(0) > bestRows) {
                    bestRows = arr.size(0);
                    embeddingTable = arr;
                }
            }
            assertNotNull(embeddingTable, "Could not find embedding table");

            // Discover IO config
            ioConfig = ModelIOConfig.discover(decoder);
            kvNames = ioConfig.getKvCacheNames();
            logitsName = ioConfig.getLogitsOutputName();

            modelsLoaded = true;
            log.info("Models loaded: decoder={} ops, embedTokens={} ops, logits={}, kvLayers={}",
                    decoder.ops().length, embedTokens.ops().length, logitsName,
                    kvNames.keyNames.size());
        } catch (Exception e) {
            log.error("Failed to load models: {}", e.getMessage(), e);
        }
    }

    @AfterAll
    public void teardown() {
        if (decoder != null) decoder.close();
        if (embedTokens != null) embedTokens.close();
    }

    @AfterEach
    public void cleanupAfterEach() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ========== Test 1: Baseline (direct decode, full outputs, output()) ==========

    @Test
    @Order(1)
    @DisplayName("1. Baseline: direct decode with graph capture (should pass)")
    public void testBasicDecodeWithGraphCapture() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags baseline = new FeatureFlags(); // all defaults = basic flow
        TestResult result = runComparison("1_BASELINE", baseline);
        testResults.put("1_BASELINE", result);

        assertTokensMatch("1_BASELINE", result);
    }

    // ========== Test 2: Logits-only recompilation ==========

    @Test
    @Order(2)
    @DisplayName("2. Logits-only recompile with graph capture")
    public void testLogitsOnlyRecompileWithGraphCapture() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.logitsOnlyRecompile = true;
        TestResult result = runComparison("2_LOGITS_ONLY_RECOMPILE", flags);
        testResults.put("2_LOGITS_ONLY_RECOMPILE", result);

        assertTokensMatch("2_LOGITS_ONLY_RECOMPILE", result);
    }

    // ========== Test 3: C++ KV scatter ==========

    @Test
    @Order(3)
    @DisplayName("3. C++ KV scatter with graph capture")
    public void testCppKvScatterWithGraphCapture() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.cppKvScatter = true;
        TestResult result = runComparison("3_CPP_KV_SCATTER", flags);
        testResults.put("3_CPP_KV_SCATTER", result);

        assertTokensMatch("3_CPP_KV_SCATTER", result);
    }

    // ========== Test 4: Logits-only + C++ KV scatter ==========

    @Test
    @Order(4)
    @DisplayName("4. Logits-only + C++ KV scatter with graph capture")
    public void testLogitsOnlyPlusCppKvScatter() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.logitsOnlyRecompile = true;
        flags.cppKvScatter = true;
        TestResult result = runComparison("4_LOGITS_ONLY_PLUS_CPP_KV", flags);
        testResults.put("4_LOGITS_ONLY_PLUS_CPP_KV", result);

        assertTokensMatch("4_LOGITS_ONLY_PLUS_CPP_KV", result);
    }

    // ========== Test 5: outputDirect() ==========

    @Test
    @Order(5)
    @DisplayName("5. outputDirect() with graph capture")
    public void testOutputDirectWithGraphCapture() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.useOutputDirect = true;
        TestResult result = runComparison("5_OUTPUT_DIRECT", flags);
        testResults.put("5_OUTPUT_DIRECT", result);

        assertTokensMatch("5_OUTPUT_DIRECT", result);
    }

    // ========== Test 6: FP16 pre-cast ==========

    @Test
    @Order(6)
    @DisplayName("6. FP16 pre-cast with graph capture")
    public void testFP16PreCastWithGraphCapture() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.fp16PreCast = true;
        TestResult result = runComparison("6_FP16_PRECAST", flags);
        testResults.put("6_FP16_PRECAST", result);

        assertTokensMatch("6_FP16_PRECAST", result);
    }

    // ========== Test 7: Full pipeline (17 tokens) ==========

    @Test
    @Order(7)
    @DisplayName("7. Full pipeline with 17-token prefill")
    public void testFullPipelineWithGraphCapture() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.logitsOnlyRecompile = true;
        flags.cppKvScatter = true;
        flags.useOutputDirect = true;
        flags.fp16PreCast = true;
        TestResult result = runComparison("7_FULL_17tok", flags);
        testResults.put("7_FULL_17tok", result);

        assertTokensMatch("7_FULL_17tok", result);
    }

    // ========== Test 8: Full pipeline with 200 tokens ==========

    @Test
    @Order(8)
    @DisplayName("8. Full pipeline with 200-token prefill")
    public void testFullPipeline200Tokens() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.logitsOnlyRecompile = true;
        flags.cppKvScatter = true;
        flags.useOutputDirect = true;
        flags.fp16PreCast = true;
        flags.prefillSize = 200;
        TestResult result = runComparison("8_FULL_200tok", flags);
        testResults.put("8_FULL_200tok", result);

        assertTokensMatch("8_FULL_200tok", result);
    }

    // ========== Test 9: Full pipeline with 500 tokens ==========

    @Test
    @Order(9)
    @DisplayName("9. Full pipeline with 500-token prefill")
    public void testFullPipeline500Tokens() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.logitsOnlyRecompile = true;
        flags.cppKvScatter = true;
        flags.useOutputDirect = true;
        flags.fp16PreCast = true;
        flags.prefillSize = 500;
        TestResult result = runComparison("9_FULL_500tok", flags);
        testResults.put("9_FULL_500tok", result);

        assertTokensMatch("9_FULL_500tok", result);
    }

    // ========== Test 10: C++ KV scatter only with 500 tokens ==========

    @Test
    @Order(10)
    @DisplayName("10. C++ KV scatter + logits-only with 500-token prefill")
    public void testCppKvScatter500Tokens() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.logitsOnlyRecompile = true;
        flags.cppKvScatter = true;
        flags.prefillSize = 500;
        TestResult result = runComparison("10_KV_SCATTER_500tok", flags);
        testResults.put("10_KV_SCATTER_500tok", result);

        assertTokensMatch("10_KV_SCATTER_500tok", result);
    }

    // ========== Test 11: Pre-compilation before prefill (mirrors VisionEmbedGraphReplayTest) ==========

    @Test
    @Order(11)
    @DisplayName("11. Pre-compile + prefill + recompile + decode (VisionEmbed flow)")
    public void testPreCompilePrefillRecompileDecode() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.logitsOnlyRecompile = true;
        flags.cppKvScatter = true;
        flags.useOutputDirect = true;
        flags.fp16PreCast = true;
        flags.prefillSize = 500;
        flags.preCompileBeforePrefill = true;  // KEY DIFFERENCE
        TestResult result = runComparison("11_PRECOMPILE_500tok", flags);
        testResults.put("11_PRECOMPILE_500tok", result);

        assertTokensMatch("11_PRECOMPILE_500tok", result);
    }

    // ========== Test 12: Pre-compile + 17 token prefill ==========

    @Test
    @Order(12)
    @DisplayName("12. Pre-compile + prefill + recompile + decode (17 tokens)")
    public void testPreCompilePrefillRecompile17() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.logitsOnlyRecompile = true;
        flags.cppKvScatter = true;
        flags.useOutputDirect = true;
        flags.fp16PreCast = true;
        flags.preCompileBeforePrefill = true;  // KEY DIFFERENCE
        TestResult result = runComparison("12_PRECOMPILE_17tok", flags);
        testResults.put("12_PRECOMPILE_17tok", result);

        assertTokensMatch("12_PRECOMPILE_17tok", result);
    }

    // ========== Test 13: More decode steps (15) to catch late divergence ==========

    @Test
    @Order(13)
    @DisplayName("13. Full pipeline with 15 decode steps (catch late divergence)")
    public void testFullPipeline15Steps() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        FeatureFlags flags = new FeatureFlags();
        flags.logitsOnlyRecompile = true;
        flags.cppKvScatter = true;
        flags.useOutputDirect = true;
        flags.fp16PreCast = true;
        flags.preCompileBeforePrefill = true;
        flags.numDecodeSteps = 15;
        TestResult result = runComparison("13_PRECOMPILE_15steps", flags);
        testResults.put("13_PRECOMPILE_15steps", result);

        assertTokensMatch("13_PRECOMPILE_15steps", result);
    }

    // ========== Test 14: Summary report ==========

    @Test
    @Order(14)
    @DisplayName("14. Summary: which feature first causes divergence")
    public void testSummaryReport() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        log.info("========================================================================");
        log.info("  DECODE LOOP GRAPH REPLAY ISOLATION SUMMARY");
        log.info("========================================================================");

        String firstFailure = null;
        for (Map.Entry<String, TestResult> entry : testResults.entrySet()) {
            String name = entry.getKey();
            TestResult r = entry.getValue();
            String status;
            if (r.allMatch) {
                status = String.format("PASS (all %d tokens match)", r.totalTokens);
            } else {
                status = String.format("DIVERGE at step %d (%d/%d match, degenerate=%s)",
                        r.firstDivergenceStep, r.matchCount, r.totalTokens, r.degenerate);
                if (firstFailure == null) firstFailure = name;
            }
            log.info("  {} | {}", String.format("%-35s", name), status);
        }

        log.info("------------------------------------------------------------------------");
        if (firstFailure != null) {
            log.info("  FIRST FAILING FEATURE: {}", firstFailure);
            TestResult failResult = testResults.get(firstFailure);
            log.info("  Baseline tokens:  {}", failResult.baselineTokens);
            log.info("  Treatment tokens: {}", failResult.treatmentTokens);
        } else {
            log.info("  NO DIVERGENCE DETECTED -- all features pass individually");
            log.info("  (The bug may require a specific combination not tested here)");
        }
        log.info("========================================================================");
    }

    // ========== Core Logic ==========

    /**
     * Feature flags controlling which StaticKvCacheDecodeLoop features to enable.
     */
    private static class FeatureFlags {
        boolean logitsOnlyRecompile = false;
        boolean cppKvScatter = false;
        boolean useOutputDirect = false;
        boolean fp16PreCast = false;
        /** Prefill token count. 0 means use the default PREFILL_TOKENS array. */
        int prefillSize = 0;
        /** When true, compile DSP plan BEFORE prefill (mirrors VisionEmbedGraphReplayTest flow). */
        boolean preCompileBeforePrefill = false;
        /** Number of decode steps. 0 means use the default NUM_DECODE_STEPS. */
        int numDecodeSteps = 0;

        @Override
        public String toString() {
            List<String> enabled = new ArrayList<>();
            if (logitsOnlyRecompile) enabled.add("logitsOnly");
            if (cppKvScatter) enabled.add("cppKvScatter");
            if (useOutputDirect) enabled.add("outputDirect");
            if (fp16PreCast) enabled.add("fp16PreCast");
            if (prefillSize > 0) enabled.add("prefill=" + prefillSize);
            if (preCompileBeforePrefill) enabled.add("preCompile");
            if (numDecodeSteps > 0) enabled.add("steps=" + numDecodeSteps);
            return enabled.isEmpty() ? "BASELINE" : String.join("+", enabled);
        }
    }

    /**
     * Result of comparing graph-capture-OFF vs graph-capture-ON for one configuration.
     */
    private static class TestResult {
        List<Integer> baselineTokens;
        List<Integer> treatmentTokens;
        List<Double> baselineChecksums;
        List<Double> treatmentChecksums;
        int totalTokens;
        int matchCount;
        boolean allMatch;
        int firstDivergenceStep = -1;
        boolean degenerate = false;
    }

    /**
     * Run both baseline (graph capture OFF) and treatment (graph capture ON) with
     * the specified feature flags, then compare results.
     */
    private TestResult runComparison(String label, FeatureFlags flags) {
        log.info("========== {} (features: {}) ==========", label, flags);

        Environment env = Nd4j.getEnvironment();

        // Save original environment
        boolean origGraphCapture = env.tritonGraphCapture();
        boolean origCompileAll = env.tritonCompileAll();
        boolean origSectionFusion = env.tritonSectionFusion();
        boolean origConsolidated = env.tritonConsolidatedArgTable();
        boolean origDirtyTracking = env.tritonArgDirtyTracking();
        boolean origCublasTf32 = env.cublasTf32Enabled();
        boolean origTritonTf32 = env.tritonTf32Enabled();
        boolean origBatchedGemm = env.dspBatchedGemm();
        boolean origFusionScoring = env.tritonFusionScoring();
        String origIncludeTypes = env.tritonIncludeTypes();
        int origNumWarps = env.tritonNumWarps();
        int origNumStages = env.tritonNumStages();

        try {
            // ---- Baseline: Triton ON, graph capture OFF ----
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(false);
            env.setTritonConsolidatedArgTable(false);
            env.setTritonArgDirtyTracking(false);
            env.setCublasTf32Enabled(false);
            env.setTritonTf32Enabled(false);
            env.setDspBatchedGemm(false);
            env.setTritonFusionScoring(true);
            log.info("[{}] Running BASELINE (graphCapture=OFF, features={})", label, flags);

            DecodeResult baseline = runDecodeSequence(
                    label + "_BASELINE", flags);
            log.info("[{}] Baseline tokens: {}", label, baseline.tokens);

            // ---- Treatment: Full OPTIMAL config (graph capture ON) ----
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(true);
            env.setTritonConsolidatedArgTable(true);
            env.setTritonArgDirtyTracking(true);
            env.setCublasTf32Enabled(true);
            env.setTritonTf32Enabled(true);
            env.setDspBatchedGemm(true);
            env.setTritonFusionScoring(false);
            env.setTritonNumWarps(4);
            env.setTritonNumStages(1);
            log.info("[{}] Running TREATMENT (graphCapture=ON, features={})", label, flags);

            DecodeResult treatment = runDecodeSequence(
                    label + "_TREATMENT", flags);
            log.info("[{}] Treatment tokens: {}", label, treatment.tokens);

            // Compare
            return compareResults(label, baseline, treatment);
        } finally {
            env.setTritonGraphCapture(origGraphCapture);
            env.setTritonCompileAll(origCompileAll);
            env.setTritonSectionFusion(origSectionFusion);
            env.setTritonConsolidatedArgTable(origConsolidated);
            env.setTritonArgDirtyTracking(origDirtyTracking);
            env.setCublasTf32Enabled(origCublasTf32);
            env.setTritonTf32Enabled(origTritonTf32);
            env.setDspBatchedGemm(origBatchedGemm);
            env.setTritonFusionScoring(origFusionScoring);
            env.setTritonIncludeTypes(origIncludeTypes);
            env.setTritonNumWarps(origNumWarps);
            env.setTritonNumStages(origNumStages);
        }
    }

    /**
     * Run the decode sequence with current env settings and the given feature flags.
     *
     * <p>Steps:</p>
     * <ol>
     *   <li>Reset model state</li>
     *   <li>Apply FP16 pre-cast if requested</li>
     *   <li>Run 17-token prefill</li>
     *   <li>Initialize static KV cache</li>
     *   <li>Recompile DSP for seqLen=1 decode (logits-only if requested)</li>
     *   <li>Configure C++ KV scatter if requested</li>
     *   <li>Run NUM_DECODE_STEPS decode steps</li>
     * </ol>
     */
    private DecodeResult runDecodeSequence(String label, FeatureFlags flags) {
        DecodeResult result = new DecodeResult();

        // Reset state
        decoder.resetSession();
        embedTokens.resetSession();
        InferenceSession.setDynamicShapePlanEnabled(true);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        // FP16 pre-casting: applies GraphOptimizer to cast FLOAT constants to HALF
        // This is done by the VisionEmbedGraphReplayTest / benchmark via
        // System.setProperty("nd4j.optimizer.fp16", "true") before OnnxModelCache.import.
        // For isolation, we do NOT re-import. Instead, we cast constant arrays in-place.
        if (flags.fp16PreCast) {
            int castCount = 0;
            for (SDVariable var : decoder.variables()) {
                if (var.getVariableType() == org.nd4j.autodiff.samediff.VariableType.CONSTANT) {
                    INDArray arr = decoder.getArrForVarName(var.name());
                    if (arr != null && arr.dataType() == DataType.FLOAT && arr.rank() >= 2) {
                        INDArray halfArr = arr.castTo(DataType.HALF);
                        decoder.associateArrayWithVariable(halfArr, var.name());
                        castCount++;
                    }
                }
            }
            log.info("[{}] FP16 pre-cast: converted {} FLOAT constants to HALF", label, castCount);
        }

        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};

        // Decide which outputs to compile with and request at decode time
        // When logitsOnlyRecompile is true, we recompile with logits-only outputs
        // (like StaticKvCacheDecodeLoop does after KV retention is configured)
        String[] decodeCompileOutputs = flags.logitsOnlyRecompile ? logitsOnlyOutputNames : fullOutputNames;
        // When cppKvScatter is ON, decode requests logits-only (C++ scatters KV internally)
        String[] decodeRequestOutputs = flags.cppKvScatter ? logitsOnlyOutputNames : fullOutputNames;

        String embedOutputName = embedTokens.outputs().get(0);

        // ---- Prefill ----
        // Use variable prefill size if specified, otherwise use the default 17-token array
        int[] prefillTokenIds = (flags.prefillSize > 0)
                ? generatePrefillTokens(flags.prefillSize)
                : PREFILL_TOKENS;

        INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                .reshape(1, prefillTokenIds.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        log.info("[{}] Prefill: {} tokens (preCompile={})", label, prefillSeqLen, flags.preCompileBeforePrefill);

        // Pre-compile DSP plan BEFORE prefill if requested.
        // This mirrors the VisionEmbedGraphReplayTest flow where BenchmarkConfigApplier.compileModels()
        // is called before StaticKvCacheDecodeLoop.decode().
        if (flags.preCompileBeforePrefill) {
            decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, fullOutputNames);
            log.info("[{}] Pre-compiled DSP plan before prefill", label);
        }

        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, embeddingTable.size(1));

        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, label + ": prefill logits null");

        // Extract first token
        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1),
                NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        result.tokens.add(nextToken);
        result.logitChecksums.add(prefillLogits.sumNumber().doubleValue());

        log.info("[{}] Prefill token: {} logitSum={}", label, nextToken, result.logitChecksums.get(0));

        // Initialize static KV cache
        StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
        kvMgr.initializeFromPrefill(prefillOutputs);

        // Close prefill outputs
        closePrefillOutputs(prefillOutputs, prefillLogits);

        // ---- Recompile for seqLen=1 decode ----
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();

        // Associate static KV buffers before compilation
        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        // Compile with the selected outputs
        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, decodeCompileOutputs);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        // Configure frozen shapes + C++ KV scatter
        boolean cppKvScatterActive = false;
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);

            if (flags.cppKvScatter && dspExec.getCurrentPlan() != null) {
                List<String> presentNames = new ArrayList<>();
                presentNames.addAll(kvNames.keyNames);
                presentNames.addAll(kvNames.valueNames);
                List<String> pastNames = new ArrayList<>();
                for (String pn : presentNames) {
                    pastNames.add(ioConfig.presentToInputName(pn));
                }
                cppKvScatterActive = dspExec.configureKvCacheRetention(
                        dspExec.getCurrentPlan(), presentNames, pastNames,
                        (int) kvMgr.getMaxKvLen(), (int) kvMgr.getCachePosition());
                log.info("[{}] C++ KV scatter: {}", label, cppKvScatterActive);

                if (cppKvScatterActive) {
                    dspExec.configureDecodeInputs(dspExec.getCurrentPlan(), (int) kvMgr.getMaxKvLen());
                }
            }
        }

        // ---- Decode steps ----
        Map<String, INDArray> reusableInputs = new HashMap<>();

        int actualDecodeSteps = flags.numDecodeSteps > 0 ? flags.numDecodeSteps : NUM_DECODE_STEPS;
        for (int step = 0; step < actualDecodeSteps; step++) {
            long pastSeqLen = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();

            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken})
                    .reshape(1, 1)
                    .castTo(DataType.LONG);

            // Get token embedding
            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            // Build decode inputs
            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, stepEmbed, tokenIdArr,
                    pastSeqLen, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, embeddingTable.size(1), reusableInputs, true);

            // Execute with the selected method
            Map<String, INDArray> outputs;
            if (flags.useOutputDirect) {
                outputs = decoder.outputDirect(decodeInputs, decodeRequestOutputs);
            } else {
                outputs = decoder.output(decodeInputs, decodeRequestOutputs);
            }

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, label + ": step " + step + " logits null");

            // Greedy sample
            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(stepLogits.size(1) - 1),
                    NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            result.tokens.add(nextToken);

            double checksum = stepLogits.sumNumber().doubleValue();
            result.logitChecksums.add(checksum);

            log.info("[{}] Step {}: token={} logitSum={} cachePos={}",
                    label, step, nextToken, checksum, cachePos);

            // KV cache management
            if (cppKvScatterActive) {
                kvMgr.advancePosition();
            } else {
                kvMgr.scatterNewEntries(outputs);
            }
        }

        return result;
    }

    // ========== Comparison ==========

    private TestResult compareResults(String label, DecodeResult baseline, DecodeResult treatment) {
        TestResult result = new TestResult();
        result.baselineTokens = baseline.tokens;
        result.treatmentTokens = treatment.tokens;
        result.baselineChecksums = baseline.logitChecksums;
        result.treatmentChecksums = treatment.logitChecksums;
        result.totalTokens = baseline.tokens.size();

        int firstDivergence = -1;
        int matchCount = 0;
        for (int i = 0; i < Math.min(baseline.tokens.size(), treatment.tokens.size()); i++) {
            if (baseline.tokens.get(i).equals(treatment.tokens.get(i))) {
                matchCount++;
            } else if (firstDivergence == -1) {
                firstDivergence = i;
            }
        }

        result.matchCount = matchCount;
        result.allMatch = (matchCount == baseline.tokens.size());
        result.firstDivergenceStep = firstDivergence;

        log.info("[{}] Match: {}/{} ({}%)", label, matchCount, baseline.tokens.size(),
                String.format("%.1f", (double) matchCount / baseline.tokens.size() * 100.0));

        if (firstDivergence >= 0) {
            String stepLabel = firstDivergence == 0 ? "prefill" : "decode step " + (firstDivergence - 1);
            log.error("[{}] DIVERGENCE at index {} ({}): baseline={} treatment={}",
                    label, firstDivergence, stepLabel,
                    baseline.tokens.get(firstDivergence), treatment.tokens.get(firstDivergence));
            log.error("[{}] Baseline tokens:  {}", label, baseline.tokens);
            log.error("[{}] Treatment tokens: {}", label, treatment.tokens);

            // Log checksum comparison around divergence
            int logStart = Math.max(0, firstDivergence - 1);
            int logEnd = Math.min(baseline.logitChecksums.size(), firstDivergence + 3);
            for (int i = logStart; i < logEnd; i++) {
                String bSum = i < baseline.logitChecksums.size()
                        ? String.valueOf(baseline.logitChecksums.get(i)) : "N/A";
                String tSum = i < treatment.logitChecksums.size()
                        ? String.valueOf(treatment.logitChecksums.get(i)) : "N/A";
                log.error("[{}] Checksum at step {}: baseline={} treatment={}", label, i, bSum, tSum);
            }

            // Check degenerate output
            boolean degenerate = true;
            for (int i = firstDivergence + 1; i < treatment.tokens.size(); i++) {
                if (!treatment.tokens.get(i).equals(treatment.tokens.get(firstDivergence))) {
                    degenerate = false;
                    break;
                }
            }
            if (degenerate && treatment.tokens.size() - firstDivergence > 2) {
                result.degenerate = true;
                log.error("[{}] Treatment output is DEGENERATE: repeating token {} from step {}",
                        label, treatment.tokens.get(firstDivergence), firstDivergence);
            }
        }

        return result;
    }

    private void assertTokensMatch(String testName, TestResult result) {
        for (int i = 0; i < result.baselineTokens.size(); i++) {
            assertEquals(result.baselineTokens.get(i), result.treatmentTokens.get(i),
                    String.format("[%s] Token mismatch at index %d (%s): baseline=%d treatment=%d. " +
                                    "Baseline logitSum=%s, Treatment logitSum=%s",
                            testName, i,
                            i == 0 ? "prefill" : "decode step " + (i - 1),
                            result.baselineTokens.get(i), result.treatmentTokens.get(i),
                            i < result.baselineChecksums.size() ? String.valueOf(result.baselineChecksums.get(i)) : "?",
                            i < result.treatmentChecksums.size() ? String.valueOf(result.treatmentChecksums.get(i)) : "?"));
        }
    }

    // ========== Data classes ==========

    private static class DecodeResult {
        List<Integer> tokens = new ArrayList<>();
        List<Double> logitChecksums = new ArrayList<>();
    }

    // ========== Helpers ==========

    /**
     * Generate a deterministic token sequence of the requested length.
     * Starts with doctag (49229), then cycles through common structural tokens.
     */
    private int[] generatePrefillTokens(int length) {
        int vocabSize = (int) embeddingTable.size(0);
        int[] tokens = new int[length];
        tokens[0] = Math.min(49229, vocabSize - 1);
        int[] seedTokens = {1, 42, 100, 256, 500, 1000, 2000, 3000, 4000, 5000,
                6000, 7000, 8000, 9000, 10000, 11126, 12000, 13000, 14000, 15000};
        for (int i = 1; i < length; i++) {
            int seedIdx = (i - 1) % seedTokens.length;
            tokens[i] = Math.min(seedTokens[seedIdx], vocabSize - 1);
        }
        return tokens;
    }

    private INDArray buildPrefillEmbeddings(int[] tokens) {
        long hiddenSize = embeddingTable.size(1);
        INDArray result = Nd4j.create(DataType.FLOAT, 1, tokens.length, hiddenSize);
        int vocabSize = (int) embeddingTable.size(0);
        for (int i = 0; i < tokens.length; i++) {
            int tokenId = tokens[i];
            if (tokenId >= vocabSize) tokenId = tokenId % vocabSize;
            result.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all())
                    .assign(embeddingTable.getRow(tokenId));
        }
        return result;
    }

    private String[] buildFullOutputNames() {
        List<String> names = new ArrayList<>();
        names.add(logitsName);
        names.addAll(kvNames.keyNames);
        names.addAll(kvNames.valueNames);
        return names.toArray(new String[0]);
    }

    private void closePrefillOutputs(Map<String, INDArray> prefillOutputs, INDArray prefillLogits) {
        for (String name : kvNames.keyNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) {
                arr.setCloseable(true);
                arr.close();
            }
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) {
                arr.setCloseable(true);
                arr.close();
            }
        }
        if (prefillLogits != null && !prefillLogits.wasClosed()) {
            prefillLogits.setCloseable(true);
            prefillLogits.close();
        }
    }

    // ========== Static KV Cache Manager ==========

    /**
     * Manages static KV cache buffers for decode steps.
     * Pre-allocates [batch, heads, maxKvLen, headDim] buffers and scatters
     * new KV entries from each decode step output.
     */
    private class StaticKvManager {
        private final DecoderUtils.KVCacheNames kvNames;
        private final long maxKvLen;
        private final Map<String, INDArray> staticKvBuffers = new HashMap<>();
        private long cachePosition;

        StaticKvManager(DecoderUtils.KVCacheNames kvNames, long maxKvLen) {
            this.kvNames = kvNames;
            this.maxKvLen = maxKvLen;
        }

        void initializeFromPrefill(Map<String, INDArray> prefillOutputs) {
            for (String keyName : kvNames.keyNames) {
                INDArray present = prefillOutputs.get(keyName);
                if (present != null) {
                    long[] shape = present.shape();
                    INDArray buf = Nd4j.zeros(DataType.FLOAT, shape[0], shape[1], maxKvLen, shape[3]);
                    long copyLen = Math.min(shape[2], maxKvLen);
                    buf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(0, copyLen), NDArrayIndex.all())
                            .assign(present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(0, copyLen), NDArrayIndex.all()));
                    staticKvBuffers.put(ioConfig.presentToInputName(keyName), buf);
                }
            }
            for (String valName : kvNames.valueNames) {
                INDArray present = prefillOutputs.get(valName);
                if (present != null) {
                    long[] shape = present.shape();
                    INDArray buf = Nd4j.zeros(DataType.FLOAT, shape[0], shape[1], maxKvLen, shape[3]);
                    long copyLen = Math.min(shape[2], maxKvLen);
                    buf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(0, copyLen), NDArrayIndex.all())
                            .assign(present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(0, copyLen), NDArrayIndex.all()));
                    staticKvBuffers.put(ioConfig.presentToInputName(valName), buf);
                }
            }
            INDArray firstKey = prefillOutputs.get(kvNames.keyNames.get(0));
            cachePosition = firstKey != null ? firstKey.size(2) : 0;
        }

        long getMaxKvLen() { return maxKvLen; }
        long getCachePosition() { return cachePosition; }
        Map<String, INDArray> getStaticKvBuffers() { return staticKvBuffers; }
        void advancePosition() { cachePosition++; }

        void scatterNewEntries(Map<String, INDArray> outputs) {
            for (String keyName : kvNames.keyNames) {
                INDArray present = outputs.get(keyName);
                if (present != null && present.size(2) > 0) {
                    String pastName = ioConfig.presentToInputName(keyName);
                    INDArray buf = staticKvBuffers.get(pastName);
                    if (buf != null) {
                        long newStart = maxKvLen;
                        long newLen = present.size(2) - newStart;
                        if (newLen > 0 && cachePosition + newLen <= maxKvLen) {
                            buf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                            NDArrayIndex.interval(cachePosition, cachePosition + newLen),
                                            NDArrayIndex.all())
                                    .assign(present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                            NDArrayIndex.interval(newStart, newStart + newLen),
                                            NDArrayIndex.all()));
                        }
                    }
                }
            }
            for (String valName : kvNames.valueNames) {
                INDArray present = outputs.get(valName);
                if (present != null && present.size(2) > 0) {
                    String pastName = ioConfig.presentToInputName(valName);
                    INDArray buf = staticKvBuffers.get(pastName);
                    if (buf != null) {
                        long newStart = maxKvLen;
                        long newLen = present.size(2) - newStart;
                        if (newLen > 0 && cachePosition + newLen <= maxKvLen) {
                            buf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                            NDArrayIndex.interval(cachePosition, cachePosition + newLen),
                                            NDArrayIndex.all())
                                    .assign(present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                            NDArrayIndex.interval(newStart, newStart + newLen),
                                            NDArrayIndex.all()));
                        }
                    }
                }
            }
            cachePosition++;
        }
    }
}
