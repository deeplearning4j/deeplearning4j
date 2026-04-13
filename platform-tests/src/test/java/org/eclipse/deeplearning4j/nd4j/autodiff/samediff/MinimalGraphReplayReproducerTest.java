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
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.StaticKvCacheDecodeLoop;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal reproducer for graph replay divergence.
 *
 * <p>Starts from the PASSING test (DecodeLoopGraphReplayIsolationTest's simplest config)
 * and progressively adds ONE difference at a time from the FAILING test
 * (VisionEmbedGraphReplayTest) to find the EXACT trigger.</p>
 *
 * <h3>Key differences between passing and failing tests:</h3>
 * <ol>
 *   <li>Pre-compilation via BenchmarkConfigApplier (sets dspAutoCompileEnabled=false)</li>
 *   <li>StaticKvCacheDecodeLoop vs manual decode pipeline</li>
 *   <li>Continuous float embeddings vs embedding table lookups</li>
 *   <li>Longer prefill sequences (~680 tokens vs 17-500 tokens)</li>
 *   <li>BenchmarkConfig.optimal() environment flags (tritonFusionScoring=false, etc.)</li>
 * </ol>
 *
 * Run:
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=MinimalGraphReplayReproducerTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 \
 *     2>&1 | tee /tmp/minimal-reproducer.log
 * </pre>
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class MinimalGraphReplayReproducerTest {

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
    private Tokenizer tokenizer;
    private boolean modelsLoaded = false;
    private long hiddenSize;

    /** Summary of results for the final report. */
    private final Map<String, String> testResults = new LinkedHashMap<>();

    @BeforeAll
    public void loadModel() throws Exception {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        // Enable graph optimizer for FP16 pre-cast (matches VisionEmbedGraphReplayTest)
        String optEnabled = System.getProperty("nd4j.optimizer.enabled");
        if (optEnabled == null || optEnabled.isEmpty()) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        String fp16Prop = System.getProperty("nd4j.optimizer.fp16");
        if (fp16Prop == null || fp16Prop.isEmpty()) {
            System.setProperty("nd4j.optimizer.fp16", "true");
        }

        try {
            var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
            var embedResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
            var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
            VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

            decoder = OnnxModelCache.importWithCache(decoderResult.getModelFile().getAbsolutePath());
            embedTokens = OnnxModelCache.importWithCache(embedResult.getModelFile().getAbsolutePath());

            tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());

            // Find embedding table
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
            hiddenSize = embeddingTable.size(1);

            ioConfig = ModelIOConfig.discover(decoder);
            kvNames = ioConfig.getKvCacheNames();
            logitsName = ioConfig.getLogitsOutputName();

            modelsLoaded = true;
            log.info("Models loaded: decoder={} ops, embed={} ops, hiddenSize={}, logits={}, kvLayers={}",
                    decoder.ops().length, embedTokens.ops().length, hiddenSize, logitsName,
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

    // ========================================================================
    // Test 1: Passing baseline (exact copy of DecodeLoopGraphReplayIsolationTest)
    // Uses: manual decode, embedding table lookups, dspAutoCompile=true, 17 tokens
    // Expected: PASS
    // ========================================================================

    @Test
    @Order(1)
    @DisplayName("1. Passing baseline (manual decode, table lookups, autoCompile=true)")
    public void testPassingBaseline() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 1: Passing baseline ===");
        DecodeResult baseline = runManualDecode("1_BASELINE_OFF", false, PREFILL_TOKENS, true);
        DecodeResult treatment = runManualDecode("1_BASELINE_ON", true, PREFILL_TOKENS, true);

        String result = compareAndReport("1_PASSING_BASELINE", baseline, treatment);
        testResults.put("1_PASSING_BASELINE", result);
        assertTokensMatch("1_PASSING_BASELINE", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 2: Use StaticKvCacheDecodeLoop instead of manual decode
    // Changes: StaticKvCacheDecodeLoop with dspAutoCompile=true
    // Expected: determines if StaticKvCacheDecodeLoop itself is the trigger
    // ========================================================================

    @Test
    @Order(2)
    @DisplayName("2. StaticKvCacheDecodeLoop with autoCompile=true (no BenchmarkConfigApplier)")
    public void testWithStaticKvCacheDecodeLoop() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 2: StaticKvCacheDecodeLoop ===");
        // SLOT_BY_SLOT baseline
        List<Integer> baselineTokens = runStaticKvLoop("2_LOOP_OFF", false, PREFILL_TOKENS);
        // Graph capture ON
        List<Integer> treatmentTokens = runStaticKvLoop("2_LOOP_ON", true, PREFILL_TOKENS);

        String result = compareAndReport("2_STATICKVLOOP", baselineTokens, treatmentTokens);
        testResults.put("2_STATICKVLOOP", result);
        assertTokensMatch("2_STATICKVLOOP", baselineTokens, treatmentTokens);
    }

    // ========================================================================
    // Test 3: BenchmarkConfigApplier pre-compilation (the key difference)
    // Changes: BenchmarkConfigApplier.apply() + compileModels() before decode
    // This sets dspAutoCompileEnabled=false and applies OPTIMAL env flags
    // Expected: if this triggers divergence, the bug is in the pre-compile flow
    // ========================================================================

    @Test
    @Order(3)
    @DisplayName("3. BenchmarkConfigApplier pre-compile + StaticKvCacheDecodeLoop")
    public void testWithBenchmarkConfigApplier() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 3: BenchmarkConfigApplier ===");
        // SLOT_BY_SLOT baseline
        List<Integer> baselineTokens = runBenchmarkConfigLoop("3_BENCH_OFF", false, PREFILL_TOKENS);
        // Graph capture ON (OPTIMAL)
        List<Integer> treatmentTokens = runBenchmarkConfigLoop("3_BENCH_ON", true, PREFILL_TOKENS);

        String result = compareAndReport("3_BENCHCONFIG", baselineTokens, treatmentTokens);
        testResults.put("3_BENCHCONFIG", result);
        assertTokensMatch("3_BENCHCONFIG", baselineTokens, treatmentTokens);
    }

    // ========================================================================
    // Test 4: Continuous float embeddings (instead of embedding table lookups)
    // Changes: Use random floats as inputs_embeds instead of integer token IDs
    // Expected: if this triggers divergence, the issue is embedding value-dependent
    // ========================================================================

    @Test
    @Order(4)
    @DisplayName("4. Continuous float embeddings with manual decode")
    public void testWithContinuousEmbeddings() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 4: Continuous float embeddings ===");
        // Create random continuous embeddings (not from table lookups)
        INDArray randomEmbeds = Nd4j.randn(DataType.FLOAT, 1, PREFILL_TOKENS.length, hiddenSize).muli(0.1);

        DecodeResult baseline = runManualDecodeWithEmbeddings("4_CONTINUOUS_OFF", false, randomEmbeds, PREFILL_TOKENS);
        DecodeResult treatment = runManualDecodeWithEmbeddings("4_CONTINUOUS_ON", true, randomEmbeds, PREFILL_TOKENS);

        String result = compareAndReport("4_CONTINUOUS_EMBEDS", baseline, treatment);
        testResults.put("4_CONTINUOUS_EMBEDS", result);
        assertTokensMatch("4_CONTINUOUS_EMBEDS", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 5: Larger prefill (680 tokens, matching VisionEmbed sequence length)
    // Changes: 680 token prefill (matching vision+text merged sequence length)
    // Expected: if this triggers divergence, the issue is length-dependent
    // ========================================================================

    @Test
    @Order(5)
    @DisplayName("5. Longer prefill (680 tokens) with manual decode")
    public void testWithLongerPrefill() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 5: Longer prefill (680 tokens) ===");
        int[] longPrefill = generatePrefillTokens(680);

        DecodeResult baseline = runManualDecode("5_LONG_OFF", false, longPrefill, true);
        DecodeResult treatment = runManualDecode("5_LONG_ON", true, longPrefill, true);

        String result = compareAndReport("5_LONG_PREFILL", baseline, treatment);
        testResults.put("5_LONG_PREFILL", result);
        assertTokensMatch("5_LONG_PREFILL", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 6: BenchmarkConfigApplier + continuous embeddings + long prefill
    // This combines the key failing-test differences
    // ========================================================================

    @Test
    @Order(6)
    @DisplayName("6. BenchmarkConfig + continuous embeddings + 680 tokens")
    public void testCombinedFailingConfig() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 6: Combined failing config ===");
        int[] longPrefill = generatePrefillTokens(680);
        INDArray randomEmbeds = Nd4j.randn(DataType.FLOAT, 1, longPrefill.length, hiddenSize).muli(0.1);

        List<Integer> baselineTokens = runBenchmarkConfigLoopWithEmbeddings(
                "6_COMBINED_OFF", false, randomEmbeds, longPrefill);
        List<Integer> treatmentTokens = runBenchmarkConfigLoopWithEmbeddings(
                "6_COMBINED_ON", true, randomEmbeds, longPrefill);

        String result = compareAndReport("6_COMBINED", baselineTokens, treatmentTokens);
        testResults.put("6_COMBINED", result);
        assertTokensMatch("6_COMBINED", baselineTokens, treatmentTokens);
    }

    // ========================================================================
    // Test 7: dspAutoCompileEnabled=false only (isolate this flag)
    // The key flag that BenchmarkConfigApplier sets differently
    // ========================================================================

    @Test
    @Order(7)
    @DisplayName("7. Manual decode with dspAutoCompileEnabled=false (isolate flag)")
    public void testAutoCompileDisabled() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 7: dspAutoCompileEnabled=false ===");
        DecodeResult baseline = runManualDecode("7_NOAUTO_OFF", false, PREFILL_TOKENS, false);
        DecodeResult treatment = runManualDecode("7_NOAUTO_ON", true, PREFILL_TOKENS, false);

        String result = compareAndReport("7_NOAUTO_COMPILE", baseline, treatment);
        testResults.put("7_NOAUTO_COMPILE", result);
        assertTokensMatch("7_NOAUTO_COMPILE", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 8: Manual decode with TIGHT maxKvLen (matching StaticKvCacheDecodeLoop)
    // The loop uses maxKvLen = prefillSeqLen + maxNewTokens = 17 + 6 = 23
    // The manual test uses MAX_KV_LEN = 2048
    // This isolates whether the tight maxKvLen triggers the divergence
    // ========================================================================

    @Test
    @Order(8)
    @DisplayName("8. Manual decode with tight maxKvLen (23, matching loop)")
    public void testTightMaxKvLen() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 8: Tight maxKvLen ===");
        long tightMaxKvLen = PREFILL_TOKENS.length + NUM_DECODE_STEPS + 1; // 17 + 6 = 23

        DecodeResult baseline = runManualDecodeWithMaxKvLen("8_TIGHT_OFF", false, PREFILL_TOKENS, true, tightMaxKvLen);
        DecodeResult treatment = runManualDecodeWithMaxKvLen("8_TIGHT_ON", true, PREFILL_TOKENS, true, tightMaxKvLen);

        String result = compareAndReport("8_TIGHT_MAXKVLEN", baseline, treatment);
        testResults.put("8_TIGHT_MAXKVLEN", result);
        assertTokensMatch("8_TIGHT_MAXKVLEN", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 9: clearNodeOutputsOnly instead of clearAllCaches before recompile
    // The loop uses session.clearNodeOutputsOnly() to avoid flushing the
    // array cache (which destroys constant DataBuffers). The manual test
    // uses session.clearAllCaches(). This isolates whether the cache
    // clearing strategy triggers divergence.
    // ========================================================================

    @Test
    @Order(9)
    @DisplayName("9. clearNodeOutputsOnly instead of clearAllCaches before recompile")
    public void test9_clearNodeOutputsOnly() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 9: clearNodeOutputsOnly ===");
        DecodeResult baseline = runManualDecodeVariant("9_CLEAR_OFF", false, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/true, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/false);
        DecodeResult treatment = runManualDecodeVariant("9_CLEAR_ON", true, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/true, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/false);

        String result = compareAndReport("9_CLEAR_NODE_OUTPUTS", baseline, treatment);
        testResults.put("9_CLEAR_NODE_OUTPUTS", result);
        assertTokensMatch("9_CLEAR_NODE_OUTPUTS", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 10: reassignDynamicShapePlanDevices after recompile
    // The loop calls decoder.reassignDynamicShapePlanDevices() after recompile
    // to update device placement with fresh memory budgets.
    // ========================================================================

    @Test
    @Order(10)
    @DisplayName("10. reassignDynamicShapePlanDevices after recompile")
    public void test10_reassignDevices() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 10: reassignDevices ===");
        DecodeResult baseline = runManualDecodeVariant("10_REASSIGN_OFF", false, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/true,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/false);
        DecodeResult treatment = runManualDecodeVariant("10_REASSIGN_ON", true, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/true,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/false);

        String result = compareAndReport("10_REASSIGN_DEVICES", baseline, treatment);
        testResults.put("10_REASSIGN_DEVICES", result);
        assertTokensMatch("10_REASSIGN_DEVICES", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 11: suppressCrossDeviceRouting around the entire decode
    // The loop wraps all decode in OpaqueDataBuffer.suppressCrossDeviceRouting(true)
    // to keep all ops on the model's home device.
    // ========================================================================

    @Test
    @Order(11)
    @DisplayName("11. suppressCrossDeviceRouting around decode")
    public void test11_suppressCrossDeviceRouting() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 11: suppressCrossDeviceRouting ===");
        DecodeResult baseline = runManualDecodeVariant("11_SUPPRESS_OFF", false, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/true, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/false);
        DecodeResult treatment = runManualDecodeVariant("11_SUPPRESS_ON", true, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/true, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/false);

        String result = compareAndReport("11_SUPPRESS_CROSS_DEV", baseline, treatment);
        testResults.put("11_SUPPRESS_CROSS_DEV", result);
        assertTokensMatch("11_SUPPRESS_CROSS_DEV", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 12: ensureExecutionDevice each step
    // The loop calls dspExec.ensureExecutionDevice() before the loop and
    // at the start of each step to re-pin the thread to the execution device.
    // ========================================================================

    @Test
    @Order(12)
    @DisplayName("12. ensureExecutionDevice each step")
    public void test12_ensureExecutionDevice() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 12: ensureExecutionDevice ===");
        DecodeResult baseline = runManualDecodeVariant("12_ENSURE_OFF", false, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/true,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/false);
        DecodeResult treatment = runManualDecodeVariant("12_ENSURE_ON", true, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/true,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/false);

        String result = compareAndReport("12_ENSURE_EXEC_DEV", baseline, treatment);
        testResults.put("12_ENSURE_EXEC_DEV", result);
        assertTokensMatch("12_ENSURE_EXEC_DEV", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 13: clearPlaceholders(false) after each step
    // The loop calls decoder.clearPlaceholders(false) after each step
    // to clean up per-step placeholder state.
    // ========================================================================

    @Test
    @Order(13)
    @DisplayName("13. clearPlaceholders(false) after each step")
    public void test13_clearPlaceholders() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 13: clearPlaceholders ===");
        DecodeResult baseline = runManualDecodeVariant("13_PLACEHOLDERS_OFF", false, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/true, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/false);
        DecodeResult treatment = runManualDecodeVariant("13_PLACEHOLDERS_ON", true, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/true, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/false);

        String result = compareAndReport("13_CLEAR_PLACEHOLDERS", baseline, treatment);
        testResults.put("13_CLEAR_PLACEHOLDERS", result);
        assertTokensMatch("13_CLEAR_PLACEHOLDERS", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 14: Mark KV buffers setCloseable(false) after prefill
    // The loop marks static KV buffers as non-closeable to prevent the DSP
    // cache from closing them. Tests whether this poisoning changes behavior.
    // ========================================================================

    @Test
    @Order(14)
    @DisplayName("14. KV buffers setCloseable(false)")
    public void test14_kvSetCloseableFalse() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 14: KV setCloseable(false) ===");
        DecodeResult baseline = runManualDecodeVariant("14_KVCLOSEABLE_OFF", false, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/true,
                /*outputThenDirect=*/false);
        DecodeResult treatment = runManualDecodeVariant("14_KVCLOSEABLE_ON", true, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/true,
                /*outputThenDirect=*/false);

        String result = compareAndReport("14_KV_CLOSEABLE", baseline, treatment);
        testResults.put("14_KV_CLOSEABLE", result);
        assertTokensMatch("14_KV_CLOSEABLE", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 15: output() for step 1 (prefill), outputDirect() for steps 2+
    // The loop uses output() for prefill/early steps and switches to
    // outputDirect() once shapes are frozen. outputDirect() skips dup() of
    // output arrays. This isolates whether the output mode switch matters.
    // ========================================================================

    @Test
    @Order(15)
    @DisplayName("15. output() then outputDirect() (mixed mode)")
    public void test15_outputThenOutputDirect() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 15: output() then outputDirect() ===");
        DecodeResult baseline = runManualDecodeVariant("15_MIXED_OFF", false, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/true);
        DecodeResult treatment = runManualDecodeVariant("15_MIXED_ON", true, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/false, /*reassignDevices=*/false,
                /*suppressCrossDevice=*/false, /*ensureExecDevice=*/false,
                /*clearPlaceholders=*/false, /*kvSetCloseableFalse=*/false,
                /*outputThenDirect=*/true);

        String result = compareAndReport("15_OUTPUT_THEN_DIRECT", baseline, treatment);
        testResults.put("15_OUTPUT_THEN_DIRECT", result);
        assertTokensMatch("15_OUTPUT_THEN_DIRECT", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 16: ALL loop features combined
    // If individual features pass but the loop fails, the bug must be in
    // the combination. This test applies ALL features simultaneously.
    // ========================================================================

    @Test
    @Order(16)
    @DisplayName("16. ALL StaticKvCacheDecodeLoop features combined")
    public void test16_allFeaturesCombined() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 16: ALL features combined ===");
        DecodeResult baseline = runManualDecodeVariant("16_ALL_OFF", false, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/true, /*reassignDevices=*/true,
                /*suppressCrossDevice=*/true, /*ensureExecDevice=*/true,
                /*clearPlaceholders=*/true, /*kvSetCloseableFalse=*/true,
                /*outputThenDirect=*/true);
        DecodeResult treatment = runManualDecodeVariant("16_ALL_ON", true, PREFILL_TOKENS, true,
                /*clearNodeOutputsOnly=*/true, /*reassignDevices=*/true,
                /*suppressCrossDevice=*/true, /*ensureExecDevice=*/true,
                /*clearPlaceholders=*/true, /*kvSetCloseableFalse=*/true,
                /*outputThenDirect=*/true);

        String result = compareAndReport("16_ALL_COMBINED", baseline, treatment);
        testResults.put("16_ALL_COMBINED", result);
        assertTokensMatch("16_ALL_COMBINED", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 17: Reusable fixed-address embedding + inputIds buffers
    // The loop uses reusableEmbeddings.assign() and reusableInputIds.putScalar()
    // for fixed-address CUDA graph stability. The manual decode creates new
    // arrays each step. This tests the reusable buffer pattern.
    // ========================================================================

    @Test
    @Order(17)
    @DisplayName("17. Reusable fixed-address embedding+inputId buffers")
    public void test17_reusableBuffers() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 17: Reusable buffers ===");
        DecodeResult baseline = runManualDecodeWithReusableBuffers("17_REUSE_OFF", false, PREFILL_TOKENS, true);
        DecodeResult treatment = runManualDecodeWithReusableBuffers("17_REUSE_ON", true, PREFILL_TOKENS, true);

        String result = compareAndReport("17_REUSABLE_BUFFERS", baseline, treatment);
        testResults.put("17_REUSABLE_BUFFERS", result);
        assertTokensMatch("17_REUSABLE_BUFFERS", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 18: Direct embedding table lookup (skip embedTokens SameDiff)
    // The loop uses embeddingTable.getRow(tokenId) directly instead of
    // running the embedTokens SameDiff graph. This tests whether the
    // embedding source matters.
    // ========================================================================

    @Test
    @Order(18)
    @DisplayName("18. Direct embedding table lookup (no embedTokens SameDiff)")
    public void test18_directEmbedLookup() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 18: Direct embed lookup ===");
        DecodeResult baseline = runManualDecodeWithDirectEmbed("18_DIRECT_OFF", false, PREFILL_TOKENS, true);
        DecodeResult treatment = runManualDecodeWithDirectEmbed("18_DIRECT_ON", true, PREFILL_TOKENS, true);

        String result = compareAndReport("18_DIRECT_EMBED", baseline, treatment);
        testResults.put("18_DIRECT_EMBED", result);
        assertTokensMatch("18_DIRECT_EMBED", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 19: output() for step 0 only (loop's step 1 = our step 0)
    // The loop uses output() for step 0 (prefill) and step 1 (first decode
    // with useDirect=false because step>=2 is false). This test uses
    // output() for the first decode step only.
    // ========================================================================

    @Test
    @Order(19)
    @DisplayName("19. output() for first decode step only")
    public void test19_outputFirstStepOnly() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 19: output() first step only ===");
        DecodeResult baseline = runManualDecodeWithFirstStepOutput("19_FIRST_OFF", false, PREFILL_TOKENS, true);
        DecodeResult treatment = runManualDecodeWithFirstStepOutput("19_FIRST_ON", true, PREFILL_TOKENS, true);

        String result = compareAndReport("19_FIRST_STEP_OUTPUT", baseline, treatment);
        testResults.put("19_FIRST_STEP_OUTPUT", result);
        assertTokensMatch("19_FIRST_STEP_OUTPUT", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 20: Reusable EMBEDDINGS only (inputIds are fresh each step)
    // Isolate whether reusableEmbeddings.assign() alone triggers divergence.
    // ========================================================================

    @Test
    @Order(20)
    @DisplayName("20. Reusable embeddings only")
    public void test20_reusableEmbeddingsOnly() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 20: Reusable embeddings only ===");
        DecodeResult baseline = runManualDecodeReusableSplit("20_EMBED_OFF", false, PREFILL_TOKENS, true,
                /*reusableEmbeddings=*/true, /*reusableInputIds=*/false);
        DecodeResult treatment = runManualDecodeReusableSplit("20_EMBED_ON", true, PREFILL_TOKENS, true,
                /*reusableEmbeddings=*/true, /*reusableInputIds=*/false);

        String result = compareAndReport("20_REUSE_EMBED_ONLY", baseline, treatment);
        testResults.put("20_REUSE_EMBED_ONLY", result);
        assertTokensMatch("20_REUSE_EMBED_ONLY", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 21: Reusable INPUT_IDS only (embeddings are fresh each step)
    // Isolate whether reusableInputIds.putScalar() alone triggers divergence.
    // ========================================================================

    @Test
    @Order(21)
    @DisplayName("21. Reusable inputIds only")
    public void test21_reusableInputIdsOnly() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 21: Reusable inputIds only ===");
        DecodeResult baseline = runManualDecodeReusableSplit("21_IDS_OFF", false, PREFILL_TOKENS, true,
                /*reusableEmbeddings=*/false, /*reusableInputIds=*/true);
        DecodeResult treatment = runManualDecodeReusableSplit("21_IDS_ON", true, PREFILL_TOKENS, true,
                /*reusableEmbeddings=*/false, /*reusableInputIds=*/true);

        String result = compareAndReport("21_REUSE_IDS_ONLY", baseline, treatment);
        testResults.put("21_REUSE_IDS_ONLY", result);
        assertTokensMatch("21_REUSE_IDS_ONLY", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 22: Reusable embeddings + executor commit after assign
    // Tests whether the divergence is caused by CUDA stream ordering:
    // assign() runs on the executor stream, graph replay on DSP stream.
    // A commit() call forces a stream sync after the assign.
    // ========================================================================

    @Test
    @Order(22)
    @DisplayName("22. Reusable embeddings + commit after assign")
    public void test22_reusableWithCommit() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available");

        log.info("=== TEST 22: Reusable embeddings + commit ===");
        DecodeResult baseline = runManualDecodeReusableWithCommit("22_COMMIT_OFF", false, PREFILL_TOKENS, true);
        DecodeResult treatment = runManualDecodeReusableWithCommit("22_COMMIT_ON", true, PREFILL_TOKENS, true);

        String result = compareAndReport("22_REUSE_WITH_COMMIT", baseline, treatment);
        testResults.put("22_REUSE_WITH_COMMIT", result);
        assertTokensMatch("22_REUSE_WITH_COMMIT", baseline.tokens, treatment.tokens);
    }

    // ========================================================================
    // Test 23: Summary
    // ========================================================================

    @Test
    @Order(23)
    @DisplayName("23. Summary of all results")
    public void testSummary() {
        log.info("========================================================================");
        log.info("  MINIMAL GRAPH REPLAY REPRODUCER SUMMARY");
        log.info("========================================================================");
        for (Map.Entry<String, String> entry : testResults.entrySet()) {
            log.info("  {} | {}", String.format("%-30s", entry.getKey()), entry.getValue());
        }
        log.info("========================================================================");
    }

    // ========================================================================
    // Core decode methods
    // ========================================================================

    /**
     * Manual decode flow (matches DecodeLoopGraphReplayIsolationTest).
     * Uses embedding table lookups for prefill.
     */
    private DecodeResult runManualDecode(String label, boolean graphCaptureOn,
                                          int[] prefillTokenIds, boolean autoCompile) {
        DecodeResult result = new DecodeResult();
        Environment env = Nd4j.getEnvironment();

        // Save and set environment
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

        try {
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(graphCaptureOn);
            env.setTritonConsolidatedArgTable(graphCaptureOn);
            env.setTritonArgDirtyTracking(graphCaptureOn);
            env.setCublasTf32Enabled(graphCaptureOn);
            env.setTritonTf32Enabled(graphCaptureOn);
            env.setDspBatchedGemm(graphCaptureOn);
            env.setTritonFusionScoring(!graphCaptureOn);

            decoder.resetSession();
            embedTokens.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            decoder.setDspAutoCompileEnabled(autoCompile);
            decoder.setDspNativeAutoCompileEnabled(autoCompile);

            INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
            runDecodeSteps(label, result, prefillEmbeds, prefillTokenIds, autoCompile);
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
        }

        return result;
    }

    /**
     * Manual decode with custom maxKvLen (to match StaticKvCacheDecodeLoop's tight sizing).
     */
    private DecodeResult runManualDecodeWithMaxKvLen(String label, boolean graphCaptureOn,
                                                      int[] prefillTokenIds, boolean autoCompile,
                                                      long customMaxKvLen) {
        DecodeResult result = new DecodeResult();
        Environment env = Nd4j.getEnvironment();

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

        try {
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(graphCaptureOn);
            env.setTritonConsolidatedArgTable(graphCaptureOn);
            env.setTritonArgDirtyTracking(graphCaptureOn);
            env.setCublasTf32Enabled(graphCaptureOn);
            env.setTritonTf32Enabled(graphCaptureOn);
            env.setDspBatchedGemm(graphCaptureOn);
            env.setTritonFusionScoring(!graphCaptureOn);

            decoder.resetSession();
            embedTokens.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            decoder.setDspAutoCompileEnabled(autoCompile);
            decoder.setDspNativeAutoCompileEnabled(autoCompile);

            INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
            runDecodeStepsWithMaxKvLen(label, result, prefillEmbeds, prefillTokenIds, autoCompile, customMaxKvLen);
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
        }

        return result;
    }

    /**
     * Manual decode with pre-supplied embeddings instead of table lookups.
     */
    private DecodeResult runManualDecodeWithEmbeddings(String label, boolean graphCaptureOn,
                                                        INDArray prefillEmbeds, int[] prefillTokenIds) {
        DecodeResult result = new DecodeResult();
        Environment env = Nd4j.getEnvironment();

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

        try {
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(graphCaptureOn);
            env.setTritonConsolidatedArgTable(graphCaptureOn);
            env.setTritonArgDirtyTracking(graphCaptureOn);
            env.setCublasTf32Enabled(graphCaptureOn);
            env.setTritonTf32Enabled(graphCaptureOn);
            env.setDspBatchedGemm(graphCaptureOn);
            env.setTritonFusionScoring(!graphCaptureOn);

            decoder.resetSession();
            embedTokens.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            decoder.setDspAutoCompileEnabled(true);
            decoder.setDspNativeAutoCompileEnabled(true);

            runDecodeSteps(label, result, prefillEmbeds, prefillTokenIds, true);
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
        }

        return result;
    }

    /**
     * Manual decode variant that adds individual StaticKvCacheDecodeLoop features one at a time.
     * Starts from the same baseline as test 1 (passing manual decode) and adds ONE feature.
     *
     * @param clearNodeOutputsOnly use clearNodeOutputsOnly() instead of clearAllCaches() before recompile
     * @param reassignDevices call decoder.reassignDynamicShapePlanDevices() after recompile
     * @param suppressCrossDevice wrap decode in OpaqueDataBuffer.suppressCrossDeviceRouting(true)
     * @param ensureExecDevice call dspExec.ensureExecutionDevice() each step
     * @param clearPlaceholders call decoder.clearPlaceholders(false) after each step
     * @param kvSetCloseableFalse mark KV buffers setCloseable(false) after prefill init
     * @param outputThenDirect use output() for step 1, outputDirect() for step 2+
     */
    private DecodeResult runManualDecodeVariant(String label, boolean graphCaptureOn,
                                                  int[] prefillTokenIds, boolean autoCompile,
                                                  boolean clearNodeOutputsOnly, boolean reassignDevices,
                                                  boolean suppressCrossDevice, boolean ensureExecDevice,
                                                  boolean clearPlaceholders, boolean kvSetCloseableFalse,
                                                  boolean outputThenDirect) {
        DecodeResult result = new DecodeResult();
        Environment env = Nd4j.getEnvironment();

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

        if (suppressCrossDevice) {
            OpaqueDataBuffer.suppressCrossDeviceRouting(true);
        }
        try {
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(graphCaptureOn);
            env.setTritonConsolidatedArgTable(graphCaptureOn);
            env.setTritonArgDirtyTracking(graphCaptureOn);
            env.setCublasTf32Enabled(graphCaptureOn);
            env.setTritonTf32Enabled(graphCaptureOn);
            env.setDspBatchedGemm(graphCaptureOn);
            env.setTritonFusionScoring(!graphCaptureOn);

            decoder.resetSession();
            embedTokens.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            decoder.setDspAutoCompileEnabled(autoCompile);
            decoder.setDspNativeAutoCompileEnabled(autoCompile);

            INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
            runDecodeStepsVariant(label, result, prefillEmbeds, prefillTokenIds, autoCompile,
                    clearNodeOutputsOnly, reassignDevices, ensureExecDevice,
                    clearPlaceholders, kvSetCloseableFalse, outputThenDirect);
        } finally {
            if (suppressCrossDevice) {
                OpaqueDataBuffer.suppressCrossDeviceRouting(false);
            }
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
        }

        return result;
    }

    /**
     * Decode steps with individual StaticKvCacheDecodeLoop features toggled.
     * This is a copy of runDecodeSteps with conditional feature application.
     */
    private void runDecodeStepsVariant(String label, DecodeResult result,
                                        INDArray prefillEmbeds, int[] prefillTokenIds,
                                        boolean autoCompile,
                                        boolean clearNodeOutputsOnly, boolean reassignDevices,
                                        boolean ensureExecDevice, boolean clearPlaceholders,
                                        boolean kvSetCloseableFalse, boolean outputThenDirect) {
        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};
        String embedOutputName = embedTokens.outputs().get(0);

        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                .reshape(1, prefillTokenIds.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        // Prefill (always uses output(), same as baseline)
        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, hiddenSize);
        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, label + ": prefill logits null");

        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1), NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        result.tokens.add(nextToken);
        log.info("[{}] Prefill token: {}", label, nextToken);

        // Initialize static KV cache
        StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
        kvMgr.initializeFromPrefill(prefillOutputs);

        // Feature: kvSetCloseableFalse — mark KV buffers non-closeable
        if (kvSetCloseableFalse) {
            for (INDArray kvBuf : kvMgr.getStaticKvBuffers().values()) {
                kvBuf.setCloseable(false);
            }
            log.info("[{}] KV buffers marked setCloseable(false)", label);
        }

        // Close prefill KV outputs
        for (String name : kvNames.keyNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        if (prefillLogits != null && !prefillLogits.wasClosed()) {
            prefillLogits.setCloseable(true); prefillLogits.close();
        }

        // Recompile for seqLen=1 decode
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();

        // Feature: clearNodeOutputsOnly vs clearAllCaches
        if (clearNodeOutputsOnly) {
            session.clearNodeOutputsOnly();
            log.info("[{}] Using clearNodeOutputsOnly() before recompile", label);
        } else {
            session.clearAllCaches();
        }

        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

        // Feature: reassignDevices
        if (reassignDevices) {
            decoder.reassignDynamicShapePlanDevices();
            log.info("[{}] Called reassignDynamicShapePlanDevices()", label);
        }

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        boolean cppKvScatterActive = false;
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);

            if (dspExec.getCurrentPlan() != null) {
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

        // Decode steps
        Map<String, INDArray> reusableInputs = new HashMap<>();
        for (int step = 0; step < NUM_DECODE_STEPS; step++) {
            long pastSeqLen2 = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();

            // Feature: ensureExecutionDevice each step
            if (ensureExecDevice && dspExec != null) {
                dspExec.ensureExecutionDevice();
            }

            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken})
                    .reshape(1, 1).castTo(DataType.LONG);

            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, stepEmbed, tokenIdArr,
                    pastSeqLen2, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, hiddenSize, reusableInputs, true);

            // Feature: outputThenDirect — use output() for step 0, outputDirect() for step 1+
            Map<String, INDArray> outputs;
            if (outputThenDirect && step >= 1) {
                outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);
            } else if (outputThenDirect && step == 0) {
                outputs = decoder.output(decodeInputs, logitsOnlyOutputNames);
            } else {
                // Baseline behavior: always outputDirect() for decode steps
                outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);
            }

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, label + ": step " + step + " logits null");

            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(stepLogits.size(1) - 1), NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            result.tokens.add(nextToken);

            log.info("[{}] Step {}: token={} cachePos={}", label, step, nextToken, cachePos);

            if (cppKvScatterActive) {
                kvMgr.advancePosition();
            } else {
                kvMgr.scatterNewEntries(outputs);
            }

            // Feature: clearPlaceholders after each step
            if (clearPlaceholders) {
                decoder.clearPlaceholders(false);
            }
        }

        // Clean up kvSetCloseableFalse — restore closeability
        if (kvSetCloseableFalse) {
            for (INDArray kvBuf : kvMgr.getStaticKvBuffers().values()) {
                kvBuf.setCloseable(true);
            }
        }
    }

    /**
     * Manual decode with reusable fixed-address embedding and inputId buffers.
     * Mimics the StaticKvCacheDecodeLoop pattern of assign()-ing into a
     * fixed buffer instead of creating new arrays each step.
     */
    private DecodeResult runManualDecodeWithReusableBuffers(String label, boolean graphCaptureOn,
                                                             int[] prefillTokenIds, boolean autoCompile) {
        DecodeResult result = new DecodeResult();
        Environment env = Nd4j.getEnvironment();

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

        try {
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(graphCaptureOn);
            env.setTritonConsolidatedArgTable(graphCaptureOn);
            env.setTritonArgDirtyTracking(graphCaptureOn);
            env.setCublasTf32Enabled(graphCaptureOn);
            env.setTritonTf32Enabled(graphCaptureOn);
            env.setDspBatchedGemm(graphCaptureOn);
            env.setTritonFusionScoring(!graphCaptureOn);

            decoder.resetSession();
            embedTokens.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            decoder.setDspAutoCompileEnabled(autoCompile);
            decoder.setDspNativeAutoCompileEnabled(autoCompile);

            INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
            runDecodeStepsWithReusableBuffers(label, result, prefillEmbeds, prefillTokenIds, autoCompile);
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
        }

        return result;
    }

    /**
     * Decode steps using reusable fixed-address buffers for embeddings and inputIds.
     * This mirrors how StaticKvCacheDecodeLoop manages these buffers.
     */
    private void runDecodeStepsWithReusableBuffers(String label, DecodeResult result,
                                                     INDArray prefillEmbeds, int[] prefillTokenIds,
                                                     boolean autoCompile) {
        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};
        String embedOutputName = embedTokens.outputs().get(0);

        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                .reshape(1, prefillTokenIds.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        // Prefill
        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, hiddenSize);
        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, label + ": prefill logits null");

        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1), NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        result.tokens.add(nextToken);
        log.info("[{}] Prefill token: {}", label, nextToken);

        // Initialize static KV cache
        StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
        kvMgr.initializeFromPrefill(prefillOutputs);

        // Close prefill KV outputs
        for (String name : kvNames.keyNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        if (prefillLogits != null && !prefillLogits.wasClosed()) {
            prefillLogits.setCloseable(true); prefillLogits.close();
        }

        // Recompile
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();

        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        boolean cppKvScatterActive = false;
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
            if (dspExec.getCurrentPlan() != null) {
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

        // Decode steps with REUSABLE BUFFERS (the key difference)
        INDArray reusableEmbeddings = null;
        INDArray reusableInputIds = null;
        Map<String, INDArray> reusableInputs = new HashMap<>();
        for (int step = 0; step < NUM_DECODE_STEPS; step++) {
            long pastSeqLen2 = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();

            // Get embedding using SameDiff (same as baseline)
            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken})
                    .reshape(1, 1).castTo(DataType.LONG);
            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            // REUSABLE BUFFER: assign into fixed-address buffer instead of using stepEmbed directly
            if (reusableEmbeddings == null) {
                reusableEmbeddings = stepEmbed.dup();
            } else {
                reusableEmbeddings.assign(stepEmbed);
            }
            if (reusableInputIds == null) {
                reusableInputIds = tokenIdArr.dup();
            } else {
                reusableInputIds.putScalar(0, 0, nextToken);
            }

            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, reusableEmbeddings, reusableInputIds,
                    pastSeqLen2, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, hiddenSize, reusableInputs, true);

            Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, label + ": step " + step + " logits null");

            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(stepLogits.size(1) - 1), NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            result.tokens.add(nextToken);
            log.info("[{}] Step {}: token={} cachePos={}", label, step, nextToken, cachePos);

            if (cppKvScatterActive) {
                kvMgr.advancePosition();
            } else {
                kvMgr.scatterNewEntries(outputs);
            }
        }
    }

    /**
     * Manual decode with direct embedding table lookup (no embedTokens SameDiff graph).
     * This mirrors how StaticKvCacheDecodeLoop uses embeddingTable.getRow(tokenId) directly.
     */
    private DecodeResult runManualDecodeWithDirectEmbed(String label, boolean graphCaptureOn,
                                                         int[] prefillTokenIds, boolean autoCompile) {
        DecodeResult result = new DecodeResult();
        Environment env = Nd4j.getEnvironment();

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

        try {
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(graphCaptureOn);
            env.setTritonConsolidatedArgTable(graphCaptureOn);
            env.setTritonArgDirtyTracking(graphCaptureOn);
            env.setCublasTf32Enabled(graphCaptureOn);
            env.setTritonTf32Enabled(graphCaptureOn);
            env.setDspBatchedGemm(graphCaptureOn);
            env.setTritonFusionScoring(!graphCaptureOn);

            decoder.resetSession();
            embedTokens.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            decoder.setDspAutoCompileEnabled(autoCompile);
            decoder.setDspNativeAutoCompileEnabled(autoCompile);

            INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
            runDecodeStepsWithDirectEmbed(label, result, prefillEmbeds, prefillTokenIds, autoCompile);
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
        }

        return result;
    }

    /**
     * Decode steps using direct embedding table lookup instead of embedTokens SameDiff.
     */
    private void runDecodeStepsWithDirectEmbed(String label, DecodeResult result,
                                                 INDArray prefillEmbeds, int[] prefillTokenIds,
                                                 boolean autoCompile) {
        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};

        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                .reshape(1, prefillTokenIds.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        // Prefill
        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, hiddenSize);
        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, label + ": prefill logits null");

        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1), NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        result.tokens.add(nextToken);
        log.info("[{}] Prefill token: {}", label, nextToken);

        // Initialize static KV cache
        StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
        kvMgr.initializeFromPrefill(prefillOutputs);

        // Close prefill KV outputs
        for (String name : kvNames.keyNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        if (prefillLogits != null && !prefillLogits.wasClosed()) {
            prefillLogits.setCloseable(true); prefillLogits.close();
        }

        // Recompile
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();

        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        boolean cppKvScatterActive = false;
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
            if (dspExec.getCurrentPlan() != null) {
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

        // Decode steps with DIRECT EMBED LOOKUP (the key difference)
        INDArray reusableEmbeddings = null;
        INDArray reusableInputIds = null;
        Map<String, INDArray> reusableInputs = new HashMap<>();
        for (int step = 0; step < NUM_DECODE_STEPS; step++) {
            long pastSeqLen2 = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();

            // DIRECT LOOKUP: use embeddingTable.getRow() instead of embedTokens.output()
            INDArray rowEmbed = embeddingTable.getRow(nextToken).reshape(1, 1, hiddenSize);
            if (reusableEmbeddings == null) {
                reusableEmbeddings = rowEmbed.dup();
            } else {
                reusableEmbeddings.assign(rowEmbed);
            }
            if (reusableInputIds == null) {
                reusableInputIds = Nd4j.createFromArray(new int[]{nextToken}).reshape(1, 1).castTo(DataType.LONG);
            } else {
                reusableInputIds.putScalar(0, 0, nextToken);
            }

            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, reusableEmbeddings, reusableInputIds,
                    pastSeqLen2, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, hiddenSize, reusableInputs, true);

            Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, label + ": step " + step + " logits null");

            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(stepLogits.size(1) - 1), NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            result.tokens.add(nextToken);
            log.info("[{}] Step {}: token={} cachePos={}", label, step, nextToken, cachePos);

            if (cppKvScatterActive) {
                kvMgr.advancePosition();
            } else {
                kvMgr.scatterNewEntries(outputs);
            }
        }
    }

    /**
     * Manual decode using output() for the first decode step, then outputDirect() after.
     * This matches the loop's step counting where the first decode step
     * (step 1 in the loop) uses output() because useDirect = step >= 2.
     */
    private DecodeResult runManualDecodeWithFirstStepOutput(String label, boolean graphCaptureOn,
                                                              int[] prefillTokenIds, boolean autoCompile) {
        DecodeResult result = new DecodeResult();
        Environment env = Nd4j.getEnvironment();

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

        try {
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(graphCaptureOn);
            env.setTritonConsolidatedArgTable(graphCaptureOn);
            env.setTritonArgDirtyTracking(graphCaptureOn);
            env.setCublasTf32Enabled(graphCaptureOn);
            env.setTritonTf32Enabled(graphCaptureOn);
            env.setDspBatchedGemm(graphCaptureOn);
            env.setTritonFusionScoring(!graphCaptureOn);

            decoder.resetSession();
            embedTokens.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            decoder.setDspAutoCompileEnabled(autoCompile);
            decoder.setDspNativeAutoCompileEnabled(autoCompile);

            INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
            runDecodeStepsFirstStepOutput(label, result, prefillEmbeds, prefillTokenIds, autoCompile);
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
        }

        return result;
    }

    /**
     * Decode steps using output() for the first decode step only.
     * In the loop, step 0 = prefill (output()), step 1 = first decode (output()),
     * step 2+ = decode (outputDirect()). This test's step 0 = loop's step 1.
     */
    private void runDecodeStepsFirstStepOutput(String label, DecodeResult result,
                                                 INDArray prefillEmbeds, int[] prefillTokenIds,
                                                 boolean autoCompile) {
        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};
        String embedOutputName = embedTokens.outputs().get(0);

        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                .reshape(1, prefillTokenIds.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        // Prefill
        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, hiddenSize);
        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, label + ": prefill logits null");

        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1), NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        result.tokens.add(nextToken);
        log.info("[{}] Prefill token: {}", label, nextToken);

        // Initialize static KV cache
        StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
        kvMgr.initializeFromPrefill(prefillOutputs);

        // Close prefill KV outputs
        for (String name : kvNames.keyNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        if (prefillLogits != null && !prefillLogits.wasClosed()) {
            prefillLogits.setCloseable(true); prefillLogits.close();
        }

        // Recompile
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();

        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        boolean cppKvScatterActive = false;
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
            if (dspExec.getCurrentPlan() != null) {
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

        // Decode steps: output() for step 0, outputDirect() for step 1+
        Map<String, INDArray> reusableInputs = new HashMap<>();
        for (int step = 0; step < NUM_DECODE_STEPS; step++) {
            long pastSeqLen2 = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();

            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken})
                    .reshape(1, 1).castTo(DataType.LONG);

            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, stepEmbed, tokenIdArr,
                    pastSeqLen2, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, hiddenSize, reusableInputs, true);

            // KEY DIFFERENCE: output() for step 0, outputDirect() for step 1+
            Map<String, INDArray> outputs;
            if (step == 0) {
                outputs = decoder.output(decodeInputs, logitsOnlyOutputNames);
                log.info("[{}] Step 0: using output() (not outputDirect)", label);
            } else {
                outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);
            }

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, label + ": step " + step + " logits null");

            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(stepLogits.size(1) - 1), NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            result.tokens.add(nextToken);
            log.info("[{}] Step {}: token={} cachePos={}", label, step, nextToken, cachePos);

            if (cppKvScatterActive) {
                kvMgr.advancePosition();
            } else {
                kvMgr.scatterNewEntries(outputs);
            }
        }
    }

    /**
     * Manual decode with reusable embeddings + commit() after assign().
     * Tests whether stream ordering between assign() and graph replay causes divergence.
     */
    private DecodeResult runManualDecodeReusableWithCommit(String label, boolean graphCaptureOn,
                                                             int[] prefillTokenIds, boolean autoCompile) {
        DecodeResult result = new DecodeResult();
        Environment env = Nd4j.getEnvironment();

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

        try {
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(graphCaptureOn);
            env.setTritonConsolidatedArgTable(graphCaptureOn);
            env.setTritonArgDirtyTracking(graphCaptureOn);
            env.setCublasTf32Enabled(graphCaptureOn);
            env.setTritonTf32Enabled(graphCaptureOn);
            env.setDspBatchedGemm(graphCaptureOn);
            env.setTritonFusionScoring(!graphCaptureOn);

            decoder.resetSession();
            embedTokens.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            decoder.setDspAutoCompileEnabled(autoCompile);
            decoder.setDspNativeAutoCompileEnabled(autoCompile);

            INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
            // Same as runDecodeStepsWithReusableBuffers but with commit() after assign()
            String[] fullOutputNames = buildFullOutputNames();
            String[] logitsOnlyOutputNames = new String[]{logitsName};
            String embedOutputName = embedTokens.outputs().get(0);

            INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                    .reshape(1, prefillTokenIds.length)
                    .castTo(DataType.LONG);
            long prefillSeqLen = prefillTokenIds.length;

            // Prefill
            Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, prefillEmbeds, inputIds,
                    0, prefillSeqLen, null, 0, 0, false, hiddenSize);
            Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
            INDArray prefillLogits = prefillOutputs.get(logitsName);
            assertNotNull(prefillLogits, label + ": prefill logits null");

            INDArray lastLogits = prefillLogits.rank() == 3
                    ? prefillLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(prefillLogits.size(1) - 1), NDArrayIndex.all())
                    : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            int nextToken = Nd4j.argMax(lastLogits).getInt(0);
            result.tokens.add(nextToken);
            log.info("[{}] Prefill token: {}", label, nextToken);

            // Initialize static KV cache
            StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
            kvMgr.initializeFromPrefill(prefillOutputs);

            for (String name : kvNames.keyNames) {
                INDArray arr = prefillOutputs.get(name);
                if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
            }
            for (String name : kvNames.valueNames) {
                INDArray arr = prefillOutputs.get(name);
                if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
            }
            if (prefillLogits != null && !prefillLogits.wasClosed()) {
                prefillLogits.setCloseable(true); prefillLogits.close();
            }

            decoder.clearDynamicShapePlanCache();
            var session = decoder.getOrCreateSession();
            session.clearAllCaches();

            Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
            for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
                if (decoder.hasVariable(e.getKey())) {
                    decoder.associateArrayWithVariable(e.getValue(), e.getKey());
                }
            }

            decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

            session = decoder.getOrCreateSession();
            DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

            boolean cppKvScatterActive = false;
            if (dspExec != null) {
                dspExec.setShapesFrozen(true);
                if (dspExec.getCurrentPlan() != null) {
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

            // Decode steps with REUSABLE BUFFERS + COMMIT after assign
            INDArray reusableEmbeddings = null;
            INDArray reusableInputIds = null;
            Map<String, INDArray> reusableInputs = new HashMap<>();
            for (int step = 0; step < NUM_DECODE_STEPS; step++) {
                long pastSeqLen2 = prefillSeqLen + step;
                long cachePos = kvMgr.getCachePosition();

                INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken})
                        .reshape(1, 1).castTo(DataType.LONG);
                Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                        Map.of("input_ids", tokenIdArr), embedOutputName);
                INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

                if (reusableEmbeddings == null) {
                    reusableEmbeddings = stepEmbed.dup();
                } else {
                    reusableEmbeddings.assign(stepEmbed);
                    // KEY FIX: commit executor to force stream sync after assign
                    Nd4j.getExecutioner().commit();
                }
                if (reusableInputIds == null) {
                    reusableInputIds = tokenIdArr.dup();
                } else {
                    reusableInputIds.putScalar(0, 0, nextToken);
                }

                Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                        decoder.inputs(), decoder, reusableEmbeddings, reusableInputIds,
                        pastSeqLen2, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                        true, hiddenSize, reusableInputs, true);

                Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);

                INDArray stepLogits = outputs.get(logitsName);
                assertNotNull(stepLogits, label + ": step " + step + " logits null");

                INDArray lastLogit = stepLogits.rank() == 3
                        ? stepLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(stepLogits.size(1) - 1), NDArrayIndex.all())
                        : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
                nextToken = Nd4j.argMax(lastLogit).getInt(0);
                result.tokens.add(nextToken);
                log.info("[{}] Step {}: token={} cachePos={}", label, step, nextToken, cachePos);

                if (cppKvScatterActive) {
                    kvMgr.advancePosition();
                } else {
                    kvMgr.scatterNewEntries(outputs);
                }
            }
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
        }

        return result;
    }

    /**
     * Manual decode with split reusable buffer control.
     * Tests reusable embeddings and reusable inputIds independently.
     */
    private DecodeResult runManualDecodeReusableSplit(String label, boolean graphCaptureOn,
                                                       int[] prefillTokenIds, boolean autoCompile,
                                                       boolean useReusableEmbeddings, boolean useReusableInputIds) {
        DecodeResult result = new DecodeResult();
        Environment env = Nd4j.getEnvironment();

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

        try {
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(graphCaptureOn);
            env.setTritonConsolidatedArgTable(graphCaptureOn);
            env.setTritonArgDirtyTracking(graphCaptureOn);
            env.setCublasTf32Enabled(graphCaptureOn);
            env.setTritonTf32Enabled(graphCaptureOn);
            env.setDspBatchedGemm(graphCaptureOn);
            env.setTritonFusionScoring(!graphCaptureOn);

            decoder.resetSession();
            embedTokens.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            decoder.setDspAutoCompileEnabled(autoCompile);
            decoder.setDspNativeAutoCompileEnabled(autoCompile);

            INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
            runDecodeStepsReusableSplit(label, result, prefillEmbeds, prefillTokenIds, autoCompile,
                    useReusableEmbeddings, useReusableInputIds);
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
        }

        return result;
    }

    /**
     * Decode steps with independent control over reusable embeddings and inputIds.
     */
    private void runDecodeStepsReusableSplit(String label, DecodeResult result,
                                              INDArray prefillEmbeds, int[] prefillTokenIds,
                                              boolean autoCompile,
                                              boolean useReusableEmbeddings, boolean useReusableInputIds) {
        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};
        String embedOutputName = embedTokens.outputs().get(0);

        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                .reshape(1, prefillTokenIds.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        // Prefill
        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, hiddenSize);
        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, label + ": prefill logits null");

        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1), NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        result.tokens.add(nextToken);
        log.info("[{}] Prefill token: {}", label, nextToken);

        // Initialize static KV cache
        StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
        kvMgr.initializeFromPrefill(prefillOutputs);

        // Close prefill KV outputs
        for (String name : kvNames.keyNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        if (prefillLogits != null && !prefillLogits.wasClosed()) {
            prefillLogits.setCloseable(true); prefillLogits.close();
        }

        // Recompile
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();

        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        boolean cppKvScatterActive = false;
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
            if (dspExec.getCurrentPlan() != null) {
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

        // Decode steps with split reusable buffer control
        INDArray reusableEmbed = null;
        INDArray reusableIds = null;
        Map<String, INDArray> reusableInputs = new HashMap<>();
        for (int step = 0; step < NUM_DECODE_STEPS; step++) {
            long pastSeqLen2 = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();

            // Get embedding
            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken})
                    .reshape(1, 1).castTo(DataType.LONG);
            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            // Reusable embeddings: assign into fixed buffer
            INDArray embedToUse;
            if (useReusableEmbeddings) {
                if (reusableEmbed == null) {
                    reusableEmbed = stepEmbed.dup();
                } else {
                    reusableEmbed.assign(stepEmbed);
                }
                embedToUse = reusableEmbed;
            } else {
                embedToUse = stepEmbed;
            }

            // Reusable inputIds: putScalar into fixed buffer
            INDArray idsToUse;
            if (useReusableInputIds) {
                if (reusableIds == null) {
                    reusableIds = tokenIdArr.dup();
                } else {
                    reusableIds.putScalar(0, 0, nextToken);
                }
                idsToUse = reusableIds;
            } else {
                idsToUse = tokenIdArr;
            }

            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, embedToUse, idsToUse,
                    pastSeqLen2, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, hiddenSize, reusableInputs, true);

            Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, label + ": step " + step + " logits null");

            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(stepLogits.size(1) - 1), NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            result.tokens.add(nextToken);
            log.info("[{}] Step {}: token={} cachePos={} reusableEmbed={} reusableIds={}",
                    label, step, nextToken, cachePos, useReusableEmbeddings, useReusableInputIds);

            if (cppKvScatterActive) {
                kvMgr.advancePosition();
            } else {
                kvMgr.scatterNewEntries(outputs);
            }
        }
    }

    /**
     * StaticKvCacheDecodeLoop-based decode WITHOUT BenchmarkConfigApplier.
     * Manually sets env flags and uses dspAutoCompile=true.
     */
    private List<Integer> runStaticKvLoop(String label, boolean graphCaptureOn, int[] prefillTokenIds) {
        Environment env = Nd4j.getEnvironment();

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

        try {
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(graphCaptureOn);
            env.setTritonConsolidatedArgTable(graphCaptureOn);
            env.setTritonArgDirtyTracking(graphCaptureOn);
            env.setCublasTf32Enabled(graphCaptureOn);
            env.setTritonTf32Enabled(graphCaptureOn);
            env.setDspBatchedGemm(graphCaptureOn);
            env.setTritonFusionScoring(!graphCaptureOn);

            decoder.resetSession();
            embedTokens.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            decoder.setDspAutoCompileEnabled(true);
            decoder.setDspNativeAutoCompileEnabled(true);

            // Build prefill embeddings from table lookups
            INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);

            ModelIOConfig decoderIOConfig = ModelIOConfig.discover(decoder);
            StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                    .decoder(decoder)
                    .embedTokens(embedTokens)
                    .tokenizer(tokenizer)
                    .ioConfig(decoderIOConfig)
                    .samplingConfig(SamplingConfig.greedy())
                    .maxNewTokens(NUM_DECODE_STEPS + 1) // +1 for prefill token
                    .hiddenSize(hiddenSize)
                    .build();

            GenerationResult genResult = loop.decode(prefillEmbeds, prefillTokenIds);
            log.info("[{}] {} tokens: text='{}'", label,
                    genResult.getGeneratedTokenCount(),
                    genResult.getText().substring(0, Math.min(80, genResult.getText().length())));

            List<Integer> tokens = new ArrayList<>();
            for (int id : genResult.getTokenIds()) {
                tokens.add(id);
            }
            return tokens;
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
        }
    }

    /**
     * BenchmarkConfigApplier + StaticKvCacheDecodeLoop (matches VisionEmbedGraphReplayTest flow).
     * Uses embedding table lookups for prefill.
     */
    private List<Integer> runBenchmarkConfigLoop(String label, boolean useOptimal, int[] prefillTokenIds) {
        INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
        return runBenchmarkConfigLoopWithEmbeddings(label, useOptimal, prefillEmbeds, prefillTokenIds);
    }

    /**
     * BenchmarkConfigApplier + StaticKvCacheDecodeLoop with pre-supplied embeddings.
     */
    private List<Integer> runBenchmarkConfigLoopWithEmbeddings(
            String label, boolean useOptimal, INDArray prefillEmbeds, int[] prefillTokenIds) {

        // Reset state (exactly as VisionEmbedGraphReplayTest)
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        Nd4j.getExecutioner().commit();

        // Build config
        BenchmarkConfig config;
        if (useOptimal) {
            config = BenchmarkConfig.optimal().maxTokens(NUM_DECODE_STEPS + 1).minDiversityPct(0);
        } else {
            config = BenchmarkConfig.create(label)
                    .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                    .maxTokens(NUM_DECODE_STEPS + 1)
                    .minDiversityPct(0);
        }

        // Apply config + compile (KEY DIFFERENCE: this sets dspAutoCompileEnabled=false for Triton)
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

        GenerationResult genResult = loop.decode(prefillEmbeds, prefillTokenIds);
        log.info("[{}] {} tokens: text='{}'", label,
                genResult.getGeneratedTokenCount(),
                genResult.getText().substring(0, Math.min(80, genResult.getText().length())));

        List<Integer> tokens = new ArrayList<>();
        for (int id : genResult.getTokenIds()) {
            tokens.add(id);
        }
        return tokens;
    }

    // ========================================================================
    // Shared decode step logic for manual decode
    // ========================================================================

    private void runDecodeSteps(String label, DecodeResult result,
                                 INDArray prefillEmbeds, int[] prefillTokenIds,
                                 boolean autoCompile) {
        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};
        String embedOutputName = embedTokens.outputs().get(0);

        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                .reshape(1, prefillTokenIds.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        // Prefill
        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, hiddenSize);
        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, label + ": prefill logits null");

        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1), NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        result.tokens.add(nextToken);
        log.info("[{}] Prefill token: {}", label, nextToken);

        // Initialize static KV cache
        StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
        kvMgr.initializeFromPrefill(prefillOutputs);

        // Close prefill KV outputs
        for (String name : kvNames.keyNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        if (prefillLogits != null && !prefillLogits.wasClosed()) {
            prefillLogits.setCloseable(true); prefillLogits.close();
        }

        // Recompile for seqLen=1 decode
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();

        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        boolean cppKvScatterActive = false;
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);

            if (dspExec.getCurrentPlan() != null) {
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

        // Decode steps
        Map<String, INDArray> reusableInputs = new HashMap<>();
        for (int step = 0; step < NUM_DECODE_STEPS; step++) {
            long pastSeqLen2 = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();

            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken})
                    .reshape(1, 1).castTo(DataType.LONG);

            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, stepEmbed, tokenIdArr,
                    pastSeqLen2, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, hiddenSize, reusableInputs, true);

            Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, label + ": step " + step + " logits null");

            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(stepLogits.size(1) - 1), NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            result.tokens.add(nextToken);

            log.info("[{}] Step {}: token={} cachePos={}", label, step, nextToken, cachePos);

            if (cppKvScatterActive) {
                kvMgr.advancePosition();
            } else {
                kvMgr.scatterNewEntries(outputs);
            }
        }
    }

    /**
     * Same as runDecodeSteps but with a custom maxKvLen (to match StaticKvCacheDecodeLoop sizing).
     */
    private void runDecodeStepsWithMaxKvLen(String label, DecodeResult result,
                                             INDArray prefillEmbeds, int[] prefillTokenIds,
                                             boolean autoCompile, long customMaxKvLen) {
        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};
        String embedOutputName = embedTokens.outputs().get(0);

        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                .reshape(1, prefillTokenIds.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        // Prefill
        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, hiddenSize);
        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, label + ": prefill logits null");

        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1), NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        result.tokens.add(nextToken);
        log.info("[{}] Prefill token: {} (maxKvLen={})", label, nextToken, customMaxKvLen);

        // Initialize static KV cache with CUSTOM maxKvLen (tight sizing like the loop)
        StaticKvManager kvMgr = new StaticKvManager(kvNames, customMaxKvLen);
        kvMgr.initializeFromPrefill(prefillOutputs);

        // Close prefill outputs
        for (String name : kvNames.keyNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        for (String name : kvNames.valueNames) {
            INDArray arr = prefillOutputs.get(name);
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        if (prefillLogits != null && !prefillLogits.wasClosed()) {
            prefillLogits.setCloseable(true); prefillLogits.close();
        }

        // Recompile for seqLen=1 decode
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();

        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        boolean cppKvScatterActive = false;
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);

            if (dspExec.getCurrentPlan() != null) {
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
                log.info("[{}] C++ KV scatter: {} (maxKvLen={})", label, cppKvScatterActive, customMaxKvLen);

                if (cppKvScatterActive) {
                    dspExec.configureDecodeInputs(dspExec.getCurrentPlan(), (int) kvMgr.getMaxKvLen());
                }
            }
        }

        // Decode steps
        Map<String, INDArray> reusableInputs = new HashMap<>();
        for (int step = 0; step < NUM_DECODE_STEPS; step++) {
            long pastSeqLen2 = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();

            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken})
                    .reshape(1, 1).castTo(DataType.LONG);

            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, stepEmbed, tokenIdArr,
                    pastSeqLen2, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, hiddenSize, reusableInputs, true);

            Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, label + ": step " + step + " logits null");

            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(stepLogits.size(1) - 1), NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            result.tokens.add(nextToken);

            log.info("[{}] Step {}: token={} cachePos={} maxKvLen={}", label, step, nextToken, cachePos, customMaxKvLen);

            if (cppKvScatterActive) {
                kvMgr.advancePosition();
            } else {
                kvMgr.scatterNewEntries(outputs);
            }
        }
    }

    // ========================================================================
    // Comparison and reporting
    // ========================================================================

    private String compareAndReport(String label, DecodeResult baseline, DecodeResult treatment) {
        return compareAndReport(label, baseline.tokens, treatment.tokens);
    }

    private String compareAndReport(String label, List<Integer> baseline, List<Integer> treatment) {
        int minLen = Math.min(baseline.size(), treatment.size());
        int matchCount = 0;
        int firstDivergence = -1;

        for (int i = 0; i < minLen; i++) {
            if (baseline.get(i).equals(treatment.get(i))) {
                matchCount++;
            } else if (firstDivergence == -1) {
                firstDivergence = i;
            }
        }

        if (firstDivergence >= 0) {
            log.error("[{}] DIVERGENCE at step {}: baseline={} treatment={}",
                    label, firstDivergence, baseline.get(firstDivergence), treatment.get(firstDivergence));
            log.error("[{}] Baseline:  {}", label, baseline);
            log.error("[{}] Treatment: {}", label, treatment);
            return String.format("DIVERGE at step %d (%d/%d match)", firstDivergence, matchCount, minLen);
        } else {
            log.info("[{}] All {} tokens MATCH", label, minLen);
            return String.format("PASS (%d tokens match)", minLen);
        }
    }

    private void assertTokensMatch(String testName, List<Integer> baseline, List<Integer> treatment) {
        int minLen = Math.min(baseline.size(), treatment.size());
        for (int i = 0; i < minLen; i++) {
            assertEquals(baseline.get(i), treatment.get(i),
                    String.format("[%s] Token mismatch at index %d (%s): baseline=%d treatment=%d",
                            testName, i, i == 0 ? "prefill" : "decode step " + (i - 1),
                            baseline.get(i), treatment.get(i)));
        }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    private INDArray buildPrefillEmbeddings(int[] tokens) {
        long hidden = embeddingTable.size(1);
        INDArray result = Nd4j.create(DataType.FLOAT, 1, tokens.length, hidden);
        int vocabSize = (int) embeddingTable.size(0);
        for (int i = 0; i < tokens.length; i++) {
            int tokenId = tokens[i] % vocabSize;
            result.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all())
                    .assign(embeddingTable.getRow(tokenId));
        }
        return result;
    }

    private int[] generatePrefillTokens(int length) {
        int vocabSize = (int) embeddingTable.size(0);
        int[] tokens = new int[length];
        tokens[0] = Math.min(49229, vocabSize - 1);
        int[] seedTokens = {1, 42, 100, 256, 500, 1000, 2000, 3000, 4000, 5000,
                6000, 7000, 8000, 9000, 10000, 11126, 12000, 13000, 14000, 15000};
        for (int i = 1; i < length; i++) {
            tokens[i] = Math.min(seedTokens[(i - 1) % seedTokens.length], vocabSize - 1);
        }
        return tokens;
    }

    private String[] buildFullOutputNames() {
        List<String> names = new ArrayList<>();
        names.add(logitsName);
        names.addAll(kvNames.keyNames);
        names.addAll(kvNames.valueNames);
        return names.toArray(new String[0]);
    }

    // ========================================================================
    // Data classes
    // ========================================================================

    private static class DecodeResult {
        List<Integer> tokens = new ArrayList<>();
    }

    // ========================================================================
    // Static KV Cache Manager (copied from DecodeLoopGraphReplayIsolationTest)
    // ========================================================================

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
