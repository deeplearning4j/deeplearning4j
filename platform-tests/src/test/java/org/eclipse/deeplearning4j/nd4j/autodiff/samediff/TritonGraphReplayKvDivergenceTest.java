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
 * Reproduction test for CUDA graph replay KV cache divergence.
 *
 * <p>Bug: After multi-token prefill + decode recompilation, graph replay produces
 * degenerate output (same token repeating) despite KV cache positions incrementing.
 * The model sees identical context each step because the Triton-compiled attention
 * kernel reads stale KV data during graph replay.</p>
 *
 * <p>Key observation from production benchmarks:</p>
 * <ul>
 *   <li>SLOT_BY_SLOT (no Triton, no graphs): CORRECT</li>
 *   <li>DIAG_TRITON_noGC (Triton kernels, no graph capture): CORRECT</li>
 *   <li>OPTIMAL (Triton + graph capture/replay): WRONG -- repeats after step 1</li>
 * </ul>
 *
 * <p>The critical difference is {@code tritonGraphCapture(true)}, which forces indirect
 * arg passing for all Triton kernels and enables the CUDA graph capture/replay flow.
 * The multi-token prefill (17+ tokens) triggers a decode recompilation path that
 * the single-token prefill in TritonGraphCaptureIsolationTest does not exercise.</p>
 *
 * <p>Test methods:</p>
 * <ol>
 *   <li>{@link #testMultiTokenPrefillGraphReplayDivergence} -- core reproduction with
 *       multi-token prefill + graph replay vs no-graph-capture baseline</li>
 *   <li>{@link #testKvCacheConsistencyDuringGraphReplay} -- verify KV cache contains
 *       fresh data after each graph replay step (not stale from capture)</li>
 *   <li>{@link #testAttentionOutputFreshness} -- compare attention output between
 *       graph replay and fresh execution to isolate stale reads</li>
 * </ol>
 *
 * <p>Run:</p>
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TritonGraphReplayKvDivergenceTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 \
 *     2>&1 | tee /tmp/kv-diverge-test.log
 * </pre>
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TritonGraphReplayKvDivergenceTest {

    /**
     * Multi-token prefill sequence. Uses 17 tokens to match the VLM benchmark
     * validation path. This exercises the prefill-to-decode recompilation that
     * single-token prefill (TritonGraphCaptureIsolationTest) does not.
     *
     * Token IDs from SmolDocling vocabulary:
     * 49229 = doctag, then a sequence of common structural tokens.
     */
    private static final int[] MULTI_TOKEN_PREFILL = {
            49229, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };

    private static final int NUM_DECODE_STEPS = 7;
    private static final long MAX_KV_LEN = 2048;

    private SameDiff decoder;
    private SameDiff embedTokens;
    private INDArray embeddingTable;
    private String logitsName;
    private String embedsName;
    private ModelIOConfig ioConfig;
    private DecoderUtils.KVCacheNames kvNames;
    private boolean modelsLoaded = false;

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
            embedsName = ioConfig.getInputEmbeddingsName();

            modelsLoaded = true;
            log.info("Models loaded: decoder={} ops, embedTokens={} ops, logits={}, embeds={}, kvLayers={}",
                    decoder.ops().length, embedTokens.ops().length, logitsName, embedsName,
                    kvNames.keyNames.size());
        } catch (Exception e) {
            log.error("Failed to load models: {}", e.getMessage(), e);
        }
    }

    @AfterAll
    public void teardownModels() {
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

    // ========== Test 1: Core Reproduction ==========

    /**
     * Core reproduction: multi-token prefill + graph replay vs no-graph-capture baseline.
     *
     * <p>This test uses a 17-token prefill sequence (matching the VLM benchmark validation
     * path), which triggers decode recompilation. The existing TritonGraphCaptureIsolationTest
     * only uses 1 token prefill and passes -- the bug requires multi-token prefill to manifest.</p>
     *
     * <p>Baseline: Triton ON, graph capture OFF (known correct from DIAG_TRITON_noGC).
     * Treatment: Triton ON, graph capture ON (known broken in OPTIMAL).</p>
     */
    @Test
    @Order(1)
    @DisplayName("Multi-token prefill + graph replay must not diverge from no-graph-capture baseline")
    public void testMultiTokenPrefillGraphReplayDivergence() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");

        Environment env = Nd4j.getEnvironment();

        // Save original values for ALL OPTIMAL flags
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
            // ---- Baseline: Triton ON, graph capture OFF (DIAG_TRITON_noGC equivalent) ----
            // Matches BenchmarkConfig.create("DIAG_TRITON_noGC") from the benchmark
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
            log.info("=== BASELINE: tritonGraphCapture=OFF (DIAG_TRITON_noGC equivalent) ===");
            DecodeResult baselineResult = runMultiTokenDecodeSequence("BASELINE_noGC",
                    MULTI_TOKEN_PREFILL, false);
            log.info("Baseline tokens: {}", baselineResult.tokens);

            // ---- Treatment: Full OPTIMAL config (CUDA graph replay + all optimizations) ----
            // Matches BenchmarkConfig.optimal() exactly
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
            log.info("=== TREATMENT: FULL OPTIMAL config (gc=ON tf32=ON batchedGemm=ON) ===");
            DecodeResult treatmentResult = runMultiTokenDecodeSequence("TREATMENT_OPTIMAL",
                    MULTI_TOKEN_PREFILL, false);
            log.info("Treatment tokens: {}", treatmentResult.tokens);

            // ---- Compare ----
            compareTokenSequences("multiTokenPrefill_OPTIMAL", baselineResult, treatmentResult);
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

    // ========== Test 2: KV Cache Consistency ==========

    /**
     * Verify KV cache contains fresh data after each graph replay decode step.
     *
     * <p>After each decode step with graph replay, read back the KV cache values at
     * the current cache position and verify they are NOT identical to the capture-time
     * values. Each step should add one new entry with distinct data.</p>
     *
     * <p>If the KV cache values at position N are identical to position N-1, the
     * attention kernel will see identical context and produce the same token -- which
     * is exactly the degenerate behavior observed.</p>
     */
    @Test
    @Order(2)
    @DisplayName("KV cache must contain fresh data after each graph replay step")
    public void testKvCacheConsistencyDuringGraphReplay() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");

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
        int origNumWarps = env.tritonNumWarps();
        int origNumStages = env.tritonNumStages();

        try {
            // Full OPTIMAL config (the broken path)
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

            log.info("=== KV CONSISTENCY TEST: FULL OPTIMAL config, multi-token prefill ===");
            DecodeResult result = runMultiTokenDecodeSequence("KV_CONSISTENCY",
                    MULTI_TOKEN_PREFILL, true);
            log.info("Tokens: {}", result.tokens);

            // Verify KV cache snapshots: each step should have distinct data
            assertFalse(result.kvSnapshotsPerStep.isEmpty(),
                    "KV snapshots must be captured when collectKvSnapshots=true");

            int staleStepCount = 0;
            for (int step = 1; step < result.kvSnapshotsPerStep.size(); step++) {
                double[] prevSnapshot = result.kvSnapshotsPerStep.get(step - 1);
                double[] currSnapshot = result.kvSnapshotsPerStep.get(step);

                // Compare the snapshots -- they should NOT be identical
                boolean identical = true;
                double maxDiff = 0;
                for (int i = 0; i < Math.min(prevSnapshot.length, currSnapshot.length); i++) {
                    double diff = Math.abs(prevSnapshot[i] - currSnapshot[i]);
                    if (diff > 1e-6) {
                        identical = false;
                    }
                    maxDiff = Math.max(maxDiff, diff);
                }
                log.info("KV step {} vs step {}: identical={} maxDiff={}", step - 1, step, identical, maxDiff);

                if (identical) {
                    staleStepCount++;
                    log.error("STALE KV DATA at step {}: KV cache at position {} is identical to position {} " +
                                    "-- graph replay did NOT write fresh KV data",
                            step, result.kvCachePositions.get(step), result.kvCachePositions.get(step - 1));
                }
            }

            assertEquals(0, staleStepCount,
                    String.format("KV cache had %d stale steps out of %d. " +
                                    "Graph replay is not writing fresh KV data at each decode position.",
                            staleStepCount, result.kvSnapshotsPerStep.size() - 1));
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

    // ========== Test 3: Attention Output Freshness ==========

    /**
     * Compare logit checksums between graph replay and fresh (no-graph) execution.
     *
     * <p>If the attention kernel reads stale KV data during graph replay, the logit
     * checksum will be identical across decode steps (because the model sees the same
     * context). In correct execution, logit checksums should differ between steps
     * because each step adds new KV data.</p>
     */
    @Test
    @Order(3)
    @DisplayName("Logit checksums must vary across decode steps during graph replay")
    public void testAttentionOutputFreshness() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");

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
        int origNumWarps = env.tritonNumWarps();
        int origNumStages = env.tritonNumStages();

        try {
            // ---- Graph replay path: FULL OPTIMAL config ----
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
            log.info("=== ATTENTION FRESHNESS: FULL OPTIMAL config ===");
            DecodeResult gcResult = runMultiTokenDecodeSequence("ATTN_FRESH_OPTIMAL",
                    MULTI_TOKEN_PREFILL, false);

            // ---- No graph capture path (for reference) ----
            env.setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
            env.setTritonGraphCapture(false);
            env.setTritonConsolidatedArgTable(false);
            env.setTritonArgDirtyTracking(false);
            env.setCublasTf32Enabled(false);
            env.setTritonTf32Enabled(false);
            env.setDspBatchedGemm(false);
            env.setTritonFusionScoring(true);
            log.info("=== ATTENTION FRESHNESS: DIAG_TRITON_noGC config ===");
            DecodeResult noGcResult = runMultiTokenDecodeSequence("ATTN_FRESH_noGC",
                    MULTI_TOKEN_PREFILL, false);

            // Check graph replay logit checksums for monotony (stale reads)
            log.info("Graph capture logit checksums: {}", gcResult.logitChecksums);
            log.info("No graph capture logit checksums: {}", noGcResult.logitChecksums);

            // In correct execution, logit checksums MUST differ between consecutive decode steps
            // because the model processes different KV cache content each step.
            int gcRepeatedChecksums = 0;
            for (int i = 2; i < gcResult.logitChecksums.size(); i++) {
                double prev = gcResult.logitChecksums.get(i - 1);
                double curr = gcResult.logitChecksums.get(i);
                if (Math.abs(prev - curr) < 1e-4) {
                    gcRepeatedChecksums++;
                    log.error("STALE ATTENTION: graph replay step {} has same logit checksum as step {}: {} vs {}",
                            i, i - 1, curr, prev);
                }
            }

            // Compare per-step logit checksums between graph-capture and no-graph-capture
            log.info("Per-step logit checksum comparison (GC vs noGC):");
            for (int i = 0; i < Math.min(gcResult.logitChecksums.size(), noGcResult.logitChecksums.size()); i++) {
                double gc = gcResult.logitChecksums.get(i);
                double noGc = noGcResult.logitChecksums.get(i);
                double diff = Math.abs(gc - noGc);
                String match = diff < 1.0 ? "MATCH" : "DIVERGE";
                log.info("  Step {}: GC={} noGC={} diff={} {}", i, gc, noGc, diff, match);
            }

            // The no-graph-capture path is known correct -- it should have varying checksums
            int noGcRepeatedChecksums = 0;
            for (int i = 2; i < noGcResult.logitChecksums.size(); i++) {
                double prev = noGcResult.logitChecksums.get(i - 1);
                double curr = noGcResult.logitChecksums.get(i);
                if (Math.abs(prev - curr) < 1e-4) {
                    noGcRepeatedChecksums++;
                }
            }

            // Sanity: no-GC path should have varying checksums
            assertTrue(noGcRepeatedChecksums <= 1,
                    "No-graph-capture path should have varying logit checksums but had " +
                            noGcRepeatedChecksums + " repeated. This would indicate a test setup problem.");

            // Key assertion: graph replay should NOT have more repeated checksums than no-GC
            assertTrue(gcRepeatedChecksums <= noGcRepeatedChecksums + 1,
                    String.format("Graph replay has %d repeated logit checksums vs %d for no-graph-capture. " +
                                    "The attention kernel is reading stale KV data during graph replay.",
                            gcRepeatedChecksums, noGcRepeatedChecksums));
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

    // ========== Core Decode Logic ==========

    /**
     * Result of a decode sequence, including diagnostic data.
     */
    private static class DecodeResult {
        List<Integer> tokens = new ArrayList<>();
        List<Double> logitChecksums = new ArrayList<>();
        List<Double> logitMaxValues = new ArrayList<>();
        List<Long> kvCachePositions = new ArrayList<>();
        /** KV cache snapshots: for each step, a small sample of KV values at the current cache position. */
        List<double[]> kvSnapshotsPerStep = new ArrayList<>();
    }

    /**
     * Run multi-token prefill + decode steps with the current environment settings.
     *
     * <p>This method exercises the full prefill-to-decode recompilation path that the
     * VLM benchmark uses. After prefill with N tokens, it recompiles the DSP plan
     * for seqLen=1 decode, freezes shapes, configures C++ KV scatter, and runs
     * multiple decode steps.</p>
     *
     * @param label human-readable label for logging
     * @param prefillTokenIds the multi-token prefill sequence
     * @param collectKvSnapshots whether to read back KV cache values after each step
     * @return decode result with tokens, checksums, and optionally KV snapshots
     */
    private DecodeResult runMultiTokenDecodeSequence(String label, int[] prefillTokenIds,
                                                      boolean collectKvSnapshots) {
        DecodeResult result = new DecodeResult();

        // Reset state completely between runs
        decoder.resetSession();
        embedTokens.resetSession();
        InferenceSession.setDynamicShapePlanEnabled(true);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        String[] fullOutputNames = buildFullOutputNames();
        String embedOutputName = embedTokens.outputs().get(0);

        // ---- Multi-token prefill ----
        INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                .reshape(1, prefillTokenIds.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        log.info("[{}] Running prefill with {} tokens", label, prefillSeqLen);

        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, embeddingTable.size(1));

        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, String.format("[%s] Prefill logits must not be null", label));

        // Extract first token from prefill
        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(prefillLogits.size(1) - 1),
                NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        result.tokens.add(nextToken);
        result.logitChecksums.add(prefillLogits.sumNumber().doubleValue());
        result.logitMaxValues.add(prefillLogits.maxNumber().doubleValue());

        log.info("[{}] Prefill token: {} logitSum={} logitMax={} prefillSeqLen={}",
                label, nextToken, result.logitChecksums.get(0), result.logitMaxValues.get(0), prefillSeqLen);

        // Initialize static KV cache from prefill outputs
        StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
        kvMgr.initializeFromPrefill(prefillOutputs);
        log.info("[{}] KV cache initialized: cachePos={} maxKvLen={}", label,
                kvMgr.getCachePosition(), kvMgr.getMaxKvLen());

        // Close prefill KV outputs
        closePrefillOutputs(prefillOutputs, prefillLogits);

        // ---- Recompile for decode (seqLen=1) ----
        // This is the critical path: multi-token prefill compiled for seqLen=17,
        // now recompile for seqLen=1 decode. The graph capture/replay happens AFTER
        // this recompilation.
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();

        // Associate static KV buffers as placeholders before compilation
        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, fullOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        // Configure frozen shapes + C++ KV scatter if DSP executor exists
        boolean cppKvScatter = false;
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
                cppKvScatter = dspExec.configureKvCacheRetention(
                        dspExec.getCurrentPlan(), presentNames, pastNames,
                        (int) kvMgr.getMaxKvLen(), (int) kvMgr.getCachePosition());
                log.info("[{}] C++ KV scatter configured: {} (prefillSeqLen={})", label, cppKvScatter, prefillSeqLen);
                if (cppKvScatter) {
                    dspExec.configureDecodeInputs(dspExec.getCurrentPlan(), (int) kvMgr.getMaxKvLen());
                }
            }
        }

        // ---- Decode steps ----
        Map<String, INDArray> reusableInputs = new HashMap<>();

        for (int step = 0; step < NUM_DECODE_STEPS; step++) {
            long pastSeqLen = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();
            result.kvCachePositions.add(cachePos);

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

            // Execute
            Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, fullOutputNames);

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, String.format("[%s] Step %d logits must not be null", label, step));

            // Greedy sample
            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(stepLogits.size(1) - 1),
                    NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            result.tokens.add(nextToken);

            double checksum = stepLogits.sumNumber().doubleValue();
            double maxVal = stepLogits.maxNumber().doubleValue();
            result.logitChecksums.add(checksum);
            result.logitMaxValues.add(maxVal);

            log.info("[{}] Step {}: token={} logitSum={} logitMax={} cachePos={}",
                    label, step, nextToken, checksum, maxVal, cachePos);

            // Collect KV cache snapshot if requested
            if (collectKvSnapshots) {
                double[] snapshot = snapshotKvAtPosition(kvMgr, cachePos);
                result.kvSnapshotsPerStep.add(snapshot);
            }

            // KV cache management
            if (!cppKvScatter) {
                kvMgr.scatterNewEntries(outputs);
            } else {
                kvMgr.advancePosition();
            }
        }

        return result;
    }

    // ========== KV Snapshot Helpers ==========

    /**
     * Take a snapshot of KV cache values at a specific position.
     * Reads the first few elements from the first key buffer at the given position.
     */
    private double[] snapshotKvAtPosition(StaticKvManager kvMgr, long position) {
        Map<String, INDArray> kvBuffers = kvMgr.getStaticKvBuffers();
        if (kvBuffers.isEmpty()) return new double[0];

        // Get first key buffer's values at this position
        String firstPastKeyName = ioConfig.presentToInputName(kvNames.keyNames.get(0));
        INDArray firstKeyBuf = kvBuffers.get(firstPastKeyName);
        if (firstKeyBuf == null || position >= firstKeyBuf.size(2)) return new double[0];

        // Extract [batch=0, head=0, position, :8] -- first 8 values of first head at this position
        int snapLen = (int) Math.min(8, firstKeyBuf.size(3));
        INDArray slice = firstKeyBuf.get(
                NDArrayIndex.point(0),
                NDArrayIndex.point(0),
                NDArrayIndex.point(position),
                NDArrayIndex.interval(0, snapLen)
        ).dup();

        double[] values = new double[snapLen];
        for (int i = 0; i < snapLen; i++) {
            values[i] = slice.getDouble(i);
        }
        return values;
    }

    // ========== Token Comparison ==========

    /**
     * Compare two decode results and assert token sequences match.
     * On mismatch, logs detailed divergence diagnostics.
     */
    private void compareTokenSequences(String testName, DecodeResult baseline, DecodeResult treatment) {
        assertEquals(baseline.tokens.size(), treatment.tokens.size(),
                String.format("[%s] Token count mismatch: baseline=%d treatment=%d",
                        testName, baseline.tokens.size(), treatment.tokens.size()));

        int firstDivergence = -1;
        int matchCount = 0;
        for (int i = 0; i < baseline.tokens.size(); i++) {
            if (baseline.tokens.get(i).equals(treatment.tokens.get(i))) {
                matchCount++;
            } else if (firstDivergence == -1) {
                firstDivergence = i;
            }
        }

        double matchRate = (double) matchCount / baseline.tokens.size() * 100.0;
        log.info("[{}] Match rate: {}/{} ({}%)", testName, matchCount, baseline.tokens.size(),
                String.format("%.1f", matchRate));

        if (firstDivergence >= 0) {
            String stepLabel = firstDivergence == 0 ? "prefill" : "decode step " + (firstDivergence - 1);
            log.error("[{}] FIRST DIVERGENCE at index {} ({}): baseline={} treatment={}",
                    testName, firstDivergence, stepLabel,
                    baseline.tokens.get(firstDivergence), treatment.tokens.get(firstDivergence));
            log.error("[{}] Full baseline:  {}", testName, baseline.tokens);
            log.error("[{}] Full treatment: {}", testName, treatment.tokens);

            // Log per-step logit checksum comparison around divergence point
            int logStart = Math.max(0, firstDivergence - 1);
            int logEnd = Math.min(baseline.logitChecksums.size(), firstDivergence + 3);
            for (int i = logStart; i < logEnd; i++) {
                String bCheck = i < baseline.logitChecksums.size()
                        ? String.valueOf(baseline.logitChecksums.get(i)) : "N/A";
                String tCheck = i < treatment.logitChecksums.size()
                        ? String.valueOf(treatment.logitChecksums.get(i)) : "N/A";
                log.error("[{}] Logit checksum at index {}: baseline={} treatment={}", testName, i, bCheck, tCheck);
            }

            // Check for degenerate output (all same token after divergence)
            boolean degenerate = true;
            for (int i = firstDivergence + 1; i < treatment.tokens.size(); i++) {
                if (!treatment.tokens.get(i).equals(treatment.tokens.get(firstDivergence))) {
                    degenerate = false;
                    break;
                }
            }
            if (degenerate && treatment.tokens.size() - firstDivergence > 2) {
                log.error("[{}] Treatment output is DEGENERATE -- all tokens from divergence point are {}. " +
                                "This matches the known CUDA graph replay KV stale-read bug.",
                        testName, treatment.tokens.get(firstDivergence));
            }
        }

        // Assert token-by-token match
        for (int i = 0; i < baseline.tokens.size(); i++) {
            assertEquals(baseline.tokens.get(i), treatment.tokens.get(i),
                    String.format("[%s] Token mismatch at index %d (%s): baseline=%d treatment=%d. " +
                                    "Baseline logitSum=%s, Treatment logitSum=%s",
                            testName, i,
                            i == 0 ? "prefill" : "decode step " + (i - 1),
                            baseline.tokens.get(i), treatment.tokens.get(i),
                            i < baseline.logitChecksums.size() ? String.valueOf(baseline.logitChecksums.get(i)) : "?",
                            i < treatment.logitChecksums.size() ? String.valueOf(treatment.logitChecksums.get(i)) : "?"));
        }
    }

    // ========== Helper Methods ==========

    private INDArray buildPrefillEmbeddings(int[] tokens) {
        long hiddenSize = embeddingTable.size(1);
        INDArray result = Nd4j.create(DataType.FLOAT, 1, tokens.length, hiddenSize);
        for (int i = 0; i < tokens.length; i++) {
            int tokenId = tokens[i];
            // Clamp to vocabulary size to avoid IndexOutOfBounds
            if (tokenId >= embeddingTable.size(0)) {
                tokenId = tokenId % (int) embeddingTable.size(0);
            }
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
