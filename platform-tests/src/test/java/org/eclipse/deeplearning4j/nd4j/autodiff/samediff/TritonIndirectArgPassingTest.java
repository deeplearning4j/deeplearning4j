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
 * Isolation test: does indirect arg passing (forced by tritonGraphCapture=true at COMPILE TIME)
 * cause wrong output, independent of whether CUDA graph capture/replay actually happens?
 *
 * <h3>Background</h3>
 * When {@code tritonGraphCapture()=true}, ALL Triton kernels are compiled with indirect arg
 * passing (pointer array + device-side load) instead of direct arg passing (kernel parameters).
 * This is the ONLY compile-time difference between the passing and failing configs.
 *
 * The indirect arg path works as follows:
 * <ul>
 *   <li>Direct args: each buffer pointer is a kernel parameter (up to 200 limit)</li>
 *   <li>Indirect args: all buffer pointers packed into a device-side int64 array,
 *       kernel unpacks via scalar loads: {@code ptr_i = load(argArray + i*8)}</li>
 * </ul>
 *
 * <h3>Test strategy</h3>
 * <ol>
 *   <li><b>Test 1 (direct args baseline):</b> tritonGraphCapture=false -- kernels compile
 *       with direct arg passing. Save per-step logit checksums and greedy tokens.</li>
 *   <li><b>Test 2 (indirect args, capture blocked):</b> tritonGraphCapture=true but
 *       captureMinExec=9999 so capture never triggers. Kernels still compile with indirect
 *       args. If tokens diverge from test 1, the bug is in indirect arg unpacking.</li>
 *   <li><b>Test 3 (indirect args, warmup step before capture):</b> tritonGraphCapture=true
 *       with normal capture timing. Check that the first decode step (executeCount=0,
 *       before capture kicks in) produces correct output via indirect args.</li>
 * </ol>
 *
 * <p>Run:</p>
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TritonIndirectArgPassingTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 \
 *     2>&1 | tee /tmp/indirect-arg-test.log
 * </pre>
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TritonIndirectArgPassingTest {

    private static final int NUM_DECODE_STEPS = 5;
    private static final long MAX_KV_LEN = 2048;
    private static final int[] PREFILL_TOKENS = {49229}; // <doctag>

    private SameDiff decoder;
    private SameDiff embedTokens;
    private INDArray embeddingTable;
    private String logitsName;
    private String embedsName;
    private ModelIOConfig ioConfig;
    private DecoderUtils.KVCacheNames kvNames;
    private boolean modelsLoaded = false;

    // Stored results from earlier tests for cross-test comparison
    private List<Integer> directArgTokens;
    private List<Double> directArgChecksums;

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

    // ========== Test 1: Direct Args Baseline ==========

    /**
     * Baseline: tritonGraphCapture=false -- kernels compile with direct arg passing.
     * All buffer pointers passed as kernel parameters.
     * This is the known-correct path (matches DIAG_TRITON_noGC config).
     */
    @Test
    @Order(1)
    @DisplayName("Direct arg passing baseline (tritonGraphCapture=false)")
    public void testDirectArgBaseline() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");

        Environment env = Nd4j.getEnvironment();

        // Save originals
        boolean origGraphCapture = env.tritonGraphCapture();
        boolean origCompileAll = env.tritonCompileAll();
        boolean origSectionFusion = env.tritonSectionFusion();
        int origCaptureMinExec = env.tritonCaptureMinExec();
        boolean origConsolidated = env.tritonConsolidatedArgTable();
        boolean origDirtyTracking = env.tritonArgDirtyTracking();

        try {
            // Direct args: tritonGraphCapture=false
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(false);   // direct args at compile time
            env.setTritonConsolidatedArgTable(false);
            env.setTritonArgDirtyTracking(false);

            log.info("=== DIRECT ARG BASELINE: tritonGraphCapture=false ===");
            log.info("  tritonCompileAll={}, sectionFusion={}, graphCapture={}, consolidatedArgTable={}",
                    env.tritonCompileAll(), env.tritonSectionFusion(),
                    env.tritonGraphCapture(), env.tritonConsolidatedArgTable());

            DecodeResult result = runDecodeSequence("DIRECT_ARGS");

            directArgTokens = result.tokens;
            directArgChecksums = result.logitChecksums;

            log.info("Direct arg tokens: {}", directArgTokens);
            log.info("Direct arg checksums: {}", directArgChecksums);

            // Sanity: tokens should not be degenerate
            assertNonDegenerate(directArgTokens, "DIRECT_ARGS");
        } finally {
            env.setTritonGraphCapture(origGraphCapture);
            env.setTritonCompileAll(origCompileAll);
            env.setTritonSectionFusion(origSectionFusion);
            env.setTritonCaptureMinExec(origCaptureMinExec);
            env.setTritonConsolidatedArgTable(origConsolidated);
            env.setTritonArgDirtyTracking(origDirtyTracking);
        }
    }

    // ========== Test 2: Indirect Args, Capture Blocked ==========

    /**
     * Key isolation test: tritonGraphCapture=true forces indirect arg compilation,
     * but captureMinExec=9999 prevents graph capture from ever triggering.
     *
     * If tokens diverge from direct arg baseline, the bug is in the indirect arg
     * pointer unpacking in the Triton kernel itself, NOT in graph replay.
     */
    @Test
    @Order(2)
    @DisplayName("Indirect arg passing, capture blocked (tritonGraphCapture=true, captureMinExec=9999)")
    public void testIndirectArgsCaptureBlocked() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");
        Assumptions.assumeTrue(directArgTokens != null, "Direct arg baseline not available -- run test 1 first");

        Environment env = Nd4j.getEnvironment();

        boolean origGraphCapture = env.tritonGraphCapture();
        boolean origCompileAll = env.tritonCompileAll();
        boolean origSectionFusion = env.tritonSectionFusion();
        int origCaptureMinExec = env.tritonCaptureMinExec();
        boolean origConsolidated = env.tritonConsolidatedArgTable();
        boolean origDirtyTracking = env.tritonArgDirtyTracking();

        try {
            // Indirect args forced, but capture blocked
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(true);     // forces indirect arg compilation
            env.setTritonCaptureMinExec(9999);   // blocks actual capture (needs 9999 execs first)
            env.setTritonConsolidatedArgTable(false);
            env.setTritonArgDirtyTracking(false);

            log.info("=== INDIRECT ARGS (CAPTURE BLOCKED): tritonGraphCapture=true, captureMinExec=9999 ===");
            log.info("  tritonCompileAll={}, sectionFusion={}, graphCapture={}, captureMinExec={}, consolidatedArgTable={}",
                    env.tritonCompileAll(), env.tritonSectionFusion(),
                    env.tritonGraphCapture(), env.tritonCaptureMinExec(),
                    env.tritonConsolidatedArgTable());

            DecodeResult result = runDecodeSequence("INDIRECT_BLOCKED");

            log.info("Indirect (blocked) tokens:    {}", result.tokens);
            log.info("Indirect (blocked) checksums: {}", result.logitChecksums);
            log.info("Direct arg baseline tokens:   {}", directArgTokens);
            log.info("Direct arg baseline checksums:{}", directArgChecksums);

            // Compare tokens
            compareTokens("indirectArgs_captureBlocked", directArgTokens, result.tokens);

            // Compare checksums (should be bitwise identical -- same kernels, just different arg passing)
            compareChecksums("indirectArgs_captureBlocked", directArgChecksums, result.logitChecksums);

        } finally {
            env.setTritonGraphCapture(origGraphCapture);
            env.setTritonCompileAll(origCompileAll);
            env.setTritonSectionFusion(origSectionFusion);
            env.setTritonCaptureMinExec(origCaptureMinExec);
            env.setTritonConsolidatedArgTable(origConsolidated);
            env.setTritonArgDirtyTracking(origDirtyTracking);
        }
    }

    // ========== Test 3: Indirect Args + Consolidated Arg Table, Capture Blocked ==========

    /**
     * Tests the consolidated arg table path (tritonConsolidatedArgTable=true) WITH
     * indirect args and capture blocked. The consolidated arg table packs ALL kernel
     * arg tables into one large pinned buffer and does a single H2D copy.
     *
     * If this diverges but test 2 does not, the bug is in consolidated arg table layout.
     */
    @Test
    @Order(3)
    @DisplayName("Indirect + consolidated arg table, capture blocked")
    public void testIndirectArgsConsolidatedCaptureBlocked() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");
        Assumptions.assumeTrue(directArgTokens != null, "Direct arg baseline not available -- run test 1 first");

        Environment env = Nd4j.getEnvironment();

        boolean origGraphCapture = env.tritonGraphCapture();
        boolean origCompileAll = env.tritonCompileAll();
        boolean origSectionFusion = env.tritonSectionFusion();
        int origCaptureMinExec = env.tritonCaptureMinExec();
        boolean origConsolidated = env.tritonConsolidatedArgTable();
        boolean origDirtyTracking = env.tritonArgDirtyTracking();

        try {
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(true);     // forces indirect arg compilation
            env.setTritonCaptureMinExec(9999);   // blocks actual capture
            env.setTritonConsolidatedArgTable(true);
            env.setTritonArgDirtyTracking(true);

            log.info("=== INDIRECT + CONSOLIDATED (CAPTURE BLOCKED) ===");
            log.info("  graphCapture={}, captureMinExec={}, consolidatedArgTable={}, argDirtyTracking={}",
                    env.tritonGraphCapture(), env.tritonCaptureMinExec(),
                    env.tritonConsolidatedArgTable(), env.tritonArgDirtyTracking());

            DecodeResult result = runDecodeSequence("INDIRECT_CONSOLIDATED_BLOCKED");

            log.info("Indirect+consolidated (blocked) tokens:    {}", result.tokens);
            log.info("Indirect+consolidated (blocked) checksums: {}", result.logitChecksums);
            log.info("Direct arg baseline tokens:                {}", directArgTokens);

            compareTokens("indirectArgs_consolidated_captureBlocked", directArgTokens, result.tokens);
            compareChecksums("indirectArgs_consolidated_captureBlocked", directArgChecksums, result.logitChecksums);

        } finally {
            env.setTritonGraphCapture(origGraphCapture);
            env.setTritonCompileAll(origCompileAll);
            env.setTritonSectionFusion(origSectionFusion);
            env.setTritonCaptureMinExec(origCaptureMinExec);
            env.setTritonConsolidatedArgTable(origConsolidated);
            env.setTritonArgDirtyTracking(origDirtyTracking);
        }
    }

    // ========== Test 4: Indirect Args with Normal Capture, Warmup Check ==========

    /**
     * Tests indirect args with normal capture timing (captureMinExec=2).
     * The first 2 decode steps execute kernels directly (warmup) before graph capture.
     * Those warmup steps use indirect args WITHOUT graph replay.
     *
     * If warmup steps match the baseline but post-capture steps diverge,
     * the bug is in graph replay (not indirect args). If warmup steps already
     * diverge, the bug is in indirect arg passing itself.
     */
    @Test
    @Order(4)
    @DisplayName("Indirect args with normal capture timing -- warmup vs post-capture divergence")
    public void testIndirectArgsNormalCaptureTiming() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");
        Assumptions.assumeTrue(directArgTokens != null, "Direct arg baseline not available -- run test 1 first");

        Environment env = Nd4j.getEnvironment();

        boolean origGraphCapture = env.tritonGraphCapture();
        boolean origCompileAll = env.tritonCompileAll();
        boolean origSectionFusion = env.tritonSectionFusion();
        int origCaptureMinExec = env.tritonCaptureMinExec();
        boolean origConsolidated = env.tritonConsolidatedArgTable();
        boolean origDirtyTracking = env.tritonArgDirtyTracking();
        boolean origAllowFallback = env.tritonAllowFallbackCapture();

        try {
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(true);     // forces indirect arg compilation
            env.setTritonCaptureMinExec(2);      // capture on 3rd execution (normal timing)
            env.setTritonConsolidatedArgTable(true);
            env.setTritonArgDirtyTracking(true);
            env.setTritonAllowFallbackCapture(true);

            log.info("=== INDIRECT ARGS + NORMAL CAPTURE (captureMinExec=2) ===");
            log.info("  graphCapture={}, captureMinExec={}, consolidatedArgTable={}, argDirtyTracking={}",
                    env.tritonGraphCapture(), env.tritonCaptureMinExec(),
                    env.tritonConsolidatedArgTable(), env.tritonArgDirtyTracking());

            DecodeResult result = runDecodeSequence("INDIRECT_NORMAL_CAPTURE");

            log.info("Indirect+capture tokens:    {}", result.tokens);
            log.info("Indirect+capture checksums: {}", result.logitChecksums);
            log.info("Direct arg baseline tokens: {}", directArgTokens);

            // Analyze divergence point relative to capture timing
            // captureMinExec=2 means steps 0,1 are warmup (indirect args, no graph replay)
            // Step 2+ may be captured/replayed
            int firstDivergence = -1;
            for (int i = 0; i < Math.min(directArgTokens.size(), result.tokens.size()); i++) {
                if (!directArgTokens.get(i).equals(result.tokens.get(i))) {
                    firstDivergence = i;
                    break;
                }
            }

            if (firstDivergence >= 0) {
                // Index 0 = prefill token, index 1 = decode step 0, etc.
                int decodeStep = firstDivergence - 1; // -1 because index 0 is prefill
                boolean duringWarmup = decodeStep < 2; // captureMinExec=2, so steps 0,1 are warmup
                log.error("DIVERGENCE at token index {} (decode step {}): baseline={} treatment={}",
                        firstDivergence, decodeStep,
                        directArgTokens.get(firstDivergence), result.tokens.get(firstDivergence));
                if (duringWarmup) {
                    log.error(">>> DIVERGENCE IS DURING WARMUP (before graph capture) <<<");
                    log.error(">>> This means the BUG is in INDIRECT ARG PASSING, not graph replay <<<");
                } else {
                    log.error(">>> DIVERGENCE IS AFTER CAPTURE BEGINS (decode step {} >= captureMinExec=2) <<<",
                            decodeStep);
                    log.error(">>> This means the BUG is likely in GRAPH REPLAY, not indirect args <<<");
                }
            } else {
                log.info("ALL tokens match baseline -- indirect args + capture are correct");
            }

            // Compare warmup steps specifically (indices 0, 1, 2 = prefill + decode steps 0,1)
            int warmupLimit = Math.min(3, Math.min(directArgTokens.size(), result.tokens.size()));
            boolean warmupMatch = true;
            for (int i = 0; i < warmupLimit; i++) {
                if (!directArgTokens.get(i).equals(result.tokens.get(i))) {
                    warmupMatch = false;
                    log.error("WARMUP token mismatch at index {}: baseline={} treatment={}",
                            i, directArgTokens.get(i), result.tokens.get(i));
                }
            }
            log.info("Warmup steps (before capture) match baseline: {}", warmupMatch);

            // Full comparison
            compareTokens("indirectArgs_normalCapture", directArgTokens, result.tokens);

        } finally {
            env.setTritonGraphCapture(origGraphCapture);
            env.setTritonCompileAll(origCompileAll);
            env.setTritonSectionFusion(origSectionFusion);
            env.setTritonCaptureMinExec(origCaptureMinExec);
            env.setTritonConsolidatedArgTable(origConsolidated);
            env.setTritonArgDirtyTracking(origDirtyTracking);
            env.setTritonAllowFallbackCapture(origAllowFallback);
        }
    }

    // ========== Test 5: Indirect Args + batchedGemm, Capture Blocked ==========

    /**
     * Tests indirect args + batchedGemm (without TF32) vs direct arg baseline.
     * TF32 is excluded because it triggers a separate Triton compiler bug
     * (arith.bitcast legalization failure) that is unrelated to indirect arg passing.
     *
     * If tests 2-3 pass but this fails, the bug is in batchedGemm interaction
     * with indirect args.
     */
    @Test
    @Order(5)
    @DisplayName("Indirect args + batchedGemm (no TF32), capture blocked")
    public void testIndirectArgsBatchedGemmCaptureBlocked() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");
        Assumptions.assumeTrue(directArgTokens != null, "Direct arg baseline not available -- run test 1 first");

        Environment env = Nd4j.getEnvironment();

        // Save ALL relevant originals
        boolean origGraphCapture = env.tritonGraphCapture();
        boolean origCompileAll = env.tritonCompileAll();
        boolean origSectionFusion = env.tritonSectionFusion();
        int origCaptureMinExec = env.tritonCaptureMinExec();
        boolean origConsolidated = env.tritonConsolidatedArgTable();
        boolean origDirtyTracking = env.tritonArgDirtyTracking();
        boolean origAllowFallback = env.tritonAllowFallbackCapture();
        boolean origBatchedGemm = env.dspBatchedGemm();

        try {
            // Indirect args + consolidated + batchedGemm, capture blocked, NO TF32
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(true);     // indirect args
            env.setTritonCaptureMinExec(9999);   // block capture
            env.setTritonConsolidatedArgTable(true);
            env.setTritonArgDirtyTracking(true);
            env.setTritonAllowFallbackCapture(true);
            env.setDspBatchedGemm(true);

            log.info("=== INDIRECT + CONSOLIDATED + BATCHED_GEMM (CAPTURE BLOCKED, no TF32) ===");
            log.info("  graphCapture={}, captureMinExec={}, consolidatedArgTable={}, batchedGemm={}",
                    env.tritonGraphCapture(), env.tritonCaptureMinExec(),
                    env.tritonConsolidatedArgTable(), env.dspBatchedGemm());

            DecodeResult result = runDecodeSequence("INDIRECT_BGEMM_BLOCKED");

            log.info("Indirect+bGemm (blocked) tokens:    {}", result.tokens);
            log.info("Indirect+bGemm (blocked) checksums: {}", result.logitChecksums);
            log.info("Direct arg baseline tokens:         {}", directArgTokens);

            compareTokens("indirectArgs_batchedGemm_captureBlocked", directArgTokens, result.tokens);
            compareChecksums("indirectArgs_batchedGemm_captureBlocked", directArgChecksums, result.logitChecksums);

        } finally {
            env.setTritonGraphCapture(origGraphCapture);
            env.setTritonCompileAll(origCompileAll);
            env.setTritonSectionFusion(origSectionFusion);
            env.setTritonCaptureMinExec(origCaptureMinExec);
            env.setTritonConsolidatedArgTable(origConsolidated);
            env.setTritonArgDirtyTracking(origDirtyTracking);
            env.setTritonAllowFallbackCapture(origAllowFallback);
            env.setDspBatchedGemm(origBatchedGemm);
        }
    }

    // ========== Core Decode Logic ==========

    /**
     * Result container for a decode sequence -- tokens + per-step logit checksums.
     */
    private static class DecodeResult {
        final List<Integer> tokens;
        final List<Double> logitChecksums;
        final List<Double> logitMaxValues;

        DecodeResult(List<Integer> tokens, List<Double> logitChecksums, List<Double> logitMaxValues) {
            this.tokens = tokens;
            this.logitChecksums = logitChecksums;
            this.logitMaxValues = logitMaxValues;
        }
    }

    /**
     * Run prefill + N decode steps, returning greedy-sampled token IDs and logit checksums.
     *
     * Uses full output names (logits + KV) with frozen shapes and C++ KV scatter.
     *
     * @param label human-readable label for log messages
     * @return DecodeResult with tokens, checksums, and max values
     */
    private DecodeResult runDecodeSequence(String label) {
        // Reset state
        decoder.resetSession();
        embedTokens.resetSession();
        InferenceSession.setDynamicShapePlanEnabled(true);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        String[] fullOutputNames = buildFullOutputNames();
        String embedOutputName = embedTokens.outputs().get(0);

        // ---- Prefill ----
        INDArray prefillEmbeds = buildPrefillEmbeddings(PREFILL_TOKENS);
        INDArray inputIds = Nd4j.createFromArray(PREFILL_TOKENS)
                .reshape(1, PREFILL_TOKENS.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = PREFILL_TOKENS.length;

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

        List<Integer> tokens = new ArrayList<>();
        List<Double> checksums = new ArrayList<>();
        List<Double> maxValues = new ArrayList<>();

        tokens.add(nextToken);
        checksums.add(prefillLogits.sumNumber().doubleValue());
        maxValues.add(prefillLogits.maxNumber().doubleValue());

        log.info("[{}] Prefill token: {} logitSum={} logitMax={}",
                label, nextToken,
                checksums.get(0), maxValues.get(0));

        // Initialize static KV cache
        StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
        kvMgr.initializeFromPrefill(prefillOutputs);

        // Close prefill outputs
        closePrefillOutputs(prefillOutputs, prefillLogits);

        // ---- Recompile for decode ----
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();
        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, fullOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        // Configure frozen shapes + C++ KV scatter
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
                log.info("[{}] C++ KV scatter configured: {}", label, cppKvScatter);
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
            tokens.add(nextToken);

            double checksum = stepLogits.sumNumber().doubleValue();
            double maxVal = stepLogits.maxNumber().doubleValue();
            checksums.add(checksum);
            maxValues.add(maxVal);

            log.info("[{}] Step {}: token={} logitSum={} logitMax={} cachePos={}",
                    label, step, nextToken, checksum, maxVal, cachePos);

            // KV cache management
            if (!cppKvScatter) {
                kvMgr.scatterNewEntries(outputs);
            } else {
                kvMgr.advancePosition();
            }
        }

        return new DecodeResult(tokens, checksums, maxValues);
    }

    // ========== Token Comparison ==========

    /**
     * Compare two token sequences and assert they match.
     * On mismatch, logs the first divergence point with full context.
     */
    private void compareTokens(String settingName, List<Integer> baseline, List<Integer> treatment) {
        assertEquals(baseline.size(), treatment.size(),
                String.format("[%s] Token count mismatch: baseline=%d treatment=%d",
                        settingName, baseline.size(), treatment.size()));

        int firstDivergence = -1;
        int matchCount = 0;
        for (int i = 0; i < baseline.size(); i++) {
            if (baseline.get(i).equals(treatment.get(i))) {
                matchCount++;
            } else if (firstDivergence == -1) {
                firstDivergence = i;
            }
        }

        double matchRate = (double) matchCount / baseline.size() * 100.0;
        log.info("[{}] Token match rate: {}/{} ({}%)", settingName, matchCount, baseline.size(),
                String.format("%.1f", matchRate));

        if (firstDivergence >= 0) {
            String stepLabel = firstDivergence == 0 ? "prefill" : "decode step " + (firstDivergence - 1);
            log.error("[{}] FIRST TOKEN DIVERGENCE at index {} ({}): baseline={} treatment={}",
                    settingName, firstDivergence, stepLabel,
                    baseline.get(firstDivergence), treatment.get(firstDivergence));
            log.error("[{}] Full baseline:  {}", settingName, baseline);
            log.error("[{}] Full treatment: {}", settingName, treatment);

            // Check if treatment output is degenerate
            boolean degenerate = true;
            for (int i = firstDivergence + 1; i < treatment.size(); i++) {
                if (!treatment.get(i).equals(treatment.get(firstDivergence))) {
                    degenerate = false;
                    break;
                }
            }
            if (degenerate && treatment.size() - firstDivergence > 2) {
                log.error("[{}] Treatment output is DEGENERATE -- all tokens from divergence point are {}",
                        settingName, treatment.get(firstDivergence));
            }
        }

        for (int i = 0; i < baseline.size(); i++) {
            assertEquals(baseline.get(i), treatment.get(i),
                    String.format("[%s] Token mismatch at index %d (%s): baseline=%d treatment=%d",
                            settingName, i,
                            i == 0 ? "prefill" : "decode step " + (i - 1),
                            baseline.get(i), treatment.get(i)));
        }
    }

    /**
     * Compare logit checksums between baseline and treatment.
     * Logs differences but does not assert exact equality (floating point).
     */
    private void compareChecksums(String settingName, List<Double> baseline, List<Double> treatment) {
        int minLen = Math.min(baseline.size(), treatment.size());
        for (int i = 0; i < minLen; i++) {
            double diff = Math.abs(baseline.get(i) - treatment.get(i));
            double relDiff = baseline.get(i) != 0 ? diff / Math.abs(baseline.get(i)) : diff;
            String stepLabel = i == 0 ? "prefill" : "step " + (i - 1);
            if (relDiff > 1e-5) {
                log.warn("[{}] Checksum divergence at {} : baseline={} treatment={} relDiff={}",
                        settingName, stepLabel, baseline.get(i), treatment.get(i),
                        String.format("%.2e", relDiff));
            } else {
                log.info("[{}] Checksum match at {} : baseline={} treatment={} relDiff={}",
                        settingName, stepLabel, baseline.get(i), treatment.get(i),
                        String.format("%.2e", relDiff));
            }
        }
    }

    /**
     * Assert that a token sequence is not degenerate (all same token after the first).
     */
    private void assertNonDegenerate(List<Integer> tokens, String label) {
        if (tokens.size() <= 2) return;
        Set<Integer> unique = new HashSet<>(tokens.subList(1, tokens.size()));
        assertTrue(unique.size() > 1,
                String.format("[%s] DEGENERATE output -- all decode tokens are %d. Full: %s",
                        label, tokens.get(1), tokens));
    }

    // ========== Helper Methods ==========

    private INDArray buildPrefillEmbeddings(int[] tokens) {
        long hiddenSize = embeddingTable.size(1);
        INDArray result = Nd4j.create(DataType.FLOAT, 1, tokens.length, hiddenSize);
        for (int i = 0; i < tokens.length; i++) {
            result.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all())
                    .assign(embeddingTable.getRow(tokens[i]));
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
