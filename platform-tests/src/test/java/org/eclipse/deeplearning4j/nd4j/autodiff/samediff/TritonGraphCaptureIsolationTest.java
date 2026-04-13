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
 * Isolation test: does tritonGraphCapture(true) cause wrong output in the
 * SmolDocling VLM decoder?
 *
 * Each test method toggles ONE setting against a known-good baseline (same model,
 * same inputs, setting OFF) and asserts greedy-decoded tokens match.
 *
 * Test methods:
 *   - testTritonGraphCaptureIsolation:       toggle tritonGraphCapture
 *   - testBatchedGemmIsolation:              toggle dspBatchedGemm
 *   - testConsolidatedArgTableIsolation:     toggle tritonConsolidatedArgTable + argDirtyTracking
 *   - testTf32Isolation:                     toggle cublasTf32 + tritonTf32
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=TritonGraphCaptureIsolationTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 \
 *     2>&1 | tee /tmp/gc-isolation.log
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TritonGraphCaptureIsolationTest {

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

    // ========== Test Methods ==========

    /**
     * Core test: does tritonGraphCapture(true) cause wrong output?
     *
     * Baseline: Triton compilation ON, graph capture OFF.
     * Treatment: Triton compilation ON, graph capture ON.
     */
    @Test
    @Order(1)
    @DisplayName("tritonGraphCapture ON vs OFF must produce identical tokens")
    public void testTritonGraphCaptureIsolation() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");

        Environment env = Nd4j.getEnvironment();

        // Save original values
        boolean origGraphCapture = env.tritonGraphCapture();
        boolean origCompileAll = env.tritonCompileAll();
        boolean origSectionFusion = env.tritonSectionFusion();

        try {
            // ---- Baseline: Triton ON, graph capture OFF ----
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(false);
            log.info("=== BASELINE: tritonGraphCapture=OFF ===");
            List<Integer> baselineTokens = runDecodeSequence("BASELINE");
            log.info("Baseline tokens: {}", baselineTokens);

            // ---- Treatment: Triton ON, graph capture ON ----
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonGraphCapture(true);
            log.info("=== TREATMENT: tritonGraphCapture=ON ===");
            List<Integer> captureTokens = runDecodeSequence("GRAPH_CAPTURE");
            log.info("Graph capture tokens: {}", captureTokens);

            // ---- Compare ----
            compareTokens("tritonGraphCapture", baselineTokens, captureTokens);
        } finally {
            env.setTritonGraphCapture(origGraphCapture);
            env.setTritonCompileAll(origCompileAll);
            env.setTritonSectionFusion(origSectionFusion);
        }
    }

    /**
     * Isolation: does dspBatchedGemm cause wrong output?
     *
     * Baseline: dspBatchedGemm OFF.
     * Treatment: dspBatchedGemm ON.
     */
    @Test
    @Order(2)
    @DisplayName("dspBatchedGemm ON vs OFF must produce identical tokens")
    public void testBatchedGemmIsolation() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");

        Environment env = Nd4j.getEnvironment();

        boolean origBatchedGemm = env.dspBatchedGemm();
        boolean origCompileAll = env.tritonCompileAll();
        boolean origSectionFusion = env.tritonSectionFusion();

        try {
            // ---- Baseline: batchedGemm OFF ----
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setDspBatchedGemm(false);
            log.info("=== BASELINE: dspBatchedGemm=OFF ===");
            List<Integer> baselineTokens = runDecodeSequence("BATCHED_GEMM_OFF");
            log.info("Baseline tokens: {}", baselineTokens);

            // ---- Treatment: batchedGemm ON ----
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setDspBatchedGemm(true);
            log.info("=== TREATMENT: dspBatchedGemm=ON ===");
            List<Integer> treatmentTokens = runDecodeSequence("BATCHED_GEMM_ON");
            log.info("BatchedGemm tokens: {}", treatmentTokens);

            // ---- Compare ----
            compareTokens("dspBatchedGemm", baselineTokens, treatmentTokens);
        } finally {
            env.setDspBatchedGemm(origBatchedGemm);
            env.setTritonCompileAll(origCompileAll);
            env.setTritonSectionFusion(origSectionFusion);
        }
    }

    /**
     * Isolation: does tritonConsolidatedArgTable + argDirtyTracking cause wrong output?
     *
     * Baseline: both OFF.
     * Treatment: both ON.
     */
    @Test
    @Order(3)
    @DisplayName("tritonConsolidatedArgTable + argDirtyTracking ON vs OFF must produce identical tokens")
    public void testConsolidatedArgTableIsolation() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");

        Environment env = Nd4j.getEnvironment();

        boolean origConsolidated = env.tritonConsolidatedArgTable();
        boolean origDirtyTracking = env.tritonArgDirtyTracking();
        boolean origCompileAll = env.tritonCompileAll();
        boolean origSectionFusion = env.tritonSectionFusion();

        try {
            // ---- Baseline: consolidated + dirty tracking OFF ----
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonConsolidatedArgTable(false);
            env.setTritonArgDirtyTracking(false);
            log.info("=== BASELINE: consolidatedArgTable=OFF, argDirtyTracking=OFF ===");
            List<Integer> baselineTokens = runDecodeSequence("CONSOLIDATED_OFF");
            log.info("Baseline tokens: {}", baselineTokens);

            // ---- Treatment: both ON ----
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setTritonConsolidatedArgTable(true);
            env.setTritonArgDirtyTracking(true);
            log.info("=== TREATMENT: consolidatedArgTable=ON, argDirtyTracking=ON ===");
            List<Integer> treatmentTokens = runDecodeSequence("CONSOLIDATED_ON");
            log.info("Consolidated tokens: {}", treatmentTokens);

            // ---- Compare ----
            compareTokens("tritonConsolidatedArgTable+argDirtyTracking", baselineTokens, treatmentTokens);
        } finally {
            env.setTritonConsolidatedArgTable(origConsolidated);
            env.setTritonArgDirtyTracking(origDirtyTracking);
            env.setTritonCompileAll(origCompileAll);
            env.setTritonSectionFusion(origSectionFusion);
        }
    }

    /**
     * Isolation: does cublasTf32 + tritonTf32 cause wrong output?
     *
     * Baseline: both OFF (FP32 precision).
     * Treatment: both ON (TF32 reduced precision).
     */
    @Test
    @Order(4)
    @DisplayName("cublasTf32 + tritonTf32 ON vs OFF must produce identical tokens")
    public void testTf32Isolation() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");

        Environment env = Nd4j.getEnvironment();

        boolean origCublasTf32 = env.cublasTf32Enabled();
        boolean origTritonTf32 = env.tritonTf32Enabled();
        boolean origCompileAll = env.tritonCompileAll();
        boolean origSectionFusion = env.tritonSectionFusion();

        try {
            // ---- Baseline: TF32 OFF (full FP32) ----
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setCublasTf32Enabled(false);
            env.setTritonTf32Enabled(false);
            log.info("=== BASELINE: cublasTf32=OFF, tritonTf32=OFF ===");
            List<Integer> baselineTokens = runDecodeSequence("TF32_OFF");
            log.info("Baseline tokens (FP32): {}", baselineTokens);

            // ---- Treatment: TF32 ON ----
            env.setTritonCompileAll(true);
            env.setTritonSectionFusion(true);
            env.setCublasTf32Enabled(true);
            env.setTritonTf32Enabled(true);
            log.info("=== TREATMENT: cublasTf32=ON, tritonTf32=ON ===");
            List<Integer> treatmentTokens = runDecodeSequence("TF32_ON");
            log.info("TF32 tokens: {}", treatmentTokens);

            // ---- Compare ----
            compareTokens("cublasTf32+tritonTf32", baselineTokens, treatmentTokens);
        } finally {
            env.setCublasTf32Enabled(origCublasTf32);
            env.setTritonTf32Enabled(origTritonTf32);
            env.setTritonCompileAll(origCompileAll);
            env.setTritonSectionFusion(origSectionFusion);
        }
    }

    // ========== Core Decode Logic ==========

    /**
     * Run prefill + N decode steps, returning greedy-sampled token IDs.
     *
     * This method resets all decoder state, runs prefill, recompiles for decode,
     * and then runs NUM_DECODE_STEPS greedy decode steps.
     *
     * @param label human-readable label for log messages
     * @return list of token IDs (prefill token + decode tokens)
     */
    private List<Integer> runDecodeSequence(String label) {
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
        tokens.add(nextToken);
        log.info("[{}] Prefill token: {} logitSum={}", label, nextToken,
                prefillLogits.sumNumber().doubleValue());

        // Initialize static KV cache
        StaticKvManager kvMgr = new StaticKvManager(kvNames, MAX_KV_LEN);
        kvMgr.initializeFromPrefill(prefillOutputs);

        // Close prefill KV outputs (logits too)
        closePrefillOutputs(prefillOutputs, prefillLogits);

        // ---- Recompile for decode ----
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();
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
            log.info("[{}] Step {}: token={} logitSum={} logitMax={} cachePos={}",
                    label, step, nextToken, checksum, maxVal, cachePos);

            // KV cache management
            if (!cppKvScatter) {
                kvMgr.scatterNewEntries(outputs);
            } else {
                kvMgr.advancePosition();
            }
        }

        return tokens;
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
        log.info("[{}] Match rate: {}/{} ({}%)", settingName, matchCount, baseline.size(),
                String.format("%.1f", matchRate));

        if (firstDivergence >= 0) {
            String stepLabel = firstDivergence == 0 ? "prefill" : "decode step " + (firstDivergence - 1);
            log.error("[{}] FIRST DIVERGENCE at index {} ({}): baseline={} treatment={}",
                    settingName, firstDivergence, stepLabel,
                    baseline.get(firstDivergence), treatment.get(firstDivergence));
            log.error("[{}] Full baseline:  {}", settingName, baseline);
            log.error("[{}] Full treatment: {}", settingName, treatment);

            // Check if treatment output is degenerate (all same token after divergence)
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
