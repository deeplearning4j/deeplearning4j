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
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DspCompilationMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression test: setShapesFrozen(true) must NOT change decode token output.
 *
 * Bug: With shapes frozen, the C++ DSP plan takes a different execution path
 * (phaseWarmup) that produces degenerate tokens compared to the unfrozen path
 * (phaseSlotBySlot).
 *
 * This test loads the SmolDocling decoder, runs prefill + decode steps WITHOUT
 * freezing, then runs the same sequence WITH freezing, and asserts the tokens match.
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestFrozenDecodeCorrectness {

    private SameDiff decoder;
    private SameDiff embedTokens;
    private INDArray embeddingTable;
    private String logitsName;
    private String embedsName;
    private String inputIdsName;
    private ModelIOConfig ioConfig;
    private DecoderUtils.KVCacheNames kvNames;
    private boolean modelsLoaded = false;

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

            // Discover IO config
            ioConfig = ModelIOConfig.discover(decoder);
            kvNames = ioConfig.getKvCacheNames();
            logitsName = ioConfig.getLogitsOutputName();
            embedsName = ioConfig.getInputEmbeddingsName();

            // Find input names
            inputIdsName = null;
            for (String inp : decoder.inputs()) {
                if (inp.contains("input_id") || inp.equals("input_ids")) {
                    inputIdsName = inp;
                    break;
                }
            }
            if (inputIdsName == null) inputIdsName = "input_ids";

            modelsLoaded = true;
            log.info("Loaded: decoder={} ops, embedTokens={} ops, logits={}, embeds={}, kvNames={}",
                    decoder.ops().length, embedTokens.ops().length, logitsName, embedsName,
                    kvNames.keyNames);
        } catch (Exception e) {
            log.error("Failed to load models: {}", e.getMessage(), e);
        }
    }

    @AfterAll
    public void cleanup() {
        if (decoder != null) decoder.close();
        if (embedTokens != null) embedTokens.close();
    }

    private INDArray buildPrefillEmbeddings(int[] tokens) {
        long hiddenSize = embeddingTable.size(1);
        INDArray result = Nd4j.create(DataType.FLOAT, 1, tokens.length, hiddenSize);
        for (int i = 0; i < tokens.length; i++) {
            result.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all())
                    .assign(embeddingTable.getRow(tokens[i]));
        }
        return result;
    }

    /**
     * Build full output names array (logits + all KV present outputs).
     */
    private String[] buildFullOutputNames() {
        List<String> names = new ArrayList<>();
        names.add(logitsName);
        names.addAll(kvNames.keyNames);
        names.addAll(kvNames.valueNames);
        return names.toArray(new String[0]);
    }

    /**
     * Run decode sequence and return tokens.
     *
     * @param freezeShapes whether to call setShapesFrozen(true) after recompile
     * @param numSteps number of decode steps
     * @return list of token IDs
     */
    private List<Integer> runDecodeSequence(boolean freezeShapes, int numSteps) {
        // Reset state
        decoder.resetSession();
        embedTokens.resetSession();
        InferenceSession.setDynamicShapePlanEnabled(true);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        String[] fullOutputNames = buildFullOutputNames();
        String embedOutputName = embedTokens.outputs().get(0);

        // ---- Prefill ----
        int[] prefillTokens = {49229}; // <doctag>
        INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokens);
        INDArray inputIds = Nd4j.createFromArray(prefillTokens).reshape(1, prefillTokens.length).castTo(DataType.LONG);
        long prefillSeqLen = prefillTokens.length;

        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, embeddingTable.size(1));

        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, "Prefill logits must not be null");

        // Get first token
        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(prefillLogits.size(1) - 1),
                    NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        List<Integer> tokens = new ArrayList<>();
        tokens.add(nextToken);
        log.info("[{}] Prefill token: {} logitSum={}", freezeShapes ? "FROZEN" : "UNFROZEN", nextToken, prefillLogits.sumNumber().doubleValue());

        // Initialize static KV
        StaticKvForTest kvMgr = new StaticKvForTest(kvNames, 2048);
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
        // Close prefill logits
        if (prefillLogits != null && !prefillLogits.wasClosed()) {
            prefillLogits.setCloseable(true); prefillLogits.close();
        }

        // ---- Recompile for decode ----
        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearAllCaches();
        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, fullOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        if (freezeShapes && dspExec != null) {
            dspExec.setShapesFrozen(true);
            dspExec.setTraceEnabled(true);
            dspExec.setExecutionTimingEnabled(true);
            log.info("[{}] Shapes frozen, planPhase={}, pointersStable={}",
                    freezeShapes ? "FROZEN" : "UNFROZEN",
                    dspExec.getPlanPhase(), dspExec.arePointersStable());

            // Configure C++ KV scatter — this is where the bug manifests.
            // When C++ KV scatter is enabled with frozen shapes, the KV cache
            // gets corrupted, producing wrong tokens from step 2 onwards.
            if (dspExec.getCurrentPlan() != null) {
                List<String> presentNames = new ArrayList<>();
                presentNames.addAll(kvNames.keyNames);
                presentNames.addAll(kvNames.valueNames);
                List<String> pastNames = new ArrayList<>();
                for (String pn : presentNames) {
                    pastNames.add(ioConfig.presentToInputName(pn));
                }
                boolean configured = dspExec.configureKvCacheRetention(
                        dspExec.getCurrentPlan(), presentNames, pastNames,
                        (int) kvMgr.getMaxKvLen(), (int) kvMgr.getCachePosition());
                log.info("[{}] C++ KV scatter configured: {} mappings={}",
                        freezeShapes ? "FROZEN" : "UNFROZEN", configured, presentNames.size());
                if (configured) {
                    dspExec.configureDecodeInputs(dspExec.getCurrentPlan(), (int) kvMgr.getMaxKvLen());
                }
            }
        }

        // ---- Decode steps ----
        Map<String, INDArray> reusableInputs = new HashMap<>();
        boolean cppKvScatter = freezeShapes && dspExec != null && dspExec.getCurrentPlan() != null;

        for (int step = 0; step < numSteps; step++) {
            long pastSeqLen = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();
            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken}).reshape(1, 1).castTo(DataType.LONG);

            // Get token embedding
            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            // Build decode inputs
            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, stepEmbed, tokenIdArr,
                    pastSeqLen, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, embeddingTable.size(1), reusableInputs, true);

            // Execute — ALWAYS use fullOutputNames so both paths get the same outputs
            Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, fullOutputNames);

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, String.format("[%s] Step %d logits must not be null",
                    freezeShapes ? "FROZEN" : "UNFROZEN", step));

            // Get token
            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(stepLogits.size(1) - 1),
                        NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            tokens.add(nextToken);

            // Logit checksum for debugging
            double checksum = stepLogits.sumNumber().doubleValue();
            double maxVal = stepLogits.maxNumber().doubleValue();
            log.info("[{}] Step {}: token={} logitSum={} logitMax={} cachePos={} cppScatter={}",
                    freezeShapes ? "FROZEN" : "UNFROZEN", step, nextToken,
                    checksum, maxVal, cachePos, cppKvScatter);

            // Java scatter only when C++ scatter is NOT configured.
            // When C++ scatter IS configured, we still need to track the cache position
            // on the Java side so subsequent steps use the correct position.
            if (!cppKvScatter) {
                kvMgr.scatterNewEntries(outputs);
            } else {
                // C++ scatter ran internally — advance Java-side position tracker
                kvMgr.advancePosition();
            }
        }

        return tokens;
    }

    @Test
    @Order(1)
    @DisplayName("Frozen decode must produce same tokens as unfrozen decode")
    public void testFrozenVsUnfrozenDecodeTokens() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int numSteps = 5;

        // Unfrozen run
        List<Integer> unfrozenTokens = runDecodeSequence(false, numSteps);
        log.info("Unfrozen tokens: {}", unfrozenTokens);

        // Frozen run
        List<Integer> frozenTokens = runDecodeSequence(true, numSteps);
        log.info("Frozen tokens: {}", frozenTokens);

        // Compare
        assertEquals(unfrozenTokens.size(), frozenTokens.size(),
                "Token count must match");

        for (int i = 0; i < unfrozenTokens.size(); i++) {
            assertEquals(unfrozenTokens.get(i), frozenTokens.get(i),
                    String.format("Token mismatch at step %d: unfrozen=%d frozen=%d",
                            i, unfrozenTokens.get(i), frozenTokens.get(i)));
        }
    }

    /**
     * Isolate: logits-only compilation + frozen + C++ KV scatter.
     * This matches the real StaticKvCacheDecodeLoop path.
     * Compares against the unfrozen baseline (fullOutputNames) from testFrozenVsUnfrozenDecodeTokens.
     */
    @Test
    @Order(2)
    @DisplayName("Logits-only compilation must produce same tokens as full-output compilation")
    public void testLogitsOnlyVsFullOutputCompilation() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int numSteps = 5;

        // Baseline: unfrozen with full outputs (known correct from test 1)
        List<Integer> baselineTokens = runDecodeSequence(false, numSteps);
        log.info("Baseline tokens (unfrozen, full outputs): {}", baselineTokens);

        // Test: logits-only compilation + frozen + C++ KV scatter
        List<Integer> logitsOnlyTokens = runDecodeSequenceLogitsOnly(numSteps);
        log.info("LogitsOnly tokens (frozen, C++ scatter): {}", logitsOnlyTokens);

        assertEquals(baselineTokens.size(), logitsOnlyTokens.size(), "Token count must match");
        for (int i = 0; i < baselineTokens.size(); i++) {
            assertEquals(baselineTokens.get(i), logitsOnlyTokens.get(i),
                    String.format("Token mismatch at step %d: baseline=%d logitsOnly=%d",
                            i, baselineTokens.get(i), logitsOnlyTokens.get(i)));
        }
    }

    /**
     * Test: does cuBLAS TF32 precision cause degenerate output in frozen decode?
     *
     * TF32 uses 10-bit mantissa (vs 23-bit for FP32), which gives ~2x throughput
     * but reduced precision. This test checks whether the reduced precision causes
     * the frozen decode path to produce degenerate (repetitive) tokens like "upsupsupsup".
     *
     * Suspected culprit per FrozenPathDegenerateIsolationTest analysis:
     * cuBLAS TF32 rounding in matmul accumulation may push borderline logits
     * across argmax decision boundaries, causing cascading token divergence.
     */
    @Test
    @Order(4)
    @DisplayName("TF32 precision must not cause degenerate frozen decode output")
    public void testTf32VsBaselineDecodeTokens() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int numSteps = 5;

        // 1) Unfrozen baseline (FP32, no TF32) — known correct
        assertFalse(Nd4j.getEnvironment().cublasTf32Enabled(), "TF32 should be off by default");
        List<Integer> baselineTokens = runDecodeSequence(false, numSteps);
        log.info("Baseline tokens (unfrozen, FP32): {}", baselineTokens);

        // 2) Frozen decode with TF32 enabled
        boolean wasTf32 = Nd4j.getEnvironment().cublasTf32Enabled();
        try {
            Nd4j.getEnvironment().setCublasTf32Enabled(true);
            assertTrue(Nd4j.getEnvironment().cublasTf32Enabled(), "TF32 should now be enabled");
            log.info("cuBLAS TF32 ENABLED — running frozen decode with reduced precision matmul");

            List<Integer> tf32Tokens = runDecodeSequenceLogitsOnly(numSteps);
            log.info("TF32 tokens (frozen, logits-only, C++ scatter): {}", tf32Tokens);

            // Check for degenerate output: all tokens the same after step 0
            boolean allSame = true;
            for (int i = 1; i < tf32Tokens.size(); i++) {
                if (!tf32Tokens.get(i).equals(tf32Tokens.get(1))) {
                    allSame = false;
                    break;
                }
            }
            assertFalse(allSame && tf32Tokens.size() > 2,
                    String.format("TF32 frozen decode produced degenerate output — all decode tokens are %d. " +
                            "Tokens: %s. This confirms cuBLAS TF32 is the culprit for frozen decode quality regression.",
                            tf32Tokens.get(1), tf32Tokens));

            // Compare with baseline — TF32 may diverge slightly but should not be wildly different
            int matchCount = 0;
            for (int i = 0; i < Math.min(baselineTokens.size(), tf32Tokens.size()); i++) {
                if (baselineTokens.get(i).equals(tf32Tokens.get(i))) matchCount++;
            }
            double matchRate = (double) matchCount / baselineTokens.size();
            log.info("TF32 vs FP32 match rate: {}/{} ({}%)", matchCount, baselineTokens.size(),
                    String.format("%.1f", matchRate * 100));

            // Log whether first token (from prefill, before frozen decode) matches
            if (!baselineTokens.isEmpty() && !tf32Tokens.isEmpty()) {
                log.info("Prefill token match: {} (baseline={}, tf32={})",
                        baselineTokens.get(0).equals(tf32Tokens.get(0)),
                        baselineTokens.get(0), tf32Tokens.get(0));
            }

            // Log divergence points
            for (int i = 0; i < Math.min(baselineTokens.size(), tf32Tokens.size()); i++) {
                if (!baselineTokens.get(i).equals(tf32Tokens.get(i))) {
                    log.info("First divergence at step {}: baseline={} tf32={}",
                            i, baselineTokens.get(i), tf32Tokens.get(i));
                    break;
                }
            }
        } finally {
            Nd4j.getEnvironment().setCublasTf32Enabled(wasTf32);
            log.info("cuBLAS TF32 restored to: {}", Nd4j.getEnvironment().cublasTf32Enabled());
        }
    }

    /**
     * Run decode with logits-only compilation — matches the real StaticKvCacheDecodeLoop path.
     * Compile with logitsOnlyOutputNames, configure C++ KV scatter, execute with logitsOnlyOutputNames.
     */
    private List<Integer> runDecodeSequenceLogitsOnly(int numSteps) {
        decoder.resetSession();
        embedTokens.resetSession();
        InferenceSession.setDynamicShapePlanEnabled(true);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};
        String embedOutputName = embedTokens.outputs().get(0);

        // ---- Prefill (always uses full outputs) ----
        int[] prefillTokens = {49229};
        INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokens);
        INDArray inputIds = Nd4j.createFromArray(prefillTokens).reshape(1, prefillTokens.length).castTo(DataType.LONG);
        long prefillSeqLen = prefillTokens.length;

        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, embeddingTable.size(1));

        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, "Prefill logits must not be null");

        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(prefillLogits.size(1) - 1),
                    NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        List<Integer> tokens = new ArrayList<>();
        tokens.add(nextToken);
        log.info("[LOGITS_ONLY] Prefill token: {} logitSum={}", nextToken, prefillLogits.sumNumber().doubleValue());

        // Initialize static KV
        StaticKvForTest kvMgr = new StaticKvForTest(kvNames, 2048);
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

        // ---- KEY DIFFERENCE: Recompile with LOGITS-ONLY outputs ----
        // This is what StaticKvCacheDecodeLoop does when cppKvEnabled=true.
        // Associate static KV buffers as placeholders
        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearNodeOutputsOnly();

        // Compile with LOGITS-ONLY outputs (the real decode loop path)
        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        assertNotNull(dspExec, "DSP executor must exist after compilation");

        // Freeze shapes
        dspExec.setShapesFrozen(true);
        log.info("[LOGITS_ONLY] Shapes frozen, planPhase={}", dspExec.getPlanPhase());

        // Configure C++ KV scatter
        if (dspExec.getCurrentPlan() != null) {
            List<String> presentNames = new ArrayList<>();
            presentNames.addAll(kvNames.keyNames);
            presentNames.addAll(kvNames.valueNames);
            List<String> pastNames = new ArrayList<>();
            for (String pn : presentNames) {
                pastNames.add(ioConfig.presentToInputName(pn));
            }
            boolean configured = dspExec.configureKvCacheRetention(
                    dspExec.getCurrentPlan(), presentNames, pastNames,
                    (int) kvMgr.getMaxKvLen(), (int) kvMgr.getCachePosition());
            log.info("[LOGITS_ONLY] C++ KV scatter configured: {} mappings={}", configured, presentNames.size());
            assertTrue(configured, "C++ KV scatter must be configurable");
        }

        // ---- Decode steps ----
        Map<String, INDArray> reusableInputs = new HashMap<>();

        for (int step = 0; step < numSteps; step++) {
            long pastSeqLen = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();
            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken}).reshape(1, 1).castTo(DataType.LONG);

            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, stepEmbed, tokenIdArr,
                    pastSeqLen, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, embeddingTable.size(1), reusableInputs, true);

            // Execute with logits-only outputs (C++ KV scatter handles KV)
            Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, String.format("[LOGITS_ONLY] Step %d logits must not be null", step));

            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(stepLogits.size(1) - 1),
                        NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            tokens.add(nextToken);

            double checksum = stepLogits.sumNumber().doubleValue();
            double maxVal = stepLogits.maxNumber().doubleValue();
            log.info("[LOGITS_ONLY] Step {}: token={} logitSum={} logitMax={} cachePos={}",
                    step, nextToken, checksum, maxVal, cachePos);

            // C++ scatter runs internally — just advance position
            kvMgr.advancePosition();
        }

        return tokens;
    }

    /**
     * FP16 weight pre-casting: cast all 2D+ constants >= 1024 elements to HALF,
     * then run frozen decode with logits-only + C++ KV scatter.
     * Compare tokens against unfrozen FP32 baseline.
     */
    @Test
    @Order(3)
    @DisplayName("FP16 weight pre-casting must produce same tokens as FP32 baseline")
    public void testFp16WeightsVsBaselineDecodeTokens() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int numSteps = 5;

        // Baseline: unfrozen FP32 (known correct from tests 1 & 2)
        List<Integer> baselineTokens = runDecodeSequence(false, numSteps);
        log.info("FP32 Baseline tokens: {}", baselineTokens);

        // FP16 run: cast weights then run frozen decode with logits-only + C++ KV scatter
        List<Integer> fp16Tokens = runDecodeSequenceFp16Weights(numSteps);
        log.info("FP16 tokens: {}", fp16Tokens);

        assertEquals(baselineTokens.size(), fp16Tokens.size(), "Token count must match");

        int matchCount = 0;
        for (int i = 0; i < baselineTokens.size(); i++) {
            if (baselineTokens.get(i).equals(fp16Tokens.get(i))) {
                matchCount++;
            } else {
                log.warn("Token DIVERGENCE at step {}: FP32={} FP16={}", i, baselineTokens.get(i), fp16Tokens.get(i));
            }
        }
        log.info("FP16 vs FP32: {}/{} tokens match", matchCount, baselineTokens.size());

        // Assert all tokens match — FP16 should not change greedy decode for this model
        for (int i = 0; i < baselineTokens.size(); i++) {
            assertEquals(baselineTokens.get(i), fp16Tokens.get(i),
                    String.format("Token mismatch at step %d: FP32=%d FP16=%d",
                            i, baselineTokens.get(i), fp16Tokens.get(i)));
        }
    }

    /**
     * Run decode with FP16 weight pre-casting + logits-only compilation + frozen + C++ KV scatter.
     */
    private List<Integer> runDecodeSequenceFp16Weights(int numSteps) {
        decoder.resetSession();
        embedTokens.resetSession();
        InferenceSession.setDynamicShapePlanEnabled(true);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        // ---- FP16 weight pre-casting ----
        // Cast CONSTANT and VARIABLE type arrays (ONNX imports store weights as VARIABLE).
        // Only cast 2D+ FLOAT arrays with >= 1024 elements (matching QuantizeConstantsToFP16).
        int castCount = 0;
        long totalElements = 0;
        Map<DataType, Integer> constantTypeCounts = new HashMap<>();
        Map<DataType, Integer> variableTypeCounts = new HashMap<>();
        Map<VariableType, Integer> varTypeCounts = new HashMap<>();
        int eligible2dPlus = 0;
        for (SDVariable var : decoder.variables()) {
            varTypeCounts.merge(var.getVariableType(), 1, Integer::sum);
            boolean isWeight = var.getVariableType() == VariableType.CONSTANT
                    || var.getVariableType() == VariableType.VARIABLE;
            if (isWeight) {
                INDArray arr = decoder.getArrForVarName(var.name());
                if (arr != null) {
                    if (var.getVariableType() == VariableType.CONSTANT) {
                        constantTypeCounts.merge(arr.dataType(), 1, Integer::sum);
                    } else {
                        variableTypeCounts.merge(arr.dataType(), 1, Integer::sum);
                    }
                    if (arr.rank() >= 2 && arr.length() >= 1024) {
                        eligible2dPlus++;
                        if (arr.dataType() == DataType.FLOAT) {
                            decoder.associateArrayWithVariable(arr.castTo(DataType.HALF), var.name());
                            castCount++;
                            totalElements += arr.length();
                        }
                    }
                }
            }
        }
        log.info("[FP16] Variable type distribution: {}", varTypeCounts);
        log.info("[FP16] CONSTANT data types: {}", constantTypeCounts);
        log.info("[FP16] VARIABLE data types: {}", variableTypeCounts);
        log.info("[FP16] Eligible 2D+ weights >= 1024 elements: {}", eligible2dPlus);
        log.info("[FP16] Cast {} weights ({} total elements) from FLOAT to HALF", castCount, totalElements);

        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};
        String embedOutputName = embedTokens.outputs().get(0);

        // ---- Prefill (always uses full outputs) ----
        int[] prefillTokens = {49229};
        INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokens);
        INDArray inputIds = Nd4j.createFromArray(prefillTokens).reshape(1, prefillTokens.length).castTo(DataType.LONG);
        long prefillSeqLen = prefillTokens.length;

        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, embeddingTable.size(1));

        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, "Prefill logits must not be null");

        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(prefillLogits.size(1) - 1),
                    NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        List<Integer> tokens = new ArrayList<>();
        tokens.add(nextToken);
        log.info("[FP16] Prefill token: {} logitSum={}", nextToken, prefillLogits.sumNumber().doubleValue());

        // Initialize static KV
        StaticKvForTest kvMgr = new StaticKvForTest(kvNames, 2048);
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

        // ---- Recompile with logits-only outputs ----
        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearNodeOutputsOnly();

        decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

        session = decoder.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        assertNotNull(dspExec, "DSP executor must exist after compilation");

        // Freeze shapes
        dspExec.setShapesFrozen(true);
        log.info("[FP16] Shapes frozen, planPhase={}", dspExec.getPlanPhase());

        // Configure C++ KV scatter
        if (dspExec.getCurrentPlan() != null) {
            List<String> presentNames = new ArrayList<>();
            presentNames.addAll(kvNames.keyNames);
            presentNames.addAll(kvNames.valueNames);
            List<String> pastNames = new ArrayList<>();
            for (String pn : presentNames) {
                pastNames.add(ioConfig.presentToInputName(pn));
            }
            boolean configured = dspExec.configureKvCacheRetention(
                    dspExec.getCurrentPlan(), presentNames, pastNames,
                    (int) kvMgr.getMaxKvLen(), (int) kvMgr.getCachePosition());
            log.info("[FP16] C++ KV scatter configured: {} mappings={}", configured, presentNames.size());
            assertTrue(configured, "C++ KV scatter must be configurable");
        }

        // ---- Decode steps ----
        Map<String, INDArray> reusableInputs = new HashMap<>();

        for (int step = 0; step < numSteps; step++) {
            long pastSeqLen = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();
            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken}).reshape(1, 1).castTo(DataType.LONG);

            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, stepEmbed, tokenIdArr,
                    pastSeqLen, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, embeddingTable.size(1), reusableInputs, true);

            Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, String.format("[FP16] Step %d logits must not be null", step));

            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(stepLogits.size(1) - 1),
                        NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            tokens.add(nextToken);

            double checksum = stepLogits.sumNumber().doubleValue();
            double maxVal = stepLogits.maxNumber().doubleValue();
            log.info("[FP16] Step {}: token={} logitSum={} logitMax={} cachePos={}",
                    step, nextToken, checksum, maxVal, cachePos);

            kvMgr.advancePosition();
        }

        // ---- Restore FP32 weights so other tests are not affected ----
        for (SDVariable var : decoder.variables()) {
            if (var.getVariableType() == VariableType.CONSTANT
                    || var.getVariableType() == VariableType.VARIABLE) {
                INDArray arr = decoder.getArrForVarName(var.name());
                if (arr != null && arr.dataType() == DataType.HALF) {
                    decoder.associateArrayWithVariable(arr.castTo(DataType.FLOAT), var.name());
                }
            }
        }
        log.info("[FP16] Restored weights back to FP32");

        return tokens;
    }

    /**
     * Combined FP16 weight pre-casting + cuBLAS TF32: matches the full OPTIMAL benchmark config.
     * Tests whether enabling BOTH precision-reducing features simultaneously breaks frozen decode.
     *
     * 1. Run unfrozen FP32 baseline (no TF32)
     * 2. Cast all 2D+ constants >= 1024 elements to HALF, enable cuBLAS TF32 + Triton TF32
     * 3. Run frozen decode with logits-only + C++ KV scatter
     * 4. Compare tokens — log divergences, flag DEGENERATE output
     */
    @Test
    @Order(5)
    @DisplayName("FP16 weights + TF32 combined must not produce degenerate decode output")
    public void testFp16PlusTf32VsBaselineDecodeTokens() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded — skipping");

        int numSteps = 5;

        // ---- Baseline: unfrozen FP32, no TF32 ----
        List<Integer> baselineTokens = runDecodeSequence(false, numSteps);
        log.info("FP32 Baseline tokens: {}", baselineTokens);

        // ---- FP16 + TF32 run ----
        List<Integer> fp16Tf32Tokens = runDecodeSequenceFp16PlusTf32(numSteps);
        log.info("FP16+TF32 tokens: {}", fp16Tf32Tokens);

        // Log full comparison
        assertEquals(baselineTokens.size(), fp16Tf32Tokens.size(), "Token count must match");
        int matchCount = 0;
        for (int i = 0; i < baselineTokens.size(); i++) {
            boolean match = baselineTokens.get(i).equals(fp16Tf32Tokens.get(i));
            if (match) matchCount++;
            log.info("Step {}: baseline={} fp16tf32={} {}", i, baselineTokens.get(i), fp16Tf32Tokens.get(i),
                    match ? "MATCH" : "DIVERGE");
        }
        log.info("FP16+TF32 vs FP32 baseline: {}/{} tokens match", matchCount, baselineTokens.size());

        // Check for DEGENERATE output: if unique token count < 30% of total
        Set<Integer> uniqueTokens = new HashSet<>(fp16Tf32Tokens);
        double uniqueRatio = (double) uniqueTokens.size() / fp16Tf32Tokens.size();
        log.info("FP16+TF32 unique tokens: {} / {} (ratio={})", uniqueTokens.size(), fp16Tf32Tokens.size(),
                String.format("%.2f", uniqueRatio));
        if (uniqueRatio < 0.3) {
            log.error("DEGENERATE output detected! Unique tokens {} / {} = {} < 0.30 threshold. Tokens: {}",
                    uniqueTokens.size(), fp16Tf32Tokens.size(), String.format("%.2f", uniqueRatio), fp16Tf32Tokens);
            fail(String.format("DEGENERATE output: only %d unique tokens out of %d (%.1f%%). " +
                    "FP16+TF32 combination produces repetitive/stuck output. Tokens: %s",
                    uniqueTokens.size(), fp16Tf32Tokens.size(), uniqueRatio * 100, fp16Tf32Tokens));
        }

        // Note: we do NOT assert exact token match — FP16+TF32 may legitimately diverge
        // from FP32 due to reduced precision. The key assertion is non-degeneracy.
        if (matchCount < baselineTokens.size()) {
            log.warn("FP16+TF32 diverged from FP32 baseline at {}/{} steps — expected with reduced precision",
                    baselineTokens.size() - matchCount, baselineTokens.size());
        }
    }

    /**
     * Run decode with FP16 weight pre-casting + cuBLAS/Triton TF32 + logits-only + frozen + C++ KV scatter.
     */
    private List<Integer> runDecodeSequenceFp16PlusTf32(int numSteps) {
        decoder.resetSession();
        embedTokens.resetSession();
        InferenceSession.setDynamicShapePlanEnabled(true);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        // ---- Enable TF32 ----
        Environment env = Nd4j.getEnvironment();
        boolean wasCublasTf32 = env.cublasTf32Enabled();
        boolean wasTritonTf32 = env.tritonTf32Enabled();
        env.setCublasTf32Enabled(true);
        env.setTritonTf32Enabled(true);
        log.info("[FP16+TF32] TF32 enabled: cuBLAS={} Triton={}", env.cublasTf32Enabled(), env.tritonTf32Enabled());

        try {
            // ---- FP16 weight pre-casting ----
            int castCount = 0;
            long totalElements = 0;
            for (SDVariable var : decoder.variables()) {
                if (var.getVariableType() == VariableType.CONSTANT) {
                    INDArray arr = decoder.getArrForVarName(var.name());
                    if (arr != null && arr.rank() >= 2 && arr.length() >= 1024 && arr.dataType() == DataType.FLOAT) {
                        decoder.associateArrayWithVariable(arr.castTo(DataType.HALF), var.name());
                        castCount++;
                        totalElements += arr.length();
                    }
                }
            }
            log.info("[FP16+TF32] Cast {} constants ({} total elements) from FLOAT to HALF", castCount, totalElements);

            String[] fullOutputNames = buildFullOutputNames();
            String[] logitsOnlyOutputNames = new String[]{logitsName};
            String embedOutputName = embedTokens.outputs().get(0);

            // ---- Prefill (always uses full outputs) ----
            int[] prefillTokens = {49229};
            INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokens);
            INDArray inputIds = Nd4j.createFromArray(prefillTokens).reshape(1, prefillTokens.length).castTo(DataType.LONG);
            long prefillSeqLen = prefillTokens.length;

            Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, prefillEmbeds, inputIds,
                    0, prefillSeqLen, null, 0, 0, false, embeddingTable.size(1));

            Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
            INDArray prefillLogits = prefillOutputs.get(logitsName);
            assertNotNull(prefillLogits, "Prefill logits must not be null");

            INDArray lastLogits = prefillLogits.rank() == 3
                    ? prefillLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(prefillLogits.size(1) - 1),
                        NDArrayIndex.all())
                    : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            int nextToken = Nd4j.argMax(lastLogits).getInt(0);
            List<Integer> tokens = new ArrayList<>();
            tokens.add(nextToken);
            log.info("[FP16+TF32] Prefill token: {} logitSum={}", nextToken, prefillLogits.sumNumber().doubleValue());

            // Initialize static KV
            StaticKvForTest kvMgr = new StaticKvForTest(kvNames, 2048);
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

            // ---- Recompile with logits-only outputs ----
            Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
            for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
                if (decoder.hasVariable(e.getKey())) {
                    decoder.associateArrayWithVariable(e.getValue(), e.getKey());
                }
            }

            decoder.clearDynamicShapePlanCache();
            var session = decoder.getOrCreateSession();
            session.clearNodeOutputsOnly();

            decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

            session = decoder.getOrCreateSession();
            DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
            assertNotNull(dspExec, "DSP executor must exist after compilation");

            // Freeze shapes
            dspExec.setShapesFrozen(true);
            log.info("[FP16+TF32] Shapes frozen, planPhase={}", dspExec.getPlanPhase());

            // Configure C++ KV scatter
            if (dspExec.getCurrentPlan() != null) {
                List<String> presentNames = new ArrayList<>();
                presentNames.addAll(kvNames.keyNames);
                presentNames.addAll(kvNames.valueNames);
                List<String> pastNames = new ArrayList<>();
                for (String pn : presentNames) {
                    pastNames.add(ioConfig.presentToInputName(pn));
                }
                boolean configured = dspExec.configureKvCacheRetention(
                        dspExec.getCurrentPlan(), presentNames, pastNames,
                        (int) kvMgr.getMaxKvLen(), (int) kvMgr.getCachePosition());
                log.info("[FP16+TF32] C++ KV scatter configured: {} mappings={}", configured, presentNames.size());
                assertTrue(configured, "C++ KV scatter must be configurable");
                if (configured) {
                    dspExec.configureDecodeInputs(dspExec.getCurrentPlan(), (int) kvMgr.getMaxKvLen());
                }
            }

            // ---- Decode steps ----
            Map<String, INDArray> reusableInputs = new HashMap<>();

            for (int step = 0; step < numSteps; step++) {
                long pastSeqLen = prefillSeqLen + step;
                long cachePos = kvMgr.getCachePosition();
                INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken}).reshape(1, 1).castTo(DataType.LONG);

                Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                        Map.of("input_ids", tokenIdArr), embedOutputName);
                INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

                Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                        decoder.inputs(), decoder, stepEmbed, tokenIdArr,
                        pastSeqLen, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                        true, embeddingTable.size(1), reusableInputs, true);

                Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, logitsOnlyOutputNames);

                INDArray stepLogits = outputs.get(logitsName);
                assertNotNull(stepLogits, String.format("[FP16+TF32] Step %d logits must not be null", step));

                INDArray lastLogit = stepLogits.rank() == 3
                        ? stepLogits.get(NDArrayIndex.point(0),
                            NDArrayIndex.point(stepLogits.size(1) - 1),
                            NDArrayIndex.all())
                        : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
                nextToken = Nd4j.argMax(lastLogit).getInt(0);
                tokens.add(nextToken);

                double checksum = stepLogits.sumNumber().doubleValue();
                double maxVal = stepLogits.maxNumber().doubleValue();
                log.info("[FP16+TF32] Step {}: token={} logitSum={} logitMax={} cachePos={}",
                        step, nextToken, checksum, maxVal, cachePos);

                kvMgr.advancePosition();
            }

            return tokens;
        } finally {
            // ---- Restore TF32 settings ----
            env.setCublasTf32Enabled(wasCublasTf32);
            env.setTritonTf32Enabled(wasTritonTf32);
            log.info("[FP16+TF32] Restored TF32: cuBLAS={} Triton={}", env.cublasTf32Enabled(), env.tritonTf32Enabled());

            // ---- Restore FP32 weights ----
            for (SDVariable var : decoder.variables()) {
                if (var.getVariableType() == VariableType.CONSTANT) {
                    INDArray arr = decoder.getArrForVarName(var.name());
                    if (arr != null && arr.dataType() == DataType.HALF) {
                        decoder.associateArrayWithVariable(arr.castTo(DataType.FLOAT), var.name());
                    }
                }
            }
            log.info("[FP16+TF32] Restored weights back to FP32");
        }
    }

    /**
     * Multi-token prefill (mimics the vision encoder's 679-token output) followed by
     * recompile to seqLen=1 decode with frozen shapes. This is the last remaining
     * untested variable between the passing isolation tests (single-token prefill)
     * and the failing VLM benchmark (multi-token prefill from vision encoder).
     *
     * The hypothesis: the DSP plan compiled for seqLen=20 prefill leaves stale state
     * that corrupts the recompiled seqLen=1 decode plan, producing degenerate output.
     */
    @Test
    @Order(6)
    @DisplayName("Multi-token prefill then frozen decode must not produce degenerate output")
    public void testMultiTokenPrefillThenFrozenDecode() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded - skipping");

        int numSteps = 5;
        int[] multiPrefillTokens = {49229, 11126, 42, 49189, 49153, 49189, 49153, 42, 11126, 49229,
                                    49229, 11126, 42, 49189, 49153, 49189, 49153, 42, 11126, 49229};

        // ---- PATH A: Unfrozen baseline with multi-token prefill ----
        log.info("========== PATH A: UNFROZEN multi-token prefill ({} tokens) ==========", multiPrefillTokens.length);
        List<Integer> unfrozenTokens = runMultiTokenPrefillDecode(multiPrefillTokens, false, numSteps);
        log.info("Unfrozen tokens (multi-token prefill): {}", unfrozenTokens);

        // ---- PATH B: Frozen + logits-only + C++ KV scatter with multi-token prefill ----
        log.info("========== PATH B: FROZEN multi-token prefill ({} tokens) ==========", multiPrefillTokens.length);
        List<Integer> frozenTokens = runMultiTokenPrefillDecode(multiPrefillTokens, true, numSteps);
        log.info("Frozen tokens (multi-token prefill): {}", frozenTokens);

        // ---- Compare ----
        assertEquals(unfrozenTokens.size(), frozenTokens.size(), "Token count must match");

        int matchCount = 0;
        for (int i = 0; i < unfrozenTokens.size(); i++) {
            boolean match = unfrozenTokens.get(i).equals(frozenTokens.get(i));
            if (match) matchCount++;
            log.info("Step {}: unfrozen={} frozen={} {}", i, unfrozenTokens.get(i), frozenTokens.get(i),
                    match ? "MATCH" : "DIVERGE");
        }
        log.info("Multi-token prefill: frozen vs unfrozen {}/{} tokens match", matchCount, unfrozenTokens.size());

        // Check for degenerate output (< 30% unique tokens = degenerate)
        Set<Integer> uniqueFrozen = new HashSet<>(frozenTokens);
        double uniqueRatio = (double) uniqueFrozen.size() / frozenTokens.size();
        log.info("Frozen unique tokens: {} / {} (ratio={})", uniqueFrozen.size(), frozenTokens.size(),
                String.format("%.2f", uniqueRatio));

        if (uniqueRatio < 0.3) {
            log.error("DEGENERATE frozen output! Unique tokens {} / {} = {} < 0.30. Tokens: {}",
                    uniqueFrozen.size(), frozenTokens.size(), String.format("%.2f", uniqueRatio), frozenTokens);
        }

        Set<Integer> uniqueUnfrozen = new HashSet<>(unfrozenTokens);
        double uniqueUnfrozenRatio = (double) uniqueUnfrozen.size() / unfrozenTokens.size();
        log.info("Unfrozen unique tokens: {} / {} (ratio={})", uniqueUnfrozen.size(), unfrozenTokens.size(),
                String.format("%.2f", uniqueUnfrozenRatio));

        // Assert non-degeneracy for frozen path
        assertFalse(uniqueRatio < 0.3,
                String.format("DEGENERATE frozen output after multi-token prefill: only %d unique tokens out of %d (%.1f%%). " +
                        "This reproduces the 'upsupsupsup' bug from the VLM benchmark. Tokens: %s",
                        uniqueFrozen.size(), frozenTokens.size(), uniqueRatio * 100, frozenTokens));

        // Assert tokens match between frozen and unfrozen
        for (int i = 0; i < unfrozenTokens.size(); i++) {
            assertEquals(unfrozenTokens.get(i), frozenTokens.get(i),
                    String.format("Token mismatch at step %d after multi-token prefill: unfrozen=%d frozen=%d. " +
                            "Full unfrozen=%s, full frozen=%s",
                            i, unfrozenTokens.get(i), frozenTokens.get(i), unfrozenTokens, frozenTokens));
        }
    }

    /**
     * Run multi-token prefill followed by seqLen=1 decode steps.
     *
     * @param prefillTokenIds tokens for prefill phase (e.g. 20 tokens)
     * @param frozen if true, recompile with logits-only + freeze + C++ KV scatter
     * @param numSteps number of decode steps after prefill
     * @return list of token IDs (prefill result + decode tokens)
     */
    private List<Integer> runMultiTokenPrefillDecode(int[] prefillTokenIds, boolean frozen, int numSteps) {
        String tag = frozen ? "FROZEN_MULTI" : "UNFROZEN_MULTI";

        // Reset state
        decoder.resetSession();
        embedTokens.resetSession();
        InferenceSession.setDynamicShapePlanEnabled(true);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        String[] fullOutputNames = buildFullOutputNames();
        String[] logitsOnlyOutputNames = new String[]{logitsName};
        String embedOutputName = embedTokens.outputs().get(0);

        // ---- Prefill with multi-token input ----
        INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds).reshape(1, prefillTokenIds.length).castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        log.info("[{}] Prefill: {} tokens, embedShape={}", tag, prefillSeqLen, Arrays.toString(prefillEmbeds.shape()));

        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, embeddingTable.size(1));

        // Prefill always uses full outputs to get KV
        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, "Prefill logits must not be null");

        // Get first token from last position of prefill
        INDArray lastLogits = prefillLogits.rank() == 3
                ? prefillLogits.get(NDArrayIndex.point(0),
                    NDArrayIndex.point(prefillLogits.size(1) - 1),
                    NDArrayIndex.all())
                : prefillLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
        int nextToken = Nd4j.argMax(lastLogits).getInt(0);
        List<Integer> tokens = new ArrayList<>();
        tokens.add(nextToken);
        log.info("[{}] Prefill result: token={} logitShape={} logitSum={}",
                tag, nextToken, Arrays.toString(prefillLogits.shape()), prefillLogits.sumNumber().doubleValue());

        // Initialize static KV from prefill (copies prefill KV into padded buffers)
        StaticKvForTest kvMgr = new StaticKvForTest(kvNames, 2048);
        kvMgr.initializeFromPrefill(prefillOutputs);
        log.info("[{}] KV initialized: cachePosition={} maxKvLen={}", tag, kvMgr.getCachePosition(), kvMgr.getMaxKvLen());

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

        // ---- Recompile for seqLen=1 decode ----
        // This is the KEY transition: DSP plan was compiled for seqLen=20 prefill,
        // now we clear it and recompile for seqLen=1 decode.
        Map<String, INDArray> staticKvBuffers = kvMgr.getStaticKvBuffers();
        for (Map.Entry<String, INDArray> e : staticKvBuffers.entrySet()) {
            if (decoder.hasVariable(e.getKey())) {
                decoder.associateArrayWithVariable(e.getValue(), e.getKey());
            }
        }

        decoder.clearDynamicShapePlanCache();
        var session = decoder.getOrCreateSession();
        session.clearNodeOutputsOnly();

        if (frozen) {
            // Frozen path: logits-only compilation + freeze + C++ KV scatter
            decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, logitsOnlyOutputNames);

            session = decoder.getOrCreateSession();
            DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
            assertNotNull(dspExec, "DSP executor must exist after compilation");

            dspExec.setShapesFrozen(true);
            log.info("[{}] Shapes frozen, planPhase={}", tag, dspExec.getPlanPhase());

            // Configure C++ KV scatter
            if (dspExec.getCurrentPlan() != null) {
                List<String> presentNames = new ArrayList<>();
                presentNames.addAll(kvNames.keyNames);
                presentNames.addAll(kvNames.valueNames);
                List<String> pastNames = new ArrayList<>();
                for (String pn : presentNames) {
                    pastNames.add(ioConfig.presentToInputName(pn));
                }
                boolean configured = dspExec.configureKvCacheRetention(
                        dspExec.getCurrentPlan(), presentNames, pastNames,
                        (int) kvMgr.getMaxKvLen(), (int) kvMgr.getCachePosition());
                log.info("[{}] C++ KV scatter configured: {} mappings={}", tag, configured, presentNames.size());
                assertTrue(configured, "C++ KV scatter must be configurable");
                if (configured) {
                    dspExec.configureDecodeInputs(dspExec.getCurrentPlan(), (int) kvMgr.getMaxKvLen());
                }
            }
        } else {
            // Unfrozen path: full-output compilation, no freeze, Java KV scatter
            decoder.compileNativeDynamicShapePlan(DspCompilationMode.MAX_AUTOTUNE, fullOutputNames);
            log.info("[{}] Compiled for decode (unfrozen, full outputs)", tag);
        }

        // ---- Decode steps ----
        Map<String, INDArray> reusableInputs = new HashMap<>();
        String[] outputNames = frozen ? logitsOnlyOutputNames : fullOutputNames;

        for (int step = 0; step < numSteps; step++) {
            long pastSeqLen = prefillSeqLen + step;
            long cachePos = kvMgr.getCachePosition();
            INDArray tokenIdArr = Nd4j.createFromArray(new long[]{nextToken}).reshape(1, 1).castTo(DataType.LONG);

            Map<String, INDArray> tokenEmbedOut = embedTokens.output(
                    Map.of("input_ids", tokenIdArr), embedOutputName);
            INDArray stepEmbed = tokenEmbedOut.get(embedOutputName);

            Map<String, INDArray> decodeInputs = DecoderUtils.buildDecoderInputMap(
                    decoder.inputs(), decoder, stepEmbed, tokenIdArr,
                    pastSeqLen, 1, kvMgr.getStaticKvBuffers(), kvMgr.getMaxKvLen(), cachePos,
                    true, embeddingTable.size(1), reusableInputs, true);

            Map<String, INDArray> outputs = decoder.outputDirect(decodeInputs, outputNames);

            INDArray stepLogits = outputs.get(logitsName);
            assertNotNull(stepLogits, String.format("[%s] Step %d logits must not be null", tag, step));

            INDArray lastLogit = stepLogits.rank() == 3
                    ? stepLogits.get(NDArrayIndex.point(0),
                        NDArrayIndex.point(stepLogits.size(1) - 1),
                        NDArrayIndex.all())
                    : stepLogits.get(NDArrayIndex.point(0), NDArrayIndex.all());
            nextToken = Nd4j.argMax(lastLogit).getInt(0);
            tokens.add(nextToken);

            double checksum = stepLogits.sumNumber().doubleValue();
            double maxVal = stepLogits.maxNumber().doubleValue();
            log.info("[{}] Step {}: token={} logitSum={} logitMax={} pastSeqLen={} cachePos={}",
                    tag, step, nextToken, checksum, maxVal, pastSeqLen, cachePos);

            if (frozen) {
                // C++ scatter runs internally - just advance position
                kvMgr.advancePosition();
            } else {
                // Java scatter for unfrozen path
                kvMgr.scatterNewEntries(outputs);
            }
        }

        return tokens;
    }

    /**
     * Minimal KV management for the test.
     */
    private class StaticKvForTest {
        private final DecoderUtils.KVCacheNames kvNames;
        private final long maxKvLen;
        private final Map<String, INDArray> staticKvBuffers = new HashMap<>();
        private long cachePosition;

        StaticKvForTest(DecoderUtils.KVCacheNames kvNames, long maxKvLen) {
            this.kvNames = kvNames;
            this.maxKvLen = maxKvLen;
        }

        void initializeFromPrefill(Map<String, INDArray> prefillOutputs) {
            for (String keyName : kvNames.keyNames) {
                INDArray present = prefillOutputs.get(keyName);
                if (present != null) {
                    long[] shape = present.shape();
                    INDArray buf = Nd4j.zeros(DataType.FLOAT, shape[0], shape[1], maxKvLen, shape[3]);
                    // Copy prefill KV to position 0
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
            // Cache position after prefill
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
                        // New entries are at position [maxKvLen..maxKvLen+seqLen-1] in present
                        // Scatter to static buffer at [cachePos..cachePos+seqLen-1]
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
