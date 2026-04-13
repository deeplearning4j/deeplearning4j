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
 * Isolation test: does large prefill size cause graph replay divergence?
 *
 * The SmolDocling VLM benchmark fails with OPTIMAL (Triton + graph capture) but a
 * simplified test with 17-token prefill passes. The real pipeline uses a 679-token
 * prefill with vision embeddings. This test runs progressively larger prefills to
 * find the threshold where graph replay diverges from no-graph-capture.
 *
 * For each prefill size:
 *   1. Baseline: Triton ON, graph capture OFF (known correct)
 *   2. Treatment: Triton ON, graph capture ON (the suspect)
 *   3. Compare greedy-decoded tokens over 5 decode steps
 *
 * Prefill sizes tested: 17, 50, 100, 200, 500
 *
 * Run:
 *   cd platform-tests && mvn test \
 *     -Dtest=LargePrefillGraphReplayTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 \
 *     2>&1 | tee /tmp/large-prefill-test.log
 */
@Slf4j
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class LargePrefillGraphReplayTest {

    private static final int NUM_DECODE_STEPS = 5;
    /**
     * Extra KV slots beyond prefill length. The static KV buffer is
     * prefillLen + EXTRA_KV_PADDING so every decode step fits.
     */
    private static final int EXTRA_KV_PADDING = 10;

    private SameDiff decoder;
    private SameDiff embedTokens;
    private INDArray embeddingTable;
    private String logitsName;
    private String embedsName;
    private ModelIOConfig ioConfig;
    private DecoderUtils.KVCacheNames kvNames;
    private boolean modelsLoaded = false;

    /** Per-size results for the summary report at the end. */
    private final Map<Integer, SizeResult> sizeResults = new LinkedHashMap<>();

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
            log.info("Embedding table: vocabSize={} hiddenSize={}", embeddingTable.size(0), embeddingTable.size(1));
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

    // ========== Progressive prefill size tests ==========

    @Test
    @Order(1)
    @DisplayName("Prefill 17 tokens: graph capture vs no graph capture")
    public void testPrefill17() {
        runPrefillSizeComparison(17);
    }

    @Test
    @Order(2)
    @DisplayName("Prefill 50 tokens: graph capture vs no graph capture")
    public void testPrefill50() {
        runPrefillSizeComparison(50);
    }

    @Test
    @Order(3)
    @DisplayName("Prefill 100 tokens: graph capture vs no graph capture")
    public void testPrefill100() {
        runPrefillSizeComparison(100);
    }

    @Test
    @Order(4)
    @DisplayName("Prefill 200 tokens: graph capture vs no graph capture")
    public void testPrefill200() {
        runPrefillSizeComparison(200);
    }

    @Test
    @Order(5)
    @DisplayName("Prefill 500 tokens: graph capture vs no graph capture")
    public void testPrefill500() {
        runPrefillSizeComparison(500);
    }

    @Test
    @Order(6)
    @DisplayName("Summary: report which prefill size first causes divergence")
    public void testSummaryReport() {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        log.info("========================================================================");
        log.info("  LARGE PREFILL GRAPH REPLAY DIVERGENCE SUMMARY");
        log.info("========================================================================");

        int firstDivergent = -1;
        for (Map.Entry<Integer, SizeResult> entry : sizeResults.entrySet()) {
            int size = entry.getKey();
            SizeResult r = entry.getValue();
            String status;
            if (r.skipped) {
                status = "SKIPPED";
            } else if (r.allMatch) {
                status = "PASS (all tokens match)";
            } else {
                status = String.format("DIVERGE at step %d (%d/%d match)",
                        r.firstDivergenceStep, r.matchCount, r.totalTokens);
                if (firstDivergent == -1) firstDivergent = size;
            }
            log.info("  prefill={} tokens | maxKvLen={} | {}",
                    String.format("%4d", size), String.format("%4d", size + EXTRA_KV_PADDING), status);
        }

        if (firstDivergent > 0) {
            log.info("------------------------------------------------------------------------");
            log.info("  FIRST DIVERGENT PREFILL SIZE: {} tokens (maxKvLen={})",
                    firstDivergent, firstDivergent + EXTRA_KV_PADDING);
            log.info("------------------------------------------------------------------------");
        } else if (!sizeResults.isEmpty()) {
            log.info("------------------------------------------------------------------------");
            log.info("  NO DIVERGENCE DETECTED at any tested prefill size.");
            log.info("------------------------------------------------------------------------");
        }
        log.info("========================================================================");
    }

    // ========== Core comparison logic ==========

    /**
     * For a given prefill size, generate random token IDs, run baseline (no graph capture)
     * and treatment (graph capture ON), compare decoded tokens.
     */
    private void runPrefillSizeComparison(int prefillSize) {
        Assumptions.assumeTrue(modelsLoaded, "Models not loaded -- skipping");
        Assumptions.assumeTrue(Nd4j.getNativeOps().isTritonAvailable(), "Triton not available -- skipping");

        log.info("========== PREFILL SIZE: {} tokens (maxKvLen={}) ==========",
                prefillSize, prefillSize + EXTRA_KV_PADDING);

        // Generate a repeatable token sequence for this prefill size.
        // Use token 49229 (doctag) as first token, then cycle through valid token IDs.
        int[] prefillTokenIds = generatePrefillTokens(prefillSize);
        long maxKvLen = prefillSize + EXTRA_KV_PADDING;

        Environment env = Nd4j.getEnvironment();

        // Save original environment values
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
            log.info("[prefill={}] Running BASELINE (graphCapture=OFF)", prefillSize);

            DecodeResult baseline = runDecodeSequence(
                    String.format("BASELINE_noGC_p%d", prefillSize),
                    prefillTokenIds, maxKvLen);
            log.info("[prefill={}] Baseline tokens: {}", prefillSize, baseline.tokens);

            // ---- Treatment: Full OPTIMAL config (graph capture ON + all optimizations) ----
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
            log.info("[prefill={}] Running TREATMENT (FULL OPTIMAL, graphCapture=ON)", prefillSize);

            DecodeResult treatment = runDecodeSequence(
                    String.format("TREATMENT_OPTIMAL_p%d", prefillSize),
                    prefillTokenIds, maxKvLen);
            log.info("[prefill={}] Treatment tokens: {}", prefillSize, treatment.tokens);

            // ---- Compare and record ----
            SizeResult result = compareAndReport(prefillSize, baseline, treatment);
            sizeResults.put(prefillSize, result);

            // Assert token match
            for (int i = 0; i < baseline.tokens.size(); i++) {
                assertEquals(baseline.tokens.get(i), treatment.tokens.get(i),
                        String.format("[prefill=%d] Token mismatch at index %d (%s): baseline=%d treatment=%d. " +
                                        "Baseline logitSum=%s, Treatment logitSum=%s. maxKvLen=%d",
                                prefillSize, i,
                                i == 0 ? "prefill" : "decode step " + (i - 1),
                                baseline.tokens.get(i), treatment.tokens.get(i),
                                i < baseline.logitChecksums.size() ? String.valueOf(baseline.logitChecksums.get(i)) : "?",
                                i < treatment.logitChecksums.size() ? String.valueOf(treatment.logitChecksums.get(i)) : "?",
                                maxKvLen));
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
            env.setTritonNumWarps(origNumWarps);
            env.setTritonNumStages(origNumStages);
        }
    }

    // ========== Decode sequence runner ==========

    /**
     * Run prefill + decode steps with the current environment settings.
     *
     * 1. Build prefill embeddings from token IDs via the embedding table
     * 2. Run prefill to get initial logits + KV cache
     * 3. Initialize static KV buffer with prefillLen + EXTRA_KV_PADDING slots
     * 4. Recompile DSP plan for seqLen=1 decode
     * 5. Freeze shapes, configure C++ KV scatter
     * 6. Run NUM_DECODE_STEPS decode steps
     */
    private DecodeResult runDecodeSequence(String label, int[] prefillTokenIds, long maxKvLen) {
        DecodeResult result = new DecodeResult();

        // Reset state completely between runs
        decoder.resetSession();
        embedTokens.resetSession();
        InferenceSession.setDynamicShapePlanEnabled(true);
        decoder.setDspAutoCompileEnabled(true);
        decoder.setDspNativeAutoCompileEnabled(true);

        String[] fullOutputNames = buildFullOutputNames();
        String embedOutputName = embedTokens.outputs().get(0);

        // ---- Prefill ----
        INDArray prefillEmbeds = buildPrefillEmbeddings(prefillTokenIds);
        INDArray inputIds = Nd4j.createFromArray(prefillTokenIds)
                .reshape(1, prefillTokenIds.length)
                .castTo(DataType.LONG);
        long prefillSeqLen = prefillTokenIds.length;

        log.info("[{}] Running prefill with {} tokens, embedShape={}", label, prefillSeqLen,
                Arrays.toString(prefillEmbeds.shape()));

        Map<String, INDArray> prefillInputs = DecoderUtils.buildDecoderInputMap(
                decoder.inputs(), decoder, prefillEmbeds, inputIds,
                0, prefillSeqLen, null, 0, 0, false, embeddingTable.size(1));

        Map<String, INDArray> prefillOutputs = decoder.output(prefillInputs, fullOutputNames);
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        assertNotNull(prefillLogits, String.format("[%s] Prefill logits must not be null", label));

        // Extract first token from last position of prefill
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
        StaticKvManager kvMgr = new StaticKvManager(kvNames, maxKvLen);
        kvMgr.initializeFromPrefill(prefillOutputs);
        log.info("[{}] KV cache initialized: cachePos={} maxKvLen={}", label,
                kvMgr.getCachePosition(), kvMgr.getMaxKvLen());

        // Close prefill outputs
        closePrefillOutputs(prefillOutputs, prefillLogits);

        // ---- Recompile for decode (seqLen=1) ----
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
                log.info("[{}] C++ KV scatter configured: {} (prefillSeqLen={} maxKvLen={})",
                        label, cppKvScatter, prefillSeqLen, maxKvLen);
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

            log.info("[{}] Step {}: token={} logitSum={} logitMax={} pastSeqLen={} cachePos={}",
                    label, step, nextToken, checksum, maxVal, pastSeqLen, cachePos);

            // KV cache management
            if (!cppKvScatter) {
                kvMgr.scatterNewEntries(outputs);
            } else {
                kvMgr.advancePosition();
            }
        }

        return result;
    }

    // ========== Comparison and reporting ==========

    private SizeResult compareAndReport(int prefillSize, DecodeResult baseline, DecodeResult treatment) {
        SizeResult result = new SizeResult();
        result.prefillSize = prefillSize;
        result.totalTokens = baseline.tokens.size();

        assertEquals(baseline.tokens.size(), treatment.tokens.size(),
                String.format("[prefill=%d] Token count mismatch: baseline=%d treatment=%d",
                        prefillSize, baseline.tokens.size(), treatment.tokens.size()));

        int firstDivergence = -1;
        int matchCount = 0;
        for (int i = 0; i < baseline.tokens.size(); i++) {
            if (baseline.tokens.get(i).equals(treatment.tokens.get(i))) {
                matchCount++;
            } else if (firstDivergence == -1) {
                firstDivergence = i;
            }
        }

        result.matchCount = matchCount;
        result.allMatch = (matchCount == baseline.tokens.size());
        result.firstDivergenceStep = firstDivergence;

        double matchRate = (double) matchCount / baseline.tokens.size() * 100.0;
        log.info("[prefill={}] Match rate: {}/{} ({}%)", prefillSize, matchCount,
                baseline.tokens.size(), String.format("%.1f", matchRate));

        if (firstDivergence >= 0) {
            String stepLabel = firstDivergence == 0 ? "prefill" : "decode step " + (firstDivergence - 1);
            log.error("[prefill={}] FIRST DIVERGENCE at index {} ({}): baseline={} treatment={}",
                    prefillSize, firstDivergence, stepLabel,
                    baseline.tokens.get(firstDivergence), treatment.tokens.get(firstDivergence));
            log.error("[prefill={}] Full baseline:  {}", prefillSize, baseline.tokens);
            log.error("[prefill={}] Full treatment: {}", prefillSize, treatment.tokens);

            // Log per-step logit checksum comparison around divergence
            int logStart = Math.max(0, firstDivergence - 1);
            int logEnd = Math.min(baseline.tokens.size(), firstDivergence + 3);
            for (int i = logStart; i < logEnd; i++) {
                String bCheck = i < baseline.logitChecksums.size()
                        ? String.valueOf(baseline.logitChecksums.get(i)) : "N/A";
                String tCheck = i < treatment.logitChecksums.size()
                        ? String.valueOf(treatment.logitChecksums.get(i)) : "N/A";
                log.error("[prefill={}] Logit checksum at index {}: baseline={} treatment={}",
                        prefillSize, i, bCheck, tCheck);
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
                log.error("[prefill={}] Treatment output is DEGENERATE -- all tokens from divergence " +
                                "point are {}. This matches the known CUDA graph replay KV stale-read bug.",
                        prefillSize, treatment.tokens.get(firstDivergence));
                result.degenerate = true;
            }
        }

        return result;
    }

    // ========== Token generation ==========

    /**
     * Generate a deterministic token sequence of the requested length.
     * Starts with doctag (49229), then cycles through common structural tokens
     * that are within the embedding table vocabulary.
     */
    private int[] generatePrefillTokens(int length) {
        int vocabSize = (int) embeddingTable.size(0);
        int[] tokens = new int[length];
        // Start with doctag
        tokens[0] = Math.min(49229, vocabSize - 1);
        // Fill remaining with cycling pattern from low token IDs (guaranteed in vocab)
        // Use a mix of tokens to avoid pathological repetition patterns
        int[] seedTokens = {1, 42, 100, 256, 500, 1000, 2000, 3000, 4000, 5000,
                6000, 7000, 8000, 9000, 10000, 11126, 12000, 13000, 14000, 15000};
        for (int i = 1; i < length; i++) {
            int seedIdx = (i - 1) % seedTokens.length;
            tokens[i] = Math.min(seedTokens[seedIdx], vocabSize - 1);
        }
        return tokens;
    }

    // ========== Helper methods ==========

    private INDArray buildPrefillEmbeddings(int[] tokens) {
        long hiddenSize = embeddingTable.size(1);
        INDArray result = Nd4j.create(DataType.FLOAT, 1, tokens.length, hiddenSize);
        int vocabSize = (int) embeddingTable.size(0);
        for (int i = 0; i < tokens.length; i++) {
            int tokenId = tokens[i];
            // Clamp to vocabulary size
            if (tokenId >= vocabSize) {
                tokenId = tokenId % vocabSize;
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

    // ========== Data classes ==========

    private static class DecodeResult {
        List<Integer> tokens = new ArrayList<>();
        List<Double> logitChecksums = new ArrayList<>();
        List<Double> logitMaxValues = new ArrayList<>();
        List<Long> kvCachePositions = new ArrayList<>();
    }

    private static class SizeResult {
        int prefillSize;
        int totalTokens;
        int matchCount;
        boolean allMatch;
        int firstDivergenceStep = -1;
        boolean degenerate = false;
        boolean skipped = false;
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
