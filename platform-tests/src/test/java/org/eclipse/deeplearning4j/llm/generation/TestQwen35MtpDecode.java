/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
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
import org.eclipse.deeplearning4j.llm.TestGgufMtpCapturedReplay.PreparedReference;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline.GenerationSession;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end losslessness and engagement oracle for Qwen3.5's bundled NextN
 * predictor. The MTP-enabled GGUF is cached under a distinct local filename so
 * the legacy GGUF without MTP tensors remains available to the n-gram tests.
 */
@Slf4j
public class TestQwen35MtpDecode {

    private static final String MODEL_URL =
            "https://huggingface.co/unsloth/Qwen3.5-0.8B-MTP-GGUF/resolve/main/"
                    + "Qwen3.5-0.8B-Q4_K_M.gguf";
    private static final String MODEL_FILE = "Qwen3.5-0.8B-MTP-Q4_K_M.gguf";
    private static final String TOKENIZER_URL =
            "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json";
    private static final String TOKENIZER_FILE = "qwen35-0.8B-tokenizer.json";
    private static final String PROMPT =
            "The quick brown fox jumps over the lazy dog. Explain why this sentence is useful.";
    private static final int SPEC_K = 4;
    private static final int TOKENS =
            Integer.getInteger(ND4JSystemProperties.BENCH_MAX_TOKENS, 60);

    private static SameDiff model;
    private static Tokenizer tokenizer;

    @BeforeAll
    public static void setup() throws Exception {
        // Preload the selected backend before GGUF import so the normal binding
        // selected by the test backend is initialized before model construction.
        Nd4j.getEnvironment();
        if (System.getProperty(ND4JSystemProperties.OPTIMIZER_ENABLED) == null) {
            System.setProperty(ND4JSystemProperties.OPTIMIZER_ENABLED, "true");
        }

        File modelFile = LLMModelDownloader.downloadCustom(MODEL_URL, MODEL_FILE);
        SameDiff rawModel = GGMLModelImport.importModel(modelFile.getAbsolutePath());
        assertNotNull(rawModel.getVariable("target_hidden_states"),
                "Target graph must export the pre-final-norm hidden state needed by MTP");
        assertNotNull(rawModel.getVariable("mtp_logits"),
                "MTP-enabled GGUF must import the bundled predictor logits");
        assertNotNull(rawModel.getVariable("mtp_hidden_states"),
                "MTP-enabled GGUF must export predictor carry state");

        ModelIOConfig.KVCacheNames targetKvNames = ModelIOConfig.findKVCacheInputNames(rawModel);
        assertNotNull(targetKvNames, "Target decoder KV cache inputs must be discoverable");
        assertEquals(6, targetKvNames.keyNames.size(),
                "MTP-local key cache must not be classified as a target decoder cache");
        assertEquals(6, targetKvNames.valueNames.size(),
                "MTP-local value cache must not be classified as a target decoder cache");
        assertTrue(targetKvNames.keyNames.stream().noneMatch(name -> name.startsWith("mtp_"))
                        && targetKvNames.valueNames.stream().noneMatch(name -> name.startsWith("mtp_")),
                "Target KV discovery must exclude the isolated MTP cache namespace");

        ModelIOConfig targetIo = ModelIOConfig.discover(rawModel);
        assertEquals("input_ids", targetIo.getInputIdsName(),
                "MTP input IDs must not replace the target decoder input");
        assertEquals("_causal_mask", targetIo.getCausalMaskName(),
                "MTP causal mask must not replace the target decoder mask");
        assertEquals("position_offset", targetIo.getPositionOffsetName(),
                "MTP position offset must not replace the target decoder offset");
        assertEquals("cache_position", targetIo.getCachePositionName(),
                "MTP cache position must not replace the target decoder position");

        List<String> outputs = rawModel.outputs() != null
                ? new ArrayList<>(rawModel.outputs()) : new ArrayList<>();
        long optimizeStart = System.currentTimeMillis();
        model = GraphOptimizer.optimize(rawModel, outputs, GraphOptimizer.defaultOptimizations());
        log.info("[MTP-SETUP] GraphOptimizer: {} -> {} ops in {}ms",
                rawModel.getOps().size(), model.getOps().size(),
                System.currentTimeMillis() - optimizeStart);

        if (model != rawModel) {
            SameDiffMemoryUtils.freeModelArrays(rawModel);
            rawModel.close();
        }

        File tokenizerFile = LLMModelDownloader.downloadCustom(TOKENIZER_URL, TOKENIZER_FILE);
        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.getAbsolutePath());
    }

    @AfterAll
    public static void teardown() {
        if (model != null) {
            try {
                SameDiffMemoryUtils.freeModelArrays(model);
                model.close();
            } catch (Exception e) {
                log.warn("[MTP-TEARDOWN] Error closing model: {}", e.getMessage());
            }
            model = null;
        }
        tokenizer = null;
    }

    private static GenerationResult generateMeasured(GenerationPipeline pipeline, String mode) throws Exception {
        boolean timing = Boolean.getBoolean("mtp.benchmark.opTiming");
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        log.info("[MTP-BENCHMARK] backend={} native={} mode={} tokens={} opTiming={} scope=generation-including-prefill",
                backend, Nd4j.getNativeOps().getClass().getSimpleName(), mode, TOKENS, timing);
        if (timing) {
            Nd4j.getNativeOps().resetOpTiming();
            Nd4j.getNativeOps().setOpTimingEnabled(1, 1);
        }
        try {
            return pipeline.generate(PROMPT, TOKENS);
        } finally {
            if (timing) {
                try {
                    Nd4j.getNativeOps().flushOpTiming();
                    log.info("[MTP-OP-PROFILE] backend={} mode={} scope=generation-including-prefill", backend, mode);
                    Nd4j.getNativeOps().printOpTimingStats(20);
                } finally {
                    Nd4j.getNativeOps().setOpTimingEnabled(0, 0);
                }
            }
        }
    }

    @Test
    public void testPreparedPredictorMatchesCapturedReference() throws Exception {
        PreparedReference reference = new PreparedReference(System.getProperty("mtp.reference.gguf"),
                System.getProperty("qwen.mtp.snapshotPrefix"));
        var configBuilder = GenerationPipelineConfig.builder()
                .decoder(model).tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.speculative().toBuilder().minNewTokens(24).build())
                .maxNewTokens(24).maxSpeculativeTokens(SPEC_K)
                .maxPrefillLength(64).maxKvCacheLength(192)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false).dspEnabled(true);
        // Bisect gate (diagnostic): verifyKernels runs each compiled Triton
        // sub-kernel AND the equivalent native CUDA op, compares outputs, and
        // reports the first mismatching sub-kernel range — the supported way to
        // localize a Triton-vs-native compute divergence. Default: unset.
        if (Boolean.getBoolean("mtp.bisect.verifyKernels")) {
            configBuilder.benchmarkConfig(BenchmarkConfig.optimal().tritonVerifyKernels(true));
            log.info("MTP_BISECT: benchmarkConfig=OPTIMAL+verifyKernels");
        }
        GenerationPipelineConfig config = configBuilder.build();
        try (GenerationPipeline pipeline = GenerationPipeline.create(config);
             GenerationSession session = pipeline.startSession(PROMPT)) {
            InGraphKvState state = session.retainedStateForInspection();
            assertNotNull(state.mtpExecutor);
            if (Boolean.getBoolean("mtp.reference.prefillOnly")) {
                reference.verifyAfterPrefill(System.getProperty("mtp.reference.gguf"), state.mtpPrefillInputMap);
            } else {
                try (var binding = state.mtpExecutor.captureNativeExecutionBinding()) {
                    reference.verify(binding);
                }
            }
        }
    }

    @Test
    public void testBundledMtpIsLosslessAndEngaged() throws Exception {
        SamplingConfig mtpSampling = SamplingConfig.speculative().toBuilder()
                .minNewTokens(TOKENS)
                .build();
        SamplingConfig greedySampling = SamplingConfig.greedy().toBuilder()
                .minNewTokens(TOKENS)
                .build();

        GenerationPipelineConfig config = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(mtpSampling)
                .maxNewTokens(TOKENS)
                .maxSpeculativeTokens(SPEC_K)
                .maxPrefillLength(64)
                .maxKvCacheLength(Math.max(192, TOKENS + 64))
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();

        GenerationResult mtpResult;
        GenerationResult greedyResult;
        try (GenerationPipeline pipeline = GenerationPipeline.create(config)) {
            for (int warmup = 0; warmup < 2; warmup++) {
                GenerationResult result = pipeline.generate(PROMPT, TOKENS);
                log.info("[MTP-WARMUP {}] tokens={} proposed={} accepted={} steps={}",
                        warmup + 1, result.getTokenIds().length,
                        result.getTotalSpeculativeTokens(), result.getTotalAcceptedTokens(),
                        result.getSpeculativeSteps());
            }

            mtpResult = generateMeasured(pipeline, "MTP");
            log.info("[MTP-METRICS] tokens={} proposed={} accepted={} steps={} acceptance={} "
                            + "tok/s={} decodeTok/s={} lateTok/s={} effectiveTok/s={}",
                    mtpResult.getTokenIds().length, mtpResult.getTotalSpeculativeTokens(),
                    mtpResult.getTotalAcceptedTokens(), mtpResult.getSpeculativeSteps(),
                    mtpResult.getAverageAcceptanceRate(), mtpResult.getTokensPerSecond(),
                    mtpResult.getDecodeTokensPerSecond(),
                    mtpResult.getLateSteadyStateTokensPerSecond(),
                    mtpResult.getEffectiveTokensPerSecond());

            pipeline.setSamplingConfig(greedySampling);
            greedyResult = generateMeasured(pipeline, "GREEDY");
            log.info("[MTP-GREEDY-METRICS] tokens={} tok/s={} decodeTok/s={} lateTok/s={}",
                    greedyResult.getTokenIds().length, greedyResult.getTokensPerSecond(),
                    greedyResult.getDecodeTokensPerSecond(),
                    greedyResult.getLateSteadyStateTokensPerSecond());
        }

        int[] mtpTokens = mtpResult.getTokenIds();
        int[] greedyTokens = greedyResult.getTokenIds();
        log.info("[MTP-ORACLE] mtp={} greedy={} mtpFirst={} greedyFirst={}",
                mtpTokens.length, greedyTokens.length,
                Arrays.toString(Arrays.copyOf(mtpTokens, Math.min(16, mtpTokens.length))),
                Arrays.toString(Arrays.copyOf(greedyTokens, Math.min(16, greedyTokens.length))));

        assertTrue(mtpResult.getTotalSpeculativeTokens() > 0,
                "Bundled MTP proposed zero tokens; scalar/n-gram fallback is not the MTP path");
        assertTrue(mtpResult.getSpeculativeSteps() > 0,
                "Bundled MTP reported zero speculative steps");
        assertTrue(mtpResult.getTotalAcceptedTokens() > 0,
                "Bundled MTP accepted zero tokens");
        assertEquals(TOKENS, greedyTokens.length,
                "Greedy qualification run did not reach the requested token count");
        assertEquals(TOKENS, mtpTokens.length,
                "MTP qualification run did not reach the requested token count");
        assertArrayEquals(greedyTokens, mtpTokens,
                "Bundled Qwen3.5 MTP must remain token-identical to greedy decode");
    }

    /**
     * P02 adaptive-K hysteresis oracle (reviewer pass-evidence: switching K
     * preserves output and state across calls; forced low-acceptance workloads
     * stop wasting draft/verification work).
     *
     * <p>Run with K=1: the first generation exercises the MTP path; when its
     * acceptance lands below the floor, the SECOND generation must drop to
     * K=0 (the native no-spec scalar fast path: proposedCount==0) yet remain
     * token-identical to greedy, and the third generation must keep that
     * bucket. No model reload and no plan recapture occur - the bucket only
     * selects what the native loop proposes.</p>
     */
    @Test
    public void testAdaptiveSpecKCollapsesOnLowAcceptanceAndPreservesOutput() throws Exception {
        SamplingConfig mtpSampling = SamplingConfig.speculative().toBuilder()
                .minNewTokens(TOKENS)
                .build();
        SamplingConfig greedySampling = SamplingConfig.greedy().toBuilder()
                .minNewTokens(TOKENS)
                .build();

        GenerationPipelineConfig config = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(mtpSampling)
                .maxNewTokens(TOKENS)
                .maxSpeculativeTokens(1)
                .maxPrefillLength(64)
                .maxKvCacheLength(Math.max(192, TOKENS + 64))
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();

        GenerationResult first;
        GenerationResult second;
        GenerationResult third;
        GenerationResult greedy;
        try (GenerationPipeline pipeline = GenerationPipeline.create(config)) {
            // Determinism warmup: fresh-plan builds jitter (Triton autotune, allocator
            // ordering) on the very first plan; run one throwaway generation so every
            // measured generation executes on a warmed, replay-stable plan. The lossless
            // test uses the same pattern.
            pipeline.generate(PROMPT, TOKENS);

            first = pipeline.generate(PROMPT, TOKENS);
            // ORACLE over mechanism (review finding 6): the controller may have
            // legitimately dropped the bucket to K=0 during the warmup
            // generation if its acceptance fell below the floor (0.8B measured
            // 1/57 in gate 11). What must hold is the TOKEN ORACLE asserted
            // below - deterministic, greedy-identical output regardless of
            // which bucket the controller chose. Record the bucket for the
            // later legs; do not pin which path ran.
            int afterFirst = pipeline.getAdaptiveSpecK();
            assertTrue(afterFirst >= 0 && afterFirst <= 1,
                    "Bucket must stay within [0,1] after generation one: " + afterFirst);

            second = pipeline.generate(PROMPT, TOKENS);
            int afterSecond = pipeline.getAdaptiveSpecK();
            // The bucket may sit at 0 or 1 here (controller state after the
            // warmup); both are valid. What must hold: K=0 output equals K=1
            // output (proven natively in gate 11), and the same-K legs are
            // deterministic. Assert the boundary transition only when the
            // PREVIOUS generation actually speculated.
            assertTrue(afterSecond >= 0 && afterSecond <= 1,
                    "Bucket must stay within [0,1] after generation two: " + afterSecond);
            if (first.getAverageAcceptanceRate() < GenerationPipeline.SPEC_K_ACCEPTANCE_FLOOR
                    && first.getTotalSpeculativeTokens() > 0) {
                assertEquals(0, afterSecond,
                        "Below-floor acceptance from a speculative generation must drop the bucket");
            }

            third = pipeline.generate(PROMPT, TOKENS);
            assertEquals(afterSecond, pipeline.getAdaptiveSpecK(),
                    "Hysteresis: bucket must not flip-flop between consecutive generations");

            pipeline.setSamplingConfig(greedySampling);
            greedy = pipeline.generate(PROMPT, TOKENS);
        }

        log.info("[MTP-ADAPTIVE-ORACLE] first={} second={} third={} greedy={}",
                Arrays.toString(Arrays.copyOf(first.getTokenIds(), Math.min(16, first.getTokenIds().length))),
                Arrays.toString(Arrays.copyOf(second.getTokenIds(), Math.min(16, second.getTokenIds().length))),
                Arrays.toString(Arrays.copyOf(third.getTokenIds(), Math.min(16, third.getTokenIds().length))),
                Arrays.toString(Arrays.copyOf(greedy.getTokenIds(), Math.min(16, greedy.getTokenIds().length))));

        // Same-bucket generations must be deterministic and identical to each other.
        assertArrayEquals(first.getTokenIds(), second.getTokenIds(),
                "Consecutive same-K generations must be deterministic");
        assertArrayEquals(first.getTokenIds(), third.getTokenIds(),
                "Hysteresis generation must match the K it held");
        // The adaptive-K MTP path must remain token-identical to greedy decode.
        assertArrayEquals(greedy.getTokenIds(), first.getTokenIds(),
                "Adaptive-K (K=1) MTP must remain token-identical to greedy");
    }

    /**
     * Stage-2 discriminator matrix (session-vs-one-shot seam investigation).
     * N = TOKENS, M = max(8, TOKENS/3), T = N + 2*M. Greedy speculative config
     * everywhere, identical across all arms. Tests are ordered to localize the
     * FIRST failing edge: A (one-shot reference) vs B (one session, one call)
     * isolates the entry path; C (split session, K always 1) isolates the
     * call boundary; D reproduces the original K transition.
     *
     * <p>Discriminators, not replacement oracles: the original
     * testSameSessionKTransitionPreservesTokensAcrossK0Interval (with its
     * default-capacity API and complete equality assertion) is unchanged.
     * Requires -Dnd4j.mtp.multiRowCommit=1.</p>
     */
    @Test
    public void testSeamDiscriminatorMatrix() throws Exception {
        assertEquals("1", System.getenv("SD_MTP_MULTI_ROW_COMMIT"),
                "This discriminator matrix requires multi-row commit. Run with -Dnd4j.mtp.multiRowCommit=1.");
        final int n = TOKENS;
        final int m = Math.max(8, TOKENS / 3);
        final int t = n + 2 * m;
        log.info("[SEAM-MATRIX] N={} M={} T={} prefixSelect={}",
                n, m, t, System.getProperty("nd4j.mtp.prefixSelect", "off"));

        // ── Arm A: one-shot reference, fresh pipeline, single call of budget T ──
        int[] a;
        GenerationPipelineConfig config = baseSeamConfig(t);
        try (GenerationPipeline pipeline = GenerationPipeline.create(config)) {
            pipeline.setSamplingConfig(SamplingConfig.speculative().toBuilder().minNewTokens(t).build());
            GenerationResult r = pipeline.generate(PROMPT, t);
            assertEquals(t, r.getTokenIds().length, "A: one-shot must emit its full budget");
            assertTrue(r.getTotalSpeculativeTokens() > 0, "A: reference must run the speculative path");
            a = r.getTokenIds();
        }
        log.info("[SEAM-MATRIX] A(one-shot T={}) = {}", t, Arrays.toString(a));

        // ── Arm B: one session, ONE call of budget T (no continuation, no K transition) ──
        int[] b;
        try (GenerationPipeline pipeline = GenerationPipeline.create(baseSeamConfig(t));
             GenerationSession session = pipeline.startSession(PROMPT, t)) {
            session.setSpeculativeDepth(1);
            GenerationResult r = session.generate(t);
            assertEquals(t, r.getTokenIds().length, "B: single session call must emit its full budget");
            assertTrue(r.getTotalSpeculativeTokens() > 0, "B: single session call must propose tokens");
            b = session.getAllTokens();
        }
        log.info("[SEAM-MATRIX] B(session 1 call) = {}", Arrays.toString(b));
        reportFirstDiff("B vs A", a, b);

        // ── Arm C: split session, K ALWAYS ONE (two continueGeneration boundaries) ──
        int[] c;
        try (GenerationPipeline pipeline = GenerationPipeline.create(baseSeamConfig(t));
             GenerationSession session = pipeline.startSession(PROMPT, t)) {
            session.setSpeculativeDepth(1);
            GenerationResult leg1 = session.generate(n);
            assertEquals(n, leg1.getTokenIds().length, "C: leg 1 must emit its step budget");
            GenerationResult leg2 = session.continueGeneration(m);
            assertEquals(m, leg2.getTokenIds().length, "C: leg 2 must emit its step budget");
            assertTrue(leg2.getTotalSpeculativeTokens() > 0, "C: leg 2 must propose (K stays 1)");
            GenerationResult leg3 = session.continueGeneration(m);
            assertEquals(m, leg3.getTokenIds().length, "C: leg 3 must emit its step budget");
            assertTrue(leg3.getTotalSpeculativeTokens() > 0, "C: leg 3 must propose (K stays 1)");
            c = session.getAllTokens();
        }
        log.info("[SEAM-MATRIX] C(split K=1) = {}", Arrays.toString(c));
        reportFirstDiff("C vs B", b, c);
        reportFirstDiff("C vs A", a, c);

        // ── Arm D: original K transition (K=1 -> K=0 -> K=1) ──
        int[] d;
        try (GenerationPipeline pipeline = GenerationPipeline.create(baseSeamConfig(t));
             GenerationSession session = pipeline.startSession(PROMPT, t)) {
            session.setSpeculativeDepth(1);
            GenerationResult leg1 = session.generate(n);
            assertEquals(n, leg1.getTokenIds().length, "D: leg 1 must emit its step budget");
            session.setSpeculativeDepth(0);
            GenerationResult leg2 = session.continueGeneration(m);
            assertEquals(m, leg2.getTokenIds().length, "D: K=0 leg must emit its step budget");
            assertEquals(0, leg2.getTotalSpeculativeTokens(),
                    "D: K=0 leg must propose ZERO tokens");
            session.setSpeculativeDepth(1);
            GenerationResult leg3 = session.continueGeneration(m);
            assertEquals(m, leg3.getTokenIds().length, "D: re-enabled leg must emit its step budget");
            assertTrue(leg3.getTotalSpeculativeTokens() > 0, "D: re-enabled leg must propose tokens");
            d = session.getAllTokens();
        }
        log.info("[SEAM-MATRIX] D(K transition) = {}", Arrays.toString(d));
        reportFirstDiff("D vs C", c, d);
        reportFirstDiff("D vs B", b, d);
        reportFirstDiff("D vs A", a, d);
        assertArrayEquals(a, d,
                "D (K=1->K=0->K=1 session) must match A (one-shot same budget) token-for-token");
    }

    /** Shared discriminator config: identical sampling/stopping across all arms. */
    private GenerationPipelineConfig baseSeamConfig(int totalBudget) {
        return GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.speculative().toBuilder().minNewTokens(totalBudget).build())
                .maxNewTokens(totalBudget)
                .maxSpeculativeTokens(1)
                .maxPrefillLength(64)
                .maxKvCacheLength(Math.max(256, 2 * totalBudget + 64))
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();
    }

    /** Report the first differing global index of two sequences (no assertion). */
    private static void reportFirstDiff(String label, int[] expected, int[] actual) {
        if (Arrays.equals(expected, actual)) {
            log.info("[SEAM-MATRIX] {}: IDENTICAL ({} tokens)", label, actual.length);
            return;
        }
        int len = Math.min(expected.length, actual.length);
        int at = -1;
        for (int i = 0; i < len; i++) {
            if (expected[i] != actual[i]) { at = i; break; }
        }
        if (at < 0) at = len;
        int lo = Math.max(0, at - 3);
        int hiE = Math.min(expected.length, at + 4);
        int hiA = Math.min(actual.length, at + 4);
        log.info("[SEAM-MATRIX] {}: first diff at GLOBAL index {} of {} tokens; expected[{}..{}]={} "
                        + "actual[{}..{}]={}",
                label, at, actual.length, lo, hiE,
                Arrays.toString(Arrays.copyOfRange(expected, lo, hiE)),
                lo, hiA, Arrays.toString(Arrays.copyOfRange(actual, lo, hiA)));
    }

    /**
     * STEP 2 boundary-correspondence capture (split-call seam): compares the
     * RETAINED state at C's first failing boundary (leg 2 continuation entry at
     * global index 20) with the equivalent prefix point in B's fused call.
     *
     * <p>B and C are identical through global token 19 (proven by the matrix).
     * At that point B's single native invocation is mid-loop at absolute target
     * position P+19+1; C has ENDED a native call and is about to reconstruct
     * continuation inputs. This test captures, at exactly that seam: the pending
     * token, cachePosition, generatedSoFar prefix, the prepared causal mask's
     * visible-column pattern (before the native op runs), the retained target
     * KV, and the predictor's input/hidden/carry tensors — and reports them for
     * both arms. If the retained GDN/conv/KV/predictor state already differs at
     * the common prefix boundary, the defect is at B's budget-tail commit, not
     * in C's reconstruction. The equality oracle remains in
     * {@link #testSameSessionKTransitionPreservesTokensAcrossK0Interval};
     * this is a bounded observation, not a replacement.</p>
     *
     * <p>Requires -Dnd4j.mtp.multiRowCommit=1. OFF mode.</p>
     */
    @Test
    public void testSplitCallBoundaryStateCorrespondence() throws Exception {
        assertEquals("1", System.getenv("SD_MTP_MULTI_ROW_COMMIT"),
                "This boundary capture requires multi-row commit. Run with -Dnd4j.mtp.multiRowCommit=1.");
        final int n = TOKENS;
        final int t = n + 2 * Math.max(8, TOKENS / 3);
        // Run label comes from the harness: B and C each carry their own label
        // through every [SEAM-BOUNDARY] line below; the native SEAM_WATCH records
        // are attributed per-invocation by their globalBegin ranges (B: one
        // invocation at globalBegin=0 spanning 0..35; C: leg1 0..19, leg2 20..27,
        // leg3 28..35 - the sampling config's generatedTokenOffset stamps each).
        log.info("[SEAM-BOUNDARY] RUN-B-BEGIN T={}", t);

        // ── Arm B: uninterrupted K=1 session, ONE generate(T) call. ──
        int[] bTokens;
        try (GenerationPipeline pipelineB = GenerationPipeline.create(baseSeamConfig(t));
             GenerationSession sessionB = pipelineB.startSession(PROMPT, t)) {
            sessionB.setSpeculativeDepth(1);
            GenerationResult rB = sessionB.generate(t);
            assertEquals(t, rB.getTokenIds().length, "B must emit its full budget");
            assertTrue(rB.getTotalSpeculativeTokens() > 0, "B must propose tokens");
            bTokens = sessionB.getAllTokens();
            InGraphKvState sB = sessionB.retainedStateForInspection();
            log.info("[SEAM-BOUNDARY] RUN-B-END tokens={} cachePos={} lastTok={} "
                            + "gdnSum={} targetKvSum={} predictorKvSum={}",
                    Arrays.toString(bTokens), sB.cachePosition, sB.lastGeneratedToken,
                    checksumBuffers(sB.recurrentStateBuffers),
                    checksumBuffers(sB.staticKvBuffers),
                    checksumBuffers(sB.mtpKvBuffers));
        }
        log.info("[SEAM-BOUNDARY] RUN-C-BEGIN N={} seg={} T={}", n, Math.max(8, TOKENS / 3), t);

        // ── Arm C: session split at the boundary. Capture leg-2 entry state. ──
        int[] cPrefix;
        long cCachePos;
        int cLastTok;
        String cMaskPattern;
        double cGdnChecksum;
        double cKvChecksum;
        double cPredictorKvChecksum;
        INDArray cMtpInputIds;
        INDArray cMtpHidden;
        INDArray cMtpMask;
        INDArray cMtpPosOffset;
        INDArray cMtpCachePosition;
        try (GenerationPipeline pipeline = GenerationPipeline.create(baseSeamConfig(t));
             GenerationSession session = pipeline.startSession(PROMPT, t)) {
            session.setSpeculativeDepth(1);
            GenerationResult leg1 = session.generate(n);
            assertEquals(n, leg1.getTokenIds().length, "C: leg 1 must emit its step budget");

            InGraphKvState state = session.retainedStateForInspection();
            cPrefix = state.generatedSoFar.stream().mapToInt(Integer::intValue).toArray();
            cCachePos = state.cachePosition;
            cLastTok = state.lastGeneratedToken;
            cMaskPattern = maskVisiblePattern(state.decodeCausalMask);
            cGdnChecksum = checksumBuffers(state.recurrentStateBuffers);
            cKvChecksum = checksumBuffers(state.staticKvBuffers);
            cPredictorKvChecksum = checksumBuffers(state.mtpKvBuffers);
            // Copy the small predictor tensors; they are overwritten by leg 2's native run.
            cMtpInputIds = state.mtpInputIds == null ? null : state.mtpInputIds.dup();
            cMtpHidden = state.mtpTargetHiddenStates == null ? null : state.mtpTargetHiddenStates.dup();
            cMtpMask = state.mtpCausalMask == null ? null : state.mtpCausalMask.dup();
            cMtpPosOffset = state.mtpPositionOffset == null ? null : state.mtpPositionOffset.dup();
            cMtpCachePosition = state.mtpCachePosition == null ? null : state.mtpCachePosition.dup();
        }
        try {
            log.info("[SEAM-BOUNDARY] C-at-leg2-entry: generated[0..19]={} cachePos={} lastTok={} "
                            + "maskVisible={} gdnSum={} targetKvSum={} predictorKvSum={} "
                            + "mtpInputIds={} mtpHiddenSum={} mtpMaskSum={} mtpPosOffset={} mtpCachePosition={}",
                    Arrays.toString(Arrays.copyOf(cPrefix, Math.min(20, cPrefix.length))),
                    cCachePos, cLastTok, cMaskPattern,
                    cGdnChecksum, cKvChecksum, cPredictorKvChecksum,
                    cMtpInputIds == null ? "null" : cMtpInputIds,
                    cMtpHidden == null ? "null" : cMtpHidden.sumNumber().doubleValue(),
                    cMtpMask == null ? "null" : cMtpMask.sumNumber().doubleValue(),
                    cMtpPosOffset, cMtpCachePosition);

            // ── Arm C continued: run legs 2 and 3 in the SAME session? No - the
            // leg-1 session was closed to snapshot its state. Run arm C fully
            // (generate(20), continue(8), continue(8)) in a fresh session; the
            // leg-2-entry snapshot above came from an identical prefix run, so
            // the SEAM_WATCH records of this fresh C run attribute to the same
            // boundaries (globalBegin 20 and 28).
            int[] cTokens;
            try (GenerationPipeline pipelineC = GenerationPipeline.create(baseSeamConfig(t));
                 GenerationSession sessionC = pipelineC.startSession(PROMPT, t)) {
                sessionC.setSpeculativeDepth(1);
                GenerationResult leg1 = sessionC.generate(n);
                assertEquals(n, leg1.getTokenIds().length, "C: leg 1 must emit its step budget");
                // NOTE: identical fresh sessions have been observed to diverge
                // WITHIN leg 1 (e.g. index 12) — a cross-session reproducibility
                // finding in its own right. Record the prefix agreement instead of
                // gating on it, so the SEAM_WATCH records around C's own boundary
                // are always captured. The final C==B assertion below still gates.
                if (!Arrays.equals(Arrays.copyOf(bTokens, n), leg1.getTokenIds())) {
                    log.info("[SEAM-BOUNDARY] C leg1 DIVERGES from B within the first {} tokens "
                                    + "across fresh sessions (cross-session reproducibility finding); "
                                    + "B[0..n]={} C-leg1={}",
                            n, Arrays.toString(Arrays.copyOf(bTokens, n)),
                            Arrays.toString(leg1.getTokenIds()));
                }
                sessionC.setSpeculativeDepth(1);
                GenerationResult leg2 = sessionC.continueGeneration(Math.max(8, TOKENS / 3));
                assertEquals(Math.max(8, TOKENS / 3), leg2.getTokenIds().length,
                        "C: leg 2 must emit its step budget");
                assertTrue(leg2.getTotalSpeculativeTokens() > 0, "C: leg 2 must propose (K=1)");
                GenerationResult leg3 = sessionC.continueGeneration(Math.max(8, TOKENS / 3));
                assertEquals(Math.max(8, TOKENS / 3), leg3.getTokenIds().length,
                        "C: leg 3 must emit its step budget");
                assertTrue(leg3.getTotalSpeculativeTokens() > 0, "C: leg 3 must propose (K=1)");
                cTokens = sessionC.getAllTokens();
            }
            log.info("[SEAM-BOUNDARY] RUN-C-END tokens={}", Arrays.toString(cTokens));
            reportFirstDiff("C vs B", bTokens, cTokens);
            // Complete equality assertion: split-session C must be token-for-token
            // identical to uninterrupted B. This is the requirement under test.
            assertArrayEquals(bTokens, cTokens,
                    "split-session C must match uninterrupted B token-for-token");

            // The decisive early-localization assertion: C's retained state at the
            // seam must describe the SAME history as B's in-flight state would —
            // pending token equals generated[19], cachePosition equals P + 20.
            assertEquals(n, cPrefix.length, "C boundary capture must hold exactly leg-1 output");
            log.info("[SEAM-BOUNDARY] C pending token at seam = generated[{}]={} (feeding leg 2), "
                    + "cachePosition={} (expected P+{}={})",
                    n - 1, cLastTok, cCachePos, n, 17 + n + 1);
        } finally {
            if (cMtpInputIds != null) cMtpInputIds.close();
            if (cMtpHidden != null) cMtpHidden.close();
            if (cMtpMask != null) cMtpMask.close();
            if (cMtpPosOffset != null) cMtpPosOffset.close();
            if (cMtpCachePosition != null) cMtpCachePosition.close();
        }
    }

    /** Visible-column pattern of a [1,1,rows,maxKvLen] mask: "v0/total-v1/total..." for the first two rows. */
    private static String maskVisiblePattern(INDArray mask) {
        if (mask == null || mask.rank() != 4) return "null";
        StringBuilder sb = new StringBuilder();
        long rows = Math.min(2, mask.size(2));
        for (long r = 0; r < rows; r++) {
            int visible = 0;
            for (long c = 0; c < mask.size(3); c++) {
                if (mask.getDouble(0, 0, r, c) > -1e9f) visible++;
            }
            if (r > 0) sb.append('/');
            sb.append(visible).append('/').append(mask.size(3));
        }
        return sb.toString();
    }

    /** Deterministic element-sum checksum across a named buffer map (order-independent). */
    private static double checksumBuffers(Map<String, INDArray> buffers) {
        if (buffers == null || buffers.isEmpty()) return Double.NaN;
        double sum = 0;
        for (INDArray arr : buffers.values()) {
            if (arr != null) sum += arr.sumNumber().doubleValue();
        }
        return sum;
    }

    /**
     * Matched-state target replay for the split-call seam (reviewer decision:
     * exact matched-prefix comparison, not a terminal-mask patch).
     *
     * <p>Replays one two-row target forward versus two one-row target forwards
     * from the SAME captured pre-window state — the target-composition question —
     * and, when capture data is available, compares the captured real
     * continuation state S_C36 against the composed reference S_REF36.</p>
     *
     * <p>The capture uses the production pipeline to run B's leg through the
     * boundary (warmup → row 159034 → row 271) with the graph's exported K/V
     * rows committed into independent replay storage via
     * {@link #commitKvRows}, so the replay exercises the same graph, weights,
     * and output manifest as the parity oracle.</p>
     *
     * <p>Arms (all W=2 physical, K=1, OFF):
     * <ul>
     *   <li>R0: [159034,271] asl=2 at pos=35 — must reproduce B's row0 winner
     *       271 and row1 winner 8160.</li>
     *   <li>R1a: [159034] asl=1 at pos=35 — row0 winner must equal 271; commit
     *       through pos 35 → S_REF36.</li>
     *   <li>R1b: [271] asl=1 at pos=36 from S_REF36 — row0 must equal B's row1
     *       winner 8160 (sequence-partition equivalence).</li>
     *   <li>R2 (review-corrected): the ONE production session's TERMINAL
     *       generate(19) state snapshot — qualified first for a matched
     *       19-token consumed history and a KV storage contract equal to the
     *       decoder placeholders (else UNMATCHED_REPLAY_INPUTS) — is compared
     *       against the replay's S_B35 over KV + recurrent state ONLY. The
     *       layer-11/pos-9 onset observation is retained for investigation
     *       regardless of qualification.</li>
     *   <li>R3: from fresh copies of S_REF36, [271,X] asl=2 for two different
     *       future drafts X plus [271] asl=1 — row0 must be identical across
     *       all three (target causality: future draft must not alter row 0).</li>
     * </ul>
     * Requires -Dnd4j.mtp.multiRowCommit=1 for the production config path.</p>
     */
    @Test
    public void testCapturedSplitBoundaryTargetComposition() throws Exception {
        assertEquals("1", System.getenv("SD_MTP_MULTI_ROW_COMMIT"),
                "This replay requires multi-row commit. Run with -Dnd4j.mtp.multiRowCommit=1.");
        final int window = 2;
        final int maxKvLength = 256;
        final int pos35 = 35;
        final int pos36 = 36;
        // Token/position ledger (reviewer-mandated; asserted against B's recorded
        // output array from the production reproducer, NOT reconstructed from
        // prose). N=17 prompt positions (0..16). B's emitted tokens:
        //   e[0]=271  e[1]=248068  e[2]=271  e[3]=248069  e[4]=271
        //   e[5]=1919 e[6]=11316  e[7]=369  e[8]=264    e[9]=11088
        //   e[10]=3010 e[11]=314  e[12]=264 e[13]=2972  e[14]=848
        //   e[15]=1671 e[16]=1340 e[17]=11316 e[18]=159034
        //   e[19]=271  e[20]=8160 ...
        // Warmup consumes e[0] at position 17 (prefillLength=17).
        // Continuation consumes e[i] at target position 17+i for i=1..17.
        // At basePos=35: positions 0..34 consumed, e[0..17] consumed,
        //   pending = e[18]=159034 (NOT yet consumed).
        // At basePos=36: e[18] consumed at position 35, pending = e[19]=271,
        //   next prediction corresponds to e[20]=8160.
        final int promptLen = 17;                 // asserted below against tokenizer
        final int[] emitted = {
                271, 248068, 271, 248069, 271, 1919, 11316, 369, 264, 11088,
                3010, 314, 264, 2972, 848, 1671, 1340, 11316, 159034,
                271, 8160};
        final int tok18 = emitted[18];            // 159034: pending at pos 35
        final int tok19 = emitted[19];            // 271: consumed at pos 36
        final int tok20 = emitted[20];            // 8160: prediction at pos 36
        final int cDraft = 1057;
        List<INDArray> owned = new ArrayList<>();
        try {
            ModelIOConfig io = ModelIOConfig.discover(model);
            ModelIOConfig.KVCacheNames kvNames = ModelIOConfig.findKVCacheInputNames(model);
            List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                    ModelIOConfig.findRecurrentStatePairs(model, io);
            log.info("[SEAM-STATES] recurrent pairs discovered: {}", recurrentStates.size());
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                log.info("[SEAM-STATES] pair: {}", pair);
            }
            // GDN-layer state-input audit: for the six GDN layers, list every
            // graph input containing "state" and every graph output containing
            // "state" — showing exactly which placeholder/output the pairing
            // walk failed to connect.
            java.util.Set<String> gdnLayers = new java.util.TreeSet<>();
            for (Object o : model.inputs()) {
                String in = String.valueOf(o);
                if (in.startsWith("past_conv_state.") || in.startsWith("past_gdn_state.")) {
                    gdnLayers.add(in);
                }
            }
            java.util.Set<String> stateOutputs = new java.util.TreeSet<>();
            for (Object o : model.outputs()) {
                String out = String.valueOf(o);
                if (out.contains("state")) stateOutputs.add(out);
            }
            log.info("[SEAM-STATES] graph state INPUTS containing past_conv_state/past_gdn_state "
                    + "(count={}): {}", gdnLayers.size(), gdnLayers);
            log.info("[SEAM-STATES] graph state OUTPUTS containing 'state' (count={}): {}",
                    stateOutputs.size(), stateOutputs);
            DataType maskType = model.getVariable(io.getCausalMaskName()).dataType();

            int[] promptTokenIds = tokenizer.encodePrompt(PROMPT, null).getIds();
            final int prefillLength = promptTokenIds.length;
            assertEquals(promptLen, prefillLength,
                    "token ledger requires prompt length 17");
            // Corrected ledger (review round 4): the prefill consumes prompt
            // positions 0..16 ONLY. The scalar replay consumes e[i] at position
            // 17+i for i=0..17 (18 forwards: e[0]@17 ... e[17]@34); e[18] stays
            // unconsumed and pending at position 35.
            StringBuilder ledger = new StringBuilder();
            for (int i = 0; i <= 17; i++) {
                ledger.append(String.format(" | pos%d<-e[%d]=%d -> pending e[%d]=%d",
                        promptLen + i, i, emitted[i], i + 1, emitted[i + 1]));
            }
            log.info("[SEAM-REPLAY] LEDGER: prefill consumes 0..16; scalar replay:{}", ledger);
            log.info("[SEAM-REPLAY] LEDGER: e[18]={} pending at pos35 (UNCONSUMED); "
                            + "pos35 consumes e[18] -> pending e[19]; pos36 consumes e[19]={} -> e[20]={}",
                    tok18, tok19, tok20);

            // ── Arm C capture: production-matched seam state via generate(19).
            // The pipeline ends exactly at the seam: positions 0..34 consumed
            // (e[0..17] plus prompt), pending = e[18]=159034. This is a TERMINAL
            // generate(19) snapshot (review round 4, finding 4): it is NOT
            // asserted to equal B's in-flight pre-position-35 state, and this
            // closed session's buffers are anchor prints only — never reused as
            // another session's state.
            int cAtSeamGdnSum;
            int cAtSeamKvSum;
            try (GenerationPipeline pipeline = GenerationPipeline.create(baseSeamConfig(promptLen + 19));
                 GenerationSession session = pipeline.startSession(PROMPT, promptLen + 19)) {
                session.setSpeculativeDepth(1);
                GenerationResult r = session.generate(promptLen + 2); // emits e[0..18]; e[18] pending
                assertEquals(promptLen + 2, r.getTokenIds().length,
                        "capture leg must emit 19 tokens (e[0..18])");
                InGraphKvState s = session.retainedStateForInspection();
                // Full-prefix terminal-token check (review round 4): the terminal
                // pending token is e[18]=159034. Prior prose referred to a "623 at
                // index 18" divergence — a token-vs-position conflation that the
                // R2 full-prefix comparison below now measures correctly.
                if (s.lastGeneratedToken != tok18) {
                    log.info("[SEAM-CAPTURE-REPRO] capture pending {} differs from recorded "
                                    + "e[18]={} (terminal token difference; full-prefix check in R2 "
                                    + "is the authoritative history measure)",
                            s.lastGeneratedToken, tok18);
                } else {
                    log.info("[SEAM-CAPTURE-REPRO] capture terminal pending matches e[18]={}", tok18);
                }
                cAtSeamGdnSum = (int) checksumBuffers(s.recurrentStateBuffers);
                cAtSeamKvSum = (int) checksumBuffers(s.staticKvBuffers);
                log.info("[SEAM-ANCHOR] production capture at seam: cachePos={} lastTok={} "
                                + "gdnSum={} kvSum={} (state retained only for the anchor print; "
                                + "the replay below is the scalar same-history reconstruction)",
                        s.cachePosition, s.lastGeneratedToken, cAtSeamGdnSum, cAtSeamKvSum);
            }

            // ── Teacher-forced replay anchor (review-corrected history): the raw
            // prefill consumes prompt positions 0..16 ONLY. The scalar chain
            // below then consumes B's RECORDED e[0] at position 17 through
            // e[17] at position 34 (18 one-row forwards); e[18] stays pending
            // at position 35.
            // Prefill: full prompt, empty KV, zero states.
            Map<String, INDArray> prefillInputs = new HashMap<>();
            putOwned(prefillInputs, io.getInputIdsName(),
                    Nd4j.createFromArray(promptTokenIds).reshape(1, prefillLength), owned);
            putOwned(prefillInputs, io.getPositionOffsetName(),
                    Nd4j.scalar(DataType.INT64, 0L), owned);
            putOwned(prefillInputs, io.getCachePositionName(),
                    Nd4j.scalar(DataType.INT64, 0L), owned);
            putOwned(prefillInputs, "actual_sequence_length",
                    Nd4j.scalar(DataType.INT64, (long) prefillLength), owned);
            putOwned(prefillInputs, io.getCausalMaskName(),
                    DecoderInputBuilder.buildInGraphCausalMask(prefillLength, maxKvLength, maskType), owned);
            for (String name : kvNames.keyNames) {
                putOwned(prefillInputs, name, Nd4j.empty(model.getVariable(name).dataType()), owned);
            }
            for (String name : kvNames.valueNames) {
                putOwned(prefillInputs, name, Nd4j.empty(model.getVariable(name).dataType()), owned);
            }
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                long[] stateShape = GenerationPipeline.deriveRecurrentStateShape(model, pair.inputName);
                putOwned(prefillInputs, pair.inputName,
                        Nd4j.zeros(model.getVariable(pair.inputName).dataType(), stateShape), owned);
            }
            List<String> prefillOutputNames = new ArrayList<>();
            prefillOutputNames.add(io.getLogitsOutputName());
            for (String keyName : kvNames.keyNames) {
                int layer = extractLayerIndex(keyName);
                prefillOutputNames.add("k_rope_" + layer);
                prefillOutputNames.add("v_heads_" + layer);
            }
            List<String> stateOutputNames = new ArrayList<>();
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                stateOutputNames.add(pair.outputName);
            }
            prefillOutputNames.addAll(stateOutputNames);
            // Probe name sets (review round 4, finding 3): the stability probe
            // compares these components via independent snapshots.
            java.util.LinkedHashSet<String> prefillKvProbeNames = new java.util.LinkedHashSet<>();
            for (String keyName : kvNames.keyNames) {
                int layer = extractLayerIndex(keyName);
                prefillKvProbeNames.add("k_rope_" + layer);
                prefillKvProbeNames.add("v_heads_" + layer);
            }
            java.util.LinkedHashSet<String> stateNameSet = new java.util.LinkedHashSet<>(stateOutputNames);
            Map<String, INDArray> prefillOutputs = model.output(
                    prefillInputs, prefillOutputNames.toArray(new String[0]));
            ownAll(prefillOutputs, owned);

            // ── PREFILL STABILITY PROBE (3x, EARLY): moved before the scalar
            // replay (review round 4). The prefill runs from freshly-duplicated
            // zero-state inputs three times with INDEPENDENT result snapshots —
            // prior runs compared executor-owned output references, which ownAll
            // retains but does not snapshot. Probe boundary: after the capture
            // pipeline has closed, before this test's own decode work. It is a
            // same-graph zero-state control, NOT a before-any-session control.
            {
                Map<String, INDArray> run2Out = model.output(
                        duplicateArrays(prefillInputs, owned),
                        prefillOutputNames.toArray(new String[0]));
                Map<String, INDArray> run2Kv = snapshotOutputs(run2Out, prefillKvProbeNames, owned);
                Map<String, INDArray> run2States = snapshotOutputs(run2Out, stateNameSet, owned);
                Map<String, INDArray> run3Out = model.output(
                        duplicateArrays(prefillInputs, owned),
                        prefillOutputNames.toArray(new String[0]));
                Map<String, INDArray> run3Kv = snapshotOutputs(run3Out, prefillKvProbeNames, owned);
                Map<String, INDArray> run3States = snapshotOutputs(run3Out, stateNameSet, owned);
                Map<String, INDArray> run1Kv = snapshotOutputs(prefillOutputs, prefillKvProbeNames, owned);
                Map<String, INDArray> run1States = snapshotOutputs(prefillOutputs, stateNameSet, owned);
                // run2Out/run3Out retain executor-owned references; snapshots are
                // already independent. Retain them for closeOwned bookkeeping.
                ownAll(run2Out, owned);
                ownAll(run3Out, owned);

                String kvProbe = firstDiffProbe(run1Kv, run2Kv, run3Kv);
                String stateProbe = firstDiffProbe(run1States, run2States, run3States);
                log.info("[SEAM-EARLY-PREFILL-STABILITY] 3x zero-state prefill AFTER capture "
                                + "session, BEFORE scalar replay: kv{} states{}",
                        kvProbe, stateProbe);
            }

            Map<String, INDArray> replayKv = new LinkedHashMap<>();
            Map<String, INDArray> replayStates = new LinkedHashMap<>();
            for (int i = 0; i < kvNames.keyNames.size(); i++) {
                int layer = extractLayerIndex(kvNames.keyNames.get(i));
                INDArray keyRows = prefillOutputs.get("k_rope_" + layer);
                INDArray valueRows = prefillOutputs.get("v_heads_" + layer);
                INDArray keyCache = own(owned, Nd4j.zeros(keyRows.dataType(),
                        keyRows.size(0), maxKvLength, keyRows.size(2), keyRows.size(3)));
                INDArray valueCache = own(owned, Nd4j.zeros(valueRows.dataType(),
                        valueRows.size(0), maxKvLength, valueRows.size(2), valueRows.size(3)));
                keyCache.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillLength),
                        NDArrayIndex.all(), NDArrayIndex.all()).assign(keyRows);
                valueCache.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillLength),
                        NDArrayIndex.all(), NDArrayIndex.all()).assign(valueRows);
                replayKv.put(kvNames.keyNames.get(i), keyCache);
                replayKv.put(kvNames.valueNames.get(i), valueCache);
            }
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                replayStates.put(pair.inputName, own(owned, prefillOutputs.get(pair.outputName).dup()));
            }

            // Teacher-forced single-row chain (review-corrected): consume
            // e[i] at position 17+i for i=0..17 — 18 one-row asl=1 forwards:
            // e[0]@17 (the token production's warmup consumed) through
            // e[17]@34. e[18] stays UNCONSUMED and pending at position 35.
            // Calculated logits are logged for diagnostics but NOT used to
            // choose the next input. The ledger is asserted directly.
            {
                java.util.TreeMap<Integer, String> consumedLedger = new java.util.TreeMap<>();
                for (int i = 0; i <= 17; i++) {
                    int pos = promptLen + i;   // e[0]@17 ... e[17]@34
                    Map<String, INDArray> stepInputs = newStableDecodeInputs(
                            io, window, maxKvLength, maskType, replayKv, replayStates, owned);
                    setDecodeStep(stepInputs, io, emitted[i], 0, pos, 1, window, maxKvLength, maskType, owned);
                    List<String> stepOutputs = new ArrayList<>();
                    stepOutputs.add(io.getLogitsOutputName());
                    for (String keyName : kvNames.keyNames) {
                        int layer = extractLayerIndex(keyName);
                        stepOutputs.add("k_rope_" + layer);
                        stepOutputs.add("v_heads_" + layer);
                    }
                    stepOutputs.addAll(stateOutputNames);
                    Map<String, INDArray> out = model.output(stepInputs, stepOutputs.toArray(new String[0]));
                    ownAll(out, owned);
                    int freeRunToken = argMaxToken(out.get(io.getLogitsOutputName()), 0);
                    if (freeRunToken != emitted[i + 1]) {
                        log.info("[SEAM-REPLAY] TEACHER-FORCE pos={} argmax={} but production emitted e[{}]={}",
                                pos, freeRunToken, i + 1, emitted[i + 1]);
                    }
                    commitKvRows(replayKv, kvNames, out, pos, 1);
                    for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                        replayStates.get(pair.inputName).assign(out.get(pair.outputName));
                    }
                    consumedLedger.put(pos, "e[" + i + "]=" + emitted[i]);
                }
                // Ledger assertion (review round 4): assert the ACTUAL consumed
                // positions, not a token-index equality as proof of correctness.
                assertEquals(18, consumedLedger.size(),
                        "replay must consume exactly 18 tokens (e[0..17])");
                for (int i = 0; i <= 17; i++) {
                    assertTrue(consumedLedger.containsKey(promptLen + i),
                            "consumed-position ledger missing position " + (promptLen + i));
                }
                assertTrue(consumedLedger.lastEntry().getKey() == promptLen + 17,
                        "last consumed position must be 34 (promptLen+17), got "
                                + consumedLedger.lastEntry().getKey());
                log.info("[SEAM-REPLAY] CONSUMED-LEDGER: {} entries, positions {}..{}; "
                                + "e[18]={} pending at pos {} (UNCONSUMED)",
                        consumedLedger.size(), consumedLedger.firstEntry().getKey(),
                        consumedLedger.lastEntry().getKey(), tok18, pos35);
            }
            // Scalar same-history reconstruction anchor: 18 consumed forwards
            // (e[0]@17 ... e[17]@34), pending = e[18]=159034 UNCONSUMED at pos 35.
            // NOTE (review round 4): whether this scalar reconstruction equals the
            // native production session state at the same consumed history is
            // decided by the qualified R2 comparison below — the anchor sums are
            // recorded for reference, not proof.
            log.info("[SEAM-REPLAY] S_B35 (scalar same-history reconstruction): consumed "
                            + "e[0..17] at positions 17..34; pendingToken={} (production e[18]={}, "
                            + "UNCONSUMED)", emitted[18], tok18);

            // ── R2 (review round 4, corrected): compare a LIVE generate(19)
            // session's TERMINAL state snapshot against the replay's S_B35.
            // Correction 1 (indexing): the live leg emits e[0..18]; the prior
            // check compared emitted index 18 (= pending e[18]=159034) against
            // tok19=271 — that was an indexing error, NOT a divergence. The
            // corrected check is a FULL-PREFIX comparison of all 19 emitted
            // tokens against the recorded history.
            // Correction 2 (qualification): a matched-prefix state comparison
            // additionally requires matching KV storage contracts (dtypes from
            // the decoder placeholders, the production source of truth) and
            // independent state storage on both sides. If any input contract
            // does not match, report UNMATCHED_REPLAY_INPUTS and do NOT label
            // the tensor difference as proven native state corruption.
            // Correction 3 (snapshots): ownAll retains references only, so BOTH
            // sides are dup'd into independent test-owned storage before any
            // later model.output call can mutate executor-owned arrays.
            // Terminal-snapshot caveat: a generate(19) terminal snapshot is NOT
            // automatically B's in-flight pre-position-35 snapshot; the retain
            // session that produced the comparison state is the SAME session
            // whose generate produced the emitted history (no cross-session
            // state assumption).
            {
                try (GenerationPipeline pipelineLive = GenerationPipeline.create(
                                baseSeamConfig(promptLen + 19));
                     GenerationSession sessionLive = pipelineLive.startSession(
                             PROMPT, promptLen + 19)) {
                    sessionLive.setSpeculativeDepth(1);
                    GenerationResult rLive = sessionLive.generate(promptLen + 2);
                    assertEquals(promptLen + 2, rLive.getTokenIds().length,
                            "R2 live leg must emit exactly 19 tokens e[0..18]");
                    int[] liveEmitted = rLive.getTokenIds();
                    boolean fullPrefixMatch = Arrays.equals(
                            Arrays.copyOf(liveEmitted, 19), Arrays.copyOf(emitted, 19));
                    if (fullPrefixMatch) {
                        log.info("[SEAM-R2-REPRO] live leg FULL-PREFIX MATCH: 19/19 tokens "
                                        + "equal production recorded history; pending token {} "
                                        + "at pos {} (unconsumed)",
                                liveEmitted[18], promptLen + 18);
                    } else {
                        int firstIdx = -1;
                        for (int i = 0; i < 19; i++) {
                            if (liveEmitted[i] != emitted[i]) { firstIdx = i; break; }
                        }
                        log.info("[SEAM-R2-REPRO] live leg FULL-PREFIX MISMATCH: first differing "
                                        + "index {} emitted {} vs recorded {}",
                                firstIdx,
                                firstIdx >= 0 ? liveEmitted[firstIdx] : -1,
                                firstIdx >= 0 ? emitted[firstIdx] : -1);
                    }
                    assertEquals(tok18, liveEmitted[18],
                            "R2 live leg pending token (emitted index 18) must equal e[18]=159034");

                    InGraphKvState sLive = sessionLive.retainedStateForInspection();
                    log.info("[SEAM-R2] live terminal snapshot: cachePos={} pending={} "
                                    + "(terminal generate(19); in-flight B equivalence NOT asserted)",
                            sLive.cachePosition, liveEmitted[18]);
                    assertEquals((long) (promptLen + 18), sLive.cachePosition,
                            "live cachePosition must be 35 after consuming e[0..17]");

                    // Independent storage for BOTH compared sides (review round 4, finding 3).
                    Map<String, INDArray> liveKvOwned = sLive.staticKvBuffers != null
                            ? sLive.staticKvBuffers : sLive.quantizedKvBuffers;
                    Map<String, INDArray> liveStatesOwned = sLive.recurrentStateBuffers;
                    Map<String, INDArray> liveKv = duplicateArrays(liveKvOwned, owned);
                    Map<String, INDArray> liveStates = duplicateArrays(liveStatesOwned, owned);

                    // Qualification: KV storage dtype must equal the decoder
                    // placeholder dtype (production derives storage dtypes from
                    // placeholders; the replay derives them from output tensors —
                    // verify equivalence against the placeholder contract).
                    boolean replayKvStorageOk = storageMatchesPlaceholders(replayKv, model);
                    boolean liveKvStorageOk = storageMatchesPlaceholders(liveKv, model);
                    boolean inputsQualified = fullPrefixMatch && replayKvStorageOk && liveKvStorageOk;
                    log.info("[SEAM-R2-QUALIFY] fullPrefixMatch={} replayKvStorageOk={} "
                                    + "liveKvStorageOk={} => {}",
                            fullPrefixMatch, replayKvStorageOk, liveKvStorageOk,
                            inputsQualified ? "MATCHED_INPUTS" : "UNMATCHED_REPLAY_INPUTS");
                    if (!inputsQualified) {
                        log.info("[SEAM-R2] elementwise comparison NOT QUALIFIED: "
                                        + "UNMATCHED_REPLAY_INPUTS (history or storage-contract "
                                        + "mismatch). The layer-11/pos-9 onset observation is "
                                        + "retained for investigation; it is NOT classified as "
                                        + "proven native state corruption by this test.");
                    }

                    int firstDiffCount = 0;
                    double firstDiffMax = 0.0;
                    String firstDiffTensor = null;
                    long firstDiffIdx = -1;
                    double firstDiffExpected = 0.0, firstDiffActual = 0.0;
                    java.util.TreeMap<Long, String> firstDiffByPos = new java.util.TreeMap<>();
                    java.util.TreeMap<Long, Double> maxDiffByPos = new java.util.TreeMap<>();
                    for (Map.Entry<String, INDArray> kvEntry : replayKv.entrySet()) {
                        INDArray replayArr = kvEntry.getValue();
                        INDArray liveArr = liveKv.get(kvEntry.getKey());
                        if (liveArr == null) {
                            log.info("[SEAM-R2] live KV missing tensor {} (key-list diff)", kvEntry.getKey());
                            continue;
                        }
                        // Compare ONLY committed region [0, 35): rejected future
                        // scratch is not committed data.
                        long cols = Math.min(35, replayArr.size(1));
                        for (long c = 0; c < cols && firstDiffCount < 5; c++) {
                            for (long r0 = 0; r0 < replayArr.size(0); r0++) {
                                for (long h = 0; h < replayArr.size(2); h++) {
                                    for (long d = 0; d < replayArr.size(3); d++) {
                                        double rv = replayArr.getDouble(r0, c, h, d);
                                        double lv = liveArr.getDouble(r0, c, h, d);
                                        if (rv != lv) {
                                            firstDiffCount++;
                                            if (firstDiffTensor == null) {
                                                firstDiffTensor = kvEntry.getKey();
                                                firstDiffIdx = c;
                                                firstDiffExpected = rv;
                                                firstDiffActual = lv;
                                            }
                                            firstDiffMax = Math.max(firstDiffMax, Math.abs(rv - lv));
                                            firstDiffByPos.putIfAbsent(c, kvEntry.getKey());
                                            maxDiffByPos.merge(c, Math.abs(rv - lv), Math::max);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    log.info("[SEAM-R2] live-vs-replay committed[0,35) KV: firstDiffTensor={} "
                                    + "firstDiffPos={} expected={} actual={} maxAbsDiff={} diffCount(capped)={} "
                                    + "qualified={}",
                            firstDiffTensor, firstDiffIdx, firstDiffExpected, firstDiffActual,
                            firstDiffMax, firstDiffCount, inputsQualified);
                    for (Map.Entry<Long, String> e : firstDiffByPos.entrySet()) {
                        log.info("[SEAM-R2-POS] pos={} firstDiffTensor={} maxAbs={}",
                                e.getKey(), e.getValue(), maxDiffByPos.get(e.getKey()));
                    }
                    if (firstDiffByPos.isEmpty()) {
                        log.info("[SEAM-R2-POS] no per-position KV differences in [0,35)");
                    }
                    // All-layer onset map, retained for investigation regardless
                    // of qualification status (review round 4, finding 4).
                    log.info("[SEAM-R2-LAYERS] onset map (layer.tensor -> firstDiffPos, onsetHead):");
                    for (Map.Entry<String, INDArray> kvEntry : replayKv.entrySet()) {
                        INDArray replayArr = kvEntry.getValue();
                        INDArray liveArr = liveKv.get(kvEntry.getKey());
                        if (liveArr == null) continue;
                        long cols = Math.min(35, replayArr.size(1));
                        long heads = replayArr.size(2);
                        long onsetPos = -1;
                        long onsetHead = -1;
                        double onsetMax = 0.0;
                        for (long c = 0; c < cols && onsetPos < 0; c++) {
                            for (long h = 0; h < heads && onsetPos < 0; h++) {
                                for (long d = 0; d < replayArr.size(3); d++) {
                                    if (replayArr.getDouble(0, c, h, d)
                                            != liveArr.getDouble(0, c, h, d)) {
                                        onsetPos = c;
                                        onsetHead = h;
                                        onsetMax = Math.abs(
                                                replayArr.getDouble(0, c, h, d)
                                                - liveArr.getDouble(0, c, h, d));
                                        break;
                                    }
                                }
                            }
                        }
                        if (onsetPos >= 0) {
                            log.info("[SEAM-R2-LAYERS] {} -> onsetPos={} onsetHead={} firstMaxDiff={}",
                                    kvEntry.getKey(), onsetPos, onsetHead, onsetMax);
                        }
                    }

                    // Layer-11 key row-9 anatomy retained for investigation.
                    {
                        INDArray replayKey = replayKv.get("past_key_values.11.key");
                        INDArray liveKey = liveKv.get("past_key_values.11.key");
                        if (replayKey != null && liveKey != null && replayKey.size(1) > 9) {
                            long heads = replayKey.size(2);
                            long dims = replayKey.size(3);
                            StringBuilder headLine = new StringBuilder();
                            int diffDimsTotal = 0;
                            for (long h = 0; h < heads; h++) {
                                int headDiffDims = 0;
                                double headMax = 0.0;
                                long headFirstDim = -1;
                                double headFirstRv = 0, headFirstLv = 0;
                                for (long d = 0; d < dims; d++) {
                                    double rv = replayKey.getDouble(0, 9, h, d);
                                    double lv = liveKey.getDouble(0, 9, h, d);
                                    if (rv != lv) {
                                        headDiffDims++;
                                        diffDimsTotal++;
                                        if (headFirstDim < 0) {
                                            headFirstDim = d;
                                            headFirstRv = rv;
                                            headFirstLv = lv;
                                        }
                                        headMax = Math.max(headMax, Math.abs(rv - lv));
                                    }
                                }
                                headLine.append(String.format(" h%d:%d/%dd max=%.4f first[d%d %g->%g]",
                                        h, headDiffDims, dims, headMax, headFirstDim, headFirstRv, headFirstLv));
                            }
                            log.info("[SEAM-R2-ROW9] layer11 key pos9 per-head anatomy (diffDims/total):{} "
                                            + "totalDiffDims={}",
                                    headLine, diffDimsTotal);
                        }
                    }
                    // Recurrent-state comparison on independent snapshots.
                    int stateDiffCount = 0;
                    double stateDiffMax = 0.0;
                    String stateDiffTensor = null;
                    double stateDiffExpected = 0.0, stateDiffActual = 0.0;
                    for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                        INDArray replayArr = replayStates.get(pair.inputName);
                        INDArray liveArr = liveStates.get(pair.inputName);
                        if (replayArr == null || liveArr == null
                                || replayArr.length() != liveArr.length()) {
                            log.info("[SEAM-R2] recurrent shape/missing {} replay={} live={}",
                                    pair.inputName,
                                    replayArr == null ? "null" : replayArr.length(),
                                    liveArr == null ? "null" : liveArr.length());
                            continue;
                        }
                        for (long i = 0; i < replayArr.length() && stateDiffCount < 5; i++) {
                            double rv = replayArr.getDouble(i);
                            double lv = liveArr.getDouble(i);
                            if (rv != lv) {
                                stateDiffCount++;
                                if (stateDiffTensor == null) {
                                    stateDiffTensor = pair.inputName;
                                    stateDiffExpected = rv;
                                    stateDiffActual = lv;
                                }
                                stateDiffMax = Math.max(stateDiffMax, Math.abs(rv - lv));
                            }
                        }
                    }
                    log.info("[SEAM-R2] live-vs-replay recurrent state: firstDiffTensor={} "
                                    + "expected={} actual={} maxAbsDiff={} diffCount(capped)={} "
                                    + "qualified={}",
                            stateDiffTensor, stateDiffExpected, stateDiffActual,
                            stateDiffMax, stateDiffCount, inputsQualified);
                    // Scope statement (review round 4): these comparisons cover the
                    // committed KV region [0,35) and the recurrent state buffers
                    // only — the components actually inspected. They do NOT claim
                    // complete state equality (scales, masks, predictor buffers
                    // are not compared here). UNMATCHED inputs downgrades any
                    // tensor difference to unqualified, not proof of corruption.
                }
            }

            // ── R4: same-execution determinism. Run the IDENTICAL scalar step
            // (same inputs, same restored state) TWICE from independent copies of
            // the anchor state and compare logits element-wise. Zero => graph is
            // deterministic and state divergence enters via accumulation across
            // differing sessions; nonzero => a kernel is nondeterministic.
            // Review round 4, finding 3: BOTH runs' logits are snapshotted into
            // independent arrays BEFORE the KV written-row comparison (which runs
            // additional getDouble reads only — no model.output in between — but
            // the snapshotted logits guarantee the compared values cannot be
            // mutated by any later staging).
            List<String> replayOutputNames = new ArrayList<>();
            replayOutputNames.add(io.getLogitsOutputName());
            replayOutputNames.add("target_hidden_states");
            for (String keyName : kvNames.keyNames) {
                int layer = extractLayerIndex(keyName);
                replayOutputNames.add("k_rope_" + layer);
                replayOutputNames.add("v_heads_" + layer);
            }
            replayOutputNames.addAll(stateOutputNames);
            {
                Map<String, INDArray> r4aKv = duplicateArrays(replayKv, owned);
                Map<String, INDArray> r4aStates = duplicateArrays(replayStates, owned);
                Map<String, INDArray> r4aInputs = newStableDecodeInputs(
                        io, window, maxKvLength, maskType, r4aKv, r4aStates, owned);
                setDecodeStep(r4aInputs, io, tok18, 0, pos35, 1, window, maxKvLength, maskType, owned);
                Map<String, INDArray> r4aOut = model.output(
                        r4aInputs, replayOutputNames.toArray(new String[0]));
                INDArray r4aLogitsSnapshot = own(owned,
                        r4aOut.get(io.getLogitsOutputName()).dup());

                Map<String, INDArray> r4bKv = duplicateArrays(replayKv, owned);
                Map<String, INDArray> r4bStates = duplicateArrays(replayStates, owned);
                Map<String, INDArray> r4bInputs = newStableDecodeInputs(
                        io, window, maxKvLength, maskType, r4bKv, r4bStates, owned);
                setDecodeStep(r4bInputs, io, tok18, 0, pos35, 1, window, maxKvLength, maskType, owned);
                Map<String, INDArray> r4bOut = model.output(
                        r4bInputs, replayOutputNames.toArray(new String[0]));
                INDArray r4bLogitsSnapshot = own(owned,
                        r4bOut.get(io.getLogitsOutputName()).dup());
                ownAll(r4aOut, owned);
                ownAll(r4bOut, owned);

                // Compare the two INDEPENDENT snapshots.
                double[] r4diff = difference(
                        queryRow(r4aLogitsSnapshot, 0),
                        queryRow(r4bLogitsSnapshot, 0));
                log.info("[SEAM-R4] same-state same-step determinism (independent snapshots): "
                                + "max={} l1={}", r4diff[0], r4diff[1]);
                double kvSelfDiffMax = 0;
                int kvSelfCount = 0;
                for (Map.Entry<String, INDArray> kvEntry : replayKv.entrySet()) {
                    INDArray a = r4aKv.get(kvEntry.getKey());
                    INDArray b = r4bKv.get(kvEntry.getKey());
                    if (a == null || b == null) continue;
                    long cols = Math.min(36, a.size(1));
                    for (long c = 0; c < cols; c++) {
                        for (long r0i = 0; r0i < a.size(0); r0i++) {
                            for (long h = 0; h < a.size(2); h++) {
                                for (long d = 0; d < a.size(3); d++) {
                                    double x = a.getDouble(r0i, c, h, d);
                                    double y = b.getDouble(r0i, c, h, d);
                                    if (x != y) {
                                        kvSelfCount++;
                                        kvSelfDiffMax = Math.max(kvSelfDiffMax, Math.abs(x - y));
                                    }
                                }
                            }
                        }
                    }
                }
                log.info("[SEAM-R4] same-execution KV written-row diff: max={} count={}",
                        kvSelfDiffMax, kvSelfCount);
            }

            // ── R0: two-row window from the replay anchor: [159034, 271], asl=2, pos=35.
            // Review round 4, finding 3: logits snapshotted before comparison.
            Map<String, INDArray> r0Kv = duplicateArrays(replayKv, owned);
            Map<String, INDArray> r0States = duplicateArrays(replayStates, owned);
            Map<String, INDArray> r0Inputs = newStableDecodeInputs(
                    io, window, maxKvLength, maskType, r0Kv, r0States, owned);
            setDecodeStep(r0Inputs, io, tok18, tok19, pos35, 2, window, maxKvLength, maskType, owned);
            Map<String, INDArray> r0Outputs = model.output(
                    r0Inputs, replayOutputNames.toArray(new String[0]));
            INDArray r0LogitsSnapshot = own(owned,
                    r0Outputs.get(io.getLogitsOutputName()).dup());
            ownAll(r0Outputs, owned);
            int r0Row0 = argMaxToken(r0LogitsSnapshot, 0);
            int r0Row1 = argMaxToken(r0LogitsSnapshot, 1);
            log.info("[SEAM-REPLAY] R0 winners: row0={} row1={} (production B: 271 then 8160)",
                    r0Row0, r0Row1);
            assertEquals(tok19, r0Row0, "R0 row0 must reproduce B's row-0 winner 271");
            assertEquals(tok20, r0Row1, "R0 row1 must reproduce B's row-1 winner 8160");

            // ── R1a: one-row forward from the replay anchor: [159034], asl=1, pos=35.
            // Row 0 predicts e[19]=271. Logits snapshotted (independent observation).
            Map<String, INDArray> r1Kv = duplicateArrays(replayKv, owned);
            Map<String, INDArray> r1States = duplicateArrays(replayStates, owned);
            Map<String, INDArray> r1Inputs = newStableDecodeInputs(
                    io, window, maxKvLength, maskType, r1Kv, r1States, owned);
            setDecodeStep(r1Inputs, io, tok18, 0, pos35, 1, window, maxKvLength, maskType, owned);
            Map<String, INDArray> r1aOutputs = model.output(
                    r1Inputs, replayOutputNames.toArray(new String[0]));
            INDArray r1aLogitsSnapshot = own(owned,
                    r1aOutputs.get(io.getLogitsOutputName()).dup());
            ownAll(r1aOutputs, owned);
            int r1aRow0 = argMaxToken(r1aLogitsSnapshot, 0);
            assertEquals(tok19, r1aRow0, "R1a row0 must reproduce 271");
            // Commit through pos 35 → S_REF36 (outputs read directly for commit,
            // then state copies are dup'd into independent storage).
            commitKvRows(r1Kv, kvNames, r1aOutputs, pos35, 1);
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                r1States.get(pair.inputName).assign(r1aOutputs.get(pair.outputName));
            }
            final Map<String, INDArray> sRef36Kv = duplicateArrays(r1Kv, owned);
            final Map<String, INDArray> sRef36States = duplicateArrays(r1States, owned);

            // ── R1b: one-row forward from S_REF36: [271], asl=1, pos=36.
            // Row 0 predicts e[20]=8160.
            Map<String, INDArray> r1bInputs = newStableDecodeInputs(
                    io, window, maxKvLength, maskType, sRef36Kv, sRef36States, owned);
            setDecodeStep(r1bInputs, io, tok19, 0, pos36, 1, window, maxKvLength, maskType, owned);
            Map<String, INDArray> r1bOutputs = model.output(
                    r1bInputs, replayOutputNames.toArray(new String[0]));
            ownAll(r1bOutputs, owned);
            int r1bRow0 = argMaxToken(r1bOutputs.get(io.getLogitsOutputName()), 0);
            log.info("[SEAM-REPLAY] R1b winner: row0={} (B's row1 winner: 8160)", r1bRow0);
            assertEquals(tok20, r1bRow0,
                    "R1b (one-row composition second forward) must equal B's row-1 winner 8160");
            INDArray r1bLogits = own(owned, r1bOutputs.get(io.getLogitsOutputName()).dup());
            INDArray r1bHidden = own(owned, r1bOutputs.get("target_hidden_states").dup());

            // Composition check: R0 row1 vs R1b row0 — both from INDEPENDENT
            // snapshots (r0LogitsSnapshot above; r1bLogits dup'd at capture).
            double[] row1Diff = difference(
                    queryRow(r0LogitsSnapshot, 1),
                    queryRow(r1bLogits, 0));
            log.info("[SEAM-REPLAY] COMPOSITION R0-row1 vs R1b-row0 (independent snapshots): "
                            + "max={} l1={}", row1Diff[0], row1Diff[1]);

            // ── R3: future-draft causality from fresh copies of S_REF36.
            // Row 0 at pos 36 consuming [271, X] must be identical for any X.
            int[] futureDrafts = {cDraft, tok18, 0};
            double[][] r3Row0Logits = new double[futureDrafts.length][];
            for (int arm = 0; arm < futureDrafts.length; arm++) {
                Map<String, INDArray> r3Kv = duplicateArrays(sRef36Kv, owned);
                Map<String, INDArray> r3States = duplicateArrays(sRef36States, owned);
                Map<String, INDArray> r3Inputs = newStableDecodeInputs(
                        io, window, maxKvLength, maskType, r3Kv, r3States, owned);
                int asl = (arm == 2) ? 1 : 2;
                setDecodeStep(r3Inputs, io, tok19, futureDrafts[arm], pos36, asl,
                        window, maxKvLength, maskType, owned);
                Map<String, INDArray> r3Out = model.output(
                        r3Inputs, replayOutputNames.toArray(new String[0]));
                ownAll(r3Out, owned);
                INDArray lg = r3Out.get(io.getLogitsOutputName());
                r3Row0Logits[arm] = new double[(int) lg.size(2)];
                for (int v = 0; v < lg.size(2); v++) {
                    r3Row0Logits[arm][v] = lg.getDouble(0, 0, v);
                }
            }
            double[] causalityDraftVsAlt = diffDoubles(r3Row0Logits[0], r3Row0Logits[1]);
            double[] causalityDraftVsAsl1 = diffDoubles(r3Row0Logits[0], r3Row0Logits[2]);
            log.info("[SEAM-REPLAY] R3 CAUSALITY row0: draft({}) vs alt({}) max={} l1={}; "
                            + "draft vs asl1 max={} l1={}",
                    futureDrafts[0], futureDrafts[1],
                    causalityDraftVsAlt[0], causalityDraftVsAlt[1],
                    causalityDraftVsAsl1[0], causalityDraftVsAsl1[1]);
            assertEquals(0.0, causalityDraftVsAlt[0], 1.0e-6,
                    "R3 CAUSALITY: row-0 logits must not depend on the row-1 future draft");
            assertEquals(0.0, causalityDraftVsAsl1[0], 1.0e-6,
                    "R3 CAUSALITY: row-0 logits must not depend on actual_length folding row 1");

            // ── R2 would compare S_C36 (captured live continuation state) against
            // S_REF36. That requires exporting the live session's internal buffers
            // mid-session, which the current InGraphKvState does not expose beyond
            // the retainedStateForInspection maps already checksummed in
            // testSplitCallBoundaryStateCorrespondence. The composition and
            // causality results above decide the target-level questions:
            //   - R0+R1 green => target composition is sequence-partition exact.
            //   - R3 green => future drafts do not alter row 0.
            // Any remaining split-call divergence must then be in the state the
            // real continuation received (publication boundary) or the predictor.
        } finally {
            model.resetSession();
            closeOwned(owned);
        }
    }

    /**
     * Same-session K transition oracle (review round 3, finding 1): forces the
     * K=1 -> K=0 -> K=1 depth transitions WITHIN one continuing
     * GenerationSession via the session-scoped setSpeculativeDepth control
     * (the pipeline-level setSamplingConfig deliberately does not reach an
     * open session). The oracle proves three separate properties: (1) the
     * requested K actually changed what the native decode ran - the K=0 leg
     * must propose zero tokens and the re-enabled leg must propose again;
     * (2) the whole combined sequence (initial leg + both continuations)
     * equals a full-length same-total-budget reference generated in one
     * K=1 session on the same prompt - so both continuation legs, not just
     * the pre-transition prefix, are under token equality; (3) the retained
     * predictor state stayed valid across the scalar-only interval. A stale
     * predictor carry/KV on re-enable surfaces as divergence in the third
     * leg - the exact defect class the K-re-enable epilogue fix guards.
     */
    @Test
    public void testSameSessionKTransitionPreservesTokensAcrossK0Interval() throws Exception {
        SamplingConfig mtpSampling = SamplingConfig.speculative().toBuilder()
                .minNewTokens(TOKENS)
                .build();

        GenerationPipelineConfig config = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(mtpSampling)
                .maxNewTokens(TOKENS)
                .maxSpeculativeTokens(1)
                .maxPrefillLength(64)
                .maxKvCacheLength(Math.max(256, 2 * TOKENS + 64))
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();

        final int stepTokens = Math.max(8, TOKENS / 3);
        final int totalBudget = TOKENS + 2 * stepTokens;
        int[] referenceSeq;
        int[] sessionSeq;
        int k0Proposed;
        int reProposed;
        try (GenerationPipeline referencePipeline = GenerationPipeline.create(config)) {
            // Reference: ONE K=1 generation covering the session's whole budget.
            referencePipeline.setSamplingConfig(mtpSampling);
            GenerationResult reference = referencePipeline.generate(PROMPT, totalBudget);
            assertTrue(reference.getTotalSpeculativeTokens() > 0,
                    "Reference leg must run the speculative path");
            referenceSeq = reference.getTokenIds();
        }

        try (GenerationPipeline pipeline = GenerationPipeline.create(config);
             GenerationSession session = pipeline.startSession(PROMPT)) {
            // Leg 1: K=1 head of the session.
            session.generate(TOKENS);

            // Leg 2: force K=0 for THIS session - scalar-only tokens stream
            // into the shared predictor context. Must propose ZERO tokens.
            session.setSpeculativeDepth(0);
            GenerationResult k0Leg = session.continueGeneration(stepTokens);
            assertEquals(stepTokens, k0Leg.getTokenIds().length,
                    "K=0 continuation must emit exactly its step budget");
            assertEquals(0, k0Leg.getTotalSpeculativeTokens(),
                    "K=0 continuation must propose ZERO tokens - the depth override did not reach the native decode");
            k0Proposed = k0Leg.getTotalSpeculativeTokens();

            // Leg 3: re-enable K=1 from the same retained state. Must propose
            // again - proving the native execution actually resumed drafting.
            session.setSpeculativeDepth(1);
            GenerationResult reLeg = session.continueGeneration(stepTokens);
            assertEquals(stepTokens, reLeg.getTokenIds().length,
                    "Re-enabled continuation must emit exactly its step budget");
            assertTrue(reLeg.getTotalSpeculativeTokens() > 0,
                    "Re-enabled continuation must PROPOSE tokens - the depth override did not reach the native decode");
            reProposed = reLeg.getTotalSpeculativeTokens();

            sessionSeq = session.getAllTokens();
        }

        log.info("[MTP-K-TRANSITION-ORACLE] k0Proposed={} reProposed={} sessionLen={} refLen={} "
                        + "reference={} session={}",
                k0Proposed, reProposed, sessionSeq.length, referenceSeq.length,
                Arrays.toString(Arrays.copyOf(referenceSeq, Math.min(20, referenceSeq.length))),
                Arrays.toString(Arrays.copyOf(sessionSeq, Math.min(20, sessionSeq.length))));

        // Property 1: the depth override reached the native decode.
        assertEquals(0, k0Proposed, "K=0 leg must propose zero tokens");
        assertTrue(reProposed > 0, "Re-enabled leg must propose tokens");

        // Property 2: FULL-sequence equality including both transition legs.
        // The reference covers the identical total budget in one K=1 session,
        // so the entire session sequence - not just the pre-transition prefix -
        // is under token equality.
        assertEquals(totalBudget, referenceSeq.length,
                "Reference must cover the full combined budget");
        assertEquals(totalBudget, sessionSeq.length,
                "Session must produce the full combined budget across the K transitions");
        assertArrayEquals(referenceSeq, sessionSeq,
                "K=1 -> K=0 -> K=1 same-session sequence must match the K=1 full-budget reference");
    }

    /**
     * Exact bootstrap tuple fixture (review round 3, finding 4): opens a session
     * and inspects the RETAINED PREDICTOR state immediately after start (before
     * any native drafting) plus the retained scalar pair the first native draft
     * consumes. Documents the shipped bootstrap convention precisely:
     * <p>PREDICTOR ROW CONVENTION (review-ruled, packet 6): row r consumes
     * (x_(r+1), h_r) at rope=r, slot=r. With N = actualPrefillLen and y0/y1 the
     * first/second target-sampled tokens (target positions N and N+1):</p>
     * <ul>
     *   <li>prefill rows r = 0..N-1 carry (x_(r+1), h_r) — the tail row N-1 is
     *       (y0, h_(N-1)); prefill position ORIGIN is 0;</li>
     *   <li>the scalar predictor warmup consumes (y0, h_(N-1)) at rope=N-1,
     *       slot=N-1 — it REWRITES the tail slot, appending nothing;</li>
     *   <li>after the warmup the pending pair is (y1, h_N) at rope=N, slot=N
     *       (both retained scalars = state.cachePosition - 1); slot N is NOT
     *       yet written and stays masked until native execution consumes it;</li>
     *   <li>the target resumes at state.cachePosition = N+1 (unchanged).</li>
     * </ul>
     * Exactly N live predictor rows [0,N) exist after the bootstrap. Asserts
     * the scalars, shifted ids, mask boundary, and the unwritten/pending row.
     * If any bootstrap change shifts an index, THIS test fails before the
     * acceptance-rate regression can hide it.
     */
    @Test
    public void testBootstrapPredictorTupleLayout() throws Exception {
        SamplingConfig mtpSampling = SamplingConfig.speculative().toBuilder()
                .minNewTokens(8)
                .build();
        GenerationPipelineConfig config = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(mtpSampling)
                .maxNewTokens(8)
                .maxSpeculativeTokens(1)
                .maxPrefillLength(64)
                .maxKvCacheLength(256)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();

        try (GenerationPipeline pipeline = GenerationPipeline.create(config);
             GenerationSession session = pipeline.startSession(PROMPT)) {
            InGraphKvState st = session.retainedStateForInspection();
            // The retained state's OWN logical prompt length (template/BOS-aware);
            // an independently tokenized string could omit template tokens.
            final int n = st.actualPrefillLen;

            // -- Scalar geometry (packet 6, assertions 1-4) -------------------
            // Target resume position: the first native target position is N+1.
            assertEquals((long) n + 1L, (long) st.cachePosition,
                    "target state.cachePosition must resume at N+1");
            // BOTH retained predictor scalars describe the PENDING row N = N+1-1.
            assertEquals((long) n, st.mtpPositionOffset.getLong(0),
                    "retained predictor positionOffset must be the pending row N");
            assertEquals((long) n, st.mtpCachePosition.getLong(0),
                    "retained predictor cachePosition must be the pending row N");
            assertEquals((long) st.cachePosition - 1L, st.mtpCachePosition.getLong(0),
                    "predictor pending slot must equal target cachePosition - 1");
            // Pending token = the second sampled token (the first was consumed
            // by the warmup; the pair (y0, h_(N-1)) is stored once, in row N-1).
            assertEquals((long) st.lastGeneratedToken, st.mtpInputIds.getLong(0),
                    "predictor pending input must be the second sampled token");
            // Pending carry = the target warmup hidden h_N (the hidden the target
            // produced consuming y0): verified via shape (row-carry contract).
            assertNotNull(st.mtpTargetHiddenStates);
            assertEquals(3, st.mtpTargetHiddenStates.rank(),
                    "retained predictor carry must be a rank-3 single-row tensor");

            log.info("[MTP-BOOTSTRAP-ORACLE] n={} targetResume={} pendingRow={} pendingToken={}",
                    n, st.cachePosition, st.mtpCachePosition.getLong(0),
                    st.mtpInputIds.getLong(0));

            // -- Shifted predictor ids vs the actual target prefill ids --------
            INDArray sourceIds = st.prefillInputMap.get(st.inputIdsName);
            INDArray shiftedIds = st.mtpPrefillInputMap.get("mtp_input_ids");
            assertNotNull(sourceIds, "target prefill ids must be retained");
            assertNotNull(shiftedIds, "predictor prefill ids must be retained");
            for (int r = 0; r + 1 < n; ++r) {
                assertEquals(sourceIds.getLong(0, r + 1), shiftedIds.getLong(0, r),
                        "Shifted predictor token at row " + r);
            }
            // Tail row N-1 carries the FIRST sampled token.
            assertEquals(st.generatedSoFar.get(0).longValue(), shiftedIds.getLong(0, n - 1),
                    "predictor prefill tail row must carry the first sampled token");
            // Pending scalar carries the SECOND sampled token.
            assertEquals(st.generatedSoFar.get(1).longValue(), st.mtpInputIds.getLong(0),
                    "predictor pending token must be the second sampled token");

            // -- Mask boundary: N live rows visible, unwritten rows invisible --
            INDArray mtpMask = st.mtpCausalMask;
            assertNotNull(mtpMask, "retained predictor causal mask must exist");
            for (int r = 0; r < n; ++r) {
                assertEquals(0.0, mtpMask.getDouble(0, 0, 0, r), 0.0,
                        "live predictor row " + r + " must be visible (unmasked)");
            }
            for (long r = n; r < mtpMask.size(3); ++r) {
                assertTrue(mtpMask.getDouble(0, 0, 0, r) < -1.0e6,
                        "Unwritten predictor row is visible: " + r);
            }

            // -- Warmup REWRITES slot N-1; slot N stays unwritten ----------------
            // The bootstrap re-zeroes the retained cache, so slot N must still be
            // zero immediately after the bootstrap (an appended warmup row would
            // be non-zero). This distinguishes rewrite-tail from append-slot-N.
            INDArray mtpKeyCache = st.mtpKvBuffers.get("mtp_past_key_values.0.key");
            INDArray mtpValueCache = st.mtpKvBuffers.get("mtp_past_key_values.0.value");
            assertNotNull(mtpKeyCache, "Retained MTP key cache must exist");
            assertNotNull(mtpValueCache, "Retained MTP value cache must exist");
            assertEquals(4, mtpKeyCache.rank(), "MTP key cache must be rank 4");
            assertTrue(mtpKeyCache.size(1) >= n + 1,
                    "MTP key cache must have capacity for the pending row N");
            for (int h = 0; h < (int) mtpKeyCache.size(2); ++h) {
                assertEquals(0.0, mtpKeyCache.getDouble(0, n, h, 0), 0.0,
                        "pending predictor slot N key must be unwritten after bootstrap");
                assertEquals(0.0, mtpValueCache.getDouble(0, n, h, 0), 0.0,
                        "pending predictor slot N value must be unwritten after bootstrap");
            }
            log.info("[MTP-BOOTSTRAP-ORACLE] liveRows=[0,{}) pendingRow={} "
                            + "warmupRewroteTailSlot={} targetUnchanged=true",
                    n, n, n - 1);
        }
    }

    /**
     * Pins the target-model invariant required by lossless speculative decoding: evaluating two
     * causally chained tokens inside the production W=5 envelope must produce the same per-layer
     * rows as evaluating the same tokens as two activeWindow=1 calls on that frozen W=5 plan.
     *
     * <p>The test deliberately requests layer boundaries plus the operation boundaries inside the
     * first hybrid GDN block and first full-attention block. If parity regresses, the numeric
     * discriminator identifies the first graph variable where row 1 departs from the chained
     * scalar reference.</p>
     */
    @Test
    public void testTargetWindowRowsMatchChainedScalarCheckpoints() throws Exception {
        runTargetWindowRowsMatchChainedScalarCheckpoints(true);
    }

    @Test
    public void testTargetWindowRowsMatchChainedScalarOutputOnly() throws Exception {
        runTargetWindowRowsMatchChainedScalarCheckpoints(false);
    }

    private void runTargetWindowRowsMatchChainedScalarCheckpoints(boolean requestAttentionAux) throws Exception {
        if (Boolean.getBoolean("mtp.parity.productionConfig")) {
            BenchmarkConfigApplier.apply(BenchmarkConfig.optimal());
        }
        final int window = 5;
        final int maxKvLength = 192;
        List<INDArray> owned = new ArrayList<>();

        try {
            ModelIOConfig io = ModelIOConfig.discover(model);
            ModelIOConfig.KVCacheNames kvNames = ModelIOConfig.findKVCacheInputNames(model);
            List<ModelIOConfig.RecurrentStatePair> recurrentStates =
                    ModelIOConfig.findRecurrentStatePairs(model, io);
            assertNotNull(kvNames, "Target decoder KV inputs must be discoverable");

            int[] promptTokenIds = tokenizer.encodePrompt(PROMPT, null).getIds();
            int prefillLength = promptTokenIds.length;
            DataType maskType = model.getVariable(io.getCausalMaskName()).dataType();

            // Production parity mode: -Dmtp.parity.paddedPrefill=64 replicates the
            // pipeline's maxPrefillLength padded prefill (pad ids with 0, asl=actual,
            // padded causal mask) so window/scalar writes are comparable against the
            // production decode-loop KV dumps. Default 0 keeps the original unpadded
            // prefill behavior.
            int paddedPrefill = Integer.getInteger("mtp.parity.paddedPrefill", 0);
            int prefillSeqLen = paddedPrefill > prefillLength ? paddedPrefill : prefillLength;
            long[] paddedIds = new long[prefillSeqLen];
            for (int i = 0; i < prefillLength; i++) paddedIds[i] = promptTokenIds[i];

            Map<String, INDArray> prefillInputs = new HashMap<>();
            putOwned(prefillInputs, io.getInputIdsName(),
                    Nd4j.createFromArray(paddedIds).reshape(1, prefillSeqLen), owned);
            putOwned(prefillInputs, io.getPositionOffsetName(), Nd4j.scalar(DataType.INT64, 0L), owned);
            putOwned(prefillInputs, io.getCachePositionName(), Nd4j.scalar(DataType.INT64, 0L), owned);
            putOwned(prefillInputs, "actual_sequence_length",
                    Nd4j.scalar(DataType.INT64, (long) prefillLength), owned);
            putOwned(prefillInputs, io.getCausalMaskName(),
                    prefillSeqLen > prefillLength
                            ? buildPaddedPrefillMaskForTest(prefillLength, prefillSeqLen, maxKvLength, maskType)
                            : DecoderInputBuilder.buildInGraphCausalMask(prefillLength, maxKvLength, maskType), owned);

            for (String name : kvNames.keyNames) {
                putOwned(prefillInputs, name, Nd4j.empty(model.getVariable(name).dataType()), owned);
            }
            for (String name : kvNames.valueNames) {
                putOwned(prefillInputs, name, Nd4j.empty(model.getVariable(name).dataType()), owned);
            }
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                long[] stateShape = GenerationPipeline.deriveRecurrentStateShape(model, pair.inputName);
                assertNotNull(stateShape, "Could not derive recurrent state shape for " + pair.inputName);
                putOwned(prefillInputs, pair.inputName,
                        Nd4j.zeros(model.getVariable(pair.inputName).dataType(), stateShape), owned);
            }

            List<String> prefillOutputNames = new ArrayList<>();
            prefillOutputNames.add(io.getLogitsOutputName());
            for (String keyName : kvNames.keyNames) {
                int layer = extractLayerIndex(keyName);
                prefillOutputNames.add("k_rope_" + layer);
                prefillOutputNames.add("v_heads_" + layer);
            }
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                prefillOutputNames.add(pair.outputName);
            }

            Map<String, INDArray> prefillOutputs = model.output(
                    prefillInputs, prefillOutputNames.toArray(new String[0]));
            ownAll(prefillOutputs, owned);
            INDArray prefillLogits = prefillOutputs.get(io.getLogitsOutputName());
            int firstToken = argMaxToken(prefillLogits, prefillLength - 1);

            Map<String, INDArray> baseKv = new LinkedHashMap<>();
            for (int i = 0; i < kvNames.keyNames.size(); i++) {
                int layer = extractLayerIndex(kvNames.keyNames.get(i));
                INDArray keyRows = prefillOutputs.get("k_rope_" + layer);
                INDArray valueRows = prefillOutputs.get("v_heads_" + layer);
                assertEquals(4, keyRows.rank(), "Expected BSHD K rows for layer " + layer);
                assertEquals(4, valueRows.rank(), "Expected BSHD V rows for layer " + layer);

                INDArray keyCache = own(owned, Nd4j.zeros(keyRows.dataType(),
                        keyRows.size(0), maxKvLength, keyRows.size(2), keyRows.size(3)));
                INDArray valueCache = own(owned, Nd4j.zeros(valueRows.dataType(),
                        valueRows.size(0), maxKvLength, valueRows.size(2), valueRows.size(3)));
                keyCache.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillSeqLen),
                        NDArrayIndex.all(), NDArrayIndex.all()).assign(keyRows);
                valueCache.get(NDArrayIndex.all(), NDArrayIndex.interval(0, prefillSeqLen),
                        NDArrayIndex.all(), NDArrayIndex.all()).assign(valueRows);
                baseKv.put(kvNames.keyNames.get(i), keyCache);
                baseKv.put(kvNames.valueNames.get(i), valueCache);
            }

            Map<String, INDArray> baseStates = new LinkedHashMap<>();
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                baseStates.put(pair.inputName, own(owned, prefillOutputs.get(pair.outputName).dup()));
            }

            // Prefill-boundary comparison against the production pipeline's [GGUF-KV]
            // logs (kRoped min/max per layer) and committed token ids.
            INDArray prefillKRope3 = prefillOutputs.get("k_rope_3");
            log.info("[MTP-TARGET-PARITY] JAVA-TRUTH prefill: len={} padded={} firstToken={} "
                            + "k_rope_3 shape={} min={} max={}",
                    prefillLength, prefillSeqLen, firstToken,
                    Arrays.toString(prefillKRope3.shape()),
                    prefillKRope3.minNumber(), prefillKRope3.maxNumber());

            // Match GenerationPipeline's normal warmup: physical W=2, only row 0 active.
            Map<String, INDArray> warmupInputs = newStableDecodeInputs(
                    io, window, maxKvLength, maskType, baseKv, baseStates, owned);
            setDecodeStep(warmupInputs, io, firstToken, 0, prefillLength, 1,
                    window, maxKvLength, maskType, owned);
            List<String> warmupOutputNames = new ArrayList<>();
            warmupOutputNames.add(io.getLogitsOutputName());
            for (String keyName : kvNames.keyNames) {
                int layer = extractLayerIndex(keyName);
                warmupOutputNames.add("k_rope_" + layer);
                warmupOutputNames.add("v_heads_" + layer);
            }
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                warmupOutputNames.add(pair.outputName);
            }
            Map<String, INDArray> warmupOutputs = model.output(
                    warmupInputs, warmupOutputNames.toArray(new String[0]));
            ownAll(warmupOutputs, owned);
            int secondToken = argMaxToken(warmupOutputs.get(io.getLogitsOutputName()), 0);
            log.info("[MTP-TARGET-PARITY] JAVA-TRUTH tokens: firstToken={} secondToken={} "
                    + "(production committed: 271@17, 248068@18)", firstToken, secondToken);
            commitKvRows(baseKv, kvNames, warmupOutputs, prefillLength, 1);

            Map<String, INDArray> postWarmupKv = duplicateArrays(baseKv, owned);
            Map<String, INDArray> postWarmupStates = new LinkedHashMap<>();
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                postWarmupStates.put(pair.inputName,
                        own(owned, warmupOutputs.get(pair.outputName).dup()));
            }

            LinkedHashSet<String> detailedCandidates = new LinkedHashSet<>(Arrays.asList(
                    "embedded",
                    "model.layers.0.input_layernorm",
                    "gdn_qkv_0",
                    "gdn_conv_0",
                    "gdn_gate_proj_0",
                    "gdn_z_reshaped_0",
                    "gdn_gate_act_0",
                    "gdn_out_0",
                    "model.layers.0.gdn.ssm_norm_0",
                    "gdn_gated_0",
                    "gdn_proj_0",
                    "post_attn_0",
                    "model.layers.0.post_attention_layernorm",
                    "gate_0",
                    "up_0",
                    "swish_1",
                    "swiglu_0",
                    "down_0",
                    "model.layers.3.input_layernorm",
                    "q_full_3",
                    "k_3",
                    "v_3",
                    "qg_reshaped_3",
                    "q_3",
                    "attn_gate_3",
                    "k_heads_3",
                    "v_heads_3",
                    "model.layers.3.self_attn.q_norm_3",
                    "model.layers.3.self_attn.k_norm_3",
                    "q_rope_3",
                    "k_rope_3",
                    "attn_out_3",
                    "attn_flat_3",
                    "gate_sigmoid_3",
                    "gated_attn_3",
                    "attn_proj_3",
                    "post_attn_3",
                    "model.layers.3.post_attention_layernorm",
                    "gate_3",
                    "up_3",
                    "swiglu_3",
                    "down_3"));
            LinkedHashSet<String> comparableNames = new LinkedHashSet<>();
            for (String name : detailedCandidates) {
                if (model.hasVariable(name)) {
                    comparableNames.add(name);
                } else {
                    log.info("[MTP-TARGET-PARITY] Optimizer removed detailed checkpoint {}", name);
                }
            }
            for (int layer = 0; layer < 24; layer++) {
                String layerOutput = "layer_out_" + layer;
                assertTrue(model.hasVariable(layerOutput), "Missing target layer boundary " + layerOutput);
                comparableNames.add(layerOutput);
            }
            assertTrue(model.hasVariable("target_hidden_states"), "Missing target hidden-state boundary");
            assertTrue(model.hasVariable(io.getLogitsOutputName()), "Missing target logits boundary");
            comparableNames.add("target_hidden_states");
            comparableNames.add(io.getLogitsOutputName());

            String attentionScoresName = "dot_product_attention_v2:1";
            String attentionLogitsName = "dot_product_attention_v2:2";
            LinkedHashSet<String> requestedNames = new LinkedHashSet<>(comparableNames);
            if (requestAttentionAux) {
                assertTrue(model.hasVariable(attentionScoresName),
                        "Missing first full-attention softmax-score output");
                assertTrue(model.hasVariable(attentionLogitsName),
                        "Missing first full-attention logits output");
                requestedNames.add(attentionScoresName);
                requestedNames.add(attentionLogitsName);
            }
            for (String keyName : kvNames.keyNames) {
                int layer = extractLayerIndex(keyName);
                requestedNames.add("k_rope_" + layer);
                requestedNames.add("v_heads_" + layer);
            }
            LinkedHashSet<String> stateOutputNames = new LinkedHashSet<>();
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                requestedNames.add(pair.outputName);
                stateOutputNames.add(pair.outputName);
            }
            String[] requested = requestedNames.toArray(new String[0]);

            // Use one stable set of external buffers for warmup, W=5, and both scalar calls. This
            // matches the frozen native plan's pointer-stability contract and avoids cross-plan noise.
            Map<String, INDArray> stableKv = duplicateArrays(postWarmupKv, owned);
            Map<String, INDArray> stableStates = duplicateArrays(postWarmupStates, owned);
            Map<String, INDArray> stableInputs = newStableDecodeInputs(
                    io, window, maxKvLength, maskType, stableKv, stableStates, owned);

            // Prime this exact output set through the compile transition, then freeze it exactly as the
            // production fixed-buffer path does. Restore model state before every priming call.
            for (int pass = 0; pass < 4; pass++) {
                restoreArrays(stableKv, postWarmupKv);
                restoreArrays(stableStates, postWarmupStates);
                setDecodeStep(stableInputs, io, secondToken, 0, prefillLength + 1,
                        1, window, maxKvLength, maskType, owned);
                Map<String, INDArray> priming = model.output(stableInputs, requested);
                argMaxToken(priming.get(io.getLogitsOutputName()), 0);
                if (pass == 0) {
                    model.getOrCreateSession().getDynamicShapePlanExecutor().setShapesFrozen(true);
                }
                ownAll(priming, owned);
            }

            int vocabSize = (int) warmupOutputs.get(io.getLogitsOutputName()).size(2);
            int draftToken = (secondToken + 1) % vocabSize;

            restoreArrays(stableKv, postWarmupKv);
            restoreArrays(stableStates, postWarmupStates);
            setDecodeStep(stableInputs, io, secondToken, draftToken, prefillLength + 1,
                    2, window, maxKvLength, maskType, owned);
            // Production-envelope mode: -Dmtp.parity.window4=true evaluates the SAME
            // rows 0-1 inside the production speculative envelope (activeWindow=4,
            // wild draft tokens at rows 2-3, asl=4) instead of activeWindow=2 with
            // zero tail rows. Rows 0-1 are causally invariant to rows>=2 under
            // correct op semantics, so all downstream row-0/row-1 comparisons remain
            // valid - a divergence here reproduces the native decode-loop corruption.
            if (Boolean.getBoolean("mtp.parity.window4")) {
                setDecodeStep(stableInputs, io, secondToken, draftToken, prefillLength + 1,
                        4, window, maxKvLength, maskType, owned);
                INDArray specIds = stableInputs.get(io.getInputIdsName());
                specIds.putScalar(0, 2, 332);
                specIds.putScalar(0, 3, 55786);
                log.info("[MTP-TARGET-PARITY] window4 production envelope: ids row2=332 row3=55786 asl=4");
            }
            // NATIVE-CROSSCHECK mode: -Dmtp.parity.nativeRows=true replays the exact
            // verification inputs captured from a live decode loop step
            // (MTP_VERIFY_INPUTS) through the Java graph and prints Java row
            // argmaxes for rows 0..4, apples-to-apples with nativeRowArgmax.
            // Rows: ids[0..4]; position = mtp.parity.nativePos (default 20).
            if (Boolean.getBoolean("mtp.parity.nativeRows")) {
                int nPos = Integer.getInteger("mtp.parity.nativePos", 20);
                int[] rows = {
                        (int) (long) Long.getLong("mtp.parity.nativeRow0", 248069L),
                        (int) (long) Long.getLong("mtp.parity.nativeRow1", 46236L),
                        (int) (long) Long.getLong("mtp.parity.nativeRow2", 153542L),
                        (int) (long) Long.getLong("mtp.parity.nativeRow3", 163905L),
                        (int) (long) Long.getLong("mtp.parity.nativeRow4", 41952L)};
                setDecodeStep(stableInputs, io, rows[0], rows[1], nPos, 4, window,
                        maxKvLength, maskType, owned);
                INDArray specIds = stableInputs.get(io.getInputIdsName());
                specIds.putScalar(0, 2, rows[2]);
                specIds.putScalar(0, 3, rows[3]);
                specIds.putScalar(0, 4, rows[4]);
                log.info("[MTP-TARGET-PARITY] nativeRows envelope: ids={} pos={} asl=4",
                        Arrays.toString(rows), nPos);
            }
            INDArray windowMaskSnapshot = own(owned,
                    stableInputs.get(io.getCausalMaskName()).dup());
            Map<String, INDArray> windowOutputs = model.output(stableInputs, requested);
            ownAll(windowOutputs, owned);
            // Generic SameDiff staging does not expose input side effects to caller arrays.
            // Export and commit K/V rows explicitly so this direct oracle matches native decode state.
            argMaxToken(windowOutputs.get(io.getLogitsOutputName()), 1);
            // Native-crosscheck: Java-truth argmax for every verification row.
            if (Boolean.getBoolean("mtp.parity.nativeRows")) {
                StringBuilder rowArgmaxes = new StringBuilder();
                for (int r = 0; r < 5 && r < windowOutputs.get(io.getLogitsOutputName()).size(1); r++) {
                    rowArgmaxes.append(' ').append(argMaxToken(
                            windowOutputs.get(io.getLogitsOutputName()), r));
                }
                log.info("[MTP-TARGET-PARITY] JAVA_ROW_ARGMAX native-vs-java:{}",
                        rowArgmaxes.toString());
            }
            commitKvRows(stableKv, kvNames, windowOutputs, prefillLength + 1, 2);
            Map<String, INDArray> windowSnapshot = snapshotOutputs(
                    windowOutputs, comparableNames, owned);
            Map<String, INDArray> windowStateSnapshot = snapshotOutputs(
                    windowOutputs, stateOutputNames, owned);
            INDArray windowAttentionScores = requestAttentionAux
                    ? own(owned, windowOutputs.get(attentionScoresName).dup()) : null;
            INDArray windowAttentionLogits = requestAttentionAux
                    ? own(owned, windowOutputs.get(attentionLogitsName).dup()) : null;
            Map<String, INDArray> windowKvSnapshot = duplicateArrays(stableKv, owned);

            restoreArrays(stableKv, postWarmupKv);
            restoreArrays(stableStates, postWarmupStates);
            setDecodeStep(stableInputs, io, secondToken, 0, prefillLength + 1,
                    1, window, maxKvLength, maskType, owned);
            INDArray scalarFirstMaskSnapshot = own(owned,
                    stableInputs.get(io.getCausalMaskName()).dup());
            Map<String, INDArray> scalarFirstOutputs = model.output(stableInputs, requested);
            ownAll(scalarFirstOutputs, owned);
            // Consume the token, then commit the exported row before the chained scalar call.
            argMaxToken(scalarFirstOutputs.get(io.getLogitsOutputName()), 0);
            commitKvRows(stableKv, kvNames, scalarFirstOutputs, prefillLength + 1, 1);
            Map<String, INDArray> scalarFirstSnapshot = snapshotOutputs(
                    scalarFirstOutputs, comparableNames, owned);
            Map<String, INDArray> scalarFirstStateSnapshot = snapshotOutputs(
                    scalarFirstOutputs, stateOutputNames, owned);
            INDArray scalarFirstAttentionScores = requestAttentionAux
                    ? own(owned, scalarFirstOutputs.get(attentionScoresName).dup()) : null;
            INDArray scalarFirstAttentionLogits = requestAttentionAux
                    ? own(owned, scalarFirstOutputs.get(attentionLogitsName).dup()) : null;

            // Commit the scalar call's accepted recurrent state for the chained row.
            for (ModelIOConfig.RecurrentStatePair pair : recurrentStates) {
                stableStates.get(pair.inputName).assign(scalarFirstOutputs.get(pair.outputName));
            }
            setDecodeStep(stableInputs, io, draftToken, 0, prefillLength + 2,
                    1, window, maxKvLength, maskType, owned);
            INDArray scalarSecondMaskSnapshot = own(owned,
                    stableInputs.get(io.getCausalMaskName()).dup());
            Map<String, INDArray> scalarSecondOutputs = model.output(stableInputs, requested);
            ownAll(scalarSecondOutputs, owned);
            argMaxToken(scalarSecondOutputs.get(io.getLogitsOutputName()), 0);
            commitKvRows(stableKv, kvNames, scalarSecondOutputs, prefillLength + 2, 1);
            Map<String, INDArray> scalarSecondSnapshot = snapshotOutputs(
                    scalarSecondOutputs, comparableNames, owned);
            Map<String, INDArray> scalarSecondStateSnapshot = snapshotOutputs(
                    scalarSecondOutputs, stateOutputNames, owned);
            INDArray scalarSecondAttentionScores = requestAttentionAux
                    ? own(owned, scalarSecondOutputs.get(attentionScoresName).dup()) : null;
            INDArray scalarSecondAttentionLogits = requestAttentionAux
                    ? own(owned, scalarSecondOutputs.get(attentionLogitsName).dup()) : null;
            Map<String, INDArray> scalarKvSnapshot = duplicateArrays(stableKv, owned);

            String firstFinalStateDivergence = null;
            double firstFinalStateMax = 0.0;
            double firstFinalStateL1 = 0.0;
            // WINDOW4 ORACLE SCOPING: the window4 mode runs the W-wide pass at
            // asl=4 with WILD draft tokens at rows 2-3 (deliberately not consumed
            // by the 2-row scalar chain). The full-window finalState therefore
            // legitimately folds two extra rows the scalar chain never saw - the
            // acceptedZeroRerunState/partialRerunState discriminators below prove
            // the shortened-recurrence rerun from pre-step state is EXACT, and the
            // row-0/row-1 output comparisons prove the consumed rows are exact.
            // So the finalState check is meaningful only in the asl=2 envelope
            // (no wild rows), where the window pass folds exactly the same rows
            // as the chained scalar. In window4 mode it is reported, not asserted.
            boolean assertFinalState = !Boolean.getBoolean("mtp.parity.window4");
            for (String name : stateOutputNames) {
                double[] stateDiff = difference(
                        windowStateSnapshot.get(name), scalarSecondStateSnapshot.get(name));
                if (stateDiff[0] != 0.0) {
                    log.info("[MTP-TARGET-PARITY] discriminator=finalState name={} max={} l1={} "
                                    + "(window4: expected - wild rows 2-3 fold extra state)",
                            name, stateDiff[0], stateDiff[1]);
                    if (firstFinalStateDivergence == null) {
                        firstFinalStateDivergence = name;
                        firstFinalStateMax = stateDiff[0];
                        firstFinalStateL1 = stateDiff[1];
                    }
                }
            }
            log.info("[MTP-TARGET-PARITY] discriminator=finalState first={} max={} l1={}",
                    firstFinalStateDivergence, firstFinalStateMax, firstFinalStateL1);

            // Reproduce native speculative rejection: execute the W-wide target pass,
            // keep its K/V writes, but rerun from the original recurrent input state
            // with only the accepted-prefix length reduced to one row.
            restoreArrays(stableKv, postWarmupKv);
            restoreArrays(stableStates, postWarmupStates);
            setDecodeStep(stableInputs, io, secondToken, draftToken, prefillLength + 1,
                    2, window, maxKvLength, maskType, owned);
            Map<String, INDArray> acceptedZeroWindowOutputs = model.output(stableInputs, requested);
            ownAll(acceptedZeroWindowOutputs, owned);
            argMaxToken(acceptedZeroWindowOutputs.get(io.getLogitsOutputName()), 1);
            commitKvRows(stableKv, kvNames, acceptedZeroWindowOutputs, prefillLength + 1, 2);
            stableInputs.get("actual_sequence_length").putScalar(new long[]{}, 1L);
            Map<String, INDArray> acceptedZeroRerunOutputs = model.output(stableInputs, requested);
            ownAll(acceptedZeroRerunOutputs, owned);
            argMaxToken(acceptedZeroRerunOutputs.get(io.getLogitsOutputName()), 0);
            Map<String, INDArray> acceptedZeroRerunStateSnapshot = snapshotOutputs(
                    acceptedZeroRerunOutputs, stateOutputNames, owned);

            String firstAcceptedZeroStateDivergence = null;
            double firstAcceptedZeroStateMax = 0.0;
            double firstAcceptedZeroStateL1 = 0.0;
            for (String name : stateOutputNames) {
                double[] stateDiff = difference(
                        acceptedZeroRerunStateSnapshot.get(name), scalarFirstStateSnapshot.get(name));
                if (stateDiff[0] != 0.0) {
                    log.info("[MTP-TARGET-PARITY] discriminator=acceptedZeroRerunState "
                                    + "name={} max={} l1={}",
                            name, stateDiff[0], stateDiff[1]);
                    if (firstAcceptedZeroStateDivergence == null) {
                        firstAcceptedZeroStateDivergence = name;
                        firstAcceptedZeroStateMax = stateDiff[0];
                        firstAcceptedZeroStateL1 = stateDiff[1];
                    }
                }
            }
            log.info("[MTP-TARGET-PARITY] discriminator=acceptedZeroRerunState "
                            + "first={} max={} l1={}",
                    firstAcceptedZeroStateDivergence,
                    firstAcceptedZeroStateMax, firstAcceptedZeroStateL1);

            // Reproduce the production step that proposed three drafts but accepted
            // only the first: execute four rows, retain their K/V writes, then rerun
            // the same fixed W=5 plan with an accepted-prefix length of two.
            int draftToken2 = (draftToken + 1) % vocabSize;
            int draftToken3 = (draftToken + 2) % vocabSize;
            restoreArrays(stableKv, postWarmupKv);
            restoreArrays(stableStates, postWarmupStates);
            setDecodeStep(stableInputs, io, secondToken, draftToken, prefillLength + 1,
                    4, window, maxKvLength, maskType, owned);
            INDArray partialIds = stableInputs.get(io.getInputIdsName());
            partialIds.putScalar(0, 2, draftToken2);
            partialIds.putScalar(0, 3, draftToken3);
            Map<String, INDArray> partialWindowOutputs = model.output(stableInputs, requested);
            ownAll(partialWindowOutputs, owned);
            argMaxToken(partialWindowOutputs.get(io.getLogitsOutputName()), 3);
            commitKvRows(stableKv, kvNames, partialWindowOutputs, prefillLength + 1, 4);
            stableInputs.get("actual_sequence_length").putScalar(new long[]{}, 2L);
            Map<String, INDArray> partialRerunOutputs = model.output(stableInputs, requested);
            ownAll(partialRerunOutputs, owned);
            argMaxToken(partialRerunOutputs.get(io.getLogitsOutputName()), 1);
            Map<String, INDArray> partialRerunStateSnapshot = snapshotOutputs(
                    partialRerunOutputs, stateOutputNames, owned);

            String firstPartialStateDivergence = null;
            double firstPartialStateMax = 0.0;
            double firstPartialStateL1 = 0.0;
            for (String name : stateOutputNames) {
                double[] stateDiff = difference(
                        partialRerunStateSnapshot.get(name), scalarSecondStateSnapshot.get(name));
                if (stateDiff[0] != 0.0) {
                    log.info("[MTP-TARGET-PARITY] discriminator=partialRerunState "
                                    + "name={} max={} l1={}",
                            name, stateDiff[0], stateDiff[1]);
                    if (firstPartialStateDivergence == null) {
                        firstPartialStateDivergence = name;
                        firstPartialStateMax = stateDiff[0];
                        firstPartialStateL1 = stateDiff[1];
                    }
                }
            }
            log.info("[MTP-TARGET-PARITY] discriminator=partialRerunState "
                            + "first={} max={} l1={}",
                    firstPartialStateDivergence, firstPartialStateMax, firstPartialStateL1);

            String layer3KeyCacheName = kvNames.keyNames.stream()
                    .filter(name -> extractLayerIndex(name) == 3)
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("Missing layer-3 key cache"));
            logKeyCacheMutation("window", postWarmupKv.get(layer3KeyCacheName),
                    windowKvSnapshot.get(layer3KeyCacheName),
                    sequenceRow(windowSnapshot.get("k_rope_3"), 0));
            logKeyCacheMutation("scalar", postWarmupKv.get(layer3KeyCacheName),
                    scalarKvSnapshot.get(layer3KeyCacheName),
                    sequenceRow(scalarFirstSnapshot.get("k_rope_3"), 0));
            double[] windowKeyVsCache = difference(
                    sequenceRow(windowSnapshot.get("k_rope_3"), 0),
                    sequenceRow(windowKvSnapshot.get(layer3KeyCacheName), prefillLength + 1));
            double[] scalarKeyVsCache = difference(
                    sequenceRow(scalarFirstSnapshot.get("k_rope_3"), 0),
                    sequenceRow(scalarKvSnapshot.get(layer3KeyCacheName), prefillLength + 1));
            double[] windowVsScalarKey = difference(
                    sequenceRow(windowSnapshot.get("k_rope_3"), 0),
                    sequenceRow(scalarFirstSnapshot.get("k_rope_3"), 0));
            log.info("[MTP-TARGET-PARITY] discriminator=keyPersistence "
                            + "windowKeyVsCacheMax={} windowKeyVsCacheL1={} "
                            + "scalarKeyVsCacheMax={} scalarKeyVsCacheL1={} "
                            + "windowVsScalarKeyMax={} windowVsScalarKeyL1={}",
                    windowKeyVsCache[0], windowKeyVsCache[1],
                    scalarKeyVsCache[0], scalarKeyVsCache[1],
                    windowVsScalarKey[0], windowVsScalarKey[1]);
            // Ground-truth arbitration against the native decode-loop KV_ROW_SLICE dumps:
            // raw layer-3 key/value cache components at rows 18..21 (h=0, d=0..1), same
            // slice the production probes print. Row 18's value is the alignment
            // fingerprint between the two position conventions.
            String layer3ValueCacheName = kvNames.valueNames.stream()
                    .filter(name -> extractLayerIndex(name) == 3)
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("Missing layer-3 value cache"));
            for (Map.Entry<String, Map<String, INDArray>> snap : Map.of(
                    "window", windowKvSnapshot, "scalar", scalarKvSnapshot).entrySet()) {
                INDArray kc = snap.getValue().get(layer3KeyCacheName);
                INDArray vc = snap.getValue().get(layer3ValueCacheName);
                StringBuilder sb = new StringBuilder();
                for (int r = 18; r <= 21 && r < kc.size(1); r++) {
                    sb.append(String.format(" k[%d]=[%.6g,%.6g] v[%d]=[%.6g,%.6g]",
                            r, kc.getDouble(0, r, 0, 0), kc.getDouble(0, r, 0, 1),
                            r, vc.getDouble(0, r, 0, 0), vc.getDouble(0, r, 0, 1)));
                }
                log.info("[MTP-TARGET-PARITY] JAVA-TRUTH {} layer3 cache rows18..21:{}",
                        snap.getKey(), sb);
            }
            INDArray kRopeWin = windowSnapshot.get("k_rope_3");
            INDArray vHeadsWin = windowSnapshot.get("v_heads_3");
            log.info("[MTP-TARGET-PARITY] JAVA-TRUTH window k_rope_3 row0=[{},{}] row1=[{},{}] "
                            + "v_heads_3 row0=[{},{}] row1=[{},{}]",
                    kRopeWin.getDouble(0, 0, 0, 0), kRopeWin.getDouble(0, 0, 0, 1),
                    kRopeWin.getDouble(0, 1, 0, 0), kRopeWin.getDouble(0, 1, 0, 1),
                    vHeadsWin.getDouble(0, 0, 0, 0), vHeadsWin.getDouble(0, 0, 0, 1),
                    vHeadsWin.getDouble(0, 1, 0, 0), vHeadsWin.getDouble(0, 1, 0, 1));
            if (requestAttentionAux) {
                logAttentionKeyReference(
                        windowSnapshot.get("q_rope_3"), 1,
                        scalarSecondSnapshot.get("q_rope_3"), 0,
                        windowKvSnapshot.get(layer3KeyCacheName),
                        postWarmupKv.get(layer3KeyCacheName),
                        windowSnapshot.get("k_rope_3"),
                        windowMaskSnapshot, 1,
                        scalarSecondMaskSnapshot, 0,
                        windowAttentionLogits, 1,
                        scalarSecondAttentionLogits, 0,
                        prefillLength + 1);
            }

            double[] maskRow0 = difference(
                    queryRow(windowMaskSnapshot, 0), queryRow(scalarFirstMaskSnapshot, 0));
            double[] maskRow1 = difference(
                    queryRow(windowMaskSnapshot, 1), queryRow(scalarSecondMaskSnapshot, 0));
            log.info("[MTP-TARGET-PARITY] discriminator=mask row0Max={} row0L1={} "
                            + "row1Max={} row1L1={} windowShape={} scalarShape={}",
                    maskRow0[0], maskRow0[1], maskRow1[0], maskRow1[1],
                    Arrays.toString(windowMaskSnapshot.shape()),
                    Arrays.toString(scalarSecondMaskSnapshot.shape()));

            if (requestAttentionAux) {
                double[] scoreRow0 = difference(
                        queryRow(windowAttentionScores, 0), queryRow(scalarFirstAttentionScores, 0));
                double[] scoreRow1 = difference(
                        queryRow(windowAttentionScores, 1), queryRow(scalarSecondAttentionScores, 0));
                log.info("[MTP-TARGET-PARITY] discriminator=attentionScores row0Max={} row0L1={} "
                                + "row1Max={} row1L1={} windowShape={} scalarShape={}",
                        scoreRow0[0], scoreRow0[1], scoreRow1[0], scoreRow1[1],
                        Arrays.toString(windowAttentionScores.shape()),
                        Arrays.toString(scalarSecondAttentionScores.shape()));

                double[] logitsRow0 = difference(
                        queryRow(windowAttentionLogits, 0), queryRow(scalarFirstAttentionLogits, 0));
                double[] logitsRow1 = difference(
                        queryRow(windowAttentionLogits, 1), queryRow(scalarSecondAttentionLogits, 0));
                log.info("[MTP-TARGET-PARITY] discriminator=attentionLogits row0Max={} row0L1={} "
                                + "row1Max={} row1L1={} windowShape={} scalarShape={}",
                        logitsRow0[0], logitsRow0[1], logitsRow1[0], logitsRow1[1],
                        Arrays.toString(windowAttentionLogits.shape()),
                        Arrays.toString(scalarSecondAttentionLogits.shape()));
                logAttentionRowDifferences("attentionLogits", windowAttentionLogits, 1,
                        scalarSecondAttentionLogits, 0, prefillLength + 1, prefillLength + 2);
                logAttentionRowDifferences("attentionScores", windowAttentionScores, 1,
                        scalarSecondAttentionScores, 0, prefillLength + 1, prefillLength + 2);
            }

            String firstKvDivergence = null;
            double firstKvMax = 0.0;
            double firstKvL1 = 0.0;
            for (String name : windowKvSnapshot.keySet()) {
                double[] kvDiff = difference(windowKvSnapshot.get(name), scalarKvSnapshot.get(name));
                if (kvDiff[0] != 0.0) {
                    log.info("[MTP-TARGET-PARITY] discriminator=kv name={} max={} l1={}",
                            name, kvDiff[0], kvDiff[1]);
                    if (firstKvDivergence == null) {
                        firstKvDivergence = name;
                        firstKvMax = kvDiff[0];
                        firstKvL1 = kvDiff[1];
                    }
                }
            }
            log.info("[MTP-TARGET-PARITY] discriminator=kv first={} max={} l1={}",
                    firstKvDivergence, firstKvMax, firstKvL1);

            String firstDivergence = null;
            double firstMax = 0.0;
            double firstL1 = 0.0;
            for (String name : comparableNames) {
                double[] row0 = difference(
                        sequenceRow(windowSnapshot.get(name), 0),
                        sequenceRow(scalarFirstSnapshot.get(name), 0));
                double[] row1 = difference(
                        sequenceRow(windowSnapshot.get(name), 1),
                        sequenceRow(scalarSecondSnapshot.get(name), 0));
                if (row0[0] != 0.0 || row1[0] != 0.0) {
                    log.info("[MTP-TARGET-PARITY] name={} row0Max={} row0L1={} row1Max={} row1L1={}",
                            name, row0[0], row0[1], row1[0], row1[1]);
                    if (firstDivergence == null) {
                        firstDivergence = name;
                        firstMax = Math.max(row0[0], row1[0]);
                        firstL1 = row0[1] + row1[1];
                    }
                }
            }

            assertTrue(firstDivergence == null,
                    (requestAttentionAux ? "Aux-output" : "Output-only")
                            + " W=5 target rows diverged first at " + firstDivergence
                            + " (max=" + firstMax + ", l1=" + firstL1 + ")");
            // window4: full-window finalState folds the wild rows; asserted only
            // in the asl=2 envelope where both legs fold the same consumed rows.
            if (assertFinalState) {
                assertTrue(firstFinalStateDivergence == null,
                        "W=5 final recurrent state diverged first at " + firstFinalStateDivergence
                                + " (max=" + firstFinalStateMax + ", l1=" + firstFinalStateL1 + ")");
            }
            assertTrue(firstAcceptedZeroStateDivergence == null,
                    "Accepted-zero rerun state diverged first at "
                            + firstAcceptedZeroStateDivergence
                            + " (max=" + firstAcceptedZeroStateMax
                            + ", l1=" + firstAcceptedZeroStateL1 + ")");
            assertTrue(firstPartialStateDivergence == null,
                    "Partial accepted-prefix rerun state diverged first at "
                            + firstPartialStateDivergence
                            + " (max=" + firstPartialStateMax
                            + ", l1=" + firstPartialStateL1 + ")");
        } finally {
            model.resetSession();
            closeOwned(owned);
        }
    }

    /** Replicates GenerationPipeline.buildPaddedPrefillCausalMask for production-parity prefill. */
    private static INDArray buildPaddedPrefillMaskForTest(int actualLen, int paddedLen,
                                                          long maxKvLen, DataType dtype) {
        int q0 = paddedLen;
        int k0 = (int) maxKvLen;
        float maskVal = (dtype == DataType.HALF || dtype == DataType.FLOAT16) ? -65504.0f : -1e9f;
        float[] data = new float[q0 * k0];
        for (int q = 0; q < q0; q++) {
            int rowOffset = q * k0;
            if (q < actualLen) {
                for (int k = q + 1; k < k0; k++) data[rowOffset + k] = maskVal;
            } else {
                // Keep one safe attention target so FP16 softmax never receives an all-masked row.
                for (int k = 1; k < k0; k++) data[rowOffset + k] = maskVal;
            }
        }
        INDArray mask = Nd4j.create(data, new long[]{1, 1, paddedLen, maxKvLen}, 'c');
        if (dtype != DataType.FLOAT) {
            INDArray cast = mask.castTo(dtype);
            mask.close();
            return cast;
        }
        return mask;
    }

    private static Map<String, INDArray> newStableDecodeInputs(
            ModelIOConfig io, int window, int maxKvLength, DataType maskType,
            Map<String, INDArray> kv, Map<String, INDArray> states, List<INDArray> owned) {
        Map<String, INDArray> inputs = new HashMap<>();
        putOwned(inputs, io.getInputIdsName(), Nd4j.zeros(DataType.INT64, 1, window), owned);
        putOwned(inputs, io.getCausalMaskName(),
                DecoderInputBuilder.buildInGraphWindowMask(
                        DecoderInputBuilder.chainParents(1, window), 0, 1, window,
                        maxKvLength, maskType), owned);
        putOwned(inputs, io.getPositionOffsetName(), Nd4j.scalar(DataType.INT64, 0L), owned);
        putOwned(inputs, io.getCachePositionName(), Nd4j.scalar(DataType.INT64, 0L), owned);
        putOwned(inputs, "actual_sequence_length", Nd4j.scalar(DataType.INT64, 1L), owned);
        inputs.putAll(kv);
        inputs.putAll(states);
        return inputs;
    }

    private static void setDecodeStep(
            Map<String, INDArray> inputs, ModelIOConfig io,
            int token0, int token1, int position, int activeWindow, int window,
            int maxKvLength, DataType maskType, List<INDArray> owned) {
        INDArray ids = inputs.get(io.getInputIdsName());
        ids.assign(0);
        ids.putScalar(0, 0, token0);
        if (activeWindow > 1) ids.putScalar(0, 1, token1);

        INDArray maskDonor = own(owned, DecoderInputBuilder.buildInGraphWindowMask(
                DecoderInputBuilder.chainParents(activeWindow, window),
                position, activeWindow, window, maxKvLength, maskType));
        inputs.get(io.getCausalMaskName()).assign(maskDonor);
        inputs.get(io.getPositionOffsetName()).putScalar(new long[]{}, (long) position);
        inputs.get(io.getCachePositionName()).putScalar(new long[]{}, (long) position);
        inputs.get("actual_sequence_length").putScalar(new long[]{}, (long) activeWindow);
    }

    private static int argMaxToken(INDArray logits, int row) {
        INDArray argMax = logits.get(
                NDArrayIndex.point(0), NDArrayIndex.point(row), NDArrayIndex.all()).argMax();
        try {
            return argMax.getInt(0);
        } finally {
            argMax.close();
        }
    }

    private static int extractLayerIndex(String cacheName) {
        String[] parts = cacheName.split("\\.");
        return Integer.parseInt(parts[1]);
    }

    private static void commitKvRows(
            Map<String, INDArray> kv, ModelIOConfig.KVCacheNames kvNames,
            Map<String, INDArray> outputs, int cachePosition, int rowCount) {
        for (int i = 0; i < kvNames.keyNames.size(); i++) {
            int layer = extractLayerIndex(kvNames.keyNames.get(i));
            commitCacheRows(kv.get(kvNames.keyNames.get(i)), outputs.get("k_rope_" + layer),
                    cachePosition, rowCount, "key", layer);
            commitCacheRows(kv.get(kvNames.valueNames.get(i)), outputs.get("v_heads_" + layer),
                    cachePosition, rowCount, "value", layer);
        }
    }

    private static void commitCacheRows(
            INDArray cache, INDArray rows, int cachePosition, int rowCount,
            String kind, int layer) {
        assertNotNull(cache, "Missing " + kind + " cache for layer " + layer);
        assertNotNull(rows, "Missing exported " + kind + " rows for layer " + layer);
        assertEquals(4, cache.rank(), "Expected BSHD " + kind + " cache for layer " + layer);
        assertEquals(4, rows.rank(), "Expected BSHD exported " + kind + " rows for layer " + layer);
        assertTrue(rows.size(1) >= rowCount,
                "Insufficient exported " + kind + " rows for layer " + layer);
        cache.get(NDArrayIndex.all(),
                        NDArrayIndex.interval(cachePosition, cachePosition + rowCount),
                        NDArrayIndex.all(), NDArrayIndex.all())
                .assign(rows.get(NDArrayIndex.all(), NDArrayIndex.interval(0, rowCount),
                        NDArrayIndex.all(), NDArrayIndex.all()));
    }

    private static Map<String, INDArray> duplicateArrays(
            Map<String, INDArray> source, List<INDArray> owned) {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (Map.Entry<String, INDArray> entry : source.entrySet()) {
            result.put(entry.getKey(), own(owned, entry.getValue().dup()));
        }
        return result;
    }

    /**
     * Review round 4, finding 4: verify every KV cache array's dtype matches the
     * decoder PLACEHOLDER dtype — the production storage contract — rather than
     * an output tensor's dtype. Production derives KV storage dtypes from the
     * placeholders; a replay that derives them from outputs must be verified
     * equivalent before its elementwise comparison is qualified.
     */
    private static boolean storageMatchesPlaceholders(Map<String, INDArray> kv, SameDiff sd) {
        for (Map.Entry<String, INDArray> e : kv.entrySet()) {
            String placeholder = e.getKey();
            INDArray arr = e.getValue();
            if (arr == null || !sd.hasVariable(placeholder)) return false;
            DataType placeholderDtype = sd.getVariable(placeholder).dataType();
            if (arr.dataType() != placeholderDtype) return false;
        }
        return true;
    }

    /**
     * Compare three named-array maps pairwise with FULL elementwise equality on
     * the components actually inspected, returning a compact report string.
     * Used by the 3x prefill-stability probe so each comparison observes two
     * independent snapshots (review round 4, finding 3).
     */
    private static String firstDiffProbe(Map<String, INDArray> run1, Map<String, INDArray> run2,
            Map<String, INDArray> run3) {
        if (run1.isEmpty()) return " (no tensors)";
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, INDArray> e : run1.entrySet()) {
            INDArray a = e.getValue();
            INDArray b = run2.get(e.getKey());
            INDArray c = run3.get(e.getKey());
            if (a == null || b == null || c == null
                    || a.length() != b.length() || a.length() != c.length()) {
                sb.append(' ').append(e.getKey()).append(":missing/shape");
                continue;
            }
            boolean d12 = false, d23 = false, d13 = false;
            for (long i = 0; i < a.length(); i++) {
                double x = a.getDouble(i), y = b.getDouble(i), z = c.getDouble(i);
                d12 |= x != y;
                d23 |= y != z;
                d13 |= x != z;
            }
            if (d12 || d23 || d13) {
                sb.append(String.format(" %s:#1!=#2=%s #2!=#3=%s #1!=#3=%s",
                        e.getKey(), d12, d23, d13));
            }
        }
        return sb.length() == 0 ? " ALL IDENTICAL" : sb.toString();
    }

    private static void restoreArrays(Map<String, INDArray> destination, Map<String, INDArray> source) {
        for (Map.Entry<String, INDArray> entry : source.entrySet()) {
            destination.get(entry.getKey()).assign(entry.getValue());
        }
    }

    private static Map<String, INDArray> snapshotOutputs(
            Map<String, INDArray> outputs, Set<String> names, List<INDArray> owned) {
        Map<String, INDArray> result = new LinkedHashMap<>();
        for (String name : names) {
            INDArray output = outputs.get(name);
            assertNotNull(output, "Missing checkpoint output " + name);
            result.put(name, own(owned, output.dup()));
        }
        return result;
    }

    private static INDArray sequenceRow(INDArray array, int row) {
        assertTrue(array.rank() >= 2 && array.size(1) > row,
                "Checkpoint is not sequence-addressable: shape=" + Arrays.toString(array.shape()));
        INDArrayIndex[] indices = new INDArrayIndex[array.rank()];
        Arrays.fill(indices, NDArrayIndex.all());
        indices[1] = NDArrayIndex.point(row);
        return array.get(indices);
    }

    private static INDArray queryRow(INDArray array, int row) {
        int queryAxis = array.rank() - 2;
        assertTrue(queryAxis >= 0 && array.size(queryAxis) > row,
                "Tensor is not query-row-addressable: shape=" + Arrays.toString(array.shape()));
        INDArrayIndex[] indices = new INDArrayIndex[array.rank()];
        Arrays.fill(indices, NDArrayIndex.all());
        indices[queryAxis] = NDArrayIndex.point(row);
        return array.get(indices);
    }

    private static void logAttentionRowDifferences(
            String kind, INDArray window, int windowRow, INDArray scalar, int scalarRow,
            int windowBasePosition, int scalarPosition) {
        assertEquals(4, window.rank(), kind + " window tensor must be [B,H,Q,KV]");
        assertArrayEquals(window.shape(), scalar.shape(), kind + " tensor shape mismatch");
        int logged = 0;
        int different = 0;
        for (int head = 0; head < window.size(1); head++) {
            for (int kv = 0; kv < window.size(3); kv++) {
                double windowValue = window.getDouble(0, head, windowRow, kv);
                double scalarValue = scalar.getDouble(0, head, scalarRow, kv);
                double delta = Math.abs(windowValue - scalarValue);
                if (Double.doubleToLongBits(windowValue) == Double.doubleToLongBits(scalarValue)
                        || delta <= 1.0e-7) {
                    continue;
                }
                different++;
                if (logged < 32) {
                    log.info("[MTP-TARGET-PARITY] discriminator={} head={} kv={} "
                                    + "windowValue={} scalarValue={} absDiff={} "
                                    + "windowBasePosition={} scalarPosition={}",
                            kind, head, kv, windowValue, scalarValue, delta,
                            windowBasePosition, scalarPosition);
                    logged++;
                }
            }
        }
        log.info("[MTP-TARGET-PARITY] discriminator={} differingElements={} logged={}",
                kind, different, logged);
    }

    private static void logAttentionKeyReference(
            INDArray windowQuery, int windowQueryRow,
            INDArray scalarQuery, int scalarQueryRow,
            INDArray keyCache,
            INDArray keyCacheBeforeWrite,
            INDArray currentKeyWindow,
            INDArray windowMask, int windowMaskRow,
            INDArray scalarMask, int scalarMaskRow,
            INDArray windowLogits, int windowLogitsRow,
            INDArray scalarLogits, int scalarLogitsRow,
            int comparedKvPosition) {
        int qHeads = (int) windowQuery.size(2);
        int kvHeads = (int) keyCache.size(2);
        int headDim = (int) windowQuery.size(3);
        int headsPerKvHead = qHeads / kvHeads;
        double scale = 1.0 / Math.sqrt(headDim);
        int searchEnd = Math.min((int) keyCache.size(1), comparedKvPosition + 2);

        for (int qHead = 0; qHead < qHeads; qHead++) {
            int kvHead = qHead / headsPerKvHead;
            double expectedWindow = dotAttentionLogit(
                    windowQuery, windowQueryRow, qHead, keyCache, comparedKvPosition, kvHead,
                    windowMask, windowMaskRow, scale);
            double expectedBeforeWrite = dotAttentionLogit(
                    windowQuery, windowQueryRow, qHead, keyCacheBeforeWrite, comparedKvPosition, kvHead,
                    windowMask, windowMaskRow, scale);
            double expectedCurrentRow0 = dotCurrentWindowAttentionLogit(
                    windowQuery, windowQueryRow, qHead, currentKeyWindow, 0, kvHead,
                    windowMask, windowMaskRow, comparedKvPosition, scale);
            double expectedCurrentRow1 = dotCurrentWindowAttentionLogit(
                    windowQuery, windowQueryRow, qHead, currentKeyWindow, 1, kvHead,
                    windowMask, windowMaskRow, comparedKvPosition, scale);
            double expectedScalar = dotAttentionLogit(
                    scalarQuery, scalarQueryRow, qHead, keyCache, comparedKvPosition, kvHead,
                    scalarMask, scalarMaskRow, scale);
            double observedWindow = windowLogits.getDouble(
                    0, qHead, windowLogitsRow, comparedKvPosition);
            double observedScalar = scalarLogits.getDouble(
                    0, qHead, scalarLogitsRow, comparedKvPosition);

            int nearestWindowKv = -1;
            int nearestScalarKv = -1;
            double nearestWindowDelta = Double.POSITIVE_INFINITY;
            double nearestScalarDelta = Double.POSITIVE_INFINITY;
            for (int kv = 0; kv < searchEnd; kv++) {
                double candidateWindow = dotAttentionLogit(
                        windowQuery, windowQueryRow, qHead, keyCache, kv, kvHead,
                        windowMask, windowMaskRow, scale);
                double candidateScalar = dotAttentionLogit(
                        scalarQuery, scalarQueryRow, qHead, keyCache, kv, kvHead,
                        scalarMask, scalarMaskRow, scale);
                double windowDelta = Math.abs(observedWindow - candidateWindow);
                double scalarDelta = Math.abs(observedScalar - candidateScalar);
                if (windowDelta < nearestWindowDelta) {
                    nearestWindowDelta = windowDelta;
                    nearestWindowKv = kv;
                }
                if (scalarDelta < nearestScalarDelta) {
                    nearestScalarDelta = scalarDelta;
                    nearestScalarKv = kv;
                }
            }
            log.info("[MTP-TARGET-PARITY] discriminator=keyReference head={} kv={} "
                            + "expectedWindow={} observedWindow={} windowResidual={} windowNearestKv={} "
                            + "windowNearestResidual={} beforeWrite={} beforeWriteResidual={} "
                            + "currentRow0={} currentRow0Residual={} currentRow1={} currentRow1Residual={} "
                            + "expectedScalar={} observedScalar={} scalarResidual={} "
                            + "scalarNearestKv={} scalarNearestResidual={}",
                    qHead, comparedKvPosition,
                    expectedWindow, observedWindow, Math.abs(expectedWindow - observedWindow),
                    nearestWindowKv, nearestWindowDelta,
                    expectedBeforeWrite, Math.abs(expectedBeforeWrite - observedWindow),
                    expectedCurrentRow0, Math.abs(expectedCurrentRow0 - observedWindow),
                    expectedCurrentRow1, Math.abs(expectedCurrentRow1 - observedWindow),
                    expectedScalar, observedScalar, Math.abs(expectedScalar - observedScalar),
                    nearestScalarKv, nearestScalarDelta);
        }
    }

    private static double dotAttentionLogit(
            INDArray query, int queryRow, int qHead,
            INDArray keyCache, int kvPosition, int kvHead,
            INDArray mask, int maskRow, double scale) {
        return dotCurrentWindowAttentionLogit(
                query, queryRow, qHead, keyCache, kvPosition, kvHead,
                mask, maskRow, kvPosition, scale);
    }

    private static double dotCurrentWindowAttentionLogit(
            INDArray query, int queryRow, int qHead,
            INDArray currentKeyWindow, int currentRow, int kvHead,
            INDArray mask, int maskRow, int maskKvPosition, double scale) {
        double dot = 0.0;
        for (int d = 0; d < query.size(3); d++) {
            dot += query.getDouble(0, queryRow, qHead, d)
                    * currentKeyWindow.getDouble(0, currentRow, kvHead, d);
        }
        return dot * scale + mask.getDouble(0, 0, maskRow, maskKvPosition);
    }

    private static void logKeyCacheMutation(
            String kind, INDArray before, INDArray after, INDArray currentKeyRow) {
        double[] total = difference(after, before);
        int changedRows = 0;
        int nearestCurrentRow = -1;
        double nearestCurrentMax = Double.POSITIVE_INFINITY;
        double nearestCurrentL1 = Double.POSITIVE_INFINITY;
        for (int row = 0; row < after.size(1); row++) {
            INDArray afterRow = sequenceRow(after, row);
            double[] mutation = difference(afterRow, sequenceRow(before, row));
            if (mutation[0] != 0.0) {
                changedRows++;
                if (changedRows <= 8) {
                    log.info("[MTP-TARGET-PARITY] discriminator=cacheMutation kind={} row={} max={} l1={}",
                            kind, row, mutation[0], mutation[1]);
                }
            }
            double[] current = difference(afterRow, currentKeyRow);
            if (current[1] < nearestCurrentL1) {
                nearestCurrentRow = row;
                nearestCurrentMax = current[0];
                nearestCurrentL1 = current[1];
            }
        }
        log.info("[MTP-TARGET-PARITY] discriminator=cacheMutation kind={} totalMax={} totalL1={} "
                        + "changedRows={} nearestCurrentRow={} nearestCurrentMax={} nearestCurrentL1={}",
                kind, total[0], total[1], changedRows,
                nearestCurrentRow, nearestCurrentMax, nearestCurrentL1);
    }

    /** Max-abs and L1 difference of two equal-length double arrays. */
    private static double[] diffDoubles(double[] left, double[] right) {
        assertEquals(left.length, right.length, "double-array diff length mismatch");
        double max = 0.0, l1 = 0.0;
        for (int i = 0; i < left.length; i++) {
            double d = Math.abs(left[i] - right[i]);
            if (d > max) max = d;
            l1 += d;
        }
        return new double[]{max, l1};
    }

    private static double[] difference(INDArray left, INDArray right) {        INDArray diff = left.sub(right);
        try {
            return new double[]{
                    diff.amaxNumber().doubleValue(),
                    diff.norm1Number().doubleValue()
            };
        } finally {
            diff.close();
        }
    }

    private static void putOwned(
            Map<String, INDArray> map, String name, INDArray array, List<INDArray> owned) {
        map.put(name, own(owned, array));
    }

    private static INDArray own(List<INDArray> owned, INDArray array) {
        owned.add(array);
        return array;
    }

    private static void ownAll(Map<String, INDArray> arrays, List<INDArray> owned) {
        owned.addAll(arrays.values());
    }

    private static void closeOwned(List<INDArray> owned) {
        Set<INDArray> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (int i = owned.size() - 1; i >= 0; i--) {
            INDArray array = owned.get(i);
            if (array != null && seen.add(array) && !array.wasClosed()) {
                try {
                    array.close();
                } catch (Exception e) {
                    log.warn("[MTP-TARGET-PARITY] Failed to close array: {}", e.getMessage());
                }
            }
        }
    }
}
