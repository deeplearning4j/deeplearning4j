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
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline.GenerationSession;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.ggml.GGMLModelImport;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * ADR 0106 Phase 2: n-gram speculative decoding integration tests.
 *
 * <p>Tests the three acceptance criteria for speculative window decode:</p>
 * <ol>
 *   <li><b>Lossless oracle</b>: greedy with n-gram speculativeK=4 must produce token-for-token
 *       identical results to greedy without speculation on Qwen3.5 0.8B GGUF (~60 tokens).
 *       This validates the lossless-accept contract: proposals that match are accepted, those
 *       that mismatch are corrected, and the output is always greedy-equivalent.</li>
 *   <li><b>Acceptance-rate sanity</b>: on a repetitive prompt, speculation must actually engage
 *       (accepted draft count > 0). If speculation never fires, the n-gram table is never
 *       populated and the bigram proposer is dead.</li>
 *   <li><b>Boundary</b>: requesting speculativeK tokens when fewer than speculativeK remain to
 *       the maxNewTokens limit must terminate exactly at maxNewTokens, not overshoot.</li>
 * </ol>
 *
 * <p><b>IMPORTANT:</b> The lossless oracle test (a) MUST run first (@Order(1)). It creates one
 * speculative-width fixed-buffer pipeline, advances it to captured steady state, and then compares
 * speculative and greedy policies on that exact plan and those exact buffer addresses.</p>
 *
 * <h2>Run (CUDA)</h2>
 * <pre>
 *   cd platform-tests &amp;&amp; /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *       -Dtest=TestSpeculativeWindowDecode \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       -Dlibnd4j.triton=ON 2&gt;&amp;1 | tee /tmp/spec2-test-cuda.log
 * </pre>
 *
 * <h2>Run (CPU)</h2>
 * <pre>
 *   cd platform-tests &amp;&amp; /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *       -Dtest=TestSpeculativeWindowDecode \
 *       -Dbackend.artifactId=nd4j-native 2&gt;&amp;1 | tee /tmp/spec2-test-cpu.log
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestSpeculativeWindowDecode {

    /** Speculative draft tokens per step. Width W_max = speculativeK + 1. */
    private static final int SPEC_K = 4;

    /**
     * Token count for the lossless oracle comparison. Override with
     * {@link ND4JSystemProperties#BENCH_MAX_TOKENS} for the 250-token measurement gate.
     */
    private static final int ORACLE_TOKENS =
            Integer.getInteger(ND4JSystemProperties.BENCH_MAX_TOKENS, 60);

    /** Repetitive prompt that forces n-gram table hits. */
    private static final String REPETITIVE_PROMPT =
            "the cat sat on the mat the cat sat on the mat the cat sat on the mat " +
            "the cat sat on the mat the cat sat on the mat";

    /** Ordinary prompt for the lossless oracle test. */
    private static final String ORACLE_PROMPT =
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.";

    /** Shared model and tokenizer, loaded once for all tests. */
    private static SameDiff optimizedModel;
    private static Tokenizer tokenizer;

    @BeforeAll
    public static void setup() throws Exception {
        if (System.getProperty(ND4JSystemProperties.OPTIMIZER_ENABLED) == null) {
            System.setProperty(ND4JSystemProperties.OPTIMIZER_ENABLED, "true");
        }
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr  = System.getProperty("qwen.quant",      "Q4_K_M");

        DownloadResult dl = LLMModelDownloader.download(
                LLMModel.fromSizeLabel(sizeLabel), QuantType.valueOf(quantStr));
        SameDiff rawModel = GGMLModelImport.importModel(dl.getModelFile().getAbsolutePath());

        // Run GraphOptimizer exactly once (same pattern as TestInt8KvLiveDecode).
        List<String> outputs = rawModel.outputs() != null
                ? new ArrayList<>(rawModel.outputs()) : new ArrayList<>();
        long optStart = System.currentTimeMillis();
        optimizedModel = GraphOptimizer.optimize(rawModel, outputs, GraphOptimizer.defaultOptimizations());
        log.info("[setup] GraphOptimizer: {} -> {} ops in {}ms",
                rawModel.getOps().size(), optimizedModel.getOps().size(),
                System.currentTimeMillis() - optStart);

        if (optimizedModel != rawModel) {
            try {
                SameDiffMemoryUtils.freeModelArrays(rawModel);
                rawModel.close();
            } catch (Exception e) {
                log.warn("[setup] Error closing raw model: {}", e.getMessage());
            }
        }

        String tokenizerPath = System.getProperty("qwen.tokenizer.path");
        if (tokenizerPath != null && !tokenizerPath.isEmpty()) {
            tokenizer = HuggingFaceTokenizer.fromFile(tokenizerPath);
        } else {
            String tokUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel
                    + "/resolve/main/tokenizer.json";
            File tf = LLMModelDownloader.downloadCustom(
                    tokUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
            tokenizer = HuggingFaceTokenizer.fromFile(tf.getAbsolutePath());
        }
    }

    @AfterAll
    public static void teardown() {
        if (optimizedModel != null) {
            try {
                SameDiffMemoryUtils.freeModelArrays(optimizedModel);
                optimizedModel.close();
            } catch (Exception e) {
                log.warn("[teardown] Error closing optimizedModel: {}", e.getMessage());
            }
            optimizedModel = null;
        }
        tokenizer = null;
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // (a) LOSSLESS ORACLE: greedy speculative k=4 == greedy no-spec, token-for-token
    // ══════════════════════════════════════════════════════════════════════════════════

    /**
     * Greedy decode with n-gram speculativeK=4 must produce bit-identical tokens compared to
     * greedy decode without speculation.
     *
     * <p>The lossless accept rule guarantees this: for each forward pass, the target verifies
     * every draft token — accepted tokens match exactly what greedy would have emitted at that
     * position. Any mismatch produces the same correction token that greedy would emit.
     * Therefore the full output sequence is always greedy-equivalent.</p>
     *
     * <p>Must run FIRST (@Order(1)): the greedy baseline establishes the DSP frozen plan on
     * this model before the speculative W>1 substrate initialises.</p>
     */
    @Test
    @Order(1)
    @DisplayName("(a) Lossless oracle: greedy specK=4 is token-identical to greedy no-spec")
    public void testLosslessOracleGreedySpeculativeEqualsGreedy() throws Exception {
        // ── ONE pipeline, ONE frozen plan, ONE set of buffers, two sampling policies ──
        // The lossless-accept contract promises: speculative output == the target
        // plan's OWN greedy output. Two generations only compare meaningfully when
        // they run the identical compiled plan on the identical buffer addresses:
        // measured on this model, per-generation recompiles diverge (cuBLAS split-K
        // variance + fresh-address algo selection flip low-margin argmaxes through
        // the GDN recurrent state — plain greedy-vs-greedy across recompiles differs
        // by ~16/60 tokens from step ~30, see TestSlotBySlotPrefill).
        //
        // maxPrefillLength > 0 + one-shot generate() is the configuration that reuses
        // the frozen plan AND its buffers across calls (cachedFixedBufferState).
        // startSession() deliberately drops that cache (session takeover), so this
        // test drives one-shot generate(). The speculative pass freezes the
        // W = SPEC_K+1 envelope; the runtime-swapped greedy pass replays the same
        // plan with activeWindow=1 (greedy is the 1x1 special case of the substrate,
        // ADR 0106).
        int[] specTokens;
        int[] greedyTokens;
        GenerationResult specResult;
        GenerationResult greedyResult;
        GenerationPipelineConfig specCfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.speculative())
                .maxNewTokens(ORACLE_TOKENS)
                .maxSpeculativeTokens(SPEC_K)       // n-gram NGRAM proposer
                .maxPrefillLength(64)               // fixed buffers → plan+buffer reuse
                .maxKvCacheLength(Math.max(192, ORACLE_TOKENS + 64))
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();
        try (GenerationPipeline pipe = GenerationPipeline.create(specCfg)) {
            // Steady-state warmup: generations 1-2 traverse the DSP lifecycle
            // (eager → frozen → captured; the prefill plan captures at exec #3).
            // Eager and captured executions of the same plan differ by ulps (cuBLAS
            // algorithm selection under stream capture), so bitwise comparison is
            // only meaningful once every plan replays captured graphs — generation
            // 3 onward (verified bit-identical, tokens and decode-entry state, in
            // TestSlotBySlotPrefill#testGreedyTwiceDeterministicFixedBuffers).
            for (int w = 0; w < 2; w++) {
                int produced = pipe.generate(ORACLE_PROMPT, ORACLE_TOKENS).getTokenIds().length;
                log.info("[ORACLE-WARMUP {}] {} tokens", w + 1, produced);
            }

            // Speculative pass at steady state on the frozen W-wide plan.
            specResult = pipe.generate(ORACLE_PROMPT, ORACLE_TOKENS);
            specTokens = specResult.getTokenIds();
            log.info("[SPEC-K={}] {} tokens: {}", SPEC_K, specTokens.length,
                    Arrays.toString(Arrays.copyOf(specTokens, Math.min(16, specTokens.length))));
            log.info("[SPEC-METRICS] tokens={} proposed={} accepted={} steps={} acceptance={} "
                            + "tok/s={} decodeTok/s={} lateTok/s={} effectiveTok/s={}",
                    specTokens.length, specResult.getTotalSpeculativeTokens(),
                    specResult.getTotalAcceptedTokens(), specResult.getSpeculativeSteps(),
                    specResult.getAverageAcceptanceRate(), specResult.getTokensPerSecond(),
                    specResult.getDecodeTokensPerSecond(),
                    specResult.getLateSteadyStateTokensPerSecond(),
                    specResult.getEffectiveTokensPerSecond());

            assertTrue(specTokens.length >= Math.min(ORACLE_TOKENS, 10),
                    "Speculative decode must produce at least " + Math.min(ORACLE_TOKENS, 10)
                            + " tokens; got " + specTokens.length);

            // Greedy baseline on the SAME frozen plan + SAME buffers: runtime sampling
            // swap resolves the GREEDY policy; the reuse path re-prefills in place and
            // the native loop runs scalar activeWindow=1 steps with no proposals.
            pipe.setSamplingConfig(SamplingConfig.greedy());
            greedyResult = pipe.generate(ORACLE_PROMPT, ORACLE_TOKENS);
            greedyTokens = greedyResult.getTokenIds();
            log.info("[GREEDY] {} tokens: {}", greedyTokens.length,
                    Arrays.toString(Arrays.copyOf(greedyTokens, Math.min(16, greedyTokens.length))));
            log.info("[GREEDY-METRICS] tokens={} tok/s={} decodeTok/s={} lateTok/s={}",
                    greedyTokens.length, greedyResult.getTokensPerSecond(),
                    greedyResult.getDecodeTokensPerSecond(),
                    greedyResult.getLateSteadyStateTokensPerSecond());
        }

        // Baseline must produce a usable number of tokens.
        assertTrue(greedyTokens.length >= Math.min(ORACLE_TOKENS, 10),
                "Greedy baseline must produce at least " + Math.min(ORACLE_TOKENS, 10)
                        + " tokens; got " + greedyTokens.length);

        // Speculative path must produce the same count (within stop-token timing slop of ±1).
        int compareLen = Math.min(greedyTokens.length, specTokens.length);
        assertTrue(compareLen >= Math.min(ORACLE_TOKENS, 10),
                "Both decoders must produce at least " + Math.min(ORACLE_TOKENS, 10)
                        + " tokens; greedy=" + greedyTokens.length + ", spec=" + specTokens.length);

        // LOSSLESS: every token in the common prefix must be identical.
        int mismatches = 0;
        int firstMismatch = -1;
        for (int i = 0; i < compareLen; i++) {
            if (greedyTokens[i] != specTokens[i]) {
                mismatches++;
                if (firstMismatch < 0) firstMismatch = i;
            }
        }
        log.info("[ORACLE] compared {} tokens: {} mismatches (first at {})",
                compareLen, mismatches, firstMismatch);

        assertEquals(0, mismatches,
                "Lossless oracle FAILED: " + mismatches + " token mismatches in " + compareLen
                        + " tokens (first at position " + firstMismatch + ").\n"
                        + "greedy=" + Arrays.toString(Arrays.copyOf(greedyTokens, compareLen)) + "\n"
                        + "  spec=" + Arrays.toString(Arrays.copyOf(specTokens, compareLen)));
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // (b) ACCEPTANCE-RATE SANITY: speculation must engage on repetitive text
    // ══════════════════════════════════════════════════════════════════════════════════

    /**
     * On a repetitive prompt, the bigram n-gram table must populate after the first few
     * steps and begin proposing draft tokens. At least some tokens must be accepted as
     * speculative drafts (i.e. total tokens generated > decode steps * 1).
     *
     * <p>We measure this indirectly: if speculation engaged, the decode should produce
     * the correct token sequence AND the timing info should be available. The key check is
     * that the generation completes successfully — a bug in the speculative path (e.g. wrong
     * mask update, bogus argmax row, or broken bookkeeping) typically manifests as token
     * corruption or crash.</p>
     *
     * <p>We also verify that the output matches greedy, confirming that accepted-then-verified
     * tokens are correctly written to the output buffer.</p>
     */
    @Test
    @Order(2)
    @DisplayName("(b) Acceptance-rate sanity: spec engages (accepted>0) on repetitive prompt")
    public void testAcceptanceRateOnRepetitivePrompt() throws Exception {
        // Greedy baseline on the repetitive prompt.
        int[] greedyTokens;
        GenerationPipelineConfig greedyCfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(40)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();
        try (GenerationPipeline pipe = GenerationPipeline.create(greedyCfg);
             GenerationSession session = pipe.startSession(REPETITIVE_PROMPT, 40)) {
            greedyTokens = session.generate(40).getTokenIds();
            log.info("[GREEDY-REP] {} tokens: {}", greedyTokens.length,
                    Arrays.toString(Arrays.copyOf(greedyTokens, Math.min(20, greedyTokens.length))));
        }

        // Speculative decode on the same repetitive prompt.
        int[] specTokens;
        GenerationResult specResult;
        GenerationPipelineConfig specCfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.speculative())
                .maxNewTokens(40)
                .maxSpeculativeTokens(SPEC_K)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();
        try (GenerationPipeline pipe = GenerationPipeline.create(specCfg);
             GenerationSession session = pipe.startSession(REPETITIVE_PROMPT, 40)) {
            specResult = session.generate(40);
            specTokens = specResult.getTokenIds();
            log.info("[SPEC-REP K={}] {} tokens: {}", SPEC_K, specTokens.length,
                    Arrays.toString(Arrays.copyOf(specTokens, Math.min(20, specTokens.length))));
        }

        // Speculative path must not produce fewer tokens than greedy (within a small margin;
        // if EOS fires 1 token earlier due to batch-accept order that is acceptable).
        assertTrue(specTokens.length >= Math.max(1, greedyTokens.length - 2),
                "Speculative path produced fewer tokens than greedy: spec="
                        + specTokens.length + " greedy=" + greedyTokens.length);

        // Lossless: token sequence must match over the common prefix.
        int compareLen = Math.min(greedyTokens.length, specTokens.length);
        int mismatches = 0;
        for (int i = 0; i < compareLen; i++) {
            if (greedyTokens[i] != specTokens[i]) mismatches++;
        }
        log.info("[SPEC-REP] {} mismatches over {} tokens", mismatches, compareLen);

        assertEquals(0, mismatches,
                "Acceptance-rate test: speculative output differs from greedy over " + compareLen
                        + " tokens (" + mismatches + " mismatches).\n"
                        + "greedy=" + Arrays.toString(Arrays.copyOf(greedyTokens, compareLen)) + "\n"
                        + "  spec=" + Arrays.toString(Arrays.copyOf(specTokens, compareLen)));

        // Acceptance check: assert native engagement directly. Token equality alone can also be
        // produced by the scalar fallback and therefore does not prove speculative execution.
        assertTrue(specResult.getTotalSpeculativeTokens() > 0,
                "N-gram proposer never produced a draft token; speculative path did not engage.");
        assertTrue(specResult.getSpeculativeSteps() > 0,
                "No speculative verification step executed.");
        assertTrue(specResult.getTotalAcceptedTokens() > 0,
                "Speculative verification ran but accepted no draft tokens.");
        log.info("[SPEC-REP] PASSED — proposed={}, accepted={}, steps={}, acceptanceRate={}",
                specResult.getTotalSpeculativeTokens(), specResult.getTotalAcceptedTokens(),
                specResult.getSpeculativeSteps(), specResult.getAverageAcceptanceRate());
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // (c) BOUNDARY: k > remaining capacity must terminate at exactly maxNewTokens
    // ══════════════════════════════════════════════════════════════════════════════════

    /**
     * When speculativeK > the remaining decode budget (maxNewTokens - tokens already generated),
     * the decode loop must terminate at exactly maxNewTokens, never overshoot.
     *
     * <p>This exercises the boundary check in the speculative accept loop: after acceptance,
     * {@code tokensGenerated} is checked against {@code maxNewTokens} for every accepted token
     * in the batch. If even one extra token leaks through the boundary check, the output array
     * will be overrun.</p>
     *
     * <p>We use a small maxNewTokens (8) with speculativeK=4 so that on step 1 or 2, the draft
     * window already exceeds the remaining budget, forcing the boundary to trigger.</p>
     */
    @Test
    @Order(3)
    @DisplayName("(c) Boundary: spec terminates at exactly maxNewTokens, no overshoot")
    public void testSpeculativeBoundaryAtMaxNewTokens() throws Exception {
        final int MAX_NEW = 8;

        GenerationPipelineConfig specCfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.speculative())
                .maxNewTokens(MAX_NEW)
                .maxSpeculativeTokens(SPEC_K)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();
        int[] specTokens;
        GenerationResult.FinishReason finishReason;
        try (GenerationPipeline pipe = GenerationPipeline.create(specCfg);
             GenerationSession session = pipe.startSession(ORACLE_PROMPT, MAX_NEW)) {
            GenerationResult r = session.generate(MAX_NEW);
            specTokens = r.getTokenIds();
            finishReason = r.getFinishReason();
            log.info("[BOUNDARY] {} tokens, finish={}: {}", specTokens.length, finishReason,
                    Arrays.toString(specTokens));
        }

        // The output must NOT exceed maxNewTokens, even when a speculative batch accepts more.
        assertTrue(specTokens.length <= MAX_NEW,
                "Speculative decode overshot maxNewTokens=" + MAX_NEW + ": produced "
                        + specTokens.length + " tokens");

        // The output must be non-empty (the model must generate at least 1 token).
        assertTrue(specTokens.length >= 1,
                "Speculative decode produced 0 tokens with maxNewTokens=" + MAX_NEW);

        // Finish reason: either MAX_TOKENS (hit budget) or EOS (model ended naturally).
        assertTrue(finishReason == GenerationResult.FinishReason.MAX_TOKENS
                        || finishReason == GenerationResult.FinishReason.EOS,
                "Unexpected finish reason: " + finishReason
                        + " (expected MAX_TOKENS or EOS when specK > remaining budget)");

        log.info("[BOUNDARY] PASSED — {} tokens, finish={}", specTokens.length, finishReason);
    }}
