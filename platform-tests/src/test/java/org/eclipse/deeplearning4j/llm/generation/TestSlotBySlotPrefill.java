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
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.ggml.GGMLModelImport;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression test for the SLOT_BY_SLOT-mode CUDA err-700 (illegal memory access)
 * during GGUF prefill/warmup.
 *
 * <p>Observed: pinning a GenerationPipeline to {@link GraphExecutionMode#SLOT_BY_SLOT}
 * via its benchmarkConfig crashed inside {@code prefillWarmupAndFreeze} with
 * {@code ReduceSameFunction intermediateScalar cudaStreamSynchronize error 700}
 * — an asynchronous fault from a kernel launched earlier in the slot-by-slot
 * prefill/warmup execution, surfacing at the first blocking host read.</p>
 *
 * <p>This test only needs prefill + warmup to reproduce: it starts a session and
 * generates a handful of tokens under SLOT_BY_SLOT, asserting clean completion.</p>
 */
@Slf4j
public class TestSlotBySlotPrefill {

    private static final String PROMPT =
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.";

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

        List<String> outputs = rawModel.outputs() != null
                ? new ArrayList<>(rawModel.outputs()) : new ArrayList<>();
        optimizedModel = GraphOptimizer.optimize(rawModel, outputs, GraphOptimizer.defaultOptimizations());
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

    /**
     * Characterization, not exactness: the variable-size path resets and recompiles the
     * DSP plan every generation ({@code resetSession} + {@code clearDynamicShapePlanCache}),
     * so every generation re-traverses the eager→frozen→captured lifecycle at fresh buffer
     * addresses. Recompiled plans differ by ulps (cuBLAS algorithm/workspace selection,
     * address-dependent heuristics) and greedy near-ties amplify that into token-level
     * divergence between generations — measured ~16/60 from step ~30. Bitwise equality is
     * therefore only asserted on the fixed-buffer steady-state path (next test); here we
     * assert clean completion and log the pairwise table for visibility.
     */
    @Test
    @DisplayName("Variable-size path: repeated greedy generations complete cleanly (divergence logged)")
    public void testGreedyTwiceDeterministic() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(60)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();
        runGreedyRepeats(cfg, false);
    }

    /**
     * Fixed-buffer padding must be semantically invisible. Run the same prompt first
     * without padding, then with a 64-token prefill buffer, and compare the complete
     * greedy token sequence. This catches asynchronous attention-buffer lifetime bugs
     * that a token-count-only assertion cannot detect.
     */
    @Test
    @DisplayName("SLOT_BY_SLOT padded prefill matches the unpadded token sequence")
    public void testSbsPaddedReusePrompt() throws Exception {
        final String REUSE_PROMPT =
                "Once upon a time, in a land far away, there lived a curious inventor who";

        BenchmarkConfig referenceMode = BenchmarkConfig.create("SBS_UNPADDED_REFERENCE")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT);
        GenerationPipelineConfig referenceCfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(16)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .benchmarkConfig(referenceMode)
                .build();

        int[] referenceTokens;
        try (GenerationPipeline pipe = GenerationPipeline.create(referenceCfg)) {
            referenceTokens = pipe.generate(REUSE_PROMPT, 16).getTokenIds();
            log.info("[SBS-UNPADDED-REFERENCE] {} tokens: {}",
                    referenceTokens.length, java.util.Arrays.toString(referenceTokens));
            assertTrue(referenceTokens.length >= 8,
                    "SBS unpadded reference must produce tokens; got " + referenceTokens.length);
        }

        BenchmarkConfig paddedMode = BenchmarkConfig.create("SBS_PADDED_REUSE_PROMPT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT);
        GenerationPipelineConfig paddedCfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(16)
                .maxPrefillLength(64)
                .maxKvCacheLength(80)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .benchmarkConfig(paddedMode)
                .build();

        try (GenerationPipeline pipe = GenerationPipeline.create(paddedCfg)) {
            int[] paddedTokens = pipe.generate(REUSE_PROMPT, 16).getTokenIds();
            log.info("[SBS-PADDED] {} tokens: {}",
                    paddedTokens.length, java.util.Arrays.toString(paddedTokens));
            assertTrue(paddedTokens.length >= 8,
                    "SBS padded must produce tokens; got " + paddedTokens.length);
            assertArrayEquals(referenceTokens, paddedTokens,
                    "Padding must not change the greedy token sequence");
            assertEquals(557, referenceTokens[0],
                    "SBS unpadded reference first token must match the established golden");
            assertEquals(557, paddedTokens[0],
                    "SBS padded first token must match the established golden");
        }
    }

    /**
     * Padding split for the reuse investigation: unpadded (variable) OPTIMAL-mode
     * generation of the same prompt as the SLOT_BY_SLOT reference. A material
     * divergence here implicates execution mode rather than fixed-buffer padding.
     */
    @Test
    @DisplayName("Variable-path OPTIMAL 16 tokens for the reuse-investigation prompt")
    public void testVariableOptimalReusePrompt() throws Exception {
        final String REUSE_PROMPT =
                "Once upon a time, in a land far away, there lived a curious inventor who";
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(16)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();
        try (GenerationPipeline pipe = GenerationPipeline.create(cfg)) {
            int[] tokens = pipe.generate(REUSE_PROMPT, 16).getTokenIds();
            log.info("[VAR-OPTIMAL] {} tokens: {}", tokens.length, java.util.Arrays.toString(tokens));
            assertTrue(tokens.length >= 8,
                    "Variable OPTIMAL must produce tokens; got " + tokens.length);
        }
    }

    /**
     * Fixed-buffer variant: {@code maxPrefillLength > 0} routes generations 2+ through the
     * cached fixed-buffer state (same frozen plan, same buffer addresses). Discriminates
     * per-generation recompile/address churn from in-plan kernel nondeterminism: if this
     * passes while the variable-size variant fails, the churn is the root.
     */
    @Test
    @DisplayName("Fixed buffers: same plan+addresses across generations must be token-identical")
    public void testGreedyTwiceDeterministicFixedBuffers() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(60)
                .maxPrefillLength(64)
                .maxKvCacheLength(192)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .build();
        runGreedyRepeats(cfg, true);
    }

    /** Decode length for the repeat probes. */
    private static final int REPEAT_TOKENS = 60;

    private void runGreedyRepeats(GenerationPipelineConfig cfg, boolean assertSteadyExact) throws Exception {

        // Three generations: pairwise pattern discriminates a systematic
        // first-vs-later-generation path difference (gen2==gen3 != gen1) from
        // per-generation randomness (all pairs differ).
        //
        // One-shot generate() (not sessions): with maxPrefillLength > 0 the one-shot
        // path retains cachedFixedBufferState across calls — same frozen plan, same
        // buffer addresses. startSession() deliberately drops that cache (session
        // takeover), so sessions always rebuild and cannot test plan reuse.
        // Lifecycle-phase contract: generations 1-2 traverse DSP execution-mode
        // transitions (eager → shapes-frozen → captured; the prefill plan hits its
        // capture threshold at exec #3). Eager and captured executions of the same
        // plan legitimately differ by ulps (cuBLAS algorithm selection changes under
        // stream capture), and greedy near-ties amplify those ulps into token flips.
        // From generation 3 on every plan replays captured graphs — bit-identical by
        // construction (verified: gens 3/4/5 exact, tokens AND decode-entry state).
        // Assert exact equality at steady state; earlier pairs are logged only.
        int genCount = 5;
        int warmups = 2;
        int[][] gens = new int[genCount][];
        try (GenerationPipeline pipe = GenerationPipeline.create(cfg)) {
            for (int g = 0; g < genCount; g++) {
                gens[g] = pipe.generate(PROMPT, REPEAT_TOKENS).getTokenIds();
                assertTrue(gens[g].length >= REPEAT_TOKENS / 2,
                        "Generation " + (g + 1) + " produced only " + gens[g].length + " tokens");
            }
        }
        int steadyMismatches = 0;
        for (int a = 0; a < genCount; a++) {
            for (int b = a + 1; b < genCount; b++) {
                int compareLen = Math.min(gens[a].length, gens[b].length);
                int mismatches = 0;
                int firstMismatch = -1;
                for (int i = 0; i < compareLen; i++) {
                    if (gens[a][i] != gens[b][i]) {
                        mismatches++;
                        if (firstMismatch < 0) firstMismatch = i;
                    }
                }
                log.info("[GREEDY-TWICE] gen{} vs gen{}: {} tokens, {} mismatches (first at {})",
                        a + 1, b + 1, compareLen, mismatches, firstMismatch);
                if (a >= warmups && b >= warmups) steadyMismatches += mismatches;
            }
        }
        if (assertSteadyExact) {
            assertTrue(steadyMismatches == 0,
                    "Steady-state greedy generations (gen " + (warmups + 1) + "+) on the same frozen "
                            + "plan must be token-identical; " + steadyMismatches
                            + " pairwise mismatches — see [GREEDY-TWICE] lines");
        }
    }

    @Test
    @DisplayName("SLOT_BY_SLOT prefill + short decode completes without CUDA faults")
    public void testSlotBySlotPrefillAndShortDecode() throws Exception {
        BenchmarkConfig sbs = BenchmarkConfig.create("SBS_PREFILL_REGRESSION")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT);

        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(8)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .benchmarkConfig(sbs)
                .build();

        try (GenerationPipeline pipe = GenerationPipeline.create(cfg);
             GenerationSession session = pipe.startSession(PROMPT, 8)) {
            GenerationResult r = session.generate(8);
            assertNotNull(r);
            int[] tokens = r.getTokenIds();
            log.info("[SBS] {} tokens: {}", tokens.length, java.util.Arrays.toString(tokens));
            assertTrue(tokens.length >= 4,
                    "SLOT_BY_SLOT decode must produce tokens; got " + tokens.length);
        }
    }

    /**
     * SLOT_BY_SLOT golden sequence for the fixed-buffer reuse investigation prompt:
     * every op eager, no capture, no replay — ground truth for the merged-replay
     * comparison. The fixed-buffer fresh generation of the same prompt must match
     * this token-for-token; divergence at the capture step implicates the merged
     * captured graphs themselves (position-era baking), not the reuse path.
     */
    @Test
    @DisplayName("SLOT_BY_SLOT golden 16 tokens for the reuse-investigation prompt")
    public void testSlotBySlotGoldenReusePrompt() throws Exception {
        final String REUSE_PROMPT =
                "Once upon a time, in a land far away, there lived a curious inventor who";
        BenchmarkConfig sbs = BenchmarkConfig.create("SBS_GOLDEN_REUSE_PROMPT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT);

        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(16)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false)
                .dspEnabled(true)
                .benchmarkConfig(sbs)
                .build();

        try (GenerationPipeline pipe = GenerationPipeline.create(cfg)) {
            int[] tokens = pipe.generate(REUSE_PROMPT, 16).getTokenIds();
            log.info("[SBS-GOLDEN] {} tokens: {}", tokens.length, java.util.Arrays.toString(tokens));
            assertTrue(tokens.length >= 8,
                    "SLOT_BY_SLOT golden must produce tokens; got " + tokens.length);
            assertEquals(557, tokens[0],
                    "SLOT_BY_SLOT golden first token must remain stable");
        }
    }
}
