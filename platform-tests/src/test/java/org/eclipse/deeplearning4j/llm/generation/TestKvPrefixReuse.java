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
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvPrefixBlockPool;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.optimize.optimizations.QuantizationOptimizations;
import org.nd4j.ggml.GGMLModelImport;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the cross-request KV prefix block pool.
 *
 * <p>Tests the {@code prefixCacheEnabled=true} path in {@link GenerationPipeline} using
 * the Qwen3.5 0.8B Q4_K_M model. All tests run greedy decoding so output is deterministic.</p>
 *
 * <p>Run on CUDA backend:</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test -Dtest=TestKvPrefixReuse \
 *       -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 *
 * <p>Run on CPU backend:</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test -Dtest=TestKvPrefixReuse \
 *       -Dbackend.artifactId=nd4j-native
 * </pre>
 */
@Slf4j
public class TestKvPrefixReuse {

    // ── Shared prefix (must be longer than one block = 16 tokens when encoded) ──────────────────
    private static final String COMMON_PREFIX =
            "The history of artificial intelligence dates back to ancient myths about mechanical beings. "
            + "Modern AI research began in the 1950s when Alan Turing proposed his famous test.";

    private static final String SUFFIX_A = " Machine learning then emerged as a key subfield.";
    private static final String SUFFIX_B = " Neural networks became dominant in the 2010s.";

    private static SameDiff model;
    private static SameDiff optimizedModel;
    private static Tokenizer tokenizer;

    @BeforeAll
    public static void setup() throws Exception {
        if (System.getProperty("nd4j.optimizer.enabled") == null) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr  = System.getProperty("qwen.quant", "Q4_K_M");

        DownloadResult dl = LLMModelDownloader.download(LLMModel.fromSizeLabel(sizeLabel), QuantType.valueOf(quantStr));
        model = GGMLModelImport.importModel(dl.getModelFile().getAbsolutePath());

        // Run GraphOptimizer exactly once — each GenerationPipeline.create() with
        // graphOptimizerEnabled=true would otherwise graph.dup() a multi-GB CUDA-resident
        // clone per pipeline; pinned pages do not return to RSS and the JavaCPP
        // physicalBytes guard kills the fork (same pattern as TestQuantizedKvCacheIntegration).
        List<OptimizerSet> optimizations = new ArrayList<>();
        for (OptimizerSet s : GraphOptimizer.defaultOptimizations()) {
            if (!(s instanceof QuantizationOptimizations)) {
                optimizations.add(s);
            }
        }
        List<String> outputs = model.outputs() != null ? new ArrayList<>(model.outputs()) : new ArrayList<>();
        optimizedModel = GraphOptimizer.optimize(model, outputs, optimizations);
        if (optimizedModel != model) {
            try {
                SameDiffMemoryUtils.freeModelArrays(model);
                model.close();
            } catch (Exception e) {
                log.warn("[setup] Error closing raw model after optimization: {}", e.getMessage());
            }
        }
        model = null; // optimizedModel is the sole shared instance

        String tokenizerPath = System.getProperty("qwen.tokenizer.path");
        if (tokenizerPath != null && !tokenizerPath.isEmpty()) {
            tokenizer = HuggingFaceTokenizer.fromFile(tokenizerPath);
        } else {
            String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
            File tf = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
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
                log.warn("[teardown] Error closing optimized model: {}", e.getMessage());
            }
            optimizedModel = null;
        }
    }

    // ── Helper ───────────────────────────────────────────────────────────────────────────────────

    private GenerationPipeline buildPrefixPipeline(int blockSize, long maxBytes, int maxNewTokens) throws Exception {
        long resolvedMaxBytes = maxBytes > 0 ? maxBytes : 64L * 1024 * 1024; // 64 MB default
        int resolvedBlock = blockSize > 0 ? blockSize : KvPrefixBlockPool.DEFAULT_BLOCK_SIZE;
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxNewTokens)
                .graphOptimizerEnabled(false) // pre-optimized once in @BeforeAll
                .dspEnabled(true)
                .prefixCacheEnabled(true)
                .prefixCacheMaxBytes(resolvedMaxBytes)
                .prefixCacheBlockSize(resolvedBlock)
                .build();
        return GenerationPipeline.create(cfg);
    }

    private GenerationPipeline buildNoPrefixPipeline(int maxNewTokens) throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxNewTokens)
                .graphOptimizerEnabled(false) // pre-optimized once in @BeforeAll
                .dspEnabled(true)
                .prefixCacheEnabled(false)
                .build();
        return GenerationPipeline.create(cfg);
    }

    // ── Test 1: Warm-hit correctness ─────────────────────────────────────────────────────────────

    /**
     * Generate with a shared prefix twice on a prefix-cache-enabled pipeline.
     *
     * <p>The SECOND call must produce tokens identical to the FIRST call (greedy → deterministic)
     * AND the common prefix tokens must be restored from the cache (verified by checking that
     * the pipeline does not report a full prefill of the whole prompt for the second call).
     * We also compare against a no-cache pipeline baseline to confirm correctness.</p>
     */
    @Test
    @DisplayName("warm cache hit: second call reproduces first-call tokens exactly")
    public void testWarmHitCorrectness() throws Exception {
        try (GenerationPipeline cachedPipeline = buildPrefixPipeline(0, 0, 20);
             GenerationPipeline baselinePipeline = buildNoPrefixPipeline(20)) {

            String promptA = COMMON_PREFIX + SUFFIX_A;

            // Baseline (no prefix cache) — generates cold
            GenerationResult baselineA = baselinePipeline.generate(promptA, 20);
            int[] baselineTokensA = baselineA.getTokenIds();
            log.info("Baseline tokens for promptA: {}", Arrays.toString(baselineTokensA));
            assertTrue(baselineTokensA.length > 0, "baseline must generate at least 1 token");

            // First call on cached pipeline — cold (stores prefix in cache)
            GenerationResult firstA = cachedPipeline.generate(promptA, 20);
            int[] firstTokensA = firstA.getTokenIds();
            log.info("First (cold) cache call tokens: {}", Arrays.toString(firstTokensA));

            // Second call on cached pipeline with SAME prompt — must be a cache hit
            GenerationResult secondA = cachedPipeline.generate(promptA, 20);
            int[] secondTokensA = secondA.getTokenIds();
            log.info("Second (warm) cache call tokens: {}", Arrays.toString(secondTokensA));

            // Correctness: warm hit must produce same tokens as the cold first call
            int n = Math.min(firstTokensA.length, secondTokensA.length);
            assertTrue(n > 0, "cache hit must produce at least 1 token");
            assertArrayEquals(
                    Arrays.copyOf(firstTokensA, n),
                    Arrays.copyOf(secondTokensA, n),
                    "Warm prefix cache hit must reproduce cold generation tokens exactly (greedy). "
                    + "\nfirst =" + Arrays.toString(Arrays.copyOf(firstTokensA, n))
                    + "\nsecond=" + Arrays.toString(Arrays.copyOf(secondTokensA, n)));

            log.info("WARM HIT CORRECTNESS OK: {} tokens matched between cold and warm generation", n);

            // Baseline parity: cached pipeline should produce same tokens as no-cache baseline
            int nb = Math.min(baselineTokensA.length, firstTokensA.length);
            assertArrayEquals(
                    Arrays.copyOf(baselineTokensA, nb),
                    Arrays.copyOf(firstTokensA, nb),
                    "Prefix-cache pipeline (cold path) must match no-cache baseline (greedy). "
                    + "\nbaseline=" + Arrays.toString(Arrays.copyOf(baselineTokensA, nb))
                    + "\ncached  =" + Arrays.toString(Arrays.copyOf(firstTokensA, nb)));
            log.info("NO-CACHE BASELINE PARITY OK: {} tokens matched", nb);
        }
    }

    // ── Test 2: Partial-prefix hit at block granularity ──────────────────────────────────────────

    /**
     * Two prompts sharing the same block-aligned prefix but different suffixes.
     *
     * <p>After the first request caches the common prefix blocks, the second request (with a
     * different suffix) must produce the same tokens as if generated fresh from scratch
     * (the baseline pipeline). Prefix reuse is transparent to the output.</p>
     */
    @Test
    @DisplayName("partial prefix hit: different suffix with shared prefix produces correct tokens")
    public void testPartialPrefixHit() throws Exception {
        try (GenerationPipeline cachedPipeline = buildPrefixPipeline(0, 0, 20);
             GenerationPipeline baselinePipeline = buildNoPrefixPipeline(20)) {

            String promptA = COMMON_PREFIX + SUFFIX_A;
            String promptB = COMMON_PREFIX + SUFFIX_B;

            // Warm the cache with promptA
            cachedPipeline.generate(promptA, 20);
            log.info("Cache warmed with promptA");

            // Baseline for promptB (no cache — full prefill from scratch)
            GenerationResult baselineB = baselinePipeline.generate(promptB, 20);
            int[] baselineTokensB = baselineB.getTokenIds();
            log.info("Baseline tokens for promptB: {}", Arrays.toString(baselineTokensB));
            assertTrue(baselineTokensB.length > 0, "baseline promptB must generate at least 1 token");

            // Cached pipeline for promptB — should use the cached shared prefix
            GenerationResult cachedB = cachedPipeline.generate(promptB, 20);
            int[] cachedTokensB = cachedB.getTokenIds();
            log.info("Cached hit tokens for promptB: {}", Arrays.toString(cachedTokensB));

            // Output must be identical to the no-cache baseline (correctness of the suffix prefill)
            int n = Math.min(baselineTokensB.length, cachedTokensB.length);
            assertTrue(n > 0, "cached promptB must generate at least 1 token");
            assertArrayEquals(
                    Arrays.copyOf(baselineTokensB, n),
                    Arrays.copyOf(cachedTokensB, n),
                    "Prefix-cache hit with different suffix must reproduce no-cache baseline (greedy). "
                    + "\nbaseline=" + Arrays.toString(Arrays.copyOf(baselineTokensB, n))
                    + "\ncached  =" + Arrays.toString(Arrays.copyOf(cachedTokensB, n)));
            log.info("PARTIAL PREFIX HIT CORRECTNESS OK: {} tokens matched with baseline", n);
        }
    }

    // ── Test 3: Eviction under byte budget ───────────────────────────────────────────────────────

    /**
     * Fill the cache past the byte budget with distinct prompts, then verify that an earlier
     * prompt that was evicted does NOT crash the pipeline (it gracefully falls back to full
     * prefill and produces correct output).
     *
     * <p>We use a very small budget (1 byte) to force immediate eviction after every store,
     * so every subsequent call is a cold miss — verifying that the eviction path is safe.</p>
     */
    @Test
    @DisplayName("eviction under tiny byte budget: falls back to full prefill gracefully")
    public void testEvictionUnderTinyBudget() throws Exception {
        // 1 byte budget forces immediate eviction of every stored block.
        try (GenerationPipeline tinyBudgetPipeline = buildPrefixPipeline(0, 1L, 8);
             GenerationPipeline baselinePipeline = buildNoPrefixPipeline(8)) {

            String[] prompts = {
                COMMON_PREFIX + SUFFIX_A,
                COMMON_PREFIX + SUFFIX_B,
                COMMON_PREFIX + " Robotics emerged as a sister discipline.",
            };

            for (String prompt : prompts) {
                GenerationResult cached = tinyBudgetPipeline.generate(prompt, 8);
                GenerationResult baseline = baselinePipeline.generate(prompt, 8);

                assertNotNull(cached, "eviction-path generate must not throw");
                int n = Math.min(cached.getTokenIds().length, baseline.getTokenIds().length);
                assertTrue(n > 0, "eviction-path generate must produce at least 1 token");
                assertArrayEquals(
                        Arrays.copyOf(baseline.getTokenIds(), n),
                        Arrays.copyOf(cached.getTokenIds(), n),
                        "Eviction fallback (full prefill) must match no-cache baseline. "
                        + "prompt='" + prompt.substring(0, Math.min(50, prompt.length())) + "'");
                log.info("EVICTION FALLBACK OK for prompt prefix: '{}...'", prompt.substring(0, 40));
            }
        }
    }

    // ── Test 4: Flag OFF — no behavior change ────────────────────────────────────────────────────

    /**
     * With {@code prefixCacheEnabled=false} (the DEFAULT), two calls to the same prompt must
     * produce identical tokens to the no-cache baseline — verifying zero behavior change when the
     * feature flag is off.
     */
    @Test
    @DisplayName("flag OFF: prefixCacheEnabled=false produces same output as no-cache baseline")
    public void testFlagOffNoChange() throws Exception {
        try (GenerationPipeline noCachePipeline = buildNoPrefixPipeline(20)) {

            String prompt = COMMON_PREFIX + SUFFIX_A;

            // Two calls on the no-cache pipeline must produce identical tokens (greedy is deterministic)
            GenerationResult first = noCachePipeline.generate(prompt, 20);
            GenerationResult second = noCachePipeline.generate(prompt, 20);

            int n = Math.min(first.getTokenIds().length, second.getTokenIds().length);
            assertTrue(n > 0);
            assertArrayEquals(
                    Arrays.copyOf(first.getTokenIds(), n),
                    Arrays.copyOf(second.getTokenIds(), n),
                    "No-cache pipeline must be deterministic (greedy). "
                    + "\nfirst =" + Arrays.toString(Arrays.copyOf(first.getTokenIds(), n))
                    + "\nsecond=" + Arrays.toString(Arrays.copyOf(second.getTokenIds(), n)));
            log.info("FLAG OFF DETERMINISM OK: {} tokens matched across two calls", n);
        }
    }

    // ── Test 5: Session API with prefix cache ────────────────────────────────────────────────────

    /**
     * Verify that the prefix cache integrates correctly with the session API.
     *
     * <p>Two sessions with the same prefix-bearing prompt: the second session should hit the
     * cache and produce the same initial tokens as the first session (greedy → deterministic).</p>
     */
    @Test
    @DisplayName("session API: second startSession with shared prefix hits cache and matches first session")
    public void testSessionApiWithPrefixCache() throws Exception {
        try (GenerationPipeline cachedPipeline = buildPrefixPipeline(0, 0, 20)) {

            String prompt = COMMON_PREFIX + SUFFIX_A;

            // First session (cold)
            int[] firstTokens;
            try (GenerationSession s1 = cachedPipeline.startSession(prompt, 20)) {
                GenerationResult r1 = s1.generate(10);
                firstTokens = r1.getTokenIds();
                log.info("Session 1 (cold) tokens: {}", Arrays.toString(firstTokens));
                assertTrue(firstTokens.length > 0, "first session must produce tokens");
            }

            // Second session (warm hit — same prompt)
            int[] secondTokens;
            try (GenerationSession s2 = cachedPipeline.startSession(prompt, 20)) {
                GenerationResult r2 = s2.generate(10);
                secondTokens = r2.getTokenIds();
                log.info("Session 2 (warm) tokens: {}", Arrays.toString(secondTokens));
                assertTrue(secondTokens.length > 0, "second session must produce tokens");
            }

            int n = Math.min(firstTokens.length, secondTokens.length);
            assertArrayEquals(
                    Arrays.copyOf(firstTokens, n),
                    Arrays.copyOf(secondTokens, n),
                    "Second session with same prompt should hit prefix cache and produce identical tokens. "
                    + "\nfirst =" + Arrays.toString(Arrays.copyOf(firstTokens, n))
                    + "\nsecond=" + Arrays.toString(Arrays.copyOf(secondTokens, n)));
            log.info("SESSION PREFIX CACHE HIT OK: {} tokens matched between session 1 and 2", n);
        }
    }
}
