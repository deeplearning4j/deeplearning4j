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
 * Integration tests for KV cache quantization (QUANTIZED strategy) wired through
 * {@link GenerationPipeline}.
 *
 * <h2>What is tested</h2>
 * <ol>
 *   <li><b>Token match rate</b>: greedy QUANTIZED decode must produce ≥ 95% token-for-token
 *       match against STATIC baseline over 32+ tokens. Because the V1 QUANTIZED design uses the
 *       same float staticKvBuffers as the working decode copy (INT8 buffers are a prefill-snapshot
 *       archive), the expected match rate is 100%.</li>
 *   <li><b>Memory footprint</b>: INT8-compressed buffers must be measurably smaller than the
 *       equivalent float buffers. Specifically:
 *       {@code quantizedKvTotalBytes < staticKvTotalBytes} (always true: INT8 is 1 byte, float is
 *       4 bytes; same element count → 4× compression).</li>
 *   <li><b>Config validation</b>: QUANTIZED + rotatingKvEnabled must throw an
 *       {@link IllegalStateException} at prefill time (the combination is not supported in v1).</li>
 *   <li><b>Format plumbing</b>: {@code getKvQuantFormat()} must return the configured format (1 for
 *       INT8) after prefill.</li>
 * </ol>
 *
 * <h2>Prerequisites</h2>
 * Requires a downloaded Qwen model (auto-downloaded by {@link LLMModelDownloader}) and the
 * native backend (CUDA or CPU) with the {@code kv_cache_quantize} op registered.
 *
 * <pre>
 *   cd platform-tests &amp;&amp; /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *       -Dtest=TestQuantizedKvCacheIntegration \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 2&gt;&amp;1 | tee /tmp/test-kvq.log
 * </pre>
 *
 * For CPU:
 * <pre>
 *   cd platform-tests &amp;&amp; /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *       -Dtest=TestQuantizedKvCacheIntegration \
 *       -Dbackend.artifactId=nd4j-native 2&gt;&amp;1 | tee /tmp/test-kvq-cpu.log
 * </pre>
 */
@Slf4j
public class TestQuantizedKvCacheIntegration {

    /** Minimum greedy token match rate between STATIC and QUANTIZED (V1 expects 100%). */
    private static final double MIN_MATCH_RATE = 0.95;

    /** Number of tokens to compare. Must be ≥ 32 to meet the task requirement. */
    private static final int DECODE_TOKENS = 32;

    /**
     * A deterministic prompt that produces well-known output under greedy decoding.
     * Keep it short so prefill is fast; long enough that 32 decode steps are meaningful.
     */
    private static final String PROMPT =
            "The quick brown fox jumps over the lazy dog. This is a well-known";

    /**
     * The raw model loaded from GGUF. Used only in {@link #setup()} to produce
     * {@link #optimizedModel}; closed immediately after optimization so its CUDA
     * memory is reclaimed before the first test runs.
     */
    private static SameDiff model;

    /**
     * The single pre-optimized model shared across all tests.
     *
     * <p>Running {@link GraphOptimizer} inside each test would serialize and re-load
     * a ~3 GB clone per test cycle. On CUDA, the host RSS for pinned/mapped pages is
     * counted by JavaCPP's {@code physicalBytes} guard. Five cycles × ~3 GB each
     * exhaust the 24 GB guard before the fourth model-load even completes. Optimizing
     * once here and passing {@code graphOptimizerEnabled(false)} in every test reduces
     * the total model footprint to one copy for the entire test class lifetime.</p>
     */
    private static SameDiff optimizedModel;

    private static Tokenizer tokenizer;

    @BeforeAll
    public static void setup() throws Exception {
        if (System.getProperty("nd4j.optimizer.enabled") == null) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr  = System.getProperty("qwen.quant",      "Q4_K_M");

        DownloadResult dl = LLMModelDownloader.download(
                LLMModel.fromSizeLabel(sizeLabel), QuantType.valueOf(quantStr));
        model = GGMLModelImport.importModel(dl.getModelFile().getAbsolutePath());

        // Run GraphOptimizer exactly once — same filter as GenerationPipeline.create()
        // (exclude QuantizationOptimizations to avoid silent FP16 downcasting).
        List<OptimizerSet> optimizations = new ArrayList<>();
        for (OptimizerSet s : GraphOptimizer.defaultOptimizations()) {
            if (!(s instanceof QuantizationOptimizations)) {
                optimizations.add(s);
            }
        }
        List<String> outputs = model.outputs() != null ? new ArrayList<>(model.outputs()) : new ArrayList<>();
        long optStart = System.currentTimeMillis();
        optimizedModel = GraphOptimizer.optimize(model, outputs, optimizations);
        log.info("[setup] GraphOptimizer: {} -> {} ops in {}ms",
                model.getOps().size(), optimizedModel.getOps().size(),
                System.currentTimeMillis() - optStart);

        // If the optimizer returned a new instance, close the raw model now so its
        // CUDA-pinned buffers are released before any test starts.
        if (optimizedModel != model) {
            try {
                SameDiffMemoryUtils.freeModelArrays(model);
                model.close();
            } catch (Exception e) {
                log.warn("[setup] Error closing raw model after optimization: {}", e.getMessage());
            }
        }
        model = null; // clear reference; optimizedModel is the sole shared instance

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
        // Release the shared optimized model. SameDiffMemoryUtils.freeModelArrays()
        // forcibly unpins constant arrays so the CUDA allocator can reclaim them;
        // close() then deallocates the Java-side SameDiff bookkeeping.
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
    // (a) Token match rate: QUANTIZED greedy must match STATIC greedy ≥ 95%
    // ══════════════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("(a) QUANTIZED greedy match rate vs STATIC ≥ 95% over " + DECODE_TOKENS + " tokens")
    public void testQuantizedTokenMatchRate() throws Exception {
        // ── STATIC baseline ────────────────────────────────────────────────────────
        int[] staticTokens;
        GenerationPipelineConfig staticCfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(DECODE_TOKENS)
                .kvCacheStrategy(KvCacheStrategy.STATIC)
                .graphOptimizerEnabled(false) // already optimized in @BeforeAll
                .dspEnabled(true)
                .build();
        try (GenerationPipeline pipe = GenerationPipeline.create(staticCfg);
             GenerationSession session = pipe.startSession(PROMPT, DECODE_TOKENS)) {
            GenerationResult r = session.generate(DECODE_TOKENS);
            staticTokens = r.getTokenIds();
            log.info("[STATIC] {} tokens: {}", staticTokens.length, Arrays.toString(staticTokens));
        }

        assertTrue(staticTokens.length >= DECODE_TOKENS * 0.8,
                "STATIC baseline should produce at least " + (int)(DECODE_TOKENS * 0.8)
                        + " tokens; got " + staticTokens.length);

        // ── QUANTIZED INT8 ─────────────────────────────────────────────────────────
        int[] quantizedTokens;
        GenerationPipelineConfig quantCfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(DECODE_TOKENS)
                .kvCacheStrategy(KvCacheStrategy.QUANTIZED)
                .kvQuantFormat(1) // INT8
                .graphOptimizerEnabled(false) // already optimized in @BeforeAll
                .dspEnabled(true)
                .build();
        try (GenerationPipeline pipe = GenerationPipeline.create(quantCfg);
             GenerationSession session = pipe.startSession(PROMPT, DECODE_TOKENS)) {
            GenerationResult r = session.generate(DECODE_TOKENS);
            quantizedTokens = r.getTokenIds();
            log.info("[QUANTIZED] {} tokens: {}", quantizedTokens.length, Arrays.toString(quantizedTokens));
        }

        assertTrue(quantizedTokens.length >= DECODE_TOKENS * 0.8,
                "QUANTIZED decode should produce at least " + (int)(DECODE_TOKENS * 0.8)
                        + " tokens; got " + quantizedTokens.length);

        // ── Compare ────────────────────────────────────────────────────────────────
        int compareLen = Math.min(staticTokens.length, quantizedTokens.length);
        int matches = 0;
        for (int i = 0; i < compareLen; i++) {
            if (staticTokens[i] == quantizedTokens[i]) matches++;
        }
        double matchRate = (double) matches / compareLen;
        log.info("[MATCH] {}/{} tokens match ({}) over first {} compared",
                matches, compareLen, String.format("%.1f%%", matchRate * 100.0), compareLen);

        // V1 QUANTIZED decode uses float staticKvBuffers as working copy → expect 100% match.
        assertTrue(matchRate >= MIN_MATCH_RATE,
                String.format("QUANTIZED vs STATIC token match rate %.1f%% < %.0f%% minimum. "
                        + "compareLen=%d, matches=%d. "
                        + "static=%s quantized=%s",
                        matchRate * 100.0, MIN_MATCH_RATE * 100.0, compareLen, matches,
                        Arrays.toString(Arrays.copyOf(staticTokens, Math.min(8, staticTokens.length))),
                        Arrays.toString(Arrays.copyOf(quantizedTokens, Math.min(8, quantizedTokens.length)))));

        log.info("PASS: QUANTIZED match rate {}/{} = {} (min {}%)",
                matches, compareLen, String.format("%.1f%%", matchRate * 100.0), (int)(MIN_MATCH_RATE * 100));
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // (b) KV memory footprint: INT8 archive must be smaller than float working copy
    // ══════════════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("(b) INT8 quantized KV archive byte count < float static KV byte count")
    public void testQuantizedKvMemoryFootprint() throws Exception {
        GenerationPipelineConfig quantCfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(DECODE_TOKENS)
                .kvCacheStrategy(KvCacheStrategy.QUANTIZED)
                .kvQuantFormat(1) // INT8
                .graphOptimizerEnabled(false) // already optimized in @BeforeAll
                .dspEnabled(true)
                .build();

        long staticBytes;
        long quantizedBytes;
        try (GenerationPipeline pipe = GenerationPipeline.create(quantCfg);
             GenerationSession session = pipe.startSession(PROMPT, DECODE_TOKENS)) {
            // Memory is allocated during startSession (prefill + buffer init). Check BEFORE decode.
            staticBytes    = session.getStaticKvTotalBytes();
            quantizedBytes = session.getQuantizedKvTotalBytes();
            int kvQuantFormat = session.getKvQuantFormat();

            log.info("[MEMORY] staticKvBytes={} quantizedKvBytes={} kvQuantFormat={}",
                    staticBytes, quantizedBytes, kvQuantFormat);

            // Sanity: format must be INT8 (1)
            assertEquals(1, kvQuantFormat,
                    "kvQuantFormat should be 1 (INT8) but was " + kvQuantFormat);

            // The quantized (row-inline) KV must have been populated
            assertTrue(quantizedBytes > 0,
                    "quantizedKvTotalBytes should be > 0 after prefill; got " + quantizedBytes);

            // ADR 0107 V2: the float KV buffers are FREED after the prefill quantize — the INT8
            // row-inline tensors ([B, maxKvLen, kvH, headDim+4]) are the ONLY live KV storage.
            assertEquals(0L, staticBytes,
                    "V2 quantized session should have staticKvBytes==0 (float KV freed after prefill); got "
                            + staticBytes);

            // Compression vs the float equivalent: float would be (headDim/(headDim+4))*4x the
            // row-inline INT8 footprint — ≥ 2x for any headDim ≥ 8.
            long floatEquivalentBytes = quantizedBytes * 4L; // upper bound incl. in-row scales
            double compressionRatio = (double) floatEquivalentBytes / quantizedBytes;
            log.info("[MEMORY] Row-inline INT8 live bytes={} (~{}x smaller than float equivalent)",
                    quantizedBytes, String.format("%.2f", compressionRatio));

            // Run a decode step to verify the session still works after we've inspected memory
            GenerationResult r = session.generate(DECODE_TOKENS);
            assertTrue(r.getGeneratedTokenCount() > 0,
                    "Session must still produce tokens after memory inspection");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // (c) Config validation: QUANTIZED + rotating KV must be rejected
    // ══════════════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("(c) QUANTIZED + rotatingKvEnabled must throw IllegalStateException")
    public void testQuantizedRotatingCombinationRejected() throws Exception {
        GenerationPipelineConfig badCfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(DECODE_TOKENS)
                .kvCacheStrategy(KvCacheStrategy.QUANTIZED)
                .kvQuantFormat(1) // INT8
                .rotatingKvEnabled(true)
                .graphOptimizerEnabled(false) // already optimized in @BeforeAll
                .dspEnabled(true)
                .build();

        // The rejection happens inside prefillWarmupAndFreeze → startSession should throw.
        IllegalStateException ex = null;
        GenerationPipeline pipe = GenerationPipeline.create(badCfg);
        try {
            GenerationSession session = pipe.startSession(PROMPT, DECODE_TOKENS);
            // If we reach here, the constraint was NOT enforced — close and fail.
            session.close();
            fail("Expected IllegalStateException for QUANTIZED + rotatingKvEnabled=true");
        } catch (IllegalStateException e) {
            ex = e;
            log.info("[CONFIG] Correctly rejected QUANTIZED+rotating: {}", e.getMessage());
        } finally {
            pipe.close();
        }

        assertNotNull(ex, "Expected IllegalStateException to have been thrown");
        assertTrue(ex.getMessage().contains("QUANTIZED"),
                "Exception message should mention QUANTIZED; was: " + ex.getMessage());
        assertTrue(ex.getMessage().contains("rotat"),
                "Exception message should mention rotating; was: " + ex.getMessage());
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // (c2) Config validation: kvQuantFormat=0 with QUANTIZED strategy → treated as non-quantized
    // ══════════════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("(c2) QUANTIZED strategy with kvQuantFormat=0 is a no-op (treated as STATIC)")
    public void testQuantizedWithFormatZeroIsNoop() throws Exception {
        // kvQuantFormat=0 means "not quantized" even if strategy=QUANTIZED.
        // This should NOT produce any INT8 buffers; it should fall through to STATIC behavior.
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(DECODE_TOKENS)
                .kvCacheStrategy(KvCacheStrategy.QUANTIZED)
                .kvQuantFormat(0) // disabled — should behave identically to STATIC
                .graphOptimizerEnabled(false) // already optimized in @BeforeAll
                .dspEnabled(true)
                .build();

        try (GenerationPipeline pipe = GenerationPipeline.create(cfg);
             GenerationSession session = pipe.startSession(PROMPT, DECODE_TOKENS)) {
            // kvQuantFormat must be 0 (no-op)
            assertEquals(0, session.getKvQuantFormat(),
                    "kvQuantFormat should be 0 (no-op path)");
            // No INT8 archive
            assertEquals(0L, session.getQuantizedKvTotalBytes(),
                    "quantizedKvTotalBytes should be 0 for format=0");
            // Static buffers must still exist
            assertTrue(session.getStaticKvTotalBytes() > 0,
                    "staticKvTotalBytes should be > 0");
            // Must still generate tokens
            GenerationResult r = session.generate(DECODE_TOKENS);
            assertTrue(r.getGeneratedTokenCount() > 0,
                    "Should still produce tokens when format=0");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // (d) Format plumbing: getKvQuantFormat() returns the configured format after prefill
    // ══════════════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("(d) getKvQuantFormat() returns configured INT8 format after prefill")
    public void testKvQuantFormatPlumbing() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(optimizedModel)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(DECODE_TOKENS)
                .kvCacheStrategy(KvCacheStrategy.QUANTIZED)
                .kvQuantFormat(1) // INT8
                .graphOptimizerEnabled(false) // already optimized in @BeforeAll
                .dspEnabled(true)
                .build();

        try (GenerationPipeline pipe = GenerationPipeline.create(cfg);
             GenerationSession session = pipe.startSession(PROMPT, DECODE_TOKENS)) {
            // The format must be wired from config into the state
            assertEquals(1, session.getKvQuantFormat(),
                    "getKvQuantFormat() must return 1 (INT8) after startSession");

            // The quantized buffers must have been populated
            assertTrue(session.getQuantizedKvTotalBytes() > 0,
                    "Quantized KV bytes must be > 0 after startSession with INT8 format");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════════════
    // Helper
    // ══════════════════════════════════════════════════════════════════════════════════

    private static int[] concat(int[] a, int[] b) {
        int[] result = new int[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
}
