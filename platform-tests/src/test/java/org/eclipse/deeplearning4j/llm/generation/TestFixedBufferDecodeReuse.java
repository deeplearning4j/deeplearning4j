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
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression coverage for the fixed-buffer decode REUSE forward-fix (ADR 0105 follow-up): a
 * {@link GenerationPipeline} configured with {@code maxPrefillLength}/{@code maxKvCacheLength}
 * (the "fixed-buffer fast path") caches its {@link InGraphKvState} across one-shot {@code generate()}
 * calls and reuses the frozen DSP plan + captured CUDA graph + stable-address KV / recurrent /
 * prefill buffers instead of re-warming every generate. This is the scenario no pre-existing test
 * covered (per the DSP/decode test survey): many generates on ONE fixed-buffer pipeline.
 *
 * <p>The fix must be behavior-preserving. This class pins three invariants:</p>
 * <ol>
 *   <li><b>reuse == fresh</b> — generate 0 builds the state fresh (no reuse); generates 1..N reuse it.
 *       Under greedy decoding on the same prompt every reused generate must be token-identical to the
 *       fresh one. A divergence means the in-place buffer refill corrupted the replay.</li>
 *   <li><b>not degenerate</b> — the fresh generate must not collapse to a short repeating loop (the
 *       symptom of the pre-fix demotion bug), guarded by a distinct-token floor.</li>
 *   <li><b>re-prefill overwrites</b> — a different prompt in the middle must produce different output,
 *       and returning to the first prompt must reproduce its original output (proves the prefill inputs
 *       are re-written in place, not left stale — the gen-3+ failure mode the fix specifically targets).</li>
 * </ol>
 *
 * <p>Uses one isolated Qwen3.5 0.8B (Q4_K_M) model dedicated to fixed-buffer pipelines, so there is no
 * variable-buffer pipeline sharing the decoder's executor (which would introduce a cross-config
 * premature-freeze artifact unrelated to the reuse path). CUDA + model download required.</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test -Dtest=TestFixedBufferDecodeReuse -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
public class TestFixedBufferDecodeReuse {

    private static final String PROMPT =
            "Once upon a time, in a land far away, there lived a curious inventor who";
    private static final String PROMPT_B =
            "The history of scientific discovery shows that the most important breakthroughs often";

    /** Decode length per generate — long enough to expose a degenerate loop, short enough to stay fast. */
    private static final int N = 16;
    /** Number of generates on one pipeline (gen 0 fresh + gens 1..7 reused). */
    private static final int GENERATES = 8;
    /** Distinct-token floor: the pre-fix degeneration collapsed to a ~4-token loop. */
    private static final int MIN_DISTINCT = 5;

    private static SameDiff model;
    private static Tokenizer tokenizer;
    private static String modelPath;

    @BeforeAll
    public static void setup() throws Exception {
        if (System.getProperty("nd4j.optimizer.enabled") == null) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr = System.getProperty("qwen.quant", "Q4_K_M");

        DownloadResult dl = LLMModelDownloader.download(LLMModel.fromSizeLabel(sizeLabel), QuantType.valueOf(quantStr));
        modelPath = dl.getModelFile().getAbsolutePath();
        model = GGMLModelImport.importModel(modelPath);

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
        model = null;
        tokenizer = null;
    }

    private static GenerationPipeline fixedBufferPipeline() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(N)
                .maxPrefillLength(64)
                .maxKvCacheLength(128)
                .graphOptimizerEnabled(true)
                .dspEnabled(true)
                .build();
        return GenerationPipeline.create(cfg);
    }

    /** Variable-buffer (no fixed-buffer fast path) pipeline — the known-correct reference decoder. */
    private static GenerationPipeline variablePipeline() throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(N)
                .graphOptimizerEnabled(true)
                .dspEnabled(true)
                .build();
        return GenerationPipeline.create(cfg);
    }

    private static int distinct(int[] a) {
        Set<Integer> s = new HashSet<>();
        for (int x : a) s.add(x);
        return s.size();
    }

    /** Greedy pipeline over a caller-supplied model, fixed-buffer or variable-buffer. */
    private static GenerationPipeline pipelineFor(SameDiff m, boolean fixed) throws Exception {
        GenerationPipelineConfig cfg = fixed
                ? GenerationPipelineConfig.builder().decoder(m).tokenizer(tokenizer)
                    .samplingConfig(SamplingConfig.greedy()).maxNewTokens(N)
                    .maxPrefillLength(64).maxKvCacheLength(128)
                    .graphOptimizerEnabled(true).dspEnabled(true).build()
                : GenerationPipelineConfig.builder().decoder(m).tokenizer(tokenizer)
                    .samplingConfig(SamplingConfig.greedy()).maxNewTokens(N)
                    .graphOptimizerEnabled(true).dspEnabled(true).build();
        return GenerationPipeline.create(cfg);
    }

    /**
     * DIAGNOSTIC (not a gate): confirm the fixed-buffer-vs-variable divergence is REAL and not a
     * shared-executor contamination artifact — each config runs on its OWN freshly-imported model, so
     * the executors are fully isolated. If fixedFresh still != variableRef here, the fixed-buffer path
     * has a genuine accuracy gap vs the reference decoder (independent of the reuse work).
     */
    @Test
    @DisplayName("DIAG: isolated fixed-buffer vs variable-buffer (separate model imports)")
    public void diagIsolatedFixedVsVariable() throws Exception {
        SameDiff mV = GGMLModelImport.importModel(modelPath);
        int[] variableRef;
        GenerationPipeline varPipe = pipelineFor(mV, false);
        try {
            variableRef = varPipe.generate(PROMPT, N).getTokenIds();
        } finally {
            varPipe.close();
        }
        // Release the variable model before importing the fixed one — three resident 0.8B models
        // (shared setup model + mV + mF) exceed the 24GB physical-memory cap.
        SameDiffMemoryUtils.freeModelArrays(mV);
        mV = null;

        SameDiff mF = GGMLModelImport.importModel(modelPath);
        int[] fixedFresh;
        GenerationPipeline fixPipe = pipelineFor(mF, true);
        try {
            fixedFresh = fixPipe.generate(PROMPT, N).getTokenIds();
        } finally {
            fixPipe.close();
        }

        log.info("[ISO] variableRef = {}", Arrays.toString(variableRef));
        log.info("[ISO] fixedFresh  = {}", Arrays.toString(fixedFresh));
        log.info("[ISO] fixed==variable? {}   (divergeIdx={})",
                Arrays.equals(fixedFresh, variableRef), firstDiff(fixedFresh, variableRef));
    }

    /** Index of the first differing element, or -1 if equal up to the shorter length. */
    private static int firstDiff(int[] a, int[] b) {
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) if (a[i] != b[i]) return i;
        return (a.length == b.length) ? -1 : n;
    }

    /** Fixed-buffer greedy pipeline with explicit padding / KV-cache sizes. */
    private static GenerationPipeline pipelineForPad(SameDiff m, int maxPrefill, int maxKv) throws Exception {
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(m).tokenizer(tokenizer).samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(N).maxPrefillLength(maxPrefill).maxKvCacheLength(maxKv)
                .graphOptimizerEnabled(true).dspEnabled(true).build();
        return GenerationPipeline.create(cfg);
    }

    /**
     * DISCRIMINATOR (numerical vs structural): the prompt is ~15 tokens. Padding it to 64 (KV=128) vs
     * 96 (KV=192) MUST produce identical greedy output — the padding positions are masked, so the real
     * tokens attend only to 0..14 either way and the decode starts at the same real position. The ONLY
     * difference is buffer/plan SHAPE, which changes DSP kernel/tile/TF32 selection but not the math.
     * <ul>
     *   <li>pad64 != pad96  ⇒ the fixed-buffer output depends on a logically-irrelevant shape ⇒ the
     *       divergence is NUMERICAL (kernel selection tips a near-tie), not a structural bug.</li>
     *   <li>pad64 == pad96  ⇒ the fixed-buffer path is shape-robust ⇒ the fixed-vs-variable gap comes
     *       from something structural (dig further: mask/position/KV handling).</li>
     * </ul>
     * Isolated model imports (free between) to avoid shared-executor contamination and the 24GB cap.
     */
    @Test
    @DisplayName("DIAG: fixed-buffer output invariance to padding length (numerical vs structural)")
    public void diagPaddingSensitivity() throws Exception {
        SameDiff m1 = GGMLModelImport.importModel(modelPath);
        int[] pad64;
        GenerationPipeline p1 = pipelineForPad(m1, 64, 128);
        try {
            pad64 = p1.generate(PROMPT, N).getTokenIds();
        } finally {
            p1.close();
        }
        SameDiffMemoryUtils.freeModelArrays(m1);
        m1 = null;

        SameDiff m2 = GGMLModelImport.importModel(modelPath);
        int[] pad96;
        GenerationPipeline p2 = pipelineForPad(m2, 96, 192);
        try {
            pad96 = p2.generate(PROMPT, N).getTokenIds();
        } finally {
            p2.close();
        }
        SameDiffMemoryUtils.freeModelArrays(m2);
        m2 = null;

        log.info("[PAD] pad64/kv128 = {}", Arrays.toString(pad64));
        log.info("[PAD] pad96/kv192 = {}", Arrays.toString(pad96));
        log.info("[PAD] equal? {}   (divergeIdx={}) -> {}",
                Arrays.equals(pad64, pad96), firstDiff(pad64, pad96),
                Arrays.equals(pad64, pad96) ? "shape-robust (structural gap elsewhere)"
                                            : "shape-sensitive => NUMERICAL near-tie");
    }

    /**
     * DIAGNOSTIC (not a gate): establish ground truth for the reuse divergence by comparing fixed-buffer
     * fresh (gen 0) and reused (gen 1) against the variable-buffer reference decoder. Runs the fixed
     * pipeline FIRST (clean, fresh executor) then the variable reference (its path resets the executor),
     * so both references are uncontaminated. Logs the three token arrays and the equality matrix.
     */
    @Test
    @DisplayName("DIAG: fixed fresh/reuse vs variable-buffer reference")
    public void diagReferenceComparison() throws Exception {
        int[] fixedFresh;
        int[] fixedReuse;
        GenerationPipeline fixedPipe = fixedBufferPipeline();
        try {
            fixedFresh = fixedPipe.generate(PROMPT, N).getTokenIds();
            fixedReuse = fixedPipe.generate(PROMPT, N).getTokenIds();
        } finally {
            fixedPipe.close();
        }

        int[] variableRef;
        GenerationPipeline varPipe = variablePipeline();
        try {
            variableRef = varPipe.generate(PROMPT, N).getTokenIds();
        } finally {
            varPipe.close();
        }

        log.info("[DIAG] variableRef = {}", Arrays.toString(variableRef));
        log.info("[DIAG] fixedFresh  = {}", Arrays.toString(fixedFresh));
        log.info("[DIAG] fixedReuse  = {}", Arrays.toString(fixedReuse));
        log.info("[DIAG] fresh==ref? {}   reuse==ref? {}   fresh==reuse? {}",
                Arrays.equals(fixedFresh, variableRef),
                Arrays.equals(fixedReuse, variableRef),
                Arrays.equals(fixedFresh, fixedReuse));
    }

    /**
     * Core gate: gen 0 (fresh, no cached state) then gens 1..7 (each reusing the cached frozen state)
     * must be token-identical under greedy decoding on the same prompt, and gen 0 must not be degenerate.
     */
    @Test
    @DisplayName("fixed-buffer reuse across generates is consistent and coherent (greedy)")
    public void reuseAcrossGeneratesIsConsistentAndCoherent() throws Exception {
        GenerationPipeline pipe = fixedBufferPipeline();
        try {
            int[] gen0 = pipe.generate(PROMPT, N).getTokenIds();      // fresh — builds + freezes the plan
            assertTrue(gen0.length >= 8,
                    "gen 0 produced too few tokens (" + gen0.length + "): " + Arrays.toString(gen0));
            assertTrue(distinct(gen0) >= MIN_DISTINCT,
                    "gen 0 looks degenerate (only " + distinct(gen0) + " distinct tokens): " + Arrays.toString(gen0));

            for (int g = 1; g < GENERATES; g++) {
                int[] genN = pipe.generate(PROMPT, N).getTokenIds();  // reuse — replays the frozen plan
                assertArrayEquals(gen0, genN,
                        "reuse gen " + g + " diverged from fresh gen 0 — in-place buffer refill corrupted "
                        + "decode.\n  fresh=" + Arrays.toString(gen0) + "\n  gen" + g + "=" + Arrays.toString(genN));
            }
            log.info("[fixed-buffer-reuse] {} generates all matched, {} distinct tokens: {}",
                    GENERATES, distinct(gen0), Arrays.toString(gen0));
        } finally {
            pipe.close();
        }
    }

    /**
     * Re-prefill overwrite: interleave a different prompt. B must differ from A, and A after B must
     * reproduce A — proving the prefill input tensors are re-written in place (not left stale), which
     * is the gen-3+ silent-degeneration failure mode the reuse fix specifically has to handle.
     */
    @Test
    @DisplayName("fixed-buffer reuse re-prefills correctly on a prompt change (A, B, A)")
    public void reusePreservesCorrectnessAcrossPromptChange() throws Exception {
        GenerationPipeline pipe = fixedBufferPipeline();
        try {
            int[] a1 = pipe.generate(PROMPT, N).getTokenIds();       // fresh
            int[] b  = pipe.generate(PROMPT_B, N).getTokenIds();      // reuse + re-prefill B
            int[] a2 = pipe.generate(PROMPT, N).getTokenIds();       // reuse + re-prefill A again

            assertArrayEquals(a1, a2,
                    "re-prefill of prompt A after B diverged — prefill-input overwrite is stale.\n"
                    + "  A(first)=" + Arrays.toString(a1) + "\n  A(after B)=" + Arrays.toString(a2));
            assertFalse(Arrays.equals(a1, b),
                    "different prompts produced identical output — re-prefill is not overwriting the KV.\n"
                    + "  A=" + Arrays.toString(a1) + "\n  B=" + Arrays.toString(b));
        } finally {
            pipe.close();
        }
    }

    /**
     * The original handoff scenario: many generates on one fixed-buffer pipeline with a sampling-config
     * swap in the middle. Greedy must remain reproducible after a creative detour on the reused plan,
     * and no demotion / exception may occur across the run.
     */
    @Test
    @DisplayName("fixed-buffer reuse survives a sampling-config swap (greedy→creative→greedy)")
    public void reuseSurvivesConfigSwap() throws Exception {
        GenerationPipeline pipe = fixedBufferPipeline();
        try {
            int[] g1 = pipe.generate(PROMPT, N).getTokenIds();       // greedy, fresh

            pipe.setSamplingConfig(SamplingConfig.creative());
            int[] creative = pipe.generate(PROMPT, N).getTokenIds(); // sampled, reuse
            assertTrue(creative.length > 0, "creative generate produced no tokens");

            pipe.setSamplingConfig(SamplingConfig.greedy());
            int[] g2 = pipe.generate(PROMPT, N).getTokenIds();       // greedy, reuse
            assertArrayEquals(g1, g2,
                    "greedy not reproducible after a creative detour on the reused plan.\n"
                    + "  g1=" + Arrays.toString(g1) + "\n  g2=" + Arrays.toString(g2));
        } finally {
            pipe.close();
        }
    }
}
