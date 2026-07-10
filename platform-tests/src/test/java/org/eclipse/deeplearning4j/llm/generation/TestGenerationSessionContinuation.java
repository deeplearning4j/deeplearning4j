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

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for {@link GenerationPipeline} "continue generation" (resume / incremental decode)
 * via {@link GenerationSession}.
 *
 * <p>The central acceptance test proves that decoding spread over multiple session calls reuses the
 * retained in-graph KV cache rather than re-prefilling: {@code generate(20)} then
 * {@code continueGeneration(20)} is token-for-token identical to a single {@code generate(40)} under
 * greedy decoding.</p>
 *
 * <p>Uses the smallest single-model GGUF (Qwen3.5 0.8B, Q4_K_M) — the same model exercised by
 * {@code TestQwen35Pipeline}. Requires the CUDA backend for the native DSP decode path:</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test -Dtest=TestGenerationSessionContinuation \
 *       -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
public class TestGenerationSessionContinuation {

    private static final String PROMPT =
            "Once upon a time, in a land far away, there lived a curious inventor who";

    private static SameDiff model;
    private static Tokenizer tokenizer;
    private static GenerationPipeline pipeline;

    @BeforeAll
    public static void setup() throws Exception {
        if (System.getProperty("nd4j.optimizer.enabled") == null) {
            System.setProperty("nd4j.optimizer.enabled", "true");
        }
        String sizeLabel = System.getProperty("qwen.model.size", "0.8B");
        String quantStr = System.getProperty("qwen.quant", "Q4_K_M");

        DownloadResult dl = LLMModelDownloader.download(LLMModel.fromSizeLabel(sizeLabel), QuantType.valueOf(quantStr));
        model = GGMLModelImport.importModel(dl.getModelFile().getAbsolutePath());

        String tokenizerPath = System.getProperty("qwen.tokenizer.path");
        if (tokenizerPath != null && !tokenizerPath.isEmpty()) {
            tokenizer = HuggingFaceTokenizer.fromFile(tokenizerPath);
        } else {
            String tokenizerUrl = "https://huggingface.co/Qwen/Qwen3.5-" + sizeLabel + "/resolve/main/tokenizer.json";
            File tf = LLMModelDownloader.downloadCustom(tokenizerUrl, "qwen35-" + sizeLabel + "-tokenizer.json");
            tokenizer = HuggingFaceTokenizer.fromFile(tf.getAbsolutePath());
        }

        // One shared pipeline (greedy) reused across tests; it does NOT own the pre-loaded model.
        GenerationPipelineConfig cfg = GenerationPipelineConfig.builder()
                .decoder(model)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(40)
                .graphOptimizerEnabled(true)
                .dspEnabled(true)
                .build();
        pipeline = GenerationPipeline.create(cfg);
    }

    @AfterAll
    public static void teardown() {
        if (pipeline != null) pipeline.close();
    }

    /**
     * ACCEPTANCE: generate(20) then continueGeneration(20) equals generate(40), token-for-token
     * (greedy) — proving continuation reuses the KV cache and is not a re-prefill.
     */
    @Test
    @DisplayName("continue(20) after generate(20) == generate(40), greedy, no re-prefill")
    public void testContinuationEqualsSingleShotGreedy() {
        // Baseline: a single 40-token greedy generation.
        GenerationResult single = pipeline.generate(PROMPT, 40);
        int[] tokens40 = single.getTokenIds();
        log.info("single-shot: {} tokens, finish={}", tokens40.length, single.getFinishReason());
        assertTrue(tokens40.length >= 20,
                "model must generate at least 20 tokens for this test; got " + tokens40.length);

        // Session on the SAME pipeline/model: generate 20, then continue 20 more.
        int[] combined;
        GenerationResult r1;
        try (GenerationSession session = pipeline.startSession(PROMPT, 40)) {
            assertEquals(0L, single.getSessionId(), "one-shot result must carry no session id");
            r1 = session.generate(20);
            log.info("session r1: {} tokens, finish={}, cachePos={}",
                    r1.getTokenIds().length, r1.getFinishReason(), session.getCachePosition());

            if (r1.isComplete()) {
                // Model hit EOS within the first 20 tokens — single-shot must have stopped identically.
                combined = r1.getTokenIds();
            } else {
                assertEquals(GenerationResult.FinishReason.MAX_TOKENS, r1.getFinishReason(),
                        "first 20-token generation should be truncated (MAX_TOKENS)");
                assertEquals(20, r1.getGeneratedTokenCount(), "first call should return exactly 20 tokens");
                assertEquals(session.getSessionId(), r1.getSessionId(), "session results carry the session id");

                GenerationResult r2 = session.continueGeneration(20);
                log.info("session r2: {} tokens, finish={}", r2.getTokenIds().length, r2.getFinishReason());
                assertTrue(r2.getGeneratedTokenCount() > 0, "continuation must produce tokens");

                combined = concat(r1.getTokenIds(), r2.getTokenIds());
                // The session's cumulative view should match the concatenation.
                assertArrayEquals(combined, session.getAllTokens(),
                        "session.getAllTokens() must equal the concatenation of per-call token ids");
            }
        }

        // Identity: session tokens == single-shot tokens over the common prefix (both greedy).
        int n = Math.min(tokens40.length, combined.length);
        assertTrue(n >= 20, "expected at least 20 comparable tokens");
        assertArrayEquals(Arrays.copyOf(tokens40, n), Arrays.copyOf(combined, n),
                "continue(20) after generate(20) must equal generate(40) token-for-token (greedy).\n"
                        + "single=" + Arrays.toString(Arrays.copyOf(tokens40, n)) + "\n"
                        + "session=" + Arrays.toString(Arrays.copyOf(combined, n)));
        log.info("IDENTITY OK: {} tokens matched across the generate/continue seam", n);
    }

    /**
     * continueToCompletion() must always terminate — bounded by KV capacity — and report a clean
     * terminal finish reason (EOS, MAX_TOKENS = context full, or REPETITION). No unbounded loop.
     */
    @Test
    @DisplayName("continueToCompletion terminates at/below capacity")
    public void testContinueToCompletionTerminates() {
        int capacity = 24;
        try (GenerationSession session = pipeline.startSession(PROMPT, capacity)) {
            GenerationResult first = session.generate(8);
            assertTrue(first.getGeneratedTokenCount() > 0);

            GenerationResult loop = session.continueToCompletion(8);
            log.info("continueToCompletion: +{} tokens, finish={}, remainingCapacity={}",
                    loop.getGeneratedTokenCount(), loop.getFinishReason(), session.getRemainingCapacity());

            assertNotNull(loop.getFinishReason());
            // Total generated (initial + loop) must never exceed the static capacity.
            assertTrue(session.getAllTokens().length <= capacity,
                    "generated tokens (" + session.getAllTokens().length + ") must not exceed capacity " + capacity);
            // Loop stopped for a real reason (not still runnable with capacity left, unless EOS).
            boolean terminal = session.isEosReached()
                    || session.getRemainingCapacity() == 0
                    || loop.getFinishReason() == GenerationResult.FinishReason.REPETITION;
            assertTrue(terminal, "continueToCompletion must stop on EOS, buffer-full, or repetition");
        }
    }

    /**
     * Lifecycle: repeated continuation works, close() releases the session, and post-close use throws;
     * a fresh session can be opened afterward (no leaked active-session state).
     */
    @Test
    @DisplayName("session lifecycle: repeated continue, close, reopen")
    public void testSessionLifecycleAndClose() {
        GenerationSession session = pipeline.startSession(PROMPT, 48);
        session.generate(6);
        for (int i = 0; i < 4 && !session.isEosReached() && session.getRemainingCapacity() > 0; i++) {
            GenerationResult r = session.continueGeneration(6);
            assertNotNull(r);
        }
        session.close();

        // Post-close use must fail fast.
        assertThrows(IllegalStateException.class, () -> session.continueGeneration(4),
                "continue after close must throw");

        // Active session was cleared → a one-shot generate and a new session both work.
        GenerationResult afterClose = pipeline.generate(PROMPT, 4);
        assertTrue(afterClose.getGeneratedTokenCount() > 0, "one-shot generate must work after session close");
        try (GenerationSession reopened = pipeline.startSession(PROMPT, 16)) {
            assertTrue(reopened.generate(4).getGeneratedTokenCount() > 0, "reopened session must work");
        }
    }

    /**
     * append(): injecting given tokens (no sampling) advances the cache position and becomes part of
     * the cumulative sequence; continuation then resumes after the injected tokens.
     */
    @Test
    @DisplayName("append injects tokens then continuation resumes after them")
    public void testAppendThenContinue() {
        try (GenerationSession session = pipeline.startSession(PROMPT, 48)) {
            session.generate(8);
            int posBefore = session.getCachePosition();
            int tokensBefore = session.getAllTokens().length;

            int[] nudge = tokenizer.encode(" Continue:", false).getIds();
            session.append(nudge);

            assertEquals(tokensBefore + nudge.length, session.getAllTokens().length,
                    "appended tokens must be part of the cumulative sequence");
            assertEquals(posBefore + nudge.length, session.getCachePosition(),
                    "cache position must advance by the number of appended tokens");
            // The tail of the cumulative tokens must be exactly the injected nudge.
            int[] all = session.getAllTokens();
            assertArrayEquals(nudge, Arrays.copyOfRange(all, all.length - nudge.length, all.length),
                    "the appended tokens must appear verbatim at the tail");

            GenerationResult after = session.continueGeneration(8);
            assertTrue(after.getGeneratedTokenCount() > 0, "continuation after append must produce tokens");
        }
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
