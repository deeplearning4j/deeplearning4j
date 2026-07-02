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

package org.eclipse.deeplearning4j.llm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.eval.PerplexityEvaluator;
import org.eclipse.deeplearning4j.llm.eval.PerplexityEvaluator.PerplexityResult;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;

import java.io.File;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Verifies that {@link PerplexityEvaluator#evaluate} works on a hybrid decoder
 * (Qwen3.5-0.8B) that has recurrent-state placeholders (past_gdn_state.N / past_conv_state.N)
 * and scalar GGUF inputs (position_offset, cache_position).
 *
 * <p>The vanilla singleton map {@code {input_ids → ids}} fails with:
 * <pre>
 *   Native executor: missing external input 'past_gdn_state.22' (index 472/523) … No fallback permitted
 * </pre>
 * This test asserts that the fixed {@link PerplexityEvaluator} — which now uses
 * {@link org.eclipse.deeplearning4j.llm.generation.DecoderInputBuilder#buildScoringInputMap} —
 * produces a finite, plausible perplexity value.
 *
 * <p>Preconditions (both are assumed, not hard failures):
 * <ul>
 *   <li>{@code ~/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf} must exist.</li>
 *   <li>A tokenizer JSON must exist at one of the cached tokenizer paths.</li>
 * </ul>
 *
 * <p>Each forward pass over a 32-token window takes ~2-4s on CPU; the test keeps the
 * window count small (contextLength=stride=32, ~300 char text) to stay under 30s.
 */
@Slf4j
public class TestPerplexityHybridModel {

    private static final String MODEL_CACHE_DIR =
            System.getProperty("user.home") + "/.cache/dl4j-llm-models";

    /** Cached path set by TestQwen35Pipeline (or the LLMModelDownloader). */
    private static final String GGUF_PATH =
            MODEL_CACHE_DIR + "/Qwen3.5-0.8B-Q4_K_M.gguf";

    /** Tokenizer downloaded alongside the model. */
    private static final String TOKENIZER_PATH_1 =
            MODEL_CACHE_DIR + "/qwen35-0.8B-tokenizer.json";
    private static final String TOKENIZER_PATH_2 =
            MODEL_CACHE_DIR + "/Qwen3.5-0.8B-serving/tokenizer.json";

    private static SameDiff decoder;
    private static Tokenizer tokenizer;

    @BeforeAll
    static void loadModel() throws Exception {
        File ggufFile = new File(GGUF_PATH);
        assumeTrue(ggufFile.exists(),
                "Skipping: GGUF not found at " + GGUF_PATH
                        + " — run TestQwen35Pipeline or LLMModelDownloader first");

        String tokPath = null;
        if (new File(TOKENIZER_PATH_1).exists()) {
            tokPath = TOKENIZER_PATH_1;
        } else if (new File(TOKENIZER_PATH_2).exists()) {
            tokPath = TOKENIZER_PATH_2;
        }
        assumeTrue(tokPath != null,
                "Skipping: no cached tokenizer found at " + TOKENIZER_PATH_1
                        + " or " + TOKENIZER_PATH_2);

        log.info("Loading GGUF model from: {}", GGUF_PATH);
        long t0 = System.currentTimeMillis();
        decoder = GGMLModelImport.importModel(GGUF_PATH, ConversionOptions.forInference());
        log.info("Model loaded in {}ms, {} ops", System.currentTimeMillis() - t0, decoder.ops().length);

        tokenizer = HuggingFaceTokenizer.fromFile(tokPath);
        log.info("Tokenizer loaded from: {} (vocab={})", tokPath, tokenizer.getVocabSize());
    }

    @Test
    void testPerplexityHybridModelIsFiniteAndPlausible() {
        // ~300 chars → ~60-80 tokens for Qwen3.5 BPE; contextLength=stride=32 → 1-2 windows
        String text = "The transformer architecture revolutionized natural language processing. "
                + "Modern language models use self-attention mechanisms to process sequences. "
                + "Hybrid architectures combine attention with state-space models for efficiency. "
                + "Recurrent state management allows processing arbitrarily long sequences.";

        log.info("Running perplexity evaluation on {} chars, context=32 stride=32", text.length());
        long t0 = System.currentTimeMillis();
        PerplexityResult result = PerplexityEvaluator.evaluate(decoder, tokenizer, text, 32, 32);
        log.info("Perplexity evaluation completed in {}ms", System.currentTimeMillis() - t0);

        log.info("Result: perplexity={} bpb={} tokens={} ms={}",
                result.getPerplexity(), result.getBitsPerByte(),
                result.getNumTokens(), result.getEvaluationTimeMs());

        assertTrue(Double.isFinite(result.getPerplexity()),
                "Perplexity must be finite, got: " + result.getPerplexity());
        assertTrue(result.getPerplexity() > 1.0,
                "Perplexity must be > 1, got: " + result.getPerplexity());
        assertTrue(result.getNumTokens() > 0,
                "Must have evaluated at least one token, got: " + result.getNumTokens());
    }
}
