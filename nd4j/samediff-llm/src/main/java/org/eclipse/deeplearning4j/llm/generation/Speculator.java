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

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for speculative token predictors in autoregressive decoding.
 *
 * <p>Speculative decoding generates multiple candidate tokens ahead of the target model,
 * then verifies them in a single forward pass. This amortizes the cost of sequential
 * token generation across multiple positions, providing throughput improvements
 * proportional to the acceptance rate.</p>
 *
 * <p>Implementations include:</p>
 * <ul>
 *   <li>{@link NgramSpeculator} - N-gram pattern matching on generated history (no model needed)</li>
 *   <li>{@link DraftModelSpeculator} - Smaller SameDiff model as draft for speculative decoding</li>
 * </ul>
 *
 * <p>Future implementations may include EAGLE (feature-level autoregression),
 * Medusa (multi-head parallel prediction), and other speculative strategies.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public interface Speculator {

    /**
     * Generate speculative token predictions based on the sequence generated so far.
     *
     * <p>Returns up to {@link #getMaxSpeculativeTokens()} candidate tokens that are
     * likely to follow the given sequence. The caller will verify these against the
     * target model's actual logits in a single batched forward pass.</p>
     *
     * @param generatedTokens the full sequence of token IDs generated so far
     * @return array of speculative token IDs (may be empty if no prediction is possible)
     */
    int[] speculate(int[] generatedTokens);

    /**
     * Verify speculative tokens against the target model's logits.
     *
     * <p>Given logits from a verification forward pass that processed the original
     * token plus K speculative tokens, determines how many speculative tokens the
     * target model agrees with via greedy decoding. Returns the count of accepted
     * tokens and a correction token (the model's prediction at the first rejected
     * position).</p>
     *
     * <p>The logits shape is {@code [1+K, vocabSize]} where position 0 corresponds to the
     * model's prediction for the first speculative position, and position K is the
     * model's prediction conditioned on all K speculative tokens.</p>
     *
     * @param speculativeTokens the K proposed speculative token IDs
     * @param logitsPerPosition logits for each position, shape {@code [1+K, vocabSize]}
     * @return a {@link SpeculationResult} with the accepted count and correction token
     */
    SpeculationResult verify(int[] speculativeTokens, INDArray logitsPerPosition);

    /**
     * Get a human-readable name for this speculator implementation.
     *
     * @return the speculator name (e.g., "ngram-3", "draft-smollm2-135m")
     */
    String getName();

    /**
     * Get the maximum number of speculative tokens this speculator can produce per step.
     *
     * @return maximum speculative tokens (K)
     */
    int getMaxSpeculativeTokens();

    /**
     * Speculate tokens for multiple sequences in a batch.
     *
     * <p>Default implementation calls {@link #speculate(int[])} for each sequence
     * independently. Subclasses that support batched draft model inference should
     * override this for efficiency.</p>
     *
     * @param generatedTokensBatch array of token sequences, one per batch element
     * @return array of speculative token arrays, one per batch element
     */
    default int[][] speculateBatch(int[][] generatedTokensBatch) {
        int[][] results = new int[generatedTokensBatch.length][];
        for (int i = 0; i < generatedTokensBatch.length; i++) {
            results[i] = speculate(generatedTokensBatch[i]);
        }
        return results;
    }

    /**
     * Result of verifying speculative tokens against the target model.
     */
    class SpeculationResult {
        /** Number of speculative tokens accepted (0 to K). */
        public final int acceptedCount;

        /**
         * The correct next token at the first rejected position. This is the argmax
         * of logits at position {@code acceptedCount} -- either the model's correction
         * after the last accepted token, or the model's prediction when all tokens
         * were rejected.
         */
        public final int correctionToken;

        public SpeculationResult(int acceptedCount, int correctionToken) {
            this.acceptedCount = acceptedCount;
            this.correctionToken = correctionToken;
        }

        @Override
        public String toString() {
            return "SpeculationResult{accepted=" + acceptedCount + ", correction=" + correctionToken + "}";
        }
    }
}
