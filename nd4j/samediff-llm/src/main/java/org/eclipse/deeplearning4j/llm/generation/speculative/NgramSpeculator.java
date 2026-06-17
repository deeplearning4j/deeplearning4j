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

package org.eclipse.deeplearning4j.llm.generation.speculative;

import java.util.ArrayList;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * N-gram based speculative token predictor for autoregressive decoding.
 *
 * <p>Instead of requiring a separate draft model, this speculator uses n-gram
 * matching on the already-generated token sequence to predict future tokens.
 * This works well for structured/repetitive outputs like DocTags, HTML, JSON,
 * code, and other formats where patterns recur.</p>
 *
 * <p>Algorithm:</p>
 * <ol>
 *   <li>Look at the last {@code ngramSize} tokens in the generated sequence</li>
 *   <li>Search for that n-gram pattern earlier in the sequence</li>
 *   <li>If found, propose the next {@code maxSpeculativeTokens} tokens that
 *       followed that pattern as speculative candidates</li>
 *   <li>The caller verifies these against the full model in a single forward pass</li>
 * </ol>
 *
 * <p>Example: if the generated sequence contains "... &lt;text&gt; Hello &lt;/text&gt; &lt;text&gt; ..."
 * and the last 2 tokens are "&lt;text&gt;", the speculator finds the earlier occurrence
 * and proposes "Hello &lt;/text&gt;" as speculative tokens.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Getter
public class NgramSpeculator {

    private final int ngramSize;
    private final int maxSpeculativeTokens;

    /**
     * Create an n-gram speculator.
     *
     * @param ngramSize size of the n-gram to match (typically 2-4)
     * @param maxSpeculativeTokens maximum number of tokens to speculate (typically 3-8)
     */
    public NgramSpeculator(int ngramSize, int maxSpeculativeTokens) {
        if (ngramSize < 1) throw new IllegalArgumentException("ngramSize must be >= 1, got " + ngramSize);
        if (maxSpeculativeTokens < 1) throw new IllegalArgumentException("maxSpeculativeTokens must be >= 1, got " + maxSpeculativeTokens);
        this.ngramSize = ngramSize;
        this.maxSpeculativeTokens = maxSpeculativeTokens;
    }

    /**
     * Create a speculator with default settings (n=3, k=5).
     */
    public NgramSpeculator() {
        this(3, 5);
    }

    /**
     * Speculate the next tokens based on n-gram matching in the generated sequence.
     *
     * <p>Searches backwards through the sequence for the most recent match of the
     * last {@code ngramSize} tokens. Returns the tokens that followed that match.</p>
     *
     * @param generatedTokens the full sequence of tokens generated so far
     * @return speculative token IDs (may be empty if no n-gram match found)
     */
    public int[] speculate(List<Integer> generatedTokens) {
        int seqLen = generatedTokens.size();
        if (seqLen < ngramSize) {
            return new int[0];
        }

        // Extract the n-gram to search for (last ngramSize tokens)
        int[] needle = new int[ngramSize];
        for (int i = 0; i < ngramSize; i++) {
            needle[i] = generatedTokens.get(seqLen - ngramSize + i);
        }

        // Search backwards for the most recent occurrence (excluding the current one)
        int searchEnd = seqLen - ngramSize;
        for (int pos = searchEnd - 1; pos >= 0; pos--) {
            boolean match = true;
            for (int j = 0; j < ngramSize; j++) {
                if (generatedTokens.get(pos + j) != needle[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // Found a match at position pos. Propose tokens starting at pos + ngramSize.
                int proposalStart = pos + ngramSize;
                int available = seqLen - proposalStart;
                // Don't include the current tail (that's what we're trying to predict beyond)
                available = Math.min(available, searchEnd - proposalStart);
                int numProposed = Math.min(available, maxSpeculativeTokens);
                if (numProposed <= 0) continue; // Match was too close to end, try earlier

                int[] proposed = new int[numProposed];
                for (int i = 0; i < numProposed; i++) {
                    proposed[i] = generatedTokens.get(proposalStart + i);
                }
                return proposed;
            }
        }

        return new int[0]; // No match found
    }

    /**
     * Speculate tokens for multiple sequences in a batch.
     *
     * <p>Returns the speculative tokens for each sequence. If a sequence has no
     * n-gram match, its entry will be empty. Only returns speculative tokens
     * up to the minimum length across all active (non-empty) sequences, so
     * the batch can be processed uniformly.</p>
     *
     * @param batchGeneratedTokens list of token lists, one per batch sequence
     * @param finished which sequences are already finished (skip speculation)
     * @return speculative tokens per sequence (may be empty arrays), and a common
     *         speculation length (0 if no speculation possible)
     */
    public SpeculationResult speculateBatch(List<List<Integer>> batchGeneratedTokens, boolean[] finished) {
        int batchSize = batchGeneratedTokens.size();
        int[][] perSequenceSpeculation = new int[batchSize][];
        int minLen = Integer.MAX_VALUE;
        int numSpeculating = 0;

        for (int i = 0; i < batchSize; i++) {
            if (finished[i]) {
                perSequenceSpeculation[i] = new int[0];
                continue;
            }
            perSequenceSpeculation[i] = speculate(batchGeneratedTokens.get(i));
            if (perSequenceSpeculation[i].length > 0) {
                minLen = Math.min(minLen, perSequenceSpeculation[i].length);
                numSpeculating++;
            }
        }

        // If not all active sequences can speculate, don't speculate at all
        // (batched verification requires all sequences to have the same length)
        int activeCount = 0;
        for (int i = 0; i < batchSize; i++) {
            if (!finished[i]) activeCount++;
        }

        if (numSpeculating < activeCount || minLen == Integer.MAX_VALUE) {
            // Can't speculate uniformly — fall back to normal decode
            return new SpeculationResult(new int[batchSize][], 0);
        }

        // Truncate all to common length
        int[][] truncated = new int[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            if (finished[i] || perSequenceSpeculation[i].length == 0) {
                truncated[i] = new int[0];
            } else {
                truncated[i] = new int[minLen];
                System.arraycopy(perSequenceSpeculation[i], 0, truncated[i], 0, minLen);
            }
        }

        return new SpeculationResult(truncated, minLen);
    }

    /**
     * Verify speculative tokens against the model's actual logits.
     *
     * <p>Given logits from a verification forward pass (which processed the
     * original token + K speculative tokens), check how many speculative tokens
     * match what the model would have actually produced via greedy decoding.</p>
     *
     * <p>The logits shape is [seqLen, vocabSize] where seqLen = 1 + K.
     * Position 0 corresponds to the real next token (model's prediction for the
     * position where speculation started). Positions 1..K are the model's
     * predictions conditioned on seeing the speculative tokens.</p>
     *
     * @param speculativeTokens the K proposed tokens
     * @param logitsPerPosition logits for each position, shape [1+K, vocabSize].
     *        Position i has logits for the token AFTER seeing tokens 0..i-1.
     * @return number of accepted speculative tokens (0 to K). The token at
     *         position {@code accepted} in logitsPerPosition is the correct
     *         next token to use (either the last accepted speculation or the
     *         model's correction).
     */
    public static int verifySpeculation(int[] speculativeTokens, float[][] logitsPerPosition) {
        // Position 0 logits predict what the model thinks the next token should be.
        // If it matches speculativeTokens[0], accept and check position 1, etc.
        int accepted = 0;
        for (int i = 0; i < speculativeTokens.length; i++) {
            // Find argmax of logitsPerPosition[i]
            float[] logits = logitsPerPosition[i];
            int modelToken = 0;
            float maxVal = logits[0];
            for (int v = 1; v < logits.length; v++) {
                if (logits[v] > maxVal) {
                    maxVal = logits[v];
                    modelToken = v;
                }
            }

            if (modelToken == speculativeTokens[i]) {
                accepted++;
            } else {
                break; // First mismatch — reject from here
            }
        }
        return accepted;
    }

    /**
     * Get the correct next token from verification logits.
     *
     * <p>After verification accepts M tokens, the correct "bonus" token is the
     * argmax of logitsPerPosition[M] (the model's prediction at the position
     * right after the last accepted token).</p>
     *
     * @param logitsAtPosition logits at the correction position
     * @return the correct next token ID
     */
    public static int getCorrectionToken(float[] logitsAtPosition) {
        int maxIdx = 0;
        float maxVal = logitsAtPosition[0];
        for (int v = 1; v < logitsAtPosition.length; v++) {
            if (logitsAtPosition[v] > maxVal) {
                maxVal = logitsAtPosition[v];
                maxIdx = v;
            }
        }
        return maxIdx;
    }

    /**
     * Result of batch speculation: per-sequence speculative tokens and common length.
     */
    @Getter
    @AllArgsConstructor
    public static class SpeculationResult {
        private final int[][] perSequenceTokens;
        private final int commonLength;

        public boolean hasSpeculation() { return commonLength > 0; }
    }
}
