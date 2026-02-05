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

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Tracks state for batch text generation with multiple sequences.
 *
 * <p>Each sequence in the batch has its own:
 * <ul>
 *   <li>Generated token list</li>
 *   <li>Finished flag (hit EOS or max tokens)</li>
 *   <li>Finish reason</li>
 * </ul>
 *
 * <p>All sequences share the same KV cache (batched along dim 0) and
 * are processed together through the decoder for efficiency.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Getter
public class BatchGenerationState {

    private final int batchSize;
    private final List<List<Integer>> generatedTokens;
    private final boolean[] finished;
    private final GenerationResult.FinishReason[] finishReasons;
    private final long[] firstTokenLatencyNanos;
    private final int eosTokenId;

    /**
     * Create a new batch generation state.
     *
     * @param batchSize number of sequences in the batch
     * @param eosTokenId end-of-sequence token ID
     */
    public BatchGenerationState(int batchSize, int eosTokenId) {
        this.batchSize = batchSize;
        this.eosTokenId = eosTokenId;
        this.generatedTokens = new ArrayList<>(batchSize);
        this.finished = new boolean[batchSize];
        this.finishReasons = new GenerationResult.FinishReason[batchSize];
        this.firstTokenLatencyNanos = new long[batchSize];

        for (int i = 0; i < batchSize; i++) {
            generatedTokens.add(new ArrayList<>());
            finishReasons[i] = GenerationResult.FinishReason.MAX_TOKENS;
        }
    }

    /**
     * Record generated tokens for this step.
     *
     * @param tokenIds array of token IDs, one per sequence
     * @param stepNanos nanoseconds since generation started
     */
    public void recordTokens(int[] tokenIds, long stepNanos) {
        for (int i = 0; i < batchSize; i++) {
            if (!finished[i]) {
                int tokenId = tokenIds[i];
                generatedTokens.get(i).add(tokenId);

                // Record first token latency
                if (generatedTokens.get(i).size() == 1) {
                    firstTokenLatencyNanos[i] = stepNanos;
                }

                // Check for EOS
                if (tokenId == eosTokenId) {
                    finished[i] = true;
                    finishReasons[i] = GenerationResult.FinishReason.EOS;
                }
            }
        }
    }

    /**
     * Mark a sequence as finished due to max tokens.
     *
     * @param index the sequence index
     */
    public void markMaxTokens(int index) {
        if (!finished[index]) {
            finished[index] = true;
            finishReasons[index] = GenerationResult.FinishReason.MAX_TOKENS;
        }
    }

    /**
     * Mark a sequence as finished due to stop sequence.
     *
     * @param index the sequence index
     */
    public void markStopSequence(int index) {
        if (!finished[index]) {
            finished[index] = true;
            finishReasons[index] = GenerationResult.FinishReason.STOP_SEQUENCE;
        }
    }

    /**
     * Check if all sequences are finished.
     *
     * @return true if all sequences have finished generating
     */
    public boolean allFinished() {
        for (boolean f : finished) {
            if (!f) return false;
        }
        return true;
    }

    /**
     * Get the number of sequences still active.
     *
     * @return count of unfinished sequences
     */
    public int activeCount() {
        int count = 0;
        for (boolean f : finished) {
            if (!f) count++;
        }
        return count;
    }

    /**
     * Check if a specific sequence is finished.
     *
     * @param index the sequence index
     * @return true if finished
     */
    public boolean isFinished(int index) {
        return finished[index];
    }

    /**
     * Get generated tokens for a sequence.
     *
     * @param index the sequence index
     * @return list of generated token IDs
     */
    public List<Integer> getTokensForSequence(int index) {
        return generatedTokens.get(index);
    }

    /**
     * Get token count for a sequence.
     *
     * @param index the sequence index
     * @return number of tokens generated
     */
    public int getTokenCount(int index) {
        return generatedTokens.get(index).size();
    }

    /**
     * Get the maximum token count across all sequences.
     *
     * @return max tokens generated by any sequence
     */
    public int getMaxTokenCount() {
        int max = 0;
        for (List<Integer> tokens : generatedTokens) {
            max = Math.max(max, tokens.size());
        }
        return max;
    }

    /**
     * Get tokens as array for a sequence.
     *
     * @param index the sequence index
     * @return int array of token IDs
     */
    public int[] getTokenArrayForSequence(int index) {
        List<Integer> tokens = generatedTokens.get(index);
        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Build generation results for all sequences.
     *
     * @param texts decoded text for each sequence
     * @param promptTokenCounts prompt token counts per sequence
     * @param totalNanos total generation time in nanoseconds
     * @return array of generation results
     */
    public GenerationResult[] buildResults(String[] texts, int[] promptTokenCounts, long totalNanos) {
        GenerationResult[] results = new GenerationResult[batchSize];
        long totalMs = totalNanos / 1_000_000;

        for (int i = 0; i < batchSize; i++) {
            int[] tokenIds = getTokenArrayForSequence(i);
            int genCount = tokenIds.length;
            int promptCount = promptTokenCounts[i];
            long firstTokenMs = firstTokenLatencyNanos[i] / 1_000_000;

            results[i] = GenerationResult.builder()
                    .text(texts[i])
                    .tokenIds(tokenIds)
                    .generatedTokenCount(genCount)
                    .promptTokenCount(promptCount)
                    .totalTokenCount(promptCount + genCount)
                    .finishReason(finishReasons[i])
                    .firstTokenLatencyMs(firstTokenMs)
                    .generationTimeMs(totalMs)
                    .tokensPerSecond(totalNanos > 0 ? (genCount * 1_000_000_000.0) / totalNanos : 0)
                    .build();
        }

        return results;
    }
}
