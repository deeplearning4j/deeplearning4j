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

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * Container for text generation results with metadata.
 *
 * <p>Provides access to the generated text along with useful metadata
 * about the generation process such as token count, finish reason,
 * and timing information.</p>
 *
 * Adam Gibson
 */
@Data
@Builder
public class GenerationResult {

    /**
     * The generated text.
     */
    private final String text;

    /**
     * The generated token IDs.
     */
    private final int[] tokenIds;

    /**
     * Number of tokens generated.
     */
    private final int generatedTokenCount;

    /**
     * Number of prompt tokens.
     */
    private final int promptTokenCount;

    /**
     * Total tokens (prompt + generated).
     */
    private final int totalTokenCount;

    /**
     * Reason generation finished.
     */
    private final FinishReason finishReason;

    /**
     * Time from start of generation to first token produced, in milliseconds.
     */
    private final long firstTokenLatencyMs;

    /**
     * Total wall-clock generation time in milliseconds.
     */
    private final long generationTimeMs;

    /**
     * Tokens per second throughput (generatedTokenCount / (generationTimeMs / 1000.0)).
     */
    private final double tokensPerSecond;

    /**
     * Decode-only throughput excluding prefill (step 0), in tokens per second.
     */
    private final double decodeTokensPerSecond;

    /**
     * Steady-state throughput after warmup/capture settling, in tokens per second.
     *
     * <p>For {@link StaticKvCacheDecodeLoop}, this is the average over fast-path
     * decode steps 3+.</p>
     */
    private final double steadyStateTokensPerSecond;

    /**
     * Late steady-state throughput after replay has fully settled, in tokens per second.
     *
     * <p>For {@link StaticKvCacheDecodeLoop}, this is the average over decode
     * steps 20+ when available.</p>
     */
    private final double lateSteadyStateTokensPerSecond;

    /**
     * Number of decode steps classified as warmup (compilation/capture overhead included).
     * Steps are classified as warmup when the DSP plan phase has not yet reached REPLAYING.
     */
    @Builder.Default
    private final int warmupStepCount = 0;

    /**
     * Maximum wall-clock time (ms) of any single warmup step.
     * Useful for understanding the worst-case compilation overhead.
     */
    @Builder.Default
    private final long maxWarmupStepMs = 0;

    /**
     * Total wall-clock time (ms) spent in warmup steps (compilation/capture).
     */
    @Builder.Default
    private final long totalWarmupMs = 0;

    /**
     * Log probabilities for each generated token (if requested).
     */
    private final List<Double> logProbs;

    // ==================== Speculative Decoding Statistics ====================

    /**
     * Total number of speculative tokens proposed across all steps.
     */
    @Builder.Default
    private final int totalSpeculativeTokens = 0;

    /**
     * Total number of speculative tokens accepted (matched model's greedy output).
     */
    @Builder.Default
    private final int totalAcceptedTokens = 0;

    /**
     * Number of speculative decode steps executed.
     */
    @Builder.Default
    private final int speculativeSteps = 0;

    /**
     * Average acceptance rate (totalAcceptedTokens / totalSpeculativeTokens).
     */
    @Builder.Default
    private final double averageAcceptanceRate = 0.0;

    /**
     * Effective tokens per second accounting for multi-token steps.
     * For speculative decoding: total tokens / wall time, where each step
     * may produce multiple tokens.
     */
    @Builder.Default
    private final double effectiveTokensPerSecond = 0.0;

    /**
     * Reason why generation stopped.
     */
    public enum FinishReason {
        /**
         * EOS token was generated.
         */
        EOS,

        /**
         * Maximum token limit reached.
         */
        MAX_TOKENS,

        /**
         * Stop sequence was matched.
         */
        STOP_SEQUENCE,

        /**
         * Generation was cancelled.
         */
        CANCELLED,

        /**
         * An error occurred.
         */
        ERROR
    }

    /**
     * Create a result for successful EOS termination.
     */
    public static GenerationResult eos(String text, int[] tokenIds, int promptTokens,
                                       long firstTokenLatencyMs, long timeMs) {
        int generated = tokenIds.length;
        return GenerationResult.builder()
                .text(text)
                .tokenIds(tokenIds)
                .generatedTokenCount(generated)
                .promptTokenCount(promptTokens)
                .totalTokenCount(promptTokens + generated)
                .finishReason(FinishReason.EOS)
                .firstTokenLatencyMs(firstTokenLatencyMs)
                .generationTimeMs(timeMs)
                .tokensPerSecond(timeMs > 0 ? (generated * 1000.0 / timeMs) : 0)
                .build();
    }

    /**
     * Create a result for successful EOS termination (backward-compatible overload).
     */
    public static GenerationResult eos(String text, int[] tokenIds, int promptTokens, long timeMs) {
        return eos(text, tokenIds, promptTokens, 0, timeMs);
    }

    /**
     * Create a result for max token limit.
     */
    public static GenerationResult maxTokens(String text, int[] tokenIds, int promptTokens,
                                              long firstTokenLatencyMs, long timeMs) {
        int generated = tokenIds.length;
        return GenerationResult.builder()
                .text(text)
                .tokenIds(tokenIds)
                .generatedTokenCount(generated)
                .promptTokenCount(promptTokens)
                .totalTokenCount(promptTokens + generated)
                .finishReason(FinishReason.MAX_TOKENS)
                .firstTokenLatencyMs(firstTokenLatencyMs)
                .generationTimeMs(timeMs)
                .tokensPerSecond(timeMs > 0 ? (generated * 1000.0 / timeMs) : 0)
                .build();
    }

    /**
     * Create a result for max token limit (backward-compatible overload).
     */
    public static GenerationResult maxTokens(String text, int[] tokenIds, int promptTokens, long timeMs) {
        return maxTokens(text, tokenIds, promptTokens, 0, timeMs);
    }

    /**
     * Create a result for stop sequence match.
     */
    public static GenerationResult stopSequence(String text, int[] tokenIds, int promptTokens,
                                                 long firstTokenLatencyMs, long timeMs) {
        int generated = tokenIds.length;
        return GenerationResult.builder()
                .text(text)
                .tokenIds(tokenIds)
                .generatedTokenCount(generated)
                .promptTokenCount(promptTokens)
                .totalTokenCount(promptTokens + generated)
                .finishReason(FinishReason.STOP_SEQUENCE)
                .firstTokenLatencyMs(firstTokenLatencyMs)
                .generationTimeMs(timeMs)
                .tokensPerSecond(timeMs > 0 ? (generated * 1000.0 / timeMs) : 0)
                .build();
    }

    /**
     * Create a result for stop sequence match (backward-compatible overload).
     */
    public static GenerationResult stopSequence(String text, int[] tokenIds, int promptTokens, long timeMs) {
        return stopSequence(text, tokenIds, promptTokens, 0, timeMs);
    }

    /**
     * Check if generation completed normally (EOS or stop sequence).
     */
    public boolean isComplete() {
        return finishReason == FinishReason.EOS || finishReason == FinishReason.STOP_SEQUENCE;
    }

    /**
     * Check if generation was truncated due to max tokens.
     */
    public boolean isTruncated() {
        return finishReason == FinishReason.MAX_TOKENS;
    }
}
