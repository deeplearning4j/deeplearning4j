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

package org.eclipse.deeplearning4j.llm.generation.sampling;

import lombok.Builder;
import lombok.Data;
import org.eclipse.deeplearning4j.llm.generation.constraint.ConstraintConfig;

/**
 * Configuration for text generation sampling strategies.
 *
 * <p>This class encapsulates all parameters that control how tokens are
 * sampled during text generation. It supports various sampling methods
 * that can be combined:</p>
 *
 * <ul>
 *   <li><b>Greedy:</b> Always select the highest probability token (temperature=0 or doSample=false)</li>
 *   <li><b>Temperature:</b> Scale logits before softmax to control randomness</li>
 *   <li><b>Top-K:</b> Only sample from the K highest probability tokens</li>
 *   <li><b>Top-P (Nucleus):</b> Sample from smallest set of tokens with cumulative probability >= p</li>
 *   <li><b>Repetition Penalty:</b> Reduce probability of recently generated tokens</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * // Creative writing config
 * SamplingConfig config = SamplingConfig.builder()
 *     .temperature(0.9)
 *     .topK(50)
 *     .topP(0.95)
 *     .doSample(true)
 *     .build();
 *
 * // Deterministic config
 * SamplingConfig greedy = SamplingConfig.greedy();
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
@Builder(toBuilder = true)
public class SamplingConfig {

    /**
     * Temperature for logit scaling.
     * Higher values (e.g., 1.5) increase randomness.
     * Lower values (e.g., 0.3) make output more deterministic.
     * A value of 0 or negative triggers greedy decoding.
     * Default: 1.0
     */
    @Builder.Default
    private double temperature = 1.0;

    /**
     * Number of highest probability tokens to consider for top-k sampling.
     * Set to 0 or negative to disable top-k filtering.
     * Common values: 40, 50, 100
     * Default: 0 (disabled)
     */
    @Builder.Default
    private int topK = 0;

    /**
     * Cumulative probability threshold for nucleus (top-p) sampling.
     * Only tokens with cumulative probability <= topP are considered.
     * Set to 1.0 to disable top-p filtering.
     * Common values: 0.9, 0.95
     * Default: 1.0 (disabled)
     */
    @Builder.Default
    private double topP = 1.0;

    /**
     * Whether to use sampling (true) or greedy decoding (false).
     * When false, always selects the highest probability token.
     * Default: true
     */
    @Builder.Default
    private boolean doSample = true;

    /**
     * Penalty applied to tokens that have already been generated.
     * Values > 1.0 discourage repetition.
     * Values < 1.0 encourage repetition.
     * Default: 1.0 (no penalty)
     */
    @Builder.Default
    private double repetitionPenalty = 1.0;

    /**
     * Explicit native periodic-tail guard. Zero disables native repetition termination and
     * preserves ordinary generate() semantics. When enabled, periods 1..this value are tested.
     */
    @Builder.Default
    private int nativeRepetitionLoopMaxPeriod = 0;

    /** Consecutive repeats required by the opt-in native periodic-tail guard. */
    @Builder.Default
    private int nativeRepetitionLoopMaxRepeats = 0;

    /**
     * Frequency penalty subtracts {@code count(token) * frequencyPenalty} from seen-token logits.
     * Default: 0.0 (disabled)
     */
    @Builder.Default
    private double frequencyPenalty = 0.0;

    /**
     * Presence penalty subtracts {@code presencePenalty} once for any token already seen.
     * Default: 0.0 (disabled)
     */
    @Builder.Default
    private double presencePenalty = 0.0;

    /**
     * Min-p adaptive filtering threshold relative to the highest-probability token.
     * Values in the 0.05-0.1 range are common; 0.0 disables the filter.
     */
    @Builder.Default
    private double minP = 0.0;

    /**
     * Typical-p (locally typical sampling) threshold — Meister et al. 2023.
     * Keeps tokens with the smallest entropy deviation |−log p_i − H| until their
     * cumulative mass >= typicalP, then masks the rest to -inf.
     * Values in (0, 1) enable the filter; 1.0 (default) = off.
     * Applied after temperature scaling and standard truncation filters.
     * Validation: must be in (0, 1] — values outside this range are rejected by validate().
     */
    @Builder.Default
    private double typicalP = 1.0;

    /**
     * XTC (Exclude Top Choices) probability — probability of applying XTC each step.
     * 0.0 (default) = always skip; 1.0 = always apply.
     * When applied: among tokens with softmax probability >= xtcThreshold, mask all
     * EXCEPT the lowest-probability one, encouraging diversity by removing the model's
     * most confident choices.
     * Validation: must be in [0, 1].
     */
    @Builder.Default
    private double xtcProbability = 0.0;

    /**
     * XTC per-token probability threshold — a token must have softmax probability >= this
     * value to be eligible for XTC exclusion. Default 0.1.
     * Validation: must be in (0, 0.5].
     */
    @Builder.Default
    private double xtcThreshold = 0.1;

    /**
     * Maximum number of tokens to generate.
     * Default: 512
     */
    @Builder.Default
    private int maxNewTokens = 512;

    /**
     * Minimum number of tokens to generate before stop tokens are allowed:
     * while under this floor, stop-token logits are suppressed at sampling
     * (standard Whisper/HF practice — greedy decode otherwise ends on a
     * marginal early EOT). Default 0 = no suppression.
     */
    @Builder.Default
    private int minNewTokens = 0;

    /**
     * Maximum number of generated tokens that template-owned output blocks may consume
     * before constrained decoding requires their closing delimiters. The block type is
     * intentionally not named here: this applies to every output block discovered from
     * the active model chat template. Default 0 disables this boundary.
     */
    @Builder.Default
    private int maxOutputBlockTokens = 0;

    /**
     * Tokens reserved at the end of the total generation budget for output-block closures
     * and the following structured payload. When the remaining budget reaches this value,
     * arbitrary block content is no longer admitted. Default 0 disables this boundary.
     */
    @Builder.Default
    private int structuredOutputTokenReserve = 0;

    /**
     * Random seed for reproducible sampling.
     * Set to null for non-deterministic behavior.
     */
    private Long seed;

    /**
     * End-of-sequence token ID.
     * Generation stops when this token is produced.
     * Default: -1 (use tokenizer's EOS)
     */
    @Builder.Default
    private int eosTokenId = -1;

    /**
     * Pad token ID for batch generation.
     * Default: -1 (use tokenizer's PAD)
     */
    @Builder.Default
    private int padTokenId = -1;

    /**
     * Decode strategy / search algorithm.
     *
     * <ul>
     *   <li>{@code AUTO} (default) — existing behavior: greedy or stochastic sampling per
     *       {@link #isGreedy()} / temperature / top-k / top-p.</li>
     *   <li>{@code GREEDY} — explicit greedy decode.</li>
     *   <li>{@code SAMPLE} — explicit stochastic sampling (temperature/top-k/top-p).</li>
     *   <li>{@code SPECULATIVE} — speculative verification over a fixed-width candidate chain.</li>
     *   <li>{@code CONTRASTIVE} — contrastive search (Su et al. 2022); requires {@link #penaltyAlpha}
     *       and {@link #contrastiveTopK}.</li>
     *   <li>{@code BEAM} — beam search; requires {@link #numBeams} &gt; 1.</li>
     * </ul>
     *
     * <p>{@code SPECULATIVE}/{@code CONTRASTIVE}/{@code BEAM} run on the masked multi-position decode
     * substrate (ADR 0106). Additive: callers that never set this get identical behavior.</p>
     */
    public enum DecodeStrategy { AUTO, GREEDY, SAMPLE, SPECULATIVE, CONTRASTIVE, BEAM }

    /** Selected decode strategy. Default {@link DecodeStrategy#AUTO} (greedy/sampling as before). */
    @Builder.Default
    private DecodeStrategy decodeStrategy = DecodeStrategy.AUTO;

    /** Beam search: number of beams (hypotheses kept). {@code <= 1} disables beam search. Default 1. */
    @Builder.Default
    private int numBeams = 1;

    /** Beam search group count for diverse beam search. Default 1 (standard beam). */
    @Builder.Default
    private int numBeamGroups = 1;

    /** Diverse beam search penalty between beam groups. Default 0.0 (disabled). */
    @Builder.Default
    private double diversityPenalty = 0.0;

    /** Number of returned sequences requested by generation config. Default 1. */
    @Builder.Default
    private int numReturnSequences = 1;

    /**
     * Beam search length penalty (HF {@code length_penalty}). {@code > 1} favors longer sequences,
     * {@code < 1} favors shorter; {@code 1.0} = pure summed log-prob. Default 1.0.
     */
    @Builder.Default
    private double lengthPenalty = 1.0;

    /**
     * Contrastive search degeneration weight α (HF {@code penalty_alpha}). The next token maximizes
     * {@code (1-α)*p(v) - α*max_j cos(h_v, h_j)} over the top-k candidates. {@code 0} disables
     * contrastive search. Default 0.
     */
    @Builder.Default
    private double penaltyAlpha = 0.0;

    /**
     * Contrastive search candidate count k (HF pairs {@code penalty_alpha} with {@code top_k}): the k
     * highest-probability tokens are re-ranked by the degeneration penalty. {@code <= 1} disables.
     * Default 0.
     */
    @Builder.Default
    private int contrastiveTopK = 0;

    /**
     * Create a greedy decoding configuration.
     * Always selects the highest probability token.
     *
     * @return greedy sampling config
     */
    public static SamplingConfig greedy() {
        return SamplingConfig.builder()
                .decodeStrategy(DecodeStrategy.GREEDY)
                .doSample(false)
                .temperature(0.0)
                .build();
    }

    /**
     * Create an explicit stochastic sampling configuration.
     *
     * @param temperature temperature for logit scaling
     * @param topK        top-k cutoff, or {@code <= 0} to disable
     * @param topP        nucleus cutoff, or {@code >= 1.0} to disable
     * @return sampling config
     */
    public static SamplingConfig sample(double temperature, int topK, double topP) {
        return SamplingConfig.builder()
                .decodeStrategy(DecodeStrategy.SAMPLE)
                .temperature(temperature)
                .topK(topK)
                .topP(topP)
                .doSample(true)
                .build();
    }

    /**
     * Create a speculative decoding configuration. The verification width is supplied by
     * {@link org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig#getMaxSpeculativeTokens()}.
     *
     * @return speculative decoding config
     */
    public static SamplingConfig speculative() {
        return SamplingConfig.builder()
                .decodeStrategy(DecodeStrategy.SPECULATIVE)
                .doSample(false)
                .temperature(0.0)
                .build();
    }

    /**
     * Create a default sampling configuration suitable for general text generation.
     * Uses temperature=0.7, top-p=0.9 for balanced creativity and coherence.
     *
     * @return default sampling config
     */
    public static SamplingConfig defaultConfig() {
        return SamplingConfig.builder()
                .decodeStrategy(DecodeStrategy.AUTO)
                .temperature(0.7)
                .topP(0.9)
                .doSample(true)
                .build();
    }

    /**
     * Create a creative writing configuration.
     * Higher temperature and top-k for more varied output.
     *
     * @return creative sampling config
     */
    public static SamplingConfig creative() {
        return SamplingConfig.builder()
                .decodeStrategy(DecodeStrategy.SAMPLE)
                .temperature(0.9)
                .topK(50)
                .topP(0.95)
                .doSample(true)
                .build();
    }

    /**
     * Create a precise/factual configuration.
     * Lower temperature for more focused, deterministic output.
     *
     * @return precise sampling config
     */
    public static SamplingConfig precise() {
        return SamplingConfig.builder()
                .decodeStrategy(DecodeStrategy.SAMPLE)
                .temperature(0.3)
                .topP(0.85)
                .doSample(true)
                .build();
    }

    /**
     * Create a configuration matching llama.cpp default sampling parameters.
     * temp=0.8, top_k=40, top_p=0.9, repeat_penalty=1.1
     *
     * @return llama.cpp-style sampling config
     */
    public static SamplingConfig llamaCppDefaults() {
        return SamplingConfig.builder()
                .decodeStrategy(DecodeStrategy.SAMPLE)
                .temperature(0.8)
                .topK(40)
                .topP(0.9)
                .repetitionPenalty(1.1)
                .doSample(true)
                .build();
    }

    /**
     * Create a beam-search configuration with the given number of beams (greedy per-beam scoring).
     *
     * @param numBeams number of beams to keep (use &gt;= 2 for a real search)
     * @return beam-search sampling config
     */
    public static SamplingConfig beam(int numBeams) {
        return SamplingConfig.builder()
                .decodeStrategy(DecodeStrategy.BEAM)
                .numBeams(numBeams)
                .doSample(false)
                .build();
    }

    /**
     * Create a contrastive-search configuration (Su et al. 2022, "A Contrastive Framework for Neural
     * Text Generation"). Typical values: {@code alpha=0.6}, {@code k=4}.
     *
     * @param penaltyAlpha degeneration weight α
     * @param k            candidate count (top-k tokens re-ranked by the degeneration penalty)
     * @return contrastive-search sampling config
     */
    public static SamplingConfig contrastive(double penaltyAlpha, int k) {
        return SamplingConfig.builder()
                .decodeStrategy(DecodeStrategy.CONTRASTIVE)
                .penaltyAlpha(penaltyAlpha)
                .contrastiveTopK(k)
                .topK(k)
                .doSample(false)
                .build();
    }

    /**
     * Check if this configuration effectively performs greedy decoding.
     *
     * @return true if greedy decoding will be used
     */
    public boolean isGreedy() {
        return decodeStrategy == DecodeStrategy.GREEDY
                || decodeStrategy == DecodeStrategy.SPECULATIVE
                || (decodeStrategy == DecodeStrategy.AUTO && (!doSample || temperature <= 0));
    }

    /**
     * Whether stochastic sampling is selected and active.
     *
     * @return true if temperature/top-k/top-p sampling should run
     */
    public boolean isSampling() {
        return decodeStrategy == DecodeStrategy.SAMPLE
                || (decodeStrategy == DecodeStrategy.AUTO && doSample && temperature > 0);
    }

    /**
     * Whether speculative decoding is selected.
     *
     * @return true if speculative verification should run
     */
    public boolean isSpeculative() {
        return decodeStrategy == DecodeStrategy.SPECULATIVE;
    }

    /**
     * Check if top-k filtering is enabled.
     *
     * @return true if top-k filtering is active
     */
    public boolean hasTopK() {
        return topK > 0;
    }

    /**
     * Check if top-p (nucleus) filtering is enabled.
     *
     * @return true if top-p filtering is active
     */
    public boolean hasTopP() {
        return topP > 0 && topP < 1.0;
    }

    /**
     * Check if min-p filtering is enabled.
     *
     * @return true if min-p filtering is active
     */
    public boolean hasMinP() {
        return minP > 0.0;
    }

    /**
     * Check if typical-p filtering is enabled.
     *
     * @return true if typical-p filtering is active
     */
    public boolean hasTypicalP() {
        return typicalP > 0.0 && typicalP < 1.0;
    }

    /**
     * Check if XTC (Exclude Top Choices) sampling is enabled.
     *
     * @return true if XTC will be applied with non-zero probability
     */
    public boolean hasXtc() {
        return xtcProbability > 0.0;
    }

    /**
     * Validate all sampling parameters; throws IllegalArgumentException on first violation.
     */
    public void validate() {
        if (typicalP <= 0.0 || typicalP > 1.0)
            throw new IllegalArgumentException("typicalP must be in (0, 1]; got: " + typicalP);
        if (xtcProbability < 0.0 || xtcProbability > 1.0)
            throw new IllegalArgumentException("xtcProbability must be in [0, 1]; got: " + xtcProbability);
        if (xtcProbability > 0.0 && (xtcThreshold <= 0.0 || xtcThreshold > 0.5))
            throw new IllegalArgumentException("xtcThreshold must be in (0, 0.5] when XTC is enabled; got: " + xtcThreshold);
        if (temperature < 0.0)
            throw new IllegalArgumentException("temperature must be >= 0; got: " + temperature);
        if (topK < 0)
            throw new IllegalArgumentException("topK must be >= 0; got: " + topK);
        if (topP < 0.0 || topP > 1.0)
            throw new IllegalArgumentException("topP must be in [0, 1]; got: " + topP);
        if (minP < 0.0 || minP > 1.0)
            throw new IllegalArgumentException("minP must be in [0, 1]; got: " + minP);
        if (nativeRepetitionLoopMaxPeriod < 0 || nativeRepetitionLoopMaxPeriod > 1024)
            throw new IllegalArgumentException("nativeRepetitionLoopMaxPeriod must be in [0, 1024]; got: "
                    + nativeRepetitionLoopMaxPeriod);
        if (nativeRepetitionLoopMaxRepeats < 0 || nativeRepetitionLoopMaxRepeats > 1024)
            throw new IllegalArgumentException("nativeRepetitionLoopMaxRepeats must be in [0, 1024]; got: "
                    + nativeRepetitionLoopMaxRepeats);
        if ((nativeRepetitionLoopMaxPeriod == 0) != (nativeRepetitionLoopMaxRepeats == 0))
            throw new IllegalArgumentException("native repetition termination requires both maxPeriod and maxRepeats");
        if (nativeRepetitionLoopMaxRepeats > 0 && nativeRepetitionLoopMaxRepeats < 2)
            throw new IllegalArgumentException("nativeRepetitionLoopMaxRepeats must be >= 2 when enabled; got: "
                    + nativeRepetitionLoopMaxRepeats);
        if (maxOutputBlockTokens < 0)
            throw new IllegalArgumentException(
                    "maxOutputBlockTokens must be >= 0; got: " + maxOutputBlockTokens);
        if (structuredOutputTokenReserve < 0)
            throw new IllegalArgumentException(
                    "structuredOutputTokenReserve must be >= 0; got: "
                            + structuredOutputTokenReserve);
    }

    /**
     * Check if repetition penalty is enabled.
     *
     * @return true if repetition penalty is active
     */
    public boolean hasRepetitionPenalty() {
        return repetitionPenalty != 1.0;
    }

    /**
     * Check if frequency penalty is enabled.
     *
     * @return true if frequency penalty is active
     */
    public boolean hasFrequencyPenalty() {
        return frequencyPenalty != 0.0;
    }

    /**
     * Check if presence penalty is enabled.
     *
     * @return true if presence penalty is active
     */
    public boolean hasPresencePenalty() {
        return presencePenalty != 0.0;
    }

    /**
     * Check if any seen-token penalty is enabled.
     *
     * @return true if repetition, frequency, or presence penalty is active
     */
    public boolean hasTokenPenalties() {
        return hasRepetitionPenalty() || hasFrequencyPenalty() || hasPresencePenalty();
    }

    /**
     * Whether beam search is selected and active (strategy {@code BEAM} with &gt; 1 beam).
     *
     * @return true if beam search should run
     */
    public boolean isBeam() {
        return decodeStrategy == DecodeStrategy.BEAM && numBeams > 1;
    }

    /**
     * Whether contrastive search is selected and active (strategy {@code CONTRASTIVE} with α &gt; 0
     * and k &gt; 1).
     *
     * @return true if contrastive search should run
     */
    public boolean isContrastive() {
        return decodeStrategy == DecodeStrategy.CONTRASTIVE && penaltyAlpha > 0.0 && contrastiveTopK > 1;
    }

    // ---- Constrained decoding ----

    /**
     * Optional constraint configuration for structured-output / constrained decoding.
     *
     * <p>When set, every token selection step runs through the Java sampling branch
     * (bypassing the native {@code autoregressive_decode} C++ loop) so the
     * {@link org.eclipse.deeplearning4j.llm.generation.constraint.ConstraintMasker}
     * can apply per-step logit masking. This has a modest throughput cost compared to
     * the native loop ({@code ~0.5–1×} tok/s on CPU vs. unconstrained; see ADR 0111)
     * but guarantees structurally valid output.
     *
     * <p>Null (default) = unconstrained; zero behavior change.
     *
     * <p>Use the factory methods on {@link ConstraintConfig} to build a config:</p>
     * <pre>{@code
     * SamplingConfig constrained = SamplingConfig.builder()
     *     .temperature(0.7)
     *     .constraintConfig(ConstraintConfig.toolCall(
     *         "ask_graph_verify", "graph_reasoning_query", "ask_graph_query"))
     *     .build();
     * }</pre>
     */
    private ConstraintConfig constraintConfig;

    /**
     * Returns {@code true} if a constraint is configured for this sampling run.
     *
     * @return true when constrained decoding is active
     */
    public boolean hasConstraint() {
        return constraintConfig != null;
    }
}
