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
@Builder
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
     * Maximum number of tokens to generate.
     * Default: 512
     */
    @Builder.Default
    private int maxNewTokens = 512;

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
     * Create a greedy decoding configuration.
     * Always selects the highest probability token.
     *
     * @return greedy sampling config
     */
    public static SamplingConfig greedy() {
        return SamplingConfig.builder()
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
                .temperature(0.8)
                .topK(40)
                .topP(0.9)
                .repetitionPenalty(1.1)
                .doSample(true)
                .build();
    }

    /**
     * Check if this configuration effectively performs greedy decoding.
     *
     * @return true if greedy decoding will be used
     */
    public boolean isGreedy() {
        return !doSample || temperature <= 0;
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
     * Check if repetition penalty is enabled.
     *
     * @return true if repetition penalty is active
     */
    public boolean hasRepetitionPenalty() {
        return repetitionPenalty != 1.0;
    }
}
