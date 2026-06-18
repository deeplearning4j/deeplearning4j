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

package org.nd4j.autodiff.samediff.config;

import lombok.*;
import lombok.experimental.SuperBuilder;
import org.nd4j.common.base.Preconditions;

/**
 * Configuration for Prefix Tuning.
 * <p>
 * Prefix Tuning prepends trainable prefix vectors to all transformer layers.
 * Unlike prompt tuning which only modifies input embeddings, prefix tuning
 * adds prefix vectors to the key and value components of each attention layer.
 * <p>
 * The prefix parameters are optimized through a separate feed-forward network
 * (reparameterization) to improve training stability. After training, the FFN
 * is discarded and only the prefix vectors are kept.
 * <p>
 * Key benefits:
 * <ul>
 *   <li>Better performance than prompt tuning for NLG tasks</li>
 *   <li>Works well in low-data settings</li>
 *   <li>~1000x fewer parameters than full fine-tuning</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>
 * PrefixTuningConfig config = PrefixTuningConfig.builder()
 *     .numVirtualTokens(30)
 *     .encoderHiddenSize(512)
 *     .prefixProjection(true)
 *     .numLayers(12)
 *     .numHeads(12)
 *     .taskType(TaskType.SEQ_2_SEQ_LM)
 *     .build();
 * </pre>
 *
 * @author Adam Gibson
 * @see PeftConfig
 * @see <a href="https://arxiv.org/abs/2101.00190">Prefix Tuning Paper</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class PrefixTuningConfig extends PeftConfig {

    /**
     * Number of virtual tokens (prefix length) to prepend.
     * Default: 30
     */
    @Builder.Default
    private int numVirtualTokens = 30;

    /**
     * Whether to use a MLP projection layer for reparameterization.
     * Recommended for training stability.
     * Default: true
     */
    @Builder.Default
    private boolean prefixProjection = true;

    /**
     * Hidden size of the encoder MLP when using prefix projection.
     * Only used if prefixProjection = true.
     * Default: 512
     */
    @Builder.Default
    private int encoderHiddenSize = 512;

    /**
     * Number of transformer layers in the model.
     * Prefix vectors are added to each layer.
     */
    private int numLayers;

    /**
     * Number of attention heads in the model.
     */
    private int numHeads;

    /**
     * Hidden size (embedding dimension) of the model.
     */
    private int hiddenSize;

    /**
     * Dropout probability for prefix vectors.
     * Default: 0.0
     */
    @Builder.Default
    private double prefixDropout = 0.0;

    /**
     * Whether this is for an encoder-decoder model.
     * If true, separate prefixes are used for encoder and decoder.
     * Default: false
     */
    @Builder.Default
    private boolean encoderDecoder = false;

    /**
     * Activation function for the prefix projection MLP.
     * Default: "tanh"
     */
    @Builder.Default
    private String activation = "tanh";

    @Override
    public PeftType getPeftType() {
        return PeftType.PREFIX_TUNING;
    }

    @Override
    public long calculateTrainableParameters(long originalParamCount) {
        // Parameters in prefix vectors for each layer
        long prefixParams = (long) numVirtualTokens * hiddenSize;

        // For prefix tuning, we add prefixes to K and V in each layer
        long perLayerParams = 2 * prefixParams; // Key and Value prefixes
        long totalPrefixParams = perLayerParams * numLayers;

        // If using prefix projection, add MLP parameters
        if (prefixProjection) {
            long mlpParams = (long) numVirtualTokens * encoderHiddenSize +
                            (long) encoderHiddenSize * 2 * hiddenSize * numLayers;
            return totalPrefixParams + mlpParams;
        }

        return totalPrefixParams;
    }

    @Override
    public void validate() {
        Preconditions.checkState(numVirtualTokens > 0,
            "Number of virtual tokens must be positive. Got: %s", numVirtualTokens);
        Preconditions.checkState(numLayers > 0,
            "Number of layers must be positive. Got: %s", numLayers);
        Preconditions.checkState(hiddenSize > 0,
            "Hidden size must be positive. Got: %s", hiddenSize);
        if (prefixProjection) {
            Preconditions.checkState(encoderHiddenSize > 0,
                "Encoder hidden size must be positive when using projection. Got: %s",
                encoderHiddenSize);
        }
        Preconditions.checkState(prefixDropout >= 0 && prefixDropout < 1,
            "Prefix dropout must be in [0, 1). Got: %s", prefixDropout);
    }

    @Override
    public String getSummary() {
        return String.format(
            "PrefixTuningConfig(numTokens=%d, layers=%d, projection=%s, hiddenSize=%d)",
            numVirtualTokens, numLayers, prefixProjection, hiddenSize);
    }

    /**
     * Create a prefix tuning configuration for a transformer model.
     *
     * @param numLayers  Number of transformer layers
     * @param numHeads   Number of attention heads
     * @param hiddenSize Model hidden size
     * @return Configured PrefixTuningConfig
     */
    public static PrefixTuningConfig forTransformer(int numLayers, int numHeads, int hiddenSize) {
        return PrefixTuningConfig.builder()
            .numVirtualTokens(30)
            .prefixProjection(true)
            .encoderHiddenSize(hiddenSize / 2)
            .numLayers(numLayers)
            .numHeads(numHeads)
            .hiddenSize(hiddenSize)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }

    /**
     * Create a prefix tuning configuration for sequence-to-sequence models.
     */
    public static PrefixTuningConfig forSeq2Seq(int numLayers, int numHeads, int hiddenSize) {
        return PrefixTuningConfig.builder()
            .numVirtualTokens(30)
            .prefixProjection(true)
            .encoderHiddenSize(hiddenSize / 2)
            .numLayers(numLayers)
            .numHeads(numHeads)
            .hiddenSize(hiddenSize)
            .encoderDecoder(true)
            .taskType(TaskType.SEQ_2_SEQ_LM)
            .build();
    }
}
