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

import java.util.List;

/**
 * Configuration for Adapter layers.
 * <p>
 * Adapter layers insert small bottleneck layers between transformer layers.
 * The adapter consists of: down-projection -> activation -> up-projection + residual.
 * <p>
 * Structure:
 * <pre>
 * input
 *   │
 *   ├──────────────────────┐
 *   ↓                      │
 * [Down-project: d → r]    │
 *   ↓                      │
 * [Activation (ReLU/GELU)] │
 *   ↓                      │
 * [Up-project: r → d]      │
 *   ↓                      │
 *   ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘ (residual)
 *   ↓
 * output
 * </pre>
 *
 * @author Adam Gibson
 * @see PeftConfig
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class AdapterConfig extends PeftConfig {

    /**
     * The bottleneck dimension of the adapter.
     * Smaller = fewer parameters but potentially less expressive.
     * Default: 64
     */
    @Builder.Default
    private int adapterSize = 64;

    /**
     * Activation function in the adapter bottleneck.
     * Options: "relu", "gelu", "silu", "tanh"
     * Default: "relu"
     */
    @Builder.Default
    private String adapterActivation = "relu";

    /**
     * Dropout probability within the adapter.
     * Default: 0.0
     */
    @Builder.Default
    private double adapterDropout = 0.0;

    /**
     * Scaling factor for the adapter output before adding residual.
     * Default: 1.0
     */
    @Builder.Default
    private double adapterScaling = 1.0;

    /**
     * Whether to add adapters after attention layers.
     * Default: true
     */
    @Builder.Default
    private boolean adapterAfterAttention = true;

    /**
     * Whether to add adapters after feed-forward layers.
     * Default: true
     */
    @Builder.Default
    private boolean adapterAfterFeedforward = true;

    /**
     * Hidden size of the model layers.
     */
    private int hiddenSize;

    /**
     * Number of layers to add adapters to.
     */
    private int numLayers;

    /**
     * Layer indices to add adapters to.
     * If null, adapters are added to all layers.
     */
    private List<Integer> layerIndices;

    @Override
    public PeftType getPeftType() {
        return PeftType.ADAPTERS;
    }

    @Override
    public long calculateTrainableParameters(long originalParamCount) {
        // Each adapter has:
        // - Down projection: hiddenSize * adapterSize
        // - Up projection: adapterSize * hiddenSize
        // - Biases: adapterSize + hiddenSize
        long paramsPerAdapter = 2L * hiddenSize * adapterSize + adapterSize + hiddenSize;

        int effectiveLayers = (layerIndices != null) ? layerIndices.size() : numLayers;
        int adaptersPerLayer = 0;
        if (adapterAfterAttention) adaptersPerLayer++;
        if (adapterAfterFeedforward) adaptersPerLayer++;

        return paramsPerAdapter * adaptersPerLayer * effectiveLayers;
    }

    @Override
    public void validate() {
        Preconditions.checkState(adapterSize > 0,
            "Adapter size must be positive. Got: %s", adapterSize);
        Preconditions.checkState(hiddenSize > 0,
            "Hidden size must be positive. Got: %s", hiddenSize);
        Preconditions.checkState(adapterDropout >= 0 && adapterDropout < 1,
            "Adapter dropout must be in [0, 1). Got: %s", adapterDropout);
        Preconditions.checkState(adapterAfterAttention || adapterAfterFeedforward,
            "At least one of adapterAfterAttention or adapterAfterFeedforward must be true");
    }

    @Override
    public String getSummary() {
        return String.format(
            "AdapterConfig(size=%d, activation=%s, layers=%d, afterAttn=%s, afterFF=%s)",
            adapterSize, adapterActivation, numLayers, adapterAfterAttention, adapterAfterFeedforward);
    }

    /**
     * Create a default adapter configuration.
     */
    public static AdapterConfig defaultConfig(int hiddenSize, int numLayers) {
        return AdapterConfig.builder()
            .adapterSize(64)
            .adapterActivation("relu")
            .hiddenSize(hiddenSize)
            .numLayers(numLayers)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }
}
