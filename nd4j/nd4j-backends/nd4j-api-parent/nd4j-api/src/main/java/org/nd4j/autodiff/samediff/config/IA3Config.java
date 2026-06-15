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

import java.util.Arrays;
import java.util.List;

/**
 * Configuration for IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).
 * <p>
 * IA³ learns vectors that rescale activation values. It's extremely parameter-efficient,
 * adding even fewer trainable parameters than LoRA while achieving competitive performance.
 * <p>
 * For each target module, IA³ learns a learned vector that element-wise multiplies
 * the activations:
 * <pre>
 * output = activation ⊙ learned_vector
 * </pre>
 * <p>
 * Typically applied to:
 * <ul>
 *   <li>Keys in attention (k_proj)</li>
 *   <li>Values in attention (v_proj)</li>
 *   <li>First feed-forward layer (up_proj / intermediate)</li>
 * </ul>
 *
 * @author Adam Gibson
 * @see PeftConfig
 * @see <a href="https://arxiv.org/abs/2205.05638">IA³ Paper</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class IA3Config extends PeftConfig {

    /**
     * Modules in the feedforward layers to apply IA³.
     * Default: ["up_proj", "down_proj"]
     */
    @Builder.Default
    private List<String> feedforwardModules = Arrays.asList("up_proj", "down_proj");

    /**
     * Whether to initialize the learned vectors to ones.
     * If true, the model starts at the pretrained behavior.
     * Default: true
     */
    @Builder.Default
    private boolean initToOne = true;

    /**
     * Hidden size for attention modules.
     */
    private int attentionHiddenSize;

    /**
     * Hidden size for feedforward modules (intermediate size).
     */
    private int feedforwardHiddenSize;

    /**
     * Default target modules for IA³.
     */
    public static final List<String> DEFAULT_TARGET_MODULES =
        Arrays.asList("k_proj", "v_proj", "up_proj");

    @Override
    public PeftType getPeftType() {
        return PeftType.IA3;
    }

    @Override
    public long calculateTrainableParameters(long originalParamCount) {
        // IA³ adds one learned vector per target module per layer
        // Vector size matches the module's output dimension
        long attnParams = 0;
        long ffParams = 0;

        if (targetModules != null) {
            for (String module : targetModules) {
                if (module.contains("k_") || module.contains("v_") ||
                    module.contains("key") || module.contains("value")) {
                    attnParams += attentionHiddenSize;
                }
            }
        }

        if (feedforwardModules != null) {
            for (String module : feedforwardModules) {
                ffParams += feedforwardHiddenSize;
            }
        }

        return attnParams + ffParams;
    }

    @Override
    public void validate() {
        Preconditions.checkState(
            (targetModules != null && !targetModules.isEmpty()) ||
            (feedforwardModules != null && !feedforwardModules.isEmpty()),
            "At least one of targetModules or feedforwardModules must be specified");
    }

    @Override
    public String getSummary() {
        return String.format(
            "IA3Config(targets=%s, ffModules=%s, initToOne=%s)",
            targetModules, feedforwardModules, initToOne);
    }

    /**
     * Create a default IA³ configuration.
     */
    public static IA3Config defaultConfig(int attentionDim, int ffDim) {
        return IA3Config.builder()
            .targetModules(DEFAULT_TARGET_MODULES)
            .feedforwardModules(Arrays.asList("up_proj"))
            .attentionHiddenSize(attentionDim)
            .feedforwardHiddenSize(ffDim)
            .initToOne(true)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }
}
