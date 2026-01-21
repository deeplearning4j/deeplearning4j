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
 * Configuration for DoRA (Weight-Decomposed Low-Rank Adaptation).
 * <p>
 * DoRA improves upon LoRA by decomposing weight updates into magnitude and direction
 * components. The direction is handled by standard LoRA matrices (A, B), while
 * magnitude is managed by a separate learnable parameter (m).
 * <p>
 * Mathematical formulation:
 * <pre>
 * W' = m * (W₀ + BA) / ||W₀ + BA||
 *
 * where:
 *   W₀ = pretrained weight (frozen)
 *   B, A = low-rank decomposition matrices (trainable)
 *   m = magnitude vector (trainable)
 *   ||·|| = column-wise norm
 * </pre>
 * <p>
 * Benefits over standard LoRA:
 * <ul>
 *   <li>Better performance especially at lower ranks</li>
 *   <li>More stable training dynamics</li>
 *   <li>No additional inference overhead (magnitude can be merged)</li>
 * </ul>
 *
 * @author Adam Gibson
 * @see LoraConfig
 * @see <a href="https://arxiv.org/abs/2402.09353">DoRA Paper</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class DoraConfig extends LoraConfig {

    /**
     * Whether to use ephemeral GPU offloading for magnitude computation.
     * Can reduce memory usage during training.
     * Default: false
     */
    @Builder.Default
    private boolean ephemeralGpuOffload = false;

    /**
     * Initialization method for magnitude vector.
     * Options: "unit" (initialize to 1), "pretrained" (use ||W₀|| column norms)
     * Default: "pretrained"
     */
    @Builder.Default
    private String magnitudeInit = "pretrained";

    /**
     * Whether to learn magnitude per-row or per-column.
     * Default: per-column (matches paper)
     */
    @Builder.Default
    private boolean magnitudePerRow = false;

    @Override
    public PeftType getPeftType() {
        return PeftType.DORA;
    }

    @Override
    public void validate() {
        super.validate();
        Preconditions.checkState(
            magnitudeInit.equals("unit") || magnitudeInit.equals("pretrained"),
            "magnitudeInit must be 'unit' or 'pretrained'. Got: %s", magnitudeInit);
    }

    @Override
    public String getSummary() {
        return String.format(
            "DoraConfig(r=%d, alpha=%d, magInit=%s, targets=%s)",
            getR(), getLoraAlpha(), magnitudeInit, getTargetModules());
    }

    /**
     * Create a default DoRA configuration.
     */
    public static DoraConfig defaultConfig(List<String> targetModules) {
        return DoraConfig.builder()
            .r(16)
            .loraAlpha(32)
            .loraDropout(0.05)
            .magnitudeInit("pretrained")
            .targetModules(targetModules)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }
}
