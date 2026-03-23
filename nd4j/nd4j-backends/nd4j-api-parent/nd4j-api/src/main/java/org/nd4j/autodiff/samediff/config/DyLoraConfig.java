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
 * Configuration for DyLoRA (Dynamic Low-Rank Adaptation).
 * <p>
 * DyLoRA trains with dynamic rank sampling: at each step, a random rank k
 * is sampled from [minRank, r] and only the first k columns/rows of A and B
 * are used. This trains a single adapter that works well across multiple ranks,
 * allowing rank selection at inference time without retraining.
 *
 * <p>Key benefit: Train once, deploy at any rank from minRank to r.</p>
 *
 * Adam Gibson
 * @see LoraConfig
 * @see PeftConfig
 * @see <a href="https://arxiv.org/abs/2210.07558">DyLoRA Paper</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class DyLoraConfig extends LoraConfig {

    /**
     * Minimum rank for dynamic sampling.
     * During training, rank k is sampled uniformly from [minRank, r].
     * Default: 1
     */
    @Builder.Default
    private int minRank = 1;

    @Override
    public PeftType getPeftType() {
        return PeftType.DYLORA;
    }

    @Override
    public void validate() {
        super.validate();
        Preconditions.checkState(minRank >= 1, "Minimum rank must be >= 1. Got: %s", minRank);
        Preconditions.checkState(minRank <= getR(), "Minimum rank must be <= r. Got minRank=%s, r=%s", minRank, getR());
    }

    @Override
    public String getSummary() {
        return String.format(
            "DyLoraConfig(r=%d, minRank=%d, alpha=%d, scaling=%.4f, dropout=%.2f, targets=%s)",
            getR(), minRank, getLoraAlpha(), getScaling(), getLoraDropout(), getTargetModules());
    }
}
