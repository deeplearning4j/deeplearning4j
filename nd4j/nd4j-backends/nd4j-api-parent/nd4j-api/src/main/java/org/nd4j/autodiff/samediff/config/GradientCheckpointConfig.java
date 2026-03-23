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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.Set;

/**
 * Configuration for gradient checkpointing (activation checkpointing).
 *
 * Gradient checkpointing trades compute for memory by discarding intermediate
 * activations during the forward pass and recomputing them during backward.
 * This can reduce peak memory usage by O(sqrt(n)) for a model with n layers.
 *
 * Supports three strategies:
 * <ul>
 *   <li><b>Every-N:</b> Checkpoint every N layers (set via {@code checkpointEveryN})</li>
 *   <li><b>Manual:</b> Checkpoint specific named variables (set via {@code checkpointVariables})</li>
 *   <li><b>Sqrt-N:</b> Automatically choose checkpoint interval as sqrt(numLayers)
 *       (use factory method {@link #sqrtN()})</li>
 * </ul>
 *
 * Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class GradientCheckpointConfig implements Serializable {

    /**
     * Checkpoint every N layers during the forward pass.
     * Activations at checkpoint boundaries are kept; intermediate ones are discarded
     * and recomputed during backward.
     * Default: 0 (disabled; use checkpointVariables or sqrtN instead)
     */
    @Builder.Default
    private int checkpointEveryN = 0;

    /**
     * Specific variable names to checkpoint.
     * When set, only these variables' activations are kept during forward pass.
     * All other intermediate activations are discarded and recomputed.
     */
    private Set<String> checkpointVariables;

    /**
     * Whether to offload checkpointed activations to CPU (host) memory.
     * Reduces GPU memory usage further at the cost of D2H/H2D transfer overhead.
     * Default: false
     */
    @Builder.Default
    private boolean offloadToHost = false;

    /**
     * Whether to use an async CUDA stream for D2H offloading.
     * Only effective when offloadToHost is true.
     * Default: true
     */
    @Builder.Default
    private boolean asyncOffload = true;

    /**
     * Maximum memory (in MB) to use for checkpoint storage.
     * 0 = unlimited.
     * Default: 0
     */
    @Builder.Default
    private long maxCheckpointMemoryMB = 0;

    /**
     * Create a sqrt(N) checkpoint configuration.
     * The checkpoint interval is automatically determined based on the number of layers.
     */
    public static GradientCheckpointConfig sqrtN() {
        return GradientCheckpointConfig.builder()
                .checkpointEveryN(-1) // sentinel: compute sqrt(N) at runtime
                .build();
    }

    /**
     * Create a manual checkpoint configuration for specific variables.
     */
    public static GradientCheckpointConfig manual(Set<String> variables) {
        return GradientCheckpointConfig.builder()
                .checkpointVariables(variables)
                .build();
    }

    /**
     * Create a checkpoint-every-N configuration.
     */
    public static GradientCheckpointConfig everyN(int n) {
        return GradientCheckpointConfig.builder()
                .checkpointEveryN(n)
                .build();
    }

    /**
     * Check if this config uses sqrt(N) strategy.
     */
    public boolean isSqrtN() {
        return checkpointEveryN == -1;
    }

    /**
     * Check if this config uses manual variable selection.
     */
    public boolean isManual() {
        return checkpointVariables != null && !checkpointVariables.isEmpty();
    }

    /**
     * Resolve the actual checkpoint interval given the total number of layers.
     */
    public int resolveInterval(int numLayers) {
        if (isSqrtN()) {
            return Math.max(1, (int) Math.sqrt(numLayers));
        }
        return checkpointEveryN > 0 ? checkpointEveryN : numLayers;
    }
}
