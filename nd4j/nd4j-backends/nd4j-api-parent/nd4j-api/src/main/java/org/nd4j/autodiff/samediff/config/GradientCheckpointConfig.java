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
 * Checkpoint storage strategy is controlled by {@link CheckpointStrategy}:
 * <ul>
 *   <li><b>RECOMPUTE:</b> Standard discard + recompute during backward</li>
 *   <li><b>OFFLOAD_HOST:</b> Synchronous D2H copy to CPU RAM during forward</li>
 *   <li><b>OFFLOAD_HOST_ASYNC:</b> Async D2H copy with H2D prefetch for ~2% overhead</li>
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
     * Strategy for checkpoint storage during training.
     */
    public enum CheckpointStrategy {
        /** Standard: discard activations during forward, recompute during backward. */
        RECOMPUTE,
        /** Synchronous D2H offload: copy checkpoints to CPU RAM during forward, sync H2D during backward. */
        OFFLOAD_HOST,
        /** Async D2H offload: non-blocking D2H during forward, prefetch H2D before backward needs them. */
        OFFLOAD_HOST_ASYNC
    }

    /**
     * Strategy for checkpoint storage.
     * Default: RECOMPUTE (standard discard + recompute)
     */
    @Builder.Default
    private CheckpointStrategy strategy = CheckpointStrategy.RECOMPUTE;

    /**
     * Prefetch distance: how many layers ahead to start H2D prefetch during backward.
     * Only effective when strategy is OFFLOAD_HOST_ASYNC.
     * Default: 2
     */
    @Builder.Default
    private int prefetchDistance = 2;

    /**
     * Whether to use pinned (page-locked) host memory for async transfers.
     * Pinned memory allows zero-copy DMA and is significantly faster for large activations.
     * Only effective when strategy is OFFLOAD_HOST or OFFLOAD_HOST_ASYNC.
     * Default: true
     */
    @Builder.Default
    private boolean pinHostMemory = true;

    /**
     * Maximum host memory budget for offloaded checkpoints in MB.
     * When exceeded, the manager falls back to RECOMPUTE for additional checkpoints.
     * 0 = unlimited.
     * Default: 0
     */
    @Builder.Default
    private long maxHostMemoryMB = 0;

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
     *
     * @deprecated Use {@link #strategy} with {@link CheckpointStrategy#OFFLOAD_HOST} or
     *             {@link CheckpointStrategy#OFFLOAD_HOST_ASYNC} instead.
     */
    @Deprecated
    @Builder.Default
    private boolean offloadToHost = false;

    /**
     * Whether to use an async CUDA stream for D2H offloading.
     * Only effective when offloadToHost is true.
     * Default: true
     *
     * @deprecated Use {@link #strategy} with {@link CheckpointStrategy#OFFLOAD_HOST_ASYNC} instead.
     */
    @Deprecated
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

    /**
     * Create an async offload configuration with default prefetch distance (2).
     * Checkpoints every layer, uses pinned memory, unlimited host budget.
     */
    public static GradientCheckpointConfig asyncOffload() {
        return builder()
                .strategy(CheckpointStrategy.OFFLOAD_HOST_ASYNC)
                .checkpointEveryN(1)
                .pinHostMemory(true)
                .prefetchDistance(2)
                .build();
    }

    /**
     * Create an async offload configuration with a custom prefetch distance.
     *
     * @param prefetchDist Number of layers ahead to begin H2D prefetch during backward.
     */
    public static GradientCheckpointConfig asyncOffload(int prefetchDist) {
        return builder()
                .strategy(CheckpointStrategy.OFFLOAD_HOST_ASYNC)
                .checkpointEveryN(1)
                .pinHostMemory(true)
                .prefetchDistance(prefetchDist)
                .build();
    }

    /**
     * Check if this config uses async offload strategy.
     */
    public boolean isAsyncOffload() {
        return strategy == CheckpointStrategy.OFFLOAD_HOST_ASYNC;
    }

    /**
     * Check if this config uses any host offload strategy (sync or async).
     */
    public boolean isAnyOffload() {
        return strategy == CheckpointStrategy.OFFLOAD_HOST
                || strategy == CheckpointStrategy.OFFLOAD_HOST_ASYNC;
    }
}
