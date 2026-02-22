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

package org.nd4j.autodiff.samediff.execution;

import lombok.Getter;
import lombok.Setter;

/**
 * Configuration for tensor parallelism across multiple GPUs.
 * <p>
 * Tensor parallelism splits individual weight matrices across GPUs,
 * enabling models larger than single-GPU memory. Each GPU holds a shard
 * of the weight and computes a partial result, then communicates via
 * AllGather (column-parallel) or AllReduce (row-parallel).
 * <p>
 * Typical usage for a transformer MLP:
 * <pre>
 *   // Column-parallel: W is split along output dimension
 *   // Each GPU holds W[:, rank*shard:(rank+1)*shard]
 *   // AllGather reassembles the full output
 *
 *   // Row-parallel: W is split along input dimension
 *   // Each GPU holds W[rank*shard:(rank+1)*shard, :]
 *   // AllReduce sums partial results
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see TensorParallelRunner
 */
@Getter
@Setter
public class TensorParallelConfig {

    /**
     * Number of GPUs participating in tensor parallelism.
     * Must evenly divide the model's hidden dimension.
     */
    private int tensorParallelSize = 1;

    /**
     * This GPU's rank in the tensor parallel group (0-indexed).
     */
    private int rank = 0;

    /**
     * Name of the communicator group for this TP group.
     * Used to identify the communication channel when multiple
     * parallel groups coexist (e.g., TP + PP).
     */
    private String commGroupName = "tp_group";

    /**
     * Whether to gather output across ranks after column-parallel linear layers.
     * When true, the output is AllGathered so each rank has the full output.
     * When false, each rank retains only its shard (useful when the next layer
     * is row-parallel and expects sharded input).
     */
    private boolean gatherOutput = true;

    /**
     * Whether to reduce (AllReduce sum) output across ranks after row-parallel
     * linear layers. When true, partial sums are combined into the full result.
     * When false, each rank retains its partial sum (for custom reduction).
     */
    private boolean reduceOutput = true;

    /**
     * Device IDs for each rank. If null, uses ranks 0..tensorParallelSize-1.
     * Allows non-contiguous GPU assignment (e.g., {0, 2, 4, 6} for 4-way TP).
     */
    private int[] deviceIds;

    /**
     * Default constructor with TP disabled (single GPU).
     */
    public TensorParallelConfig() {
    }

    /**
     * Create a TP config for the given size and rank.
     *
     * @param tpSize number of GPUs in the TP group
     * @param rank   this GPU's rank (0-indexed)
     */
    private TensorParallelConfig(int tpSize, int rank) {
        if (tpSize < 1) {
            throw new IllegalArgumentException("Tensor parallel size must be >= 1, got " + tpSize);
        }
        if (rank < 0 || rank >= tpSize) {
            throw new IllegalArgumentException("Rank must be in [0, " + (tpSize - 1) + "], got " + rank);
        }
        this.tensorParallelSize = tpSize;
        this.rank = rank;
    }

    /**
     * Static factory method.
     *
     * @param tpSize number of GPUs in the TP group
     * @param rank   this GPU's rank (0-indexed)
     * @return configured TensorParallelConfig
     */
    public static TensorParallelConfig create(int tpSize, int rank) {
        return new TensorParallelConfig(tpSize, rank);
    }

    /**
     * Returns the device ID for the given rank. Falls back to rank index
     * if no explicit device mapping is set.
     *
     * @param rankIndex the rank to look up
     * @return the CUDA device ID
     */
    public int deviceIdForRank(int rankIndex) {
        if (deviceIds != null && rankIndex < deviceIds.length) {
            return deviceIds[rankIndex];
        }
        return rankIndex;
    }

    /**
     * Returns the device ID for this rank.
     */
    public int myDeviceId() {
        return deviceIdForRank(rank);
    }

    /**
     * Returns true if tensor parallelism is actually enabled (size > 1).
     */
    public boolean isEnabled() {
        return tensorParallelSize > 1;
    }

    /**
     * Builder-style setter for commGroupName.
     */
    public TensorParallelConfig withCommGroupName(String name) {
        this.commGroupName = name;
        return this;
    }

    /**
     * Builder-style setter for gatherOutput.
     */
    public TensorParallelConfig withGatherOutput(boolean gather) {
        this.gatherOutput = gather;
        return this;
    }

    /**
     * Builder-style setter for reduceOutput.
     */
    public TensorParallelConfig withReduceOutput(boolean reduce) {
        this.reduceOutput = reduce;
        return this;
    }

    /**
     * Builder-style setter for deviceIds.
     */
    public TensorParallelConfig withDeviceIds(int... ids) {
        this.deviceIds = ids;
        return this;
    }

    @Override
    public String toString() {
        return "TensorParallelConfig{" +
                "tpSize=" + tensorParallelSize +
                ", rank=" + rank +
                ", comm='" + commGroupName + '\'' +
                ", gather=" + gatherOutput +
                ", reduce=" + reduceOutput +
                '}';
    }
}
