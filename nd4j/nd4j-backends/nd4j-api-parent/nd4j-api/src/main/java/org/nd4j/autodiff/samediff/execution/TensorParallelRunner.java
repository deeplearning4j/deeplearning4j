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
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Orchestrates SameDiff model execution across tensor-parallel ranks.
 * <p>
 * This runner takes a full model and a {@link TensorParallelConfig}, then:
 * <ol>
 *   <li>{@link #shardWeights()} - Splits weight matrices across TP ranks.
 *       Column-parallel layers split along the output dimension; row-parallel
 *       layers split along the input dimension.</li>
 *   <li>{@link #execute(Map)} - Runs the forward pass on this rank's GPU,
 *       with AllGather/AllReduce communication as needed.</li>
 *   <li>{@link #close()} - Releases sharded weight buffers and communication handles.</li>
 * </ol>
 * <p>
 * Weight sharding convention:
 * <pre>
 *   Column-parallel (gate, up projections):
 *     Full W: [in_features, out_features]
 *     Shard:  W[:, rank * shard_size : (rank+1) * shard_size]
 *
 *   Row-parallel (down projection):
 *     Full W: [in_features, out_features]
 *     Shard:  W[rank * shard_size : (rank+1) * shard_size, :]
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see TensorParallelConfig
 */
@Slf4j
public class TensorParallelRunner implements Closeable {

    @Getter
    private final SameDiff model;

    @Getter
    private final TensorParallelConfig config;

    /**
     * Maps variable names to their sharding strategy.
     * "column" = split along last dimension, "row" = split along first dimension.
     */
    private final Map<String, ShardStrategy> shardStrategies = new LinkedHashMap<>();

    /**
     * Tracks which variables have been sharded (to avoid double-sharding).
     */
    private final Map<String, INDArray> originalWeights = new HashMap<>();

    private boolean weightsSharded = false;

    /**
     * Collective communicator for AllReduce/AllGather operations.
     * Lazily initialized via {@link CollectiveCommunicatorFactory}.
     */
    private CollectiveCommunicator communicator;

    /**
     * Tracks which output variable names need post-execution AllGather
     * (outputs of column-parallel ops) or AllReduce (outputs of row-parallel ops).
     */
    private final Map<String, ShardStrategy> outputCommunicationNeeds = new LinkedHashMap<>();

    /**
     * Sharding strategy for a weight variable.
     */
    public enum ShardStrategy {
        /** Split along last (output) dimension -- used for column-parallel (gate/up projections) */
        COLUMN,
        /** Split along first (input) dimension -- used for row-parallel (down projection) */
        ROW,
        /** Replicate on all ranks (no sharding) -- used for embeddings, norms, biases */
        REPLICATE
    }

    /**
     * Create a tensor parallel runner.
     *
     * @param model  the SameDiff model (will be modified in-place during shardWeights)
     * @param config tensor parallelism configuration
     */
    public TensorParallelRunner(SameDiff model, TensorParallelConfig config) {
        if (model == null) {
            throw new IllegalArgumentException("Model must not be null");
        }
        if (config == null) {
            throw new IllegalArgumentException("Config must not be null");
        }
        this.model = model;
        this.config = config;
    }

    /**
     * Register a weight variable for column-parallel sharding.
     * The weight will be split along the last dimension when {@link #shardWeights()} is called.
     *
     * @param variableName the SameDiff variable name for the weight
     * @return this runner for chaining
     */
    public TensorParallelRunner markColumnParallel(String variableName) {
        shardStrategies.put(variableName, ShardStrategy.COLUMN);
        return this;
    }

    /**
     * Register a weight variable for row-parallel sharding.
     * The weight will be split along the first dimension when {@link #shardWeights()} is called.
     *
     * @param variableName the SameDiff variable name for the weight
     * @return this runner for chaining
     */
    public TensorParallelRunner markRowParallel(String variableName) {
        shardStrategies.put(variableName, ShardStrategy.ROW);
        return this;
    }

    /**
     * Register a weight variable to be replicated (not sharded).
     * This is the default for variables not explicitly registered.
     *
     * @param variableName the SameDiff variable name
     * @return this runner for chaining
     */
    public TensorParallelRunner markReplicate(String variableName) {
        shardStrategies.put(variableName, ShardStrategy.REPLICATE);
        return this;
    }

    /**
     * Automatically detect and register sharding strategies based on naming conventions.
     * <p>
     * Convention:
     * <ul>
     *   <li>Names containing "gate_proj" or "up_proj" or "q_proj" or "k_proj" or "v_proj" -> COLUMN</li>
     *   <li>Names containing "down_proj" or "o_proj" -> ROW</li>
     *   <li>Everything else -> REPLICATE</li>
     * </ul>
     *
     * @return this runner for chaining
     */
    public TensorParallelRunner autoDetectSharding() {
        for (SDVariable var : model.variables()) {
            String name = var.name();
            if (name.contains("gate_proj") || name.contains("up_proj") ||
                    name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj")) {
                shardStrategies.put(name, ShardStrategy.COLUMN);
            } else if (name.contains("down_proj") || name.contains("o_proj")) {
                shardStrategies.put(name, ShardStrategy.ROW);
            }
            // Everything else is implicitly REPLICATE (not sharded)
        }
        return this;
    }

    /**
     * Shard model weights across TP ranks according to registered strategies.
     * <p>
     * After this call, each weight variable in the model holds only this rank's shard.
     * Original weights are cached for potential restoration via {@link #restoreWeights()}.
     *
     * @throws IllegalStateException if weights have already been sharded
     * @throws IllegalArgumentException if a weight dimension is not evenly divisible by TP size
     */
    public void shardWeights() {
        if (weightsSharded) {
            throw new IllegalStateException("Weights have already been sharded. Call restoreWeights() first.");
        }
        if (!config.isEnabled()) {
            log.info("Tensor parallelism is disabled (size=1), skipping sharding");
            return;
        }

        int tpSize = config.getTensorParallelSize();
        int rank = config.getRank();

        for (Map.Entry<String, ShardStrategy> entry : shardStrategies.entrySet()) {
            String varName = entry.getKey();
            ShardStrategy strategy = entry.getValue();

            if (strategy == ShardStrategy.REPLICATE) {
                continue;
            }

            SDVariable sdVar = model.getVariable(varName);
            if (sdVar == null) {
                log.warn("Variable '{}' not found in model, skipping sharding", varName);
                continue;
            }

            INDArray fullWeight = sdVar.getArr();
            if (fullWeight == null) {
                log.warn("Variable '{}' has no array data, skipping sharding", varName);
                continue;
            }

            // Cache original for restoration
            originalWeights.put(varName, fullWeight);

            INDArray shard;
            if (strategy == ShardStrategy.COLUMN) {
                // Split along last dimension
                int lastDim = fullWeight.rank() - 1;
                long dimSize = fullWeight.size(lastDim);
                if (dimSize % tpSize != 0) {
                    throw new IllegalArgumentException(
                            "Cannot shard '" + varName + "' (column): dimension " + lastDim +
                                    " size " + dimSize + " is not divisible by TP size " + tpSize);
                }
                long shardSize = dimSize / tpSize;
                long start = rank * shardSize;
                // Use INDArray interval slicing along the last dimension
                shard = sliceAlongDimension(fullWeight, lastDim, start, start + shardSize);
            } else {
                // ROW: split along first dimension
                long dimSize = fullWeight.size(0);
                if (dimSize % tpSize != 0) {
                    throw new IllegalArgumentException(
                            "Cannot shard '" + varName + "' (row): dimension 0 size " +
                                    dimSize + " is not divisible by TP size " + tpSize);
                }
                long shardSize = dimSize / tpSize;
                long start = rank * shardSize;
                shard = sliceAlongDimension(fullWeight, 0, start, start + shardSize);
            }

            // Force contiguous copy (views from get() may have stale device buffers)
            INDArray contiguousShard = shard.dup();
            model.associateArrayWithVariable(contiguousShard, sdVar);

            log.debug("Sharded '{}' ({}) to shape {} on rank {}", varName, strategy,
                    java.util.Arrays.toString(contiguousShard.shape()), rank);
        }

        weightsSharded = true;
        log.info("Weight sharding complete for rank {} / {} (sharded {} variables)",
                rank, tpSize, shardStrategies.size());
    }

    /**
     * Initialize the collective communicator for this runner.
     * Called automatically on first execute() if TP is enabled.
     * Uses {@link CollectiveCommunicatorFactory} to select the best
     * available backend (NCCL for GPU, local fallback for CPU).
     */
    public void initCommunicator() {
        if (communicator != null) {
            return;
        }
        communicator = CollectiveCommunicatorFactory.create(config);
        log.info("Initialized {} communicator for TP rank {}/{}",
                communicator.backendName(), config.getRank(), config.getTensorParallelSize());
    }

    /**
     * Set a custom collective communicator (for testing or custom backends).
     *
     * @param comm the communicator to use
     */
    public void setCommunicator(CollectiveCommunicator comm) {
        if (this.communicator != null) {
            this.communicator.close();
        }
        this.communicator = comm;
    }

    /**
     * Register an output variable that needs AllGather after execution.
     * Typically the output of a column_parallel_linear op.
     *
     * @param outputVarName the output variable name
     * @return this runner for chaining
     */
    public TensorParallelRunner markOutputNeedsAllGather(String outputVarName) {
        outputCommunicationNeeds.put(outputVarName, ShardStrategy.COLUMN);
        return this;
    }

    /**
     * Register an output variable that needs AllReduce after execution.
     * Typically the output of a row_parallel_linear op.
     *
     * @param outputVarName the output variable name
     * @return this runner for chaining
     */
    public TensorParallelRunner markOutputNeedsAllReduce(String outputVarName) {
        outputCommunicationNeeds.put(outputVarName, ShardStrategy.ROW);
        return this;
    }

    /**
     * Execute the model forward pass with tensor-parallel communication.
     * <p>
     * This runs the SameDiff graph on this rank's device. After execution,
     * outputs that need AllGather or AllReduce are communicated via NCCL.
     *
     * @param inputs input variables (same on all ranks for non-sharded inputs)
     * @return output variable map (with communicated results)
     */
    public Map<String, INDArray> execute(Map<String, INDArray> inputs) {
        if (!weightsSharded && config.isEnabled()) {
            log.warn("Weights have not been sharded yet. Call shardWeights() before execute().");
        }

        // Lazily initialize communicator
        if (config.isEnabled() && communicator == null) {
            initCommunicator();
        }

        // Execute on this rank's device
        Map<String, INDArray> outputs = model.output(inputs, model.outputs().toArray(new String[0]));

        // Apply collective communication to outputs that need it
        if (config.isEnabled() && communicator != null && !outputCommunicationNeeds.isEmpty()) {
            for (Map.Entry<String, ShardStrategy> entry : outputCommunicationNeeds.entrySet()) {
                String varName = entry.getKey();
                INDArray output = outputs.get(varName);
                if (output == null) continue;

                if (entry.getValue() == ShardStrategy.COLUMN && config.isGatherOutput()) {
                    // AllGather along last dimension for column-parallel outputs
                    INDArray gathered = communicator.allGatherAlongDim(output, -1);
                    outputs.put(varName, gathered);
                    log.debug("AllGathered output '{}' from {} to {}", varName,
                            java.util.Arrays.toString(output.shape()),
                            java.util.Arrays.toString(gathered.shape()));
                } else if (entry.getValue() == ShardStrategy.ROW && config.isReduceOutput()) {
                    // AllReduce sum for row-parallel outputs
                    communicator.allReduceSum(output);
                    log.debug("AllReduced output '{}'", varName);
                }
            }
        }

        return outputs;
    }

    /**
     * Execute and return a single named output.
     *
     * @param inputs     input variable map
     * @param outputName the output variable name to return
     * @return the output array
     */
    public INDArray execute(Map<String, INDArray> inputs, String outputName) {
        Map<String, INDArray> outputs = execute(inputs);
        INDArray result = outputs.get(outputName);
        if (result == null) {
            throw new IllegalArgumentException("Output '" + outputName + "' not found. Available: " + outputs.keySet());
        }
        return result;
    }

    /**
     * Restore original (un-sharded) weights. Useful for saving the full model
     * or switching to a different TP configuration.
     */
    public void restoreWeights() {
        if (!weightsSharded) {
            return;
        }

        for (Map.Entry<String, INDArray> entry : originalWeights.entrySet()) {
            SDVariable sdVar = model.getVariable(entry.getKey());
            if (sdVar != null) {
                model.associateArrayWithVariable(entry.getValue(), sdVar);
            }
        }
        originalWeights.clear();
        weightsSharded = false;
        log.info("Restored original un-sharded weights");
    }

    /**
     * Slice an array along a specific dimension.
     */
    private static INDArray sliceAlongDimension(INDArray arr, int dimension, long fromInclusive, long toExclusive) {
        // Build interval indices for each dimension
        // For the target dimension, use the range [from, to); for others, use all
        if (arr.rank() == 2) {
            if (dimension == 0) {
                return arr.get(Nd4j.createFromArray(new long[]{fromInclusive, toExclusive}));
            } else {
                // For 2D column slicing, use interval notation
                return arr.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.interval(fromInclusive, toExclusive)
                );
            }
        }
        // General case: build index array
        org.nd4j.linalg.indexing.INDArrayIndex[] indices =
                new org.nd4j.linalg.indexing.INDArrayIndex[arr.rank()];
        for (int d = 0; d < arr.rank(); d++) {
            if (d == dimension) {
                indices[d] = org.nd4j.linalg.indexing.NDArrayIndex.interval(fromInclusive, toExclusive);
            } else {
                indices[d] = org.nd4j.linalg.indexing.NDArrayIndex.all();
            }
        }
        return arr.get(indices);
    }

    /**
     * Get the collective communicator (may be null if not yet initialized).
     */
    public CollectiveCommunicator getCommunicator() {
        return communicator;
    }

    @Override
    public void close() {
        restoreWeights();
        shardStrategies.clear();
        outputCommunicationNeeds.clear();
        if (communicator != null) {
            communicator.close();
            communicator = null;
        }
        log.info("TensorParallelRunner closed for rank {}", config.getRank());
    }
}
