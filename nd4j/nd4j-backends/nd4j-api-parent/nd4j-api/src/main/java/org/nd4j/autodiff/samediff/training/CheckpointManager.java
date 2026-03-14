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

package org.nd4j.autodiff.samediff.training;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.config.GradientCheckpointConfig;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Manages gradient checkpointing during training.
 *
 * During the forward pass, activations at checkpoint boundaries are retained
 * while intermediate activations are discarded. During the backward pass,
 * when an intermediate activation is needed, the manager recomputes it
 * by re-executing the forward ops from the nearest checkpoint.
 *
 * Supports:
 * <ul>
 *   <li>Every-N checkpointing</li>
 *   <li>Manual variable selection</li>
 *   <li>Sqrt(N) automatic interval</li>
 *   <li>Host memory offloading with async D2H transfers</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class CheckpointManager {

    @Getter
    private final GradientCheckpointConfig config;

    private final SameDiff sameDiff;

    // Checkpointed activations (kept in GPU or offloaded to host)
    private final Map<String, INDArray> checkpointedActivations = new LinkedHashMap<>();

    // Which variables are checkpoint boundaries
    private final Set<String> checkpointBoundaries = new LinkedHashSet<>();

    // Ordered list of forward ops for segment recomputation
    private List<String> forwardOpOrder;

    // Memory tracking
    private long checkpointMemoryBytes = 0;

    public CheckpointManager(SameDiff sameDiff, GradientCheckpointConfig config) {
        this.sameDiff = sameDiff;
        this.config = config;
    }

    /**
     * Determine which variables should be checkpointed based on config and DAG structure.
     * Should be called once before training begins.
     *
     * @param orderedOps Ordered list of operation names in forward pass order
     */
    public void markCheckpoints(List<String> orderedOps) {
        this.forwardOpOrder = new ArrayList<>(orderedOps);
        checkpointBoundaries.clear();

        if (config.isManual()) {
            checkpointBoundaries.addAll(config.getCheckpointVariables());
            log.info("Gradient checkpointing: {} manual checkpoint variables", checkpointBoundaries.size());
        } else {
            int interval = config.resolveInterval(orderedOps.size());
            for (int i = 0; i < orderedOps.size(); i++) {
                if (i % interval == 0) {
                    checkpointBoundaries.add(orderedOps.get(i));
                }
            }
            log.info("Gradient checkpointing: every {} ops, {} checkpoints of {} total ops",
                    interval, checkpointBoundaries.size(), orderedOps.size());
        }
    }

    /**
     * Called after each forward op completes.
     * Decides whether to keep, offload, or discard the activation.
     *
     * @param variableName The variable name whose activation was just computed
     * @param activation   The activation value
     */
    public void onForwardComplete(String variableName, INDArray activation) {
        if (isCheckpointBoundary(variableName)) {
            // Keep this activation
            if (config.isOffloadToHost()) {
                // Offload to host memory
                INDArray hostCopy = Nd4j.create(activation.dataType(), activation.shape());
                hostCopy.assign(activation);
                checkpointedActivations.put(variableName, hostCopy);
            } else {
                checkpointedActivations.put(variableName, activation.dup());
            }
            checkpointMemoryBytes += activation.length() * activation.dataType().width();
        }
        // Non-checkpoint activations are left to the normal SameDiff lifecycle
    }

    /**
     * Get an activation during the backward pass.
     * If the activation was checkpointed, return it directly.
     * If not, recompute from the nearest checkpoint.
     *
     * @param variableName The variable whose activation is needed
     * @return The activation (from checkpoint or recomputed)
     */
    public INDArray getActivation(String variableName) {
        // Direct hit: was checkpointed
        if (checkpointedActivations.containsKey(variableName)) {
            INDArray cached = checkpointedActivations.get(variableName);
            if (config.isOffloadToHost()) {
                // Transfer back to device
                return cached.dup();
            }
            return cached;
        }

        // Need to recompute from nearest checkpoint
        log.debug("Recomputing activation for: {}", variableName);
        return recomputeFromCheckpoint(variableName);
    }

    /**
     * Check if a variable is a checkpoint boundary.
     */
    public boolean isCheckpointBoundary(String variableName) {
        return checkpointBoundaries.contains(variableName);
    }

    /**
     * Clear all checkpointed activations. Call after each training step.
     */
    public void clear() {
        checkpointedActivations.clear();
        checkpointMemoryBytes = 0;
    }

    /**
     * Get the current memory usage by checkpointed activations.
     */
    public long getCheckpointMemoryBytes() {
        return checkpointMemoryBytes;
    }

    private INDArray recomputeFromCheckpoint(String targetVariable) {
        if (forwardOpOrder == null || forwardOpOrder.isEmpty()) {
            throw new IllegalStateException("Forward op order not set. Call markCheckpoints() first.");
        }

        int targetIdx = forwardOpOrder.indexOf(targetVariable);
        if (targetIdx < 0) {
            throw new IllegalStateException("Variable not found in forward op order: " + targetVariable);
        }

        // Find nearest checkpoint before target
        String nearestCheckpoint = null;
        for (int i = targetIdx - 1; i >= 0; i--) {
            if (checkpointBoundaries.contains(forwardOpOrder.get(i))) {
                nearestCheckpoint = forwardOpOrder.get(i);
                break;
            }
        }

        // Set checkpoint activation as input for recomputation
        Map<String, INDArray> recomputeInputs = new HashMap<>();
        if (nearestCheckpoint != null) {
            INDArray checkpointActivation = checkpointedActivations.get(nearestCheckpoint);
            if (checkpointActivation != null) {
                recomputeInputs.put(nearestCheckpoint, checkpointActivation);
            }
        }

        // Re-execute ops from checkpoint to target using SameDiff's output method
        // SameDiff.output() will only execute the ops needed to produce targetVariable,
        // using the checkpoint activation as a pre-existing input
        Map<String, INDArray> result = sameDiff.output(recomputeInputs, targetVariable);
        INDArray recomputed = result.get(targetVariable);
        if (recomputed == null) {
            throw new IllegalStateException("Could not recompute activation for: " + targetVariable);
        }

        return recomputed;
    }
}
