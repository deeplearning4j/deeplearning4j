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

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Estimates memory requirements for SameDiff model variables and DSP plan activations.
 * Used by {@link DevicePlacementPlanner} to make informed device placement decisions.
 */
public class ModelMemoryEstimator {

    /**
     * Per-variable memory estimate.
     */
    @Data
    @AllArgsConstructor
    public static class MemoryEstimate {
        private final String variableName;
        private final long estimatedBytes;
        private final boolean constant;
        private final boolean activation;
    }

    /**
     * Estimate total weight (constant + variable) memory for a SameDiff model.
     * Uses exact array sizes since these arrays already exist.
     *
     * @param sd the SameDiff instance
     * @return total bytes of constant and variable arrays
     */
    public static long estimateWeightMemory(SameDiff sd) {
        long total = 0;
        for (SDVariable var : sd.variables()) {
            VariableType type = var.getVariableType();
            if (type == VariableType.CONSTANT || type == VariableType.VARIABLE) {
                INDArray arr = var.getArr();
                if (arr != null) {
                    total += arr.length() * arr.dataType().width();
                }
            }
        }
        return total;
    }

    /**
     * Estimate memory for all variables in the model, categorized by type.
     *
     * @param sd the SameDiff instance
     * @return list of per-variable memory estimates
     */
    public static List<MemoryEstimate> estimateAll(SameDiff sd) {
        List<MemoryEstimate> estimates = new ArrayList<>();
        for (SDVariable var : sd.variables()) {
            VariableType type = var.getVariableType();
            boolean isConstant = (type == VariableType.CONSTANT || type == VariableType.VARIABLE);
            boolean isActivation = (type == VariableType.ARRAY);

            long bytes = 0;
            INDArray arr = var.getArr();
            if (arr != null) {
                bytes = arr.length() * arr.dataType().width();
            } else if (var.getShape() != null) {
                // Estimate from shape metadata if array not materialized
                long[] shape = var.getShape();
                long elements = 1;
                for (long dim : shape) {
                    if (dim > 0) elements *= dim;
                }
                bytes = elements * (var.dataType() != null ? var.dataType().width() : 4);
            }

            estimates.add(new MemoryEstimate(var.name(), bytes, isConstant, isActivation));
        }
        return estimates;
    }

    /**
     * Estimate peak live activation memory for a range of slots in a DSP plan.
     * Uses the {@code releaseAtStep} liveness data to track which outputs are live
     * at each step, computing the maximum concurrent activation memory.
     *
     * @param plan      the compiled DSP plan
     * @param startSlot inclusive start slot index
     * @param endSlot   exclusive end slot index
     * @return estimated peak activation bytes in the slot range
     */
    public static long estimatePeakMemory(DynamicShapePlan plan, int startSlot, int endSlot) {
        DynamicShapeSlot[] slots = plan.getSlots();
        int[][] releaseAtStep = plan.getReleaseAtStep();
        if (slots == null || slots.length == 0) return 0;

        int actualEnd = Math.min(endSlot, slots.length);
        int actualStart = Math.max(startSlot, 0);

        // Track estimated bytes per output slot
        long[] outputBytes = new long[plan.getTotalOutputSlots()];
        // Estimate each slot's output size from its data type (assume scalar minimum)
        for (int i = actualStart; i < actualEnd; i++) {
            DynamicShapeSlot slot = slots[i];
            int[] outIndices = slot.getOutputSlotIndices();
            if (outIndices != null) {
                for (int outIdx : outIndices) {
                    // Conservative estimate: use float32 (4 bytes) as default
                    // Actual sizes depend on shape inference which hasn't run yet
                    outputBytes[outIdx] = 4; // placeholder per-element, will be multiplied by shape
                }
            }
        }

        // Walk slots and track live memory using releaseAtStep
        long currentLive = 0;
        long peakLive = 0;

        for (int step = actualStart; step < actualEnd; step++) {
            // Add outputs produced by this step
            DynamicShapeSlot slot = slots[step];
            int[] outIndices = slot.getOutputSlotIndices();
            if (outIndices != null) {
                for (int outIdx : outIndices) {
                    currentLive += outputBytes[outIdx];
                }
            }

            peakLive = Math.max(peakLive, currentLive);

            // Release outputs that die after this step
            if (releaseAtStep != null && step < releaseAtStep.length) {
                int[] toRelease = releaseAtStep[step];
                if (toRelease != null) {
                    for (int relIdx : toRelease) {
                        currentLive -= outputBytes[relIdx];
                    }
                }
            }
        }

        return peakLive;
    }

    private ModelMemoryEstimator() {
        // Utility class
    }
}
