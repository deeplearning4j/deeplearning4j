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

import lombok.Data;
import org.nd4j.linalg.api.ops.OpContext;

import java.io.Closeable;
import java.util.Map;
import java.util.Set;

/**
 * A compiled execution plan for autoregressive inference with dynamic shapes.
 *
 * <p>Unlike {@link ExecutionPlan} which pre-allocates a fixed workspace for static shapes,
 * DynamicShapePlan handles the case where shapes change every step (e.g., growing KV cache).
 * It pre-compiles the graph wiring (input/output index mapping, liveness schedule) once,
 * then uses flat array-indexed slots instead of string-keyed HashMaps on each step.</p>
 *
 * <p>Key components:</p>
 * <ul>
 *   <li>{@code slots} — pre-compiled per-op descriptors with integer-indexed wiring</li>
 *   <li>{@code releaseAtStep} — pre-computed liveness: which output slots to release after each step</li>
 *   <li>{@code opContextPool} — pre-allocated OpContext pool (one per slot, reused across calls)</li>
 *   <li>{@code externalInputKeys} — ordered array of constant/variable/placeholder names for external resolution</li>
 * </ul>
 *
 * @see DynamicShapeSlot
 * @see DynamicShapePlanCompiler
 * @see DynamicShapePlanExecutor
 */
@Data
public class DynamicShapePlan implements Closeable {

    /** Pre-compiled per-op descriptors in execution order. */
    private final DynamicShapeSlot[] slots;

    /** Total number of flat output slots (for INDArray[] allocation). */
    private final int totalOutputSlots;

    /**
     * Release schedule: {@code releaseAtStep[i]} contains the flat output slot indices
     * that become dead after step {@code i} completes. Empty array if nothing to release.
     */
    private final int[][] releaseAtStep;

    /** Pre-allocated OpContext pool, one per slot. Reused across execute() calls. */
    private final OpContext[] opContextPool;

    /**
     * Ordered array of external input variable names (constants, variables, placeholders).
     * External inputs are referenced by negative indices in DynamicShapeSlot.inputSourceIndices:
     * {@code -(index + 1)} maps into this array.
     */
    private final String[] externalInputKeys;

    /** The set of output variable names this plan was compiled for. */
    private final Set<String> requestedOutputs;

    /**
     * Pre-built mapping from requested output variable name to its flat output slot index.
     * Enables O(1) output collection instead of O(outputs * slots) linear search.
     */
    private final Map<String, Integer> outputNameToSlotIndex;

    /** Whether any slot has control flow ops (plan is invalid if true — should not be created). */
    private final boolean hasControlFlowOps;

    /**
     * Get a human-readable summary of this plan.
     */
    public String getSummary() {
        int totalReleasable = 0;
        for (int[] releases : releaseAtStep) {
            totalReleasable += releases.length;
        }
        return "DynamicShapePlan{slots=" + slots.length +
                ", outputSlots=" + totalOutputSlots +
                ", externalInputs=" + externalInputKeys.length +
                ", releasableSlots=" + totalReleasable +
                ", outputs=" + requestedOutputs +
                "}";
    }

    @Override
    public void close() {
        if (opContextPool != null) {
            for (OpContext ctx : opContextPool) {
                if (ctx != null) {
                    try {
                        ctx.close();
                    } catch (Exception e) {
                        // ignore cleanup errors
                    }
                }
            }
        }
    }
}
