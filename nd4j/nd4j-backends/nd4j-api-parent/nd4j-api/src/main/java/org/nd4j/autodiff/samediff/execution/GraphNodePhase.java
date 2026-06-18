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

/**
 * GraphNodePhase — Unified 3-state lifecycle for ALL graph nodes in the DSP system.
 *
 * <p>Every node in the execution graph (RootGraph, SubgraphNode, OpNode) progresses
 * through the same lifecycle:</p>
 * <pre>
 *   BUILDING ────────────────────────► SEALED
 *        │     warmup + compile +          │
 *        │     capture all succeed     steady-state
 *        │                             replay only
 *        │
 *        └───────────────────────────► FAILED
 *                permanent failure       slot-by-slot fallback
 * </pre>
 *
 * <p>Level-specific semantics:</p>
 * <ul>
 *   <li><b>RootGraph (plan)</b>: BUILDING = any child still building; SEALED = all children sealed</li>
 *   <li><b>SubgraphNode (segment)</b>: BUILDING = warmup/compile/capture in progress; SEALED = replay ready</li>
 *   <li><b>OpNode (slot)</b>: BUILDING = shape inference active; SEALED = shape + buffer stable</li>
 * </ul>
 *
 * <p>Key insight: FROZEN_CONSTANT and VIEW_PRODUCER are NOT lifecycle states — they are
 * property flags on a SEALED OpNode. OOM_DEFERRED is NOT a lifecycle state — it is a
 * retry property on a BUILDING SubgraphNode.</p>
 *
 * <p>Mirrors the C++ {@code GraphNodePhase} enum in DspGraphTypes.h.</p>
 */
public enum GraphNodePhase {
    /** Construction in progress (warmup, compile, capture — all sub-phases) */
    BUILDING(0),
    /** Immutable steady-state (replay only, all state frozen) */
    SEALED(1),
    /** Terminal failure (slot-by-slot fallback for this node) */
    FAILED(2);

    private final int code;

    GraphNodePhase(int code) {
        this.code = code;
    }

    public int getCode() {
        return code;
    }

    /**
     * Convert a native code to a GraphNodePhase enum value.
     */
    public static GraphNodePhase fromCode(int code) {
        switch (code) {
            case 0: return BUILDING;
            case 1: return SEALED;
            case 2: return FAILED;
            default: return FAILED;
        }
    }

    /**
     * Convert a PlanPhase to the unified GraphNodePhase.
     */
    public static GraphNodePhase fromPlanPhase(PlanPhase planPhase) {
        if (planPhase == null) return BUILDING;
        switch (planPhase) {
            case REPLAYING: return SEALED;
            default: return BUILDING;
        }
    }

    /**
     * Convert a SlotState to the unified GraphNodePhase.
     */
    public static GraphNodePhase fromSlotState(SlotState slotState) {
        if (slotState == null) return BUILDING;
        switch (slotState) {
            case FROZEN:
            case FROZEN_CONSTANT:
                return SEALED;
            default:
                return BUILDING;
        }
    }

    /**
     * Convert an ExecutionPhase (segment-level) to the unified GraphNodePhase.
     */
    public static GraphNodePhase fromExecutionPhase(ExecutionPhase execPhase) {
        if (execPhase == null) return BUILDING;
        switch (execPhase) {
            case REPLAYING: return SEALED;
            case SLOT_BY_SLOT: return FAILED;
            default: return BUILDING;
        }
    }

    /**
     * Check if this phase means the node is ready for steady-state execution.
     */
    public boolean isSealed() {
        return this == SEALED;
    }

    /**
     * Check if this phase means the node is still being constructed.
     */
    public boolean isBuilding() {
        return this == BUILDING;
    }

    /**
     * Check if this phase means the node has permanently failed.
     */
    public boolean isFailed() {
        return this == FAILED;
    }
}
