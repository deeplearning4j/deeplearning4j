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
 * PlanPhase tracks the plan-level lifecycle phase for the entire DSP execution plan.
 *
 * <p>Enforces a strict progression that makes assumptions easier at each level:</p>
 * <pre>
 *   SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING
 * </pre>
 *
 * <p>Each phase guarantees everything from prior phases plus additional invariants:</p>
 * <ul>
 *   <li><b>SLOT_BY_SLOT</b>: No assumptions. Shapes may change, pointers may move.</li>
 *   <li><b>SHAPES_FROZEN</b>: All output shapes are constant. Shape inference skipped.</li>
 *   <li><b>REPLAYING</b>: Shapes frozen + pointers stable + graph replay active.
 *       Per-segment generation counters gate capture readiness.</li>
 * </ul>
 *
 * <p>Phase is automatically advanced by the C++ execute() based on observed stability.
 * Can be set backward (e.g., unfreeze → SLOT_BY_SLOT).</p>
 *
 * <p>Mirrors the C++ {@code PlanPhase} enum in NativeDynamicShapePlan.h.</p>
 */
public enum PlanPhase {
    /** No guarantees — shapes and pointers may change */
    SLOT_BY_SLOT(0),
    /** Shapes are constant across executions */
    SHAPES_FROZEN(1),
    /** Steady state — shapes frozen + pointers stable + graph replay active */
    REPLAYING(2),

    /**
     * @deprecated Removed — pointer stability is now tracked per-segment via generation counters.
     * Kept for backward compatibility with code that references this value.
     */
    @Deprecated
    POINTERS_STABLE(2);

    private final int nativeCode;

    PlanPhase(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    public int getNativeCode() {
        return nativeCode;
    }

    /**
     * Convert a native code to a PlanPhase enum value.
     * Returns null for invalid codes.
     */
    public static PlanPhase fromNativeCode(int code) {
        for (PlanPhase phase : values()) {
            if (phase.nativeCode == code) return phase;
        }
        return null;
    }

    /**
     * Check if this phase is at least at the given level.
     * For example, {@code REPLAYING.isAtLeast(SHAPES_FROZEN)} returns true.
     */
    public boolean isAtLeast(PlanPhase other) {
        return this.nativeCode >= other.nativeCode;
    }
}
