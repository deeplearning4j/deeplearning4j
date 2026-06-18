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
 * SlotState tracks the per-operation slot lifecycle within a DSP plan.
 *
 * <p>State transitions (ordered — each state includes all prior guarantees):</p>
 * <pre>
 *   WARMUP → SHAPE_CACHED → FROZEN → FROZEN_CONSTANT
 * </pre>
 *
 * <p>Backward transitions:</p>
 * <ul>
 *   <li>Any → WARMUP: Plan invalidation (shape change)</li>
 *   <li>FROZEN/FROZEN_CONSTANT → SHAPE_CACHED: Unfreeze</li>
 * </ul>
 *
 * <p>Mirrors the C++ {@code NativeSlot::SlotState} enum in NativeDynamicShapePlan.h.</p>
 */
public enum SlotState {
    /** First execution (shape inference + view detection) */
    WARMUP(0),
    /** Shape cache populated, view status determined */
    SHAPE_CACHED(1),
    /** Shapes frozen, context reuse enabled */
    FROZEN(2),
    /** Output never changes, skip execution entirely */
    FROZEN_CONSTANT(3);

    private final int nativeCode;

    SlotState(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    public int getNativeCode() {
        return nativeCode;
    }

    /**
     * Convert a native code to a SlotState enum value.
     * Returns null for invalid codes.
     */
    public static SlotState fromNativeCode(int code) {
        for (SlotState state : values()) {
            if (state.nativeCode == code) return state;
        }
        return null;
    }

    /**
     * Check if this state is at least at the given level.
     */
    public boolean isAtLeast(SlotState other) {
        return this.nativeCode >= other.nativeCode;
    }
}
