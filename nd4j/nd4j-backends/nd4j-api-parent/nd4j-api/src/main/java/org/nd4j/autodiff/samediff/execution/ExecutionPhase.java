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
 * ExecutionPhase tracks the ACTUAL runtime execution mode of a DSP segment.
 *
 * Unlike {@link GraphExecutionMode} (which is the user's PREFERENCE for which
 * backend to use), ExecutionPhase tracks what a segment is ACTUALLY doing right now.
 *
 * Lifecycle for capturable segments:
 *   WARMUP → COMPILING → COMPILED → REPLAYING
 *
 * Non-capturable segments stay at SLOT_BY_SLOT always.
 *
 * This enables programmatic assertions about execution stage at both Java and C++ levels.
 */
public enum ExecutionPhase {
    /** First execution — slot-by-slot for shape population */
    WARMUP(0),
    /** Backend is compiling (Triton, NVRTC, CUDA graph capture, oneDNN, etc.) */
    COMPILING(1),
    /** Compiled, first post-compile execution */
    COMPILED(2),
    /** Steady state — graph replay or compiled kernel reuse */
    REPLAYING(3),
    /** Non-capturable segment — always slot-by-slot */
    SLOT_BY_SLOT(4);

    private final int nativeCode;

    ExecutionPhase(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    public int getNativeCode() {
        return nativeCode;
    }

    /**
     * Convert a native code to an ExecutionPhase enum value.
     * Returns null for invalid codes.
     */
    public static ExecutionPhase fromNativeCode(int code) {
        for (ExecutionPhase phase : values()) {
            if (phase.nativeCode == code) return phase;
        }
        return null;
    }
}
