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
 * Controls how the DSP (Dynamic Shape Plan) executor runs graph segments.
 * Set via {@link org.nd4j.autodiff.samediff.SameDiff#setGraphExecutionMode(GraphExecutionMode)}.
 *
 * <p>Backends are tried in priority order when AUTO is selected.</p>
 * <p>CUDA builds: Triton → NVRTC → PTX → CUDA Graphs → slot-by-slot.</p>
 * <p>Non-CUDA builds: Triton (if HIP/Level Zero available) → oneDNN/ACL CPU graph backends
 * → slot-by-slot.</p>
 * <p>Forcing a specific mode skips the others.</p>
 */
public enum GraphExecutionMode {

    /**
     * Automatic backend selection (default). Tries GPU JIT backends first
     * (Triton → NVRTC → PTX), then CUDA graph capture/replay.
     * On non-CUDA builds, AUTO tries Triton first (if available), then CPU graph
     * backends (oneDNN/ACL), then slot-by-slot.
     */
    AUTO(0),

    /**
     * Execute each op individually. No fusion, no graph capture.
     * Useful as a correctness baseline.
     */
    SLOT_BY_SLOT(1),

    /**
     * CUDA graph capture and replay. Records a sequence of kernel launches
     * into a CUDA graph on first execution, then replays the graph on
     * subsequent executions with near-zero launch overhead.
     * On non-CUDA builds, this maps to CPU graph replay backends
     * (oneDNN Graph / ARM ACL dynamic fusion).
     */
    CUDA_GRAPHS(2),

    /**
     * NVRTC JIT compilation. Generates CUDA C++ source for fusible element-wise
     * segments, compiles with NVRTC at runtime, loads and launches the fused kernel.
     * On non-CUDA builds, this maps to TRITON.
     */
    NVRTC_JIT(3),

    /**
     * PTX template backend. Generates PTX assembly text directly for fusible
     * element-wise segments. Fastest "compilation" (string concatenation),
     * but code is less optimized than NVRTC.
     * On non-CUDA builds, this maps to TRITON.
     */
    PTX_JIT(4),

    /**
     * Triton MLIR backend. Builds MLIR IR for fusible segments, compiles through
     * the full Triton pipeline (TTIR → TTGIR → LLVM → PTX), loads and launches.
     * Produces the most optimized fused kernels.
     */
    TRITON(5);

    private final int nativeCode;

    GraphExecutionMode(int nativeCode) {
        this.nativeCode = nativeCode;
    }

    /**
     * Returns the integer code passed to the C++ NativeDynamicShapePlan.
     */
    public int getNativeCode() {
        return nativeCode;
    }

    /**
     * Look up a mode from its native integer code.
     */
    public static GraphExecutionMode fromNativeCode(int code) {
        for (GraphExecutionMode m : values()) {
            if (m.nativeCode == code) return m;
        }
        return AUTO;
    }
}
