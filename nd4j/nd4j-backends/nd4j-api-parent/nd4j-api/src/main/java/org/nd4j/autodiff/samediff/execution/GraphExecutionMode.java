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
 * <p>ROCm builds: HIP Graphs (mirror of CUDA graph capture/replay).</p>
 * <p>Non-CUDA builds: Triton (if compiled) → MLX (Apple Silicon) → oneDNN → ACL → NNAPI
 * → ARM_HYBRID → MLIR CPU JIT → slot-by-slot.</p>
 * <p>Cross-platform GPU: Level Zero (Intel), Vulkan (cross-vendor), Metal (Apple).</p>
 * <p>Forcing a specific mode skips the others.</p>
 */
public enum GraphExecutionMode {

    /**
     * Automatic backend selection (default). Tries GPU JIT backends first
     * (Triton → NVRTC → PTX), then CUDA graph capture/replay.
     * On non-CUDA builds, AUTO tries Triton first (if available), then CPU graph
     * backends (MLX/oneDNN/ACL/NNAPI/ARM_HYBRID/MLIR), then slot-by-slot.
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
    TRITON(5),

    /**
     * MLX Apple Silicon backend. Uses Metal Performance Shaders via MLX
     * for fusible segments on Apple Silicon devices.
     */
    MLX(6),

    /**
     * ARM Hybrid backend. Uses MLIR with ARM-specific optimizations (NEON,
     * SVE, dot product) for CPU, with optional Vulkan GPU offload for
     * compute-heavy ops (matmul, conv2d) on ARM Mali/Adreno GPUs.
     */
    ARM_HYBRID(7),

    /**
     * Android NNAPI backend. Routes segments through Android's Neural Networks
     * API to leverage hardware accelerators (Hexagon DSP, Mali GPU, NPU)
     * available on the device. Only available on Android API 27+.
     */
    NNAPI(8),

    /**
     * HIP graph capture and replay (AMD ROCm). Mirrors CUDA graph semantics
     * using hipStreamBeginCapture/hipGraphLaunch. Near-identical API surface
     * to CUDA graphs — records kernel launches into a hipGraph_t, instantiates
     * into hipGraphExec_t, and replays with minimal launch overhead.
     */
    HIP_GRAPHS(9),

    /**
     * Intel Level Zero mutable command list replay. Records a command list once,
     * then replays it with optional kernel argument mutation between replays
     * via zeCommandListUpdateMutableCommandsExp(). Experimental API (Level Zero 1.11+).
     */
    LEVEL_ZERO(10),

    /**
     * Vulkan compute command buffer replay. Records a VkCommandBuffer once
     * with VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT and resubmits it
     * for each replay. Cross-platform (AMD, Intel, ARM Mali, Qualcomm Adreno).
     */
    VULKAN(11),

    /**
     * Metal indirect command buffer (ICB) replay. Pre-encodes compute
     * dispatches into an MTLIndirectCommandBuffer and replays via
     * executeCommandsInBuffer. Apple GPU only (Apple Silicon).
     */
    METAL(12),

    /**
     * TPU XLA compilation caching via PJRT. Compiles fusible segments to HLO
     * (High-Level Operations) modules, caches compiled executables via
     * PJRT_Client_Compile, and re-executes cached binaries on subsequent calls.
     * Google Cloud TPU v4/v5 only.
     */
    TPU(13),

    /**
     * Hexagon-MLIR NPU compilation + command list replay. Compiles fusible
     * segments to MLIR targeting Qualcomm Hexagon NPU via hexagon-mlir,
     * stages data through TCM (Tightly Coupled Memory), and dispatches
     * HVX vector operations. Qualcomm Snapdragon SoCs only.
     */
    HEXAGON(14);

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
