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

package org.nd4j.linalg.metal;

import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Apple Metal / MLX backend for ND4J.
 *
 * <p>Provides GPU-accelerated tensor operations on Apple Silicon (M1-M5+) via two complementary
 * native layers:</p>
 * <ul>
 *   <li><b>MLX graph backend</b> ({@code MlxGraphBackend.cpp}) — builds lazy
 *       {@code mlx::core::array} computation graphs covering element-wise, reduction, matmul,
 *       normalization, convolution, and fused attention. Evaluated via {@code mlx::core::eval()}
 *       which dispatches to Metal GPU shaders. Highest-priority CPU+GPU backend on macOS arm64.</li>
 *   <li><b>Metal ICB replay</b> ({@code MetalReplayHandle.mm}) — pre-encodes compute dispatches
 *       into a {@code MTLIndirectCommandBuffer} (the CUDA-graph-replay analog on Metal) and
 *       replays via {@code executeCommandsInBuffer:withRange:} with near-zero re-encoding
 *       overhead. Enabled by GraphExecutionMode.METAL.</li>
 * </ul>
 *
 * <p>Per-op eager execution (the "gaps" path) uses the existing MPS platform helpers under
 * {@code include/ops/declarable/platform/mps/} (mps_conv, mps_blas, mps_attention, etc.).</p>
 *
 * <h3>Requirements</h3>
 * <ul>
 *   <li>macOS arm64 (Apple Silicon, M1 or later)</li>
 *   <li>macOS 13+ (Ventura; macOS 14+ for full ICB Private-mode support)</li>
 *   <li>Xcode / Metal SDK — required for native compile only, not runtime</li>
 *   <li>MLX framework (libmlx) installed at a known path or via -DSD_MLX=ON cmake build</li>
 * </ul>
 *
 * <h3>Current status</h3>
 * <p>{@link #canRun()} returns {@code false} until native JavaCPP bindings for the MLX/MPS
 * native layer are generated and the native .dylib is present. All Java-side plumbing
 * (SPI registration, properties, module-info, smoke test) is complete and builds on Linux/CI.</p>
 *
 * <p>This mirrors the TPU backend pattern: {@code JTpuBackend.canRun()} is also a stub
 * returning false until PJRT bindings land.</p>
 *
 * @see org.nd4j.common.config.ND4JSystemProperties#GRAPH_EXECUTION_MODE
 */
public class JMetalBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(JMetalBackend.class);

    private static final String LINALG_PROPS = "/nd4j-metal.properties";

    /**
     * System property checked at JVM startup to force-disable the Metal backend
     * even when native bindings are present. Useful for debugging on macOS with
     * only CPU execution desired.
     */
    public static final String METAL_DISABLED_PROPERTY = "nd4j.metal.disabled";

    @Override
    public boolean isAvailable() {
        try {
            return canRun();
        } catch (Exception e) {
            log.debug("Metal backend not available: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public boolean canRun() {
        // Hard-disable via system property
        if (Boolean.getBoolean(METAL_DISABLED_PROPERTY)) {
            log.debug("Metal backend disabled by system property {}", METAL_DISABLED_PROPERTY);
            return false;
        }

        try {
            // Requires macOS arm64 at minimum
            String os   = System.getProperty("os.name", "").toLowerCase();
            String arch = System.getProperty("os.arch", "").toLowerCase();
            if (!os.contains("mac") || (!arch.equals("aarch64") && !arch.equals("arm64"))) {
                log.debug("Metal backend requires macOS arm64 (os={}, arch={})", os, arch);
                return false;
            }

            // TODO: uncomment when JavaCPP native bindings are generated
            // Class.forName("org.nd4j.linalg.metal.bindings.Nd4jMetal");
            // int deviceCount = NativeOpsHolder.getInstance().getDeviceNativeOps().getMetalDeviceCount();
            // return deviceCount > 0;

            log.info("Metal backend detected macOS arm64 — native bindings not yet generated; "
                    + "canRun() returns false until nd4j-metal-preset JavaCPP bindings land.");
            return false;
        } catch (UnsatisfiedLinkError e) {
            log.debug("Metal native library not found: {}", e.getMessage());
            return false;
        } catch (Exception e) {
            log.debug("Error checking Metal availability: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public int getPriority() {
        // Higher priority than CPU (0) and TPU (GPU+10), to prefer Metal when available on macOS.
        // Lower than CUDA (GPU+20 range) since this is an Apple-only path.
        return BACKEND_PRIORITY_GPU + 15;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, JMetalBackend.class.getClassLoader());
    }

    @Override
    public Class<?> getNDArrayClass() {
        // Until Metal-specific NDArray subclass exists, route through BaseNDArray.
        // Operations are dispatched via platform helpers (MPS) at the libnd4j C++ layer.
        return org.nd4j.linalg.api.ndarray.BaseNDArray.class;
    }

    @Override
    public Environment getEnvironment() {
        // MetalEnvironment does not yet implement the Environment interface — the
        // native bindings are required first. canRun() returns false so this is
        // unreachable in practice until bindings land.
        throw new UnsupportedOperationException(
                "Metal backend environment requires MLX/MPS native bindings (not yet generated). "
                        + "See JMetalBackend for status and ADR 0XXX for the design.");
    }

    @Override
    public boolean allowsOrder() {
        return false;
    }

    @Override
    public String buildInfo() {
        return "ND4J Metal Backend (MLX/MPS/ICB — Apple Silicon)";
    }

    @Override
    public void logBackendInit() {
        log.info("ND4J Metal backend initialized");
        log.info("  MLX graph backend:     enabled (MlxGraphBackend — lazy eval via mlx::core)");
        log.info("  MPS per-op helpers:    enabled (mps_*.mm — gap eager execution)");
        log.info("  Metal ICB replay:      enabled (MetalReplayHandle.mm — island capture/replay)");
        log.info("  Target:               Apple Silicon (M1-M5+), macOS 13+");
    }
}
