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

package org.nd4j.linalg.hexagon;

/**
 * Hexagon NPU device information holder.
 *
 * NOTE: this class intentionally does NOT implement {@link org.nd4j.linalg.factory.Environment}
 * yet. The original scaffolding implemented an old, much smaller revision of that interface and
 * bit-rotted (the module was unbuildable — see ADR 0102). A real Environment implementation only
 * makes sense once the hexagon-mlir bindings exist to answer environment queries; until then
 * {@link HexagonBackend#getEnvironment()} throws UnsupportedOperationException and this class only
 * reports static NPU information derived from environment variables and architectural constants.
 *
 * See ADR 0088 (Hexagon MLIR Backend) for the binding plan and HexagonBackendSmokeTest for the
 * contract tests.
 */
public class HexagonEnvironment {

    private static HexagonEnvironment instance;

    private HexagonEnvironment() {
        // Private constructor for singleton
    }

    public static synchronized HexagonEnvironment getInstance() {
        if (instance == null) {
            instance = new HexagonEnvironment();
        }
        return instance;
    }

    /**
     * Get the Hexagon NPU version string (e.g., "v68", "v69", "v73", "v75").
     * @return NPU version, or "v73" when not detected
     */
    public String getNpuVersion() {
        return System.getenv().getOrDefault("HEXAGON_NPU_VERSION", "v73");
    }

    /**
     * HVX vector register width in bytes.
     * @return 128 — HVX operates on 128-byte vectors on all supported NPU versions
     */
    public int getHvxVectorWidth() {
        return 128;
    }

    /**
     * Get TCM (Tightly Coupled Memory) capacity in bytes for the detected NPU version.
     * Default sizes until the hexagon-mlir runtime can query the device.
     */
    public long getTcmCapacity() {
        String version = getNpuVersion();
        switch (version) {
            case "v75":
                return 1024L * 1024;      // 1 MB
            case "v73":
                return 512L * 1024;       // 512 KB
            case "v68":
            case "v69":
            default:
                return 256L * 1024;       // 256 KB
        }
    }

    /**
     * Check if INT8 is the preferred data type.
     * @return true — Hexagon NPUs are optimized for INT8 quantized inference
     */
    public boolean prefersInt8() {
        return true;
    }
}
