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

import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Hexagon NPU backend for ND4J using hexagon-mlir.
 *
 * This backend enables tensor operations on Qualcomm Hexagon NPU accelerators
 * found in Snapdragon SoCs. It uses the hexagon-mlir compiler (BSD-3 licensed,
 * open-sourced by Qualcomm in December 2025) to compile MLIR to HVX vector
 * operations.
 *
 * Features:
 * - HVX 128-byte vector operations for parallel computation
 * - TCM (Tightly Coupled Memory) staging for low-latency data access
 * - DMA-based data transfer between DDR and TCM
 * - Command list recording and replay for minimal dispatch overhead
 * - Full compatibility with ND4J/SameDiff APIs
 *
 * Usage:
 * To use the Hexagon backend, ensure:
 * 1. You are running on a Qualcomm Snapdragon SoC with Hexagon NPU
 * 2. hexagon-mlir runtime library is installed
 * 3. This backend JAR is on the classpath
 */
public class HexagonBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(HexagonBackend.class);

    private static final String LINALG_PROPS = "/nd4j-hexagon.properties";

    @Override
    public boolean isAvailable() {
        try {
            return canRun();
        } catch (Exception e) {
            log.debug("Hexagon backend not available: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public boolean canRun() {
        try {
            // Check for Hexagon NPU via native library
            // This would verify:
            // 1. hexagon-mlir runtime library is loadable
            // 2. Hexagon NPU is present and accessible
            // 3. HVX capabilities are available

            // For now, return false until native bindings are generated
            log.info("Checking Hexagon NPU availability...");

            // TODO: Uncomment when native bindings are ready
            // int npuCount = NativeOpsHolder.getInstance().getDeviceNativeOps().getHexagonDeviceCount();
            // return npuCount > 0;

            return false;
        } catch (UnsatisfiedLinkError e) {
            log.debug("hexagon-mlir native library not found: {}", e.getMessage());
            return false;
        } catch (Exception e) {
            log.debug("Error checking Hexagon availability: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public int getPriority() {
        // Higher priority than CPU (0), slightly above TPU for on-device inference
        return BACKEND_PRIORITY_GPU + 5;
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, HexagonBackend.class.getClassLoader());
    }

    @Override
    public Class<?> getNDArrayClass() {
        return org.nd4j.linalg.api.ndarray.BaseNDArray.class;
    }

    @Override
    public Environment getEnvironment() {
        // HexagonEnvironment does not implement the Environment interface until the
        // hexagon-mlir bindings land — this backend cannot be selected while canRun()
        // returns false, so this path is unreachable in practice.
        throw new UnsupportedOperationException(
                "Hexagon backend environment requires hexagon-mlir native bindings (not yet implemented). "
                        + "See HexagonEnvironment for NPU device info and ADR 0088 / ADR 0102.");
    }

    @Override
    public boolean allowsOrder() {
        return false;
    }

    @Override
    public String buildInfo() {
        return "ND4J Hexagon NPU Backend (hexagon-mlir)";
    }

    @Override
    public void logBackendInit() {
        log.info("ND4J Hexagon NPU backend initialized");
        log.info("  Runtime: hexagon-mlir");
        log.info("  Target: Qualcomm Hexagon NPU (HVX)");
    }
}
