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

package org.nd4j.linalg.jzluda;

import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.Resource;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

/**
 * ZLUDA Backend for ND4J
 *
 * This backend enables running CUDA-based ND4J operations on AMD and Intel GPUs
 * through the ZLUDA transpiler. ZLUDA translates CUDA API calls to:
 * - HIP/ROCm for AMD GPUs
 * - Level Zero for Intel GPUs
 *
 * Requirements:
 * - ZLUDA_PATH environment variable must point to ZLUDA installation
 * - For AMD: ROCm toolkit must be installed
 * - For Intel: oneAPI Level Zero runtime must be installed
 */
public class JZludaBackend extends Nd4jBackend {

    private static final Logger log = LoggerFactory.getLogger(JZludaBackend.class);
    private static final String LINALG_PROPS = "/nd4j-jzluda.properties";

    /**
     * ZLUDA target GPU vendors
     */
    public enum ZludaTarget {
        AMD,    // ROCm/HIP backend
        INTEL,  // Level Zero backend
        UNKNOWN
    }

    private ZludaTarget detectedTarget = ZludaTarget.UNKNOWN;
    private boolean zludaAvailable = false;

    public JZludaBackend() {
        // Check ZLUDA availability on construction
        zludaAvailable = checkZludaAvailable();
        if (zludaAvailable) {
            detectedTarget = detectTarget();
            log.info("ZLUDA backend initialized for {} GPUs", detectedTarget);
        }
    }

    @Override
    public boolean isAvailable() {
        return zludaAvailable && detectedTarget != ZludaTarget.UNKNOWN;
    }

    @Override
    public boolean canRun() {
        if (!isAvailable()) {
            return false;
        }

        try {
            // Verify the target GPU is actually available
            switch (detectedTarget) {
                case AMD:
                    return checkAmdGpuAvailable();
                case INTEL:
                    return checkIntelGpuAvailable();
                default:
                    return false;
            }
        } catch (Exception e) {
            log.warn("ZLUDA GPU check failed: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public int getPriority() {
        // Lower priority than native CUDA (100), higher than CPU (0)
        // This ensures native CUDA is preferred when available
        return BACKEND_PRIORITY_GPU - 10;  // 90
    }

    @Override
    public Resource getConfigurationResource() {
        return new ClassPathResource(LINALG_PROPS, JZludaBackend.class.getClassLoader());
    }

    @Override
    public Class<?> getNDArrayClass() {
        // Reuse CUDA NDArray class since ZLUDA is CUDA API-compatible
        try {
            return Class.forName("org.nd4j.linalg.jcublas.JCublasNDArray");
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("CUDA NDArray class not found - nd4j-cuda dependency required", e);
        }
    }

    @Override
    public Environment getEnvironment() {
        return ZludaEnvironment.getInstance();
    }

    /**
     * Get the detected ZLUDA target GPU vendor
     */
    public ZludaTarget getTarget() {
        return detectedTarget;
    }

    /**
     * Check if ZLUDA runtime is available
     */
    private boolean checkZludaAvailable() {
        String zludaPath = System.getenv("ZLUDA_PATH");
        if (zludaPath == null || zludaPath.isEmpty()) {
            log.debug("ZLUDA_PATH environment variable not set");
            return false;
        }

        File zludaDir = new File(zludaPath);
        if (!zludaDir.exists() || !zludaDir.isDirectory()) {
            log.debug("ZLUDA_PATH does not point to valid directory: {}", zludaPath);
            return false;
        }

        // Check for ZLUDA library
        File libDir = new File(zludaDir, "lib");
        if (libDir.exists()) {
            File libCuda = new File(libDir, "libcuda.so");
            File libNvCuda = new File(libDir, "libnvcuda.so");
            if (libCuda.exists() || libNvCuda.exists()) {
                log.info("Found ZLUDA installation at: {}", zludaPath);
                return true;
            }
        }

        log.debug("ZLUDA library not found in: {}", zludaPath);
        return false;
    }

    /**
     * Detect the target GPU vendor
     */
    private ZludaTarget detectTarget() {
        // First check for explicit target setting
        String target = System.getenv("ZLUDA_TARGET");
        if (target != null) {
            if (target.equalsIgnoreCase("AMD")) {
                return ZludaTarget.AMD;
            } else if (target.equalsIgnoreCase("INTEL")) {
                return ZludaTarget.INTEL;
            }
        }

        // Auto-detect based on available GPUs
        if (checkAmdGpuAvailable()) {
            return ZludaTarget.AMD;
        }
        if (checkIntelGpuAvailable()) {
            return ZludaTarget.INTEL;
        }

        return ZludaTarget.UNKNOWN;
    }

    /**
     * Check if AMD GPU is available via ROCm
     */
    private boolean checkAmdGpuAvailable() {
        try {
            // Try rocminfo command
            ProcessBuilder pb = new ProcessBuilder("rocminfo");
            pb.redirectErrorStream(true);
            Process process = pb.start();

            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line);
                }
            }

            int exitCode = process.waitFor();
            if (exitCode == 0 && output.toString().contains("gfx")) {
                log.debug("AMD GPU detected via rocminfo");
                return true;
            }
        } catch (Exception e) {
            log.debug("AMD GPU detection failed: {}", e.getMessage());
        }

        // Check for ROCm path as fallback
        String rocmPath = System.getenv("ROCM_PATH");
        if (rocmPath == null) {
            rocmPath = "/opt/rocm";
        }
        return new File(rocmPath).exists();
    }

    /**
     * Check if Intel GPU is available via Level Zero
     */
    private boolean checkIntelGpuAvailable() {
        try {
            // Try sycl-ls command (from oneAPI)
            ProcessBuilder pb = new ProcessBuilder("sycl-ls");
            pb.redirectErrorStream(true);
            Process process = pb.start();

            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line);
                }
            }

            int exitCode = process.waitFor();
            if (exitCode == 0 && output.toString().toLowerCase().contains("intel")) {
                log.debug("Intel GPU detected via sycl-ls");
                return true;
            }
        } catch (Exception e) {
            log.debug("Intel GPU detection via sycl-ls failed: {}", e.getMessage());
        }

        // Check for oneAPI path as fallback
        String oneapiPath = System.getenv("ONEAPI_ROOT");
        if (oneapiPath == null) {
            oneapiPath = "/opt/intel/oneapi";
        }
        return new File(oneapiPath).exists();
    }

    @Override
    public String toString() {
        return "ZLUDA Backend [target=" + detectedTarget + ", available=" + zludaAvailable + "]";
    }
}
