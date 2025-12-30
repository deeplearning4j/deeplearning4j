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

import org.nd4j.linalg.factory.Environment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ZLUDA Environment Configuration
 *
 * Provides environment information for the ZLUDA backend, including
 * device information, memory limits, and runtime configuration.
 */
public class ZludaEnvironment implements Environment {

    private static final Logger log = LoggerFactory.getLogger(ZludaEnvironment.class);

    private static ZludaEnvironment INSTANCE;

    private final JZludaBackend.ZludaTarget target;
    private final String zludaPath;

    private ZludaEnvironment() {
        this.zludaPath = System.getenv("ZLUDA_PATH");
        this.target = detectTarget();
        log.info("ZLUDA Environment initialized: target={}, path={}", target, zludaPath);
    }

    public static synchronized ZludaEnvironment getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new ZludaEnvironment();
        }
        return INSTANCE;
    }

    private JZludaBackend.ZludaTarget detectTarget() {
        String targetEnv = System.getenv("ZLUDA_TARGET");
        if (targetEnv != null) {
            if (targetEnv.equalsIgnoreCase("AMD")) {
                return JZludaBackend.ZludaTarget.AMD;
            } else if (targetEnv.equalsIgnoreCase("INTEL")) {
                return JZludaBackend.ZludaTarget.INTEL;
            }
        }
        return JZludaBackend.ZludaTarget.UNKNOWN;
    }

    @Override
    public int numDevices() {
        // ZLUDA exposes devices through CUDA API
        // For now, return 1 as default
        return 1;
    }

    @Override
    public boolean isDebug() {
        return Boolean.parseBoolean(System.getenv("ZLUDA_DEBUG"));
    }

    @Override
    public boolean isVerbose() {
        return Boolean.parseBoolean(System.getenv("ZLUDA_VERBOSE"));
    }

    @Override
    public boolean isProfiling() {
        return Boolean.parseBoolean(System.getenv("ZLUDA_PROFILING"));
    }

    @Override
    public boolean isDebugAndVerbose() {
        return isDebug() && isVerbose();
    }

    @Override
    public int tadThreshold() {
        String threshold = System.getenv("ZLUDA_TAD_THRESHOLD");
        if (threshold != null) {
            try {
                return Integer.parseInt(threshold);
            } catch (NumberFormatException e) {
                log.warn("Invalid ZLUDA_TAD_THRESHOLD value: {}", threshold);
            }
        }
        return 1;  // Default threshold
    }

    @Override
    public int elementwiseThreshold() {
        String threshold = System.getenv("ZLUDA_ELEMENTWISE_THRESHOLD");
        if (threshold != null) {
            try {
                return Integer.parseInt(threshold);
            } catch (NumberFormatException e) {
                log.warn("Invalid ZLUDA_ELEMENTWISE_THRESHOLD value: {}", threshold);
            }
        }
        return 32768;  // Default threshold
    }

    @Override
    public int maxThreads() {
        String maxThreads = System.getenv("ZLUDA_MAX_THREADS");
        if (maxThreads != null) {
            try {
                return Integer.parseInt(maxThreads);
            } catch (NumberFormatException e) {
                log.warn("Invalid ZLUDA_MAX_THREADS value: {}", maxThreads);
            }
        }
        return Runtime.getRuntime().availableProcessors();
    }

    @Override
    public int maxMasterThreads() {
        return maxThreads();
    }

    @Override
    public void setMaxThreads(int max) {
        log.warn("setMaxThreads not supported in ZLUDA environment");
    }

    @Override
    public void setMaxMasterThreads(int max) {
        log.warn("setMaxMasterThreads not supported in ZLUDA environment");
    }

    @Override
    public boolean isCPU() {
        return false;
    }

    @Override
    public void setDebug(boolean debug) {
        log.warn("setDebug not supported in ZLUDA environment");
    }

    @Override
    public void setVerbose(boolean verbose) {
        log.warn("setVerbose not supported in ZLUDA environment");
    }

    @Override
    public void setProfiling(boolean profiling) {
        log.warn("setProfiling not supported in ZLUDA environment");
    }

    @Override
    public void setLeaksDetector(boolean detector) {
        log.warn("setLeaksDetector not supported in ZLUDA environment");
    }

    @Override
    public boolean isLeaksDetectorEnabled() {
        return false;
    }

    @Override
    public long getDeviceFreeMemory(int deviceId) {
        // This would need native bindings to query actual device memory
        // Return a reasonable default for now
        return 4L * 1024 * 1024 * 1024;  // 4GB default
    }

    @Override
    public long getDeviceTotalMemory(int deviceId) {
        return 8L * 1024 * 1024 * 1024;  // 8GB default
    }

    @Override
    public long getCachedMemory(int deviceId) {
        return 0;
    }

    /**
     * Get the ZLUDA target backend
     */
    public JZludaBackend.ZludaTarget getTarget() {
        return target;
    }

    /**
     * Get the ZLUDA installation path
     */
    public String getZludaPath() {
        return zludaPath;
    }

    /**
     * Check if MIOpen is available (for AMD targets)
     */
    public boolean isMIOpenAvailable() {
        if (target != JZludaBackend.ZludaTarget.AMD) {
            return false;
        }
        String rocmPath = System.getenv("ROCM_PATH");
        if (rocmPath == null) {
            rocmPath = "/opt/rocm";
        }
        return new java.io.File(rocmPath, "lib/libMIOpen.so").exists();
    }

    /**
     * Check if oneDNN is available (for Intel targets)
     */
    public boolean isOneDNNAvailable() {
        if (target != JZludaBackend.ZludaTarget.INTEL) {
            return false;
        }
        String oneapiPath = System.getenv("ONEAPI_ROOT");
        if (oneapiPath == null) {
            oneapiPath = "/opt/intel/oneapi";
        }
        return new java.io.File(oneapiPath).exists();
    }

    @Override
    public String toString() {
        return "ZludaEnvironment{" +
                "target=" + target +
                ", zludaPath='" + zludaPath + '\'' +
                ", miopen=" + isMIOpenAvailable() +
                ", onednn=" + isOneDNNAvailable() +
                '}';
    }
}
