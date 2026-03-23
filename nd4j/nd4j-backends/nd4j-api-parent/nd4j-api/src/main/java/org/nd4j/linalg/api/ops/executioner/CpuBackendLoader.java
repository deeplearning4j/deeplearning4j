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

package org.nd4j.linalg.api.ops.executioner;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.nativeblas.NativeOps;

import java.lang.reflect.Constructor;

/**
 * Utility class for loading the CPU backend as a secondary backend.
 * This enables CPU fallback execution when CUDA is the primary backend
 * but CPU execution is needed for spillover or specific ops.
 *
 * <p>Usage:</p>
 * <pre>{@code
 * // Check if CPU backend is available
 * if (CpuBackendLoader.isCpuBackendAvailable()) {
 *     // Load CPU executioner as secondary
 *     OpExecutioner cpuExecutioner = CpuBackendLoader.loadCpuExecutioner();
 *
 *     // Register with DeviceAwareOpExecutioner
 *     deviceAwareExecutioner.registerBackendExecutioner(DeviceType.CPU, cpuExecutioner);
 * }
 * }</pre>
 *
 * Adam Gibson
 */
@Slf4j
public class CpuBackendLoader {

    private static final String CPU_NATIVE_OPS_CLASS = "org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu";
    private static final String CPU_EXECUTIONER_CLASS = "org.nd4j.linalg.cpu.nativecpu.ops.NativeOpExecutioner";

    private static volatile NativeOps cpuNativeOps;
    private static volatile OpExecutioner cpuExecutioner;
    private static volatile Boolean cpuAvailable;
    private static final Object LOCK = new Object();

    /**
     * Check if the CPU backend is available on the classpath.
     *
     * @return true if the CPU backend JAR is present
     */
    public static boolean isCpuBackendAvailable() {
        if (cpuAvailable != null) {
            return cpuAvailable;
        }

        synchronized (LOCK) {
            if (cpuAvailable != null) {
                return cpuAvailable;
            }

            try {
                Class<?> clazz = ND4JClassLoading.loadClassByName(CPU_NATIVE_OPS_CLASS);
                cpuAvailable = clazz != null;
                if (cpuAvailable) {
                    log.debug("CPU backend (nd4j-native) is available on classpath");
                } else {
                    log.debug("CPU backend (nd4j-native) is NOT available on classpath");
                }
            } catch (Exception e) {
                log.debug("CPU backend not available: {}", e.getMessage());
                cpuAvailable = false;
            }
        }
        return cpuAvailable;
    }

    /**
     * Load the CPU NativeOps instance.
     * This loads and initializes the native CPU library.
     *
     * @return the CPU NativeOps instance, or null if not available
     */
    public static NativeOps loadCpuNativeOps() {
        if (!isCpuBackendAvailable()) {
            return null;
        }

        if (cpuNativeOps != null) {
            return cpuNativeOps;
        }

        synchronized (LOCK) {
            if (cpuNativeOps != null) {
                return cpuNativeOps;
            }

            try {
                Class<?> nativeOpsClass = ND4JClassLoading.loadClassByName(CPU_NATIVE_OPS_CLASS);
                if (nativeOpsClass == null) {
                    log.warn("Could not load CPU NativeOps class: {}", CPU_NATIVE_OPS_CLASS);
                    return null;
                }

                // Instantiate NativeOps - this loads the native library
                cpuNativeOps = (NativeOps) nativeOpsClass.getDeclaredConstructor().newInstance();

                // Initialize devices and functions
                try {
                    cpuNativeOps.initializeDevicesAndFunctions();
                    log.info("CPU NativeOps initialized successfully for secondary backend");
                } catch (Exception e) {
                    log.warn("CPU NativeOps initialization warning: {}", e.getMessage());
                    // Continue anyway - some initialization may have succeeded
                }

                return cpuNativeOps;

            } catch (Exception e) {
                log.error("Failed to load CPU NativeOps: {}", e.getMessage());
                if (log.isDebugEnabled()) {
                    log.debug("Full stack trace:", e);
                }
                return null;
            }
        }
    }

    /**
     * Load and create a CPU OpExecutioner as a secondary backend.
     * This executioner can be used for CPU fallback execution.
     *
     * @return the CPU OpExecutioner, or null if not available
     */
    public static OpExecutioner loadCpuExecutioner() {
        if (cpuExecutioner != null) {
            return cpuExecutioner;
        }

        NativeOps nativeOps = loadCpuNativeOps();
        if (nativeOps == null) {
            return null;
        }

        synchronized (LOCK) {
            if (cpuExecutioner != null) {
                return cpuExecutioner;
            }

            try {
                Class<?> executionerClass = ND4JClassLoading.loadClassByName(CPU_EXECUTIONER_CLASS);
                if (executionerClass == null) {
                    log.warn("Could not load CPU executioner class: {}", CPU_EXECUTIONER_CLASS);
                    return null;
                }

                // Look for the secondary backend constructor: NativeOpExecutioner(NativeOps, boolean)
                Constructor<?> constructor = null;
                try {
                    constructor = executionerClass.getConstructor(NativeOps.class, boolean.class);
                } catch (NoSuchMethodException e) {
                    log.warn("CPU executioner doesn't have secondary backend constructor. " +
                            "Falling back to default constructor (may cause issues).");
                }

                if (constructor != null) {
                    // Use the secondary backend constructor
                    cpuExecutioner = (OpExecutioner) constructor.newInstance(nativeOps, true);
                } else {
                    // Fall back to default constructor (less safe)
                    cpuExecutioner = (OpExecutioner) executionerClass.getDeclaredConstructor().newInstance();
                }

                log.info("CPU OpExecutioner loaded successfully as secondary backend");
                return cpuExecutioner;

            } catch (Exception e) {
                log.error("Failed to load CPU OpExecutioner: {}", e.getMessage());
                if (log.isDebugEnabled()) {
                    log.debug("Full stack trace:", e);
                }
                return null;
            }
        }
    }

    /**
     * Check if the CPU executioner has been loaded.
     */
    public static boolean isCpuExecutionerLoaded() {
        return cpuExecutioner != null;
    }

    /**
     * Get the loaded CPU executioner without attempting to load it.
     *
     * @return the CPU executioner if loaded, null otherwise
     */
    public static OpExecutioner getCpuExecutioner() {
        return cpuExecutioner;
    }

    /**
     * Get the loaded CPU NativeOps without attempting to load it.
     *
     * @return the CPU NativeOps if loaded, null otherwise
     */
    public static NativeOps getCpuNativeOps() {
        return cpuNativeOps;
    }

    /**
     * Reset the loader state (primarily for testing).
     */
    public static void reset() {
        synchronized (LOCK) {
            cpuNativeOps = null;
            cpuExecutioner = null;
            cpuAvailable = null;
        }
    }
}
