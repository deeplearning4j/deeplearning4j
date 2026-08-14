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
import org.nd4j.linalg.factory.InitializationController;
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
    private static volatile Throwable cpuInitializationFailure;
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

        if (cpuInitializationFailure != null) {
            throw InitializationController.propagate(cpuInitializationFailure);
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

                // Construct and prove the candidate before publishing it. CPU device
                // initialization may be a native no-op, but any linkage or runtime
                // failure still makes this binding unusable.
                NativeOps candidate =
                        (NativeOps) nativeOpsClass.getDeclaredConstructor().newInstance();
                candidate.initializeDevicesAndFunctions();
                cpuNativeOps = candidate;
                log.info("CPU NativeOps initialized successfully for secondary backend");
                return candidate;

            } catch (Throwable failure) {
                if (cpuInitializationFailure == null) {
                    cpuInitializationFailure = failure;
                }
                log.error("Failed to initialize CPU NativeOps", failure);
                throw InitializationController.propagate(cpuInitializationFailure);
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

                // A secondary executioner must share the proven NativeOps authority.
                // Falling back to the default constructor creates an independent native
                // binding and breaks process-wide initialization ownership.
                Constructor<?> constructor =
                        executionerClass.getConstructor(NativeOps.class, boolean.class);
                OpExecutioner candidate =
                        (OpExecutioner) constructor.newInstance(nativeOps, true);
                cpuExecutioner = candidate;

                log.info("CPU OpExecutioner loaded successfully as secondary backend");
                return candidate;

            } catch (Throwable failure) {
                log.error("Failed to initialize CPU OpExecutioner", failure);
                throw InitializationController.propagate(failure);
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
            cpuInitializationFailure = null;
        }
    }
}
