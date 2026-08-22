/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.backends;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.nativeblas.NativeOps;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Enumeration;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Production wiring checks for the StableHLO/PJRT TPU backend. */
@Slf4j
@Tag(TagNames.TPU)
@Tag(TagNames.BACKEND_DISCOVERY)
@DisplayName("TPU backend wiring (StableHLO/PJRT)")
public class TpuBackendSmokeTest {

    private static final String TPU_BACKEND_CLASS = "org.nd4j.linalg.jtpu.JTpuBackend";
    private static final String TPU_BINDING_CLASS = "org.nd4j.linalg.jtpu.bindings.Nd4jTpu";
    private static final String TPU_HELPER_CLASS = "org.nd4j.presets.tpu.Nd4jTpuHelper";
    private static final String BACKEND_SERVICE =
            "META-INF/services/org.nd4j.linalg.factory.Nd4jBackend";

    @Test
    @DisplayName("TPU backend, generated NativeOps binding, and SPI are packaged")
    public void tpuBackendBindingAndSpiArePackaged() throws Exception {
        Class<?> backendClass = assertDoesNotThrow(() -> Class.forName(TPU_BACKEND_CLASS),
                "nd4j-tpu is not on the classpath; run from platform-tests with -Ptest-tpu");
        Class<?> bindingClass = assertDoesNotThrow(() -> Class.forName(TPU_BINDING_CLASS),
                "The generated Nd4jTpu binding is missing from the TPU artifact");
        Class<?> helperClass = Class.forName(TPU_HELPER_CLASS);

        assertTrue(NativeOps.class.isAssignableFrom(bindingClass),
                "Nd4jTpu must implement the shared NativeOps ABI");
        assertTrue(NativeOps.class.isAssignableFrom(helperClass),
                "Nd4jTpuHelper must declare the NativeOps contract");
        assertTrue(isBackendRegistered(),
                "JTpuBackend is not registered through the Nd4jBackend SPI");

        Object backend = backendClass.getDeclaredConstructor().newInstance();
        boolean canRun = (Boolean) backendClass.getMethod("canRun").invoke(backend);
        Object binding = bindingClass.getDeclaredConstructor().newInstance();
        bindingClass.getMethod("initializeDevicesAndFunctions").invoke(binding);
        int deviceCount = (Integer) bindingClass.getMethod("getAvailableDevices").invoke(binding);
        assertEquals(deviceCount > 0, canRun,
                "Backend availability and the native TPU device authority disagree");

        if (canRun) {
            Environment environment = (Environment) backendClass
                    .getMethod("getEnvironment").invoke(backend);
            assertNotNull(environment);
            boolean debug = environment.isDebug();
            environment.setDebug(debug);
            assertEquals(debug, environment.isDebug());
            assertNotNull(environment.memory());
            String cacheDir = environment.tritonCacheDir();
            environment.setTritonCacheDir(cacheDir);
            assertEquals(cacheDir, environment.tritonCacheDir());
            Object discovered = backendClass.getMethod("discoverDevices").invoke(backend);
            assertTrue(discovered instanceof java.util.List
                            && !((java.util.List<?>) discovered).isEmpty(),
                    "A runnable TPU backend must publish its devices");
        }
        log.info("TPU native probe: canRun={}, devices={}", canRun, deviceCount);
    }

    @Test
    @DisplayName("Non-TPU PJRT plugins cannot select the TPU backend")
    public void nonTpuPjrtPluginCannotSelectBackend() throws Exception {
        String configured = System.getenv("PJRT_PLUGIN_LIBRARY_PATH");
        if (configured == null || configured.contains("${")) return;
        String lower = configured.toLowerCase();
        if (!lower.contains("cpu") && !lower.contains("rocm")
                && !lower.contains("gpu_plugin")) return;
        Class<?> backendClass = Class.forName(TPU_BACKEND_CLASS);
        Object backend = backendClass.getDeclaredConstructor().newInstance();
        assertFalse((Boolean) backendClass.getMethod("canRun").invoke(backend),
                "A configured non-TPU PJRT plugin must not be mislabeled as TPU");
    }

    @Test
    @DisplayName("TPU Environment delegates native and Java-only contracts")
    public void tpuEnvironmentContractIsUsableWithoutHardware() throws Exception {
        Class<?> environmentClass = Class.forName("org.nd4j.linalg.jtpu.TpuEnvironment");
        Environment environment = (Environment) environmentClass
                .getMethod("getInstance").invoke(null);
        assertNotNull(environment);
        assertNotNull(environment.memory());

        boolean verbose = environment.isVerbose();
        environment.setVerbose(verbose);
        assertEquals(verbose, environment.isVerbose());

        boolean trace = environment.isFuncTracePrintJavaOnly();
        environment.setFuncTracePrintJavaOnly(!trace);
        assertEquals(!trace, environment.isFuncTracePrintJavaOnly());
        environment.setFuncTracePrintJavaOnly(trace);

        String cacheDir = environment.tritonCacheDir();
        environment.setTritonCacheDir(cacheDir);
        assertEquals(cacheDir, environment.tritonCacheDir());
    }

    private boolean isBackendRegistered() throws Exception {
        Enumeration<URL> serviceFiles = Thread.currentThread()
                .getContextClassLoader().getResources(BACKEND_SERVICE);
        while (serviceFiles.hasMoreElements()) {
            URL url = serviceFiles.nextElement();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(url.openStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (TPU_BACKEND_CLASS.equals(line.trim())) return true;
                }
            }
        }
        return false;
    }

}
