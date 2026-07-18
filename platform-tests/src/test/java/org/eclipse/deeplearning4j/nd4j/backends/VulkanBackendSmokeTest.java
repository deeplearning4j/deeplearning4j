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
package org.eclipse.deeplearning4j.nd4j.backends;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ops.CustomOp;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Enumeration;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Smoke tests for the Vulkan compute backend wiring. See ADR 0110 (Vulkan Backend).
 *
 * Selected by {@code -Ptest-vulkan} (group vulkan), which puts nd4j-vulkan on the classpath.
 * VulkanBackend.canRun() returns false until the JavaCPP bindings are generated against a
 * libnd4jvulkan chip build AND the nd4j-vulkan.properties runtime wiring (native.ops) is
 * activated, so these tests validate the layers that exist today:
 *
 * 1. classpath + SPI registration of the Vulkan backend,
 * 2. that the packaged backend bindings resolve the runtime Vulkan loader (libvulkan.so.1)
 *    in-process — on GPU-less CI hosts, mesa-vulkan-drivers (lavapipe) provides a software ICD,
 * 3. the nd4j-vulkan.properties contract that gates canRun().
 *
 * All nd4j-vulkan classes are accessed reflectively so this class compiles without the
 * nd4j-vulkan dependency on the classpath.
 */
@Slf4j
@Tag(TagNames.VULKAN)
@Tag(TagNames.BACKEND_DISCOVERY)
@DisplayName("Vulkan compute backend wiring smoke tests")
public class VulkanBackendSmokeTest {

    private static final String VULKAN_BACKEND_CLASS = "org.nd4j.linalg.vulkan.VulkanBackend";
    private static final String VULKAN_BINDINGS_CLASS = "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";
    private static final String BACKEND_SERVICE = "META-INF/services/org.nd4j.linalg.factory.Nd4jBackend";
    private static final String LINALG_PROPS = "/nd4j-vulkan.properties";

    @Test
    @DisplayName("nd4j-vulkan is on the classpath and SPI-registered")
    public void vulkanBackendOnClasspathAndRegistered() throws Exception {
        Class<?> backendClass = assertDoesNotThrow(() -> Class.forName(VULKAN_BACKEND_CLASS),
                "nd4j-vulkan is not on the classpath — run from platform-tests with -Ptest-vulkan "
                        + "(and install it first: mvn install -DskipTests -Pvulkan -pl :nd4j-vulkan-preset,:nd4j-vulkan)");

        boolean registered = false;
        Enumeration<URL> serviceFiles = Thread.currentThread().getContextClassLoader().getResources(BACKEND_SERVICE);
        while (serviceFiles.hasMoreElements() && !registered) {
            URL url = serviceFiles.nextElement();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(url.openStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().equals(VULKAN_BACKEND_CLASS)) {
                        registered = true;
                        log.info("VulkanBackend SPI registration found in {}", url);
                        break;
                    }
                }
            }
        }
        assertTrue(registered, "VulkanBackend is on the classpath but not registered in " + BACKEND_SERVICE
                + " — Nd4jBackend discovery can never select it");

        // Constructing and probing the backend must not throw even without a Vulkan device
        Object backend = backendClass.getDeclaredConstructor().newInstance();
        Object canRun = backendClass.getMethod("canRun").invoke(backend);
        Object priority = backendClass.getMethod("getPriority").invoke(backend);
        log.info("VulkanBackend.canRun() = {}, priority = {}", canRun, priority);

        boolean bindingsPresent;
        try {
            Class.forName(VULKAN_BINDINGS_CLASS, false, backendClass.getClassLoader());
            bindingsPresent = true;
        } catch (ClassNotFoundException e) {
            bindingsPresent = false;
        }
        if (!bindingsPresent) {
            assertEquals(Boolean.FALSE, canRun,
                    "VulkanBackend.canRun() must be false while the Nd4jVulkan bindings class is absent — "
                            + "a backend without native bindings must never win discovery over CPU");
        } else {
            log.info("Nd4jVulkan bindings class present — nd4j-vulkan.properties runtime wiring is active");
            // canRun() adds a CUDA-parity device gate: true iff >=1 Vulkan device
            // enumerates. Compute the same expectation independently so the contract
            // holds on GPU boxes AND ICD-less CI hosts.
            boolean expectRunnable;
            try {
                Class<?> bindings = Class.forName(VULKAN_BINDINGS_CLASS, true, backendClass.getClassLoader());
                Object nativeOps = bindings.getDeclaredConstructor().newInstance();
                expectRunnable = ((Integer) bindings.getMethod("getAvailableDevices").invoke(nativeOps)) >= 1;
            } catch (Throwable t) {
                log.info("Bindings failed to load natives ({}); expecting canRun()==false", t.toString());
                expectRunnable = false;
            }
            assertEquals(expectRunnable, canRun,
                    "VulkanBackend.canRun() must equal (device count >= 1) when bindings + "
                            + "native.ops wiring are present — no silent degradation either direction");
        }
    }

    @Test
    @DisplayName("Vulkan runtime loader resolves through the packaged backend bindings")
    public void vulkanLoaderLoads() {
        assertDoesNotThrow(() -> {
            Class<?> bindings = Class.forName(VULKAN_BINDINGS_CLASS, true,
                    Thread.currentThread().getContextClassLoader());
            Object nativeOps = bindings.getDeclaredConstructor().newInstance();
            int deviceCount = (Integer) bindings.getMethod("getAvailableDevices").invoke(nativeOps);
            log.info("Vulkan runtime loader resolved through backend bindings; {} device(s) enumerated",
                    deviceCount);
        }, "Failed to initialize the packaged Vulkan bindings and runtime loader");
    }

    @Test
    @DisplayName("nd4j-vulkan.properties contract gates canRun()")
    public void vulkanPropertiesContract() throws Exception {
        Class<?> backendClass;
        try {
            backendClass = Class.forName(VULKAN_BACKEND_CLASS);
        } catch (ClassNotFoundException e) {
            assumeTrue(false, "nd4j-vulkan not on the classpath — skipping properties contract check");
            return;
        }

        Properties props = new Properties();
        try (InputStream is = backendClass.getResourceAsStream(LINALG_PROPS)) {
            assertNotNull(is, LINALG_PROPS + " must be packaged in the nd4j-vulkan jar");
            props.load(is);
        }

        assertEquals("VULKAN", props.getProperty("device.type"),
                "device.type must be the named VULKAN token — a generic GPU value collides with "
                        + "CUDA detection in Nd4j initialization");
        assertNotNull(props.getProperty("real.class.double"), "real.class.double is required by Nd4j init");
        assertEquals("c", props.getProperty("ndarray.order"), "ndarray.order is required by Nd4j init");
        assertEquals("org.nd4j.linalg.vulkan.ops.executioner.VulkanExecutioner",
                props.getProperty("opexec"), "Vulkan must use its own device executioner");
        assertNull(props.getProperty("blaslapackdelegator"),
                "Vulkan must not configure a host BLAS/LAPACK execution service");
        for (String key : new String[]{"real.class.double", "affinitymanager", "memorymanager",
                "workspacemanager", "databufferfactory", "ndarrayfactory.class",
                "constantsprovider", "opexec", "random", "blas.ops"}) {
            String implementation = props.getProperty(key);
            assertNotNull(implementation, key + " must be configured");
            assertFalse(implementation.contains("cpu.nativecpu"),
                    key + " must not delegate to the CPU backend: " + implementation);
        }

        boolean bindingsPresent;
        try {
            Class.forName(VULKAN_BINDINGS_CLASS, false, backendClass.getClassLoader());
            bindingsPresent = true;
        } catch (ClassNotFoundException e) {
            bindingsPresent = false;
        }
        if (!bindingsPresent) {
            // Bindings class absent: a half-wired state (native.ops set without bindings) would
            // make canRun() try to load a class that does not exist. Verify native.ops is unset.
            try {
                boolean canRunResult = (Boolean) backendClass.getMethod("canRun")
                        .invoke(backendClass.getDeclaredConstructor().newInstance());
                assertEquals(false, canRunResult,
                        "VulkanBackend.canRun() must return false when bindings class is absent, "
                                + "regardless of native.ops wiring in properties");
            } catch (Exception reflEx) {
                throw new AssertionError("Failed to invoke canRun() reflectively: " + reflEx.getMessage(), reflEx);
            }
        } else {
            // Bindings class present: native.ops must be wired so canRun() returns true
            assertNotNull(props.getProperty("native.ops"),
                    "native.ops must be set in nd4j-vulkan.properties when the Nd4jVulkan bindings "
                            + "class is present — without it canRun() returns false and the backend "
                            + "can never be selected over CPU");
        }
    }

    @Test
    @DisplayName("Vulkan executioner owns eager device dispatch without CPU inheritance")
    public void vulkanExecutionerIsDeviceOnly() throws Exception {
        Class<?> executionerClass = Class.forName(
                "org.nd4j.linalg.vulkan.ops.executioner.VulkanExecutioner");
        Object executioner = executionerClass.getDeclaredConstructor().newInstance();
        assertEquals("VULKAN", executionerClass.getMethod("type").invoke(executioner).toString(),
                "Vulkan must not report CPU or CUDA executioner identity");

        Class<?> defaultExecutioner = Class.forName(
                "org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner");
        assertEquals(defaultExecutioner, executionerClass.getSuperclass(),
                "VulkanExecutioner must extend the backend-neutral executioner boundary");
        assertEquals(executionerClass,
                executionerClass.getDeclaredMethod("exec", CustomOp.class).getDeclaringClass(),
                "VulkanExecutioner must own the eager custom-op dispatch entry point");
    }

}
