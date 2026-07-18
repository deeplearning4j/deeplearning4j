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
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;

import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Verifies the VulkanDeviceManager NativeOps device-management contract:
 *   - getAvailableDevices() returns the correct count (0 if no Vulkan ICD available)
 *   - getDeviceName(0) non-empty when a device is present
 *   - getDeviceTotalMemory(0) > 0 (largest DEVICE_LOCAL heap)
 *   - getDeviceFreeMemory(0) >= 0 and <= totalMemory
 *   - getDevice() returns 0 (thread-local default)
 *   - setDevice(0) returns 1 (success)
 *
 * The test loads the Vulkan NativeOps class directly via reflection so it does NOT
 * depend on Nd4j initialising with VulkanBackend as primary.  All tests skip
 * automatically when no Vulkan device is enumerated (getAvailableDevices() == 0),
 * e.g. on CI machines without a Vulkan ICD.
 *
 * Run with (nd4j-vulkan jar on classpath via -Ptest-vulkan or explicit dependency):
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Ptest-vulkan \
 *     -Dtest=VulkanDeviceManagementTest 2>&1 | tee /tmp/vulkan-devmgmt-test.log
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("VulkanDeviceManager NativeOps device-management contract")
public class VulkanDeviceManagementTest {

    /** Fully-qualified name of the JavaCPP-generated Vulkan bindings class. */
    private static final String VULKAN_BINDINGS_CLASS = "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";

    /**
     * The Vulkan NativeOps singleton, loaded reflectively.
     * Stays {@code null} when the Vulkan bindings jar is absent or the native library
     * cannot be loaded (no Vulkan ICD installed).  All tests call
     * {@code assumeTrue(nativeOps != null)} via {@link #vulkanAvailable()}.
     */
    private static Object nativeOps;

    /** Set to true when Vulkan is present AND at least one device was enumerated. */
    private static boolean vulkanDevicePresent = false;

    @BeforeAll
    static void initNativeOps() {
        try {
            // Load the Vulkan bindings class — this triggers the native library load.
            Class<?> bindingsClass = Class.forName(VULKAN_BINDINGS_CLASS);
            // Obtain the singleton instance (JavaCPP-generated classes are singleton-like;
            // we use the no-arg constructor which is exposed as a static method or constructor).
            nativeOps = bindingsClass.getDeclaredConstructor().newInstance();
            log.info("Loaded Vulkan NativeOps: {}", bindingsClass.getName());

            // Query device count — 0 means no Vulkan ICD available on this machine.
            int count = (int) bindingsClass.getMethod("getAvailableDevices").invoke(nativeOps);
            log.info("Vulkan getAvailableDevices() = {}", count);
            vulkanDevicePresent = (count > 0);
        } catch (Exception e) {
            // Bindings jar absent, native library not found, or Vulkan loader not present.
            log.warn("VulkanDeviceManagementTest: Vulkan NativeOps unavailable ({}): {}",
                     e.getClass().getSimpleName(), e.getMessage());
            nativeOps = null;
            vulkanDevicePresent = false;
        }
    }

    // ── skip helper ───────────────────────────────────────────────────────────

    /**
     * All tests call this helper to skip gracefully when Vulkan is absent.
     * This avoids an ERROR classification for an environment issue.
     */
    private static void vulkanAvailable() {
        assumeTrue(nativeOps != null,
                "Vulkan NativeOps class (" + VULKAN_BINDINGS_CLASS + ") not available "
                + "— nd4j-vulkan jar absent or native library failed to load.");
    }

    private static void vulkanDeviceAvailable() {
        vulkanAvailable();
        assumeTrue(vulkanDevicePresent,
                "getAvailableDevices() returned 0 — no Vulkan devices enumerated. "
                + "Ensure a Vulkan ICD (NVIDIA driver or Mesa lavapipe) is installed "
                + "and the nd4j-vulkan chip library was built with HAVE_VULKAN=1.");
    }

    // ── reflection helpers ────────────────────────────────────────────────────

    private int callInt0(String method) throws Exception {
        return (int) nativeOps.getClass().getMethod(method).invoke(nativeOps);
    }

    private int callInt1(String method, int arg) throws Exception {
        return (int) nativeOps.getClass().getMethod(method, int.class).invoke(nativeOps, arg);
    }

    private String callString1(String method, int arg) throws Exception {
        return (String) nativeOps.getClass().getMethod(method, int.class).invoke(nativeOps, arg);
    }

    private long callLong1(String method, int arg) throws Exception {
        return (long) nativeOps.getClass().getMethod(method, int.class).invoke(nativeOps, arg);
    }

    private long callLong0(String method) throws Exception {
        return (long) nativeOps.getClass().getMethod(method).invoke(nativeOps);
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("getAvailableDevices() returns >= 0 (0 when no Vulkan ICD)")
    void testAvailableDevices() throws Exception {
        vulkanAvailable();
        int count = callInt0("getAvailableDevices");
        log.info("getAvailableDevices() = {}", count);
        assertTrue(count >= 0, "getAvailableDevices() must be >= 0, got " + count);
        // Log a note when count == 0 so CI output is informative.
        if (count == 0) {
            log.info("No Vulkan devices enumerated — vkCreateInstance may have failed. "
                   + "This is expected on machines without a Vulkan ICD.");
        }
    }

    @Test
    @DisplayName("getDeviceName(0) returns a non-empty string")
    void testDeviceName() throws Exception {
        vulkanDeviceAvailable();
        String name = callString1("getDeviceName", 0);
        log.info("getDeviceName(0) = '{}'", name);
        assertNotNull(name, "getDeviceName(0) returned null");
        assertFalse(name.isEmpty(), "getDeviceName(0) returned an empty string");
    }

    @Test
    @DisplayName("getDeviceTotalMemory(0) returns a positive value")
    void testDeviceTotalMemory() throws Exception {
        vulkanDeviceAvailable();
        long total = callLong1("getDeviceTotalMemory", 0);
        log.info("getDeviceTotalMemory(0) = {} bytes ({} MB)", total, total / (1024 * 1024));
        assertTrue(total > 0, "getDeviceTotalMemory(0) must be > 0, got " + total);
    }

    @Test
    @DisplayName("getDeviceFreeMemory(0) is non-negative and <= total")
    void testDeviceFreeMemory() throws Exception {
        vulkanDeviceAvailable();
        long total = callLong1("getDeviceTotalMemory", 0);
        long free  = callLong1("getDeviceFreeMemory",  0);
        log.info("getDeviceFreeMemory(0)  = {} bytes ({} MB)", free,  free  / (1024 * 1024));
        log.info("getDeviceTotalMemory(0) = {} bytes ({} MB)", total, total / (1024 * 1024));
        assertTrue(free >= 0,    "getDeviceFreeMemory(0) must be >= 0, got " + free);
        assertTrue(free <= total, "getDeviceFreeMemory(0) " + free + " must not exceed total " + total);
    }

    @Test
    @DisplayName("getDeviceFreeMemoryDefault() is non-negative")
    void testDeviceFreeMemoryDefault() throws Exception {
        vulkanDeviceAvailable();
        long freeDefault = callLong0("getDeviceFreeMemoryDefault");
        log.info("getDeviceFreeMemoryDefault() = {} bytes ({} MB)", freeDefault, freeDefault / (1024 * 1024));
        assertTrue(freeDefault >= 0, "getDeviceFreeMemoryDefault() must be >= 0, got " + freeDefault);
    }

    @Test
    @DisplayName("getDevice() returns 0 by default (thread-local default)")
    void testGetDevice() throws Exception {
        vulkanDeviceAvailable();
        int device = callInt0("getDevice");
        log.info("getDevice() = {}", device);
        assertEquals(0, device, "Default thread-local device must be 0");
    }

    @Test
    @DisplayName("setDevice(0) succeeds and getDevice() reflects the change")
    void testSetDevice() throws Exception {
        vulkanDeviceAvailable();
        int result = callInt1("setDevice", 0);
        log.info("setDevice(0) = {}", result);
        assertEquals(1, result, "setDevice(0) must return 1 (success), got " + result);

        int device = callInt0("getDevice");
        assertEquals(0, device, "getDevice() must return 0 after setDevice(0)");
    }

    @Test
    @DisplayName("getDeviceMajor(0) and getDeviceMinor(0) are non-negative")
    void testDeviceVersion() throws Exception {
        vulkanDeviceAvailable();
        int major = callInt1("getDeviceMajor", 0);
        int minor = callInt1("getDeviceMinor", 0);
        log.info("Vulkan API version: {}.{}", major, minor);
        assertTrue(major >= 0, "getDeviceMajor(0) must be >= 0, got " + major);
        assertTrue(minor >= 0, "getDeviceMinor(0) must be >= 0, got " + minor);
        // Vulkan 1.0 is the minimum; 1.1+ is ubiquitous on any hardware enumerated
        assertTrue(major >= 1, "Vulkan API major version must be >= 1, got " + major);
    }
}
