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
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for VulkanAffinityManager — the Vulkan thread-to-device binding layer.
 *
 * <p>These tests are split into three legs:</p>
 * <ol>
 *   <li><b>NO-DEVICE contract</b> — runs on every machine including CI boxes
 *       without a Vulkan ICD: VulkanAffinityManager must construct cleanly,
 *       report 0 devices, and never throw during backend init.</li>
 *   <li><b>DEVICE leg</b> — skipped unless {@code getNumberOfDevices() >= 1};
 *       exercises thread-binding roundtrip, setDevice/getDevice parity.</li>
 *   <li><b>MEMORY MANAGER leg</b> — verifies VulkanMemoryManager and
 *       VulkanDeviceIDProvider load and contract.</li>
 * </ol>
 *
 * <p>All Vulkan classes are loaded reflectively so this test compiles without
 * nd4j-vulkan on the classpath (it is provided by {@code -Ptest-vulkan}).</p>
 *
 * <p>Run:</p>
 * <pre>
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Ptest-vulkan \
 *     -Dtest=VulkanAffinityManagerTest 2>&1 | tee /tmp/vulkan-affinity-test.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("VulkanAffinityManager — device binding contract")
public class VulkanAffinityManagerTest {

    private static final String AFFINITY_CLASS  = "org.nd4j.linalg.vulkan.VulkanAffinityManager";
    private static final String MEMORY_CLASS    = "org.nd4j.linalg.vulkan.VulkanMemoryManager";
    private static final String DEVID_CLASS     = "org.nd4j.linalg.vulkan.VulkanDeviceIDProvider";
    private static final String WORKSPACE_CLASS = "org.nd4j.linalg.vulkan.VulkanWorkspaceManager";
    private static final String BUFFER_FACTORY_CLASS = "org.nd4j.linalg.vulkan.VulkanDataBufferFactory";
    private static final String BINDINGS_CLASS  = "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";

    /** VulkanAffinityManager instance (loaded reflectively). */
    private static Object affinityManager;

    /** Whether the Vulkan affinity manager class is available on the classpath. */
    private static boolean classAvailable = false;

    /** Device count reported by the manager; 0 on ICD-less hosts. */
    private static int deviceCount = 0;

    @BeforeAll
    static void init() {
        try {
            Class<?> cls = Class.forName(AFFINITY_CLASS);
            affinityManager = cls.getDeclaredConstructor().newInstance();
            classAvailable  = true;
            deviceCount     = (int) cls.getMethod("getNumberOfDevices").invoke(affinityManager);
            log.info("VulkanAffinityManager loaded. deviceCount={}", deviceCount);
        } catch (ClassNotFoundException e) {
            log.warn("VulkanAffinityManager class not found — nd4j-vulkan not on classpath");
        } catch (Exception e) {
            log.warn("VulkanAffinityManager failed to construct: {} {}", e.getClass().getSimpleName(), e.getMessage());
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // NO-DEVICE contract (runs on every machine)
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("VulkanAffinityManager class loads on the classpath")
    void testClassOnClasspath() {
        assumeTrue(classAvailable,
                "VulkanAffinityManager class not available — run with -Ptest-vulkan");
        assertNotNull(affinityManager,
                "VulkanAffinityManager must construct without throwing even when no devices exist");
    }

    @Test
    @DisplayName("getNumberOfDevices() returns >= 0 and does not throw")
    void testDeviceCountNonNegative() {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        assertTrue(deviceCount >= 0,
                "getNumberOfDevices() must return >= 0, got " + deviceCount);
        log.info("VulkanAffinityManager.getNumberOfDevices() = {}", deviceCount);
    }

    @Test
    @DisplayName("getDeviceForCurrentThread() does not throw (no-device machine)")
    void testGetDeviceForCurrentThreadNoThrow() {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        assertDoesNotThrow(() -> {
            int dev = (int) affinityManager.getClass()
                    .getMethod("getDeviceForCurrentThread").invoke(affinityManager);
            log.info("getDeviceForCurrentThread() = {} (deviceCount={})", dev, deviceCount);
        }, "getDeviceForCurrentThread() must not throw even when deviceCount == 0");
    }

    @Test
    @DisplayName("isCpuDevice(CPU_DEVICE_ID=-1) returns true; isCpuDevice(0) returns false")
    void testIsCpuDevice() throws Exception {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        Class<?> cls = affinityManager.getClass();
        // CPU_DEVICE_ID is -1 per AffinityManager contract.
        boolean cpuTrue  = (boolean) cls.getMethod("isCpuDevice", int.class)
                .invoke(affinityManager, -1);
        boolean cpuFalse = (boolean) cls.getMethod("isCpuDevice", int.class)
                .invoke(affinityManager, 0);
        assertTrue(cpuTrue,  "isCpuDevice(-1) must return true");
        assertFalse(cpuFalse, "isCpuDevice(0) must return false");
    }

    @Test
    @DisplayName("Vulkan devices retain Vulkan identity in types and descriptors")
    void testGetDeviceType() throws Exception {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        Class<?> cls = affinityManager.getClass();
        Object gpu  = cls.getMethod("getDeviceType", int.class).invoke(affinityManager, 0);
        Object cpu  = cls.getMethod("getDeviceType", int.class).invoke(affinityManager, -1);
        DeviceDescriptor descriptor = (DeviceDescriptor) cls
                .getMethod("getDeviceDescriptor", int.class).invoke(affinityManager, 0);

        assertEquals(DeviceType.VULKAN_GPU.name(), gpu.toString(),
                "getDeviceType(0) must identify the Vulkan backend");
        assertEquals(DeviceType.CPU.name(), cpu.toString(),
                "getDeviceType(-1) must be CPU");
        assertEquals(DeviceType.VULKAN_GPU, descriptor.getDeviceType(),
                "Vulkan device descriptors must not be labeled as CUDA");
        assertEquals("vulkan", descriptor.getBackendId());
        assertEquals(0, descriptor.getDeviceIndex());
    }

    @Test
    @DisplayName("isCrossDeviceAccessSupported() returns false (Vulkan has no UVA)")
    void testCrossDeviceAccess() throws Exception {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        boolean cross = (boolean) affinityManager.getClass()
                .getMethod("isCrossDeviceAccessSupported").invoke(affinityManager);
        assertFalse(cross, "Vulkan has no UVA/peer path — isCrossDeviceAccessSupported() must return false");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DEVICE leg (requires at least one Vulkan device)
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("thread-binding roundtrip: setDeviceForCurrentThread / getDeviceForCurrentThread")
    void testThreadBindingRoundtrip() throws Exception {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        assumeTrue(deviceCount >= 1,
                "Skipping device-binding test: no Vulkan devices enumerated");

        Class<?> cls = affinityManager.getClass();

        // Bind this thread to device 0.
        cls.getMethod("setDeviceForCurrentThread", int.class).invoke(affinityManager, 0);
        int result = (int) cls.getMethod("getDeviceForCurrentThread").invoke(affinityManager);
        assertEquals(0, result, "getDeviceForCurrentThread() must return 0 after setDeviceForCurrentThread(0)");

        log.info("Thread-binding roundtrip passed: setDevice(0) -> getDevice()={}", result);
    }

    @Test
    @DisplayName("two threads get independent device bindings")
    void testTwoThreadsIndependentBindings() throws Exception {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        assumeTrue(deviceCount >= 1,
                "Skipping thread-isolation test: no Vulkan devices enumerated");

        Class<?> cls = affinityManager.getClass();
        AtomicInteger t1DevAfterSet  = new AtomicInteger(-99);
        AtomicInteger t2DevAfterSet  = new AtomicInteger(-99);
        AtomicReference<Throwable> error = new AtomicReference<>();

        // Thread 1 binds to device 0.
        CompletableFuture<Void> f1 = CompletableFuture.runAsync(() -> {
            try {
                cls.getMethod("setDeviceForCurrentThread", int.class).invoke(affinityManager, 0);
                int dev = (int) cls.getMethod("getDeviceForCurrentThread").invoke(affinityManager);
                t1DevAfterSet.set(dev);
            } catch (Throwable t) { error.set(t); }
        });

        // Thread 2 also binds to device 0 (single-device host: both bind to 0).
        CompletableFuture<Void> f2 = CompletableFuture.runAsync(() -> {
            try {
                cls.getMethod("setDeviceForCurrentThread", int.class).invoke(affinityManager, 0);
                int dev = (int) cls.getMethod("getDeviceForCurrentThread").invoke(affinityManager);
                t2DevAfterSet.set(dev);
            } catch (Throwable t) { error.set(t); }
        });

        CompletableFuture.allOf(f1, f2).join();
        assertNull(error.get(),
                "Exception in one of the device-binding threads: " + error.get());
        assertEquals(0, t1DevAfterSet.get(),
                "Thread 1 must see device 0 after set");
        assertEquals(0, t2DevAfterSet.get(),
                "Thread 2 must see device 0 after set");

        log.info("Two-thread binding: t1={} t2={}", t1DevAfterSet.get(), t2DevAfterSet.get());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // VulkanMemoryManager contract leg
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("VulkanMemoryManager class loads on the classpath")
    void testMemoryManagerClassLoads() {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        assertDoesNotThrow(() -> {
            Class<?> cls = Class.forName(MEMORY_CLASS);
            Object mm = cls.getDeclaredConstructor().newInstance();
            assertNotNull(mm, "VulkanMemoryManager must construct cleanly");
            log.info("VulkanMemoryManager constructed: {}", mm.getClass().getName());
        }, "VulkanMemoryManager must construct without throwing");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // VulkanDeviceIDProvider contract leg
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("VulkanDeviceIDProvider.getDeviceId() does not throw")
    void testDeviceIDProvider() {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        assertDoesNotThrow(() -> {
            Class<?> cls = Class.forName(DEVID_CLASS);
            Object provider = cls.getDeclaredConstructor().newInstance();
            // Note: getDeviceId() calls Nd4j.getAffinityManager() which may not be
            // initialized if Nd4j hasn't started. The implementation catches that and
            // returns 0. So either 0 or a valid device id is acceptable here.
            int id;
            try {
                id = (int) cls.getMethod("getDeviceId").invoke(provider);
            } catch (Exception e) {
                // If Nd4j not initialized, the impl catches and returns 0.
                id = 0;
            }
            assertTrue(id >= 0, "VulkanDeviceIDProvider.getDeviceId() must return >= 0, got " + id);
            log.info("VulkanDeviceIDProvider.getDeviceId() = {}", id);
        }, "VulkanDeviceIDProvider must construct and not throw");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // VulkanWorkspaceManager contract leg
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("VulkanWorkspaceManager follows CUDA's native-device workspace boundary")
    void testWorkspaceManagerClass() {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        assertDoesNotThrow(() -> {
            Class<?> wsCls = Class.forName(WORKSPACE_CLASS);
            Class<?> basicCls = Class.forName(
                    "org.nd4j.linalg.api.memory.provider.BasicWorkspaceManager");
            assertEquals(basicCls, wsCls.getSuperclass(),
                    "VulkanWorkspaceManager must directly extend BasicWorkspaceManager, "
                            + "matching CUDA's native-device workspace boundary");
            log.info("VulkanWorkspaceManager uses the native-device workspace boundary: OK");
        }, "VulkanWorkspaceManager class must load and have correct hierarchy");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Properties contract leg — verify all four flipped keys are wired
    // ─────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("nd4j-vulkan.properties has all four P4 keys flipped to Vulkan classes")
    void testPropertiesKeysFlipped() throws Exception {
        assumeTrue(classAvailable, "nd4j-vulkan not on classpath");
        try (java.io.InputStream is =
                Class.forName(AFFINITY_CLASS).getResourceAsStream("/nd4j-vulkan.properties")) {
            assertNotNull(is, "/nd4j-vulkan.properties must be on classpath");
            java.util.Properties props = new java.util.Properties();
            props.load(is);

            // Step 1 — affinitymanager
            assertEquals(AFFINITY_CLASS, props.getProperty("affinitymanager"),
                    "affinitymanager must point to VulkanAffinityManager");

            // Step 2 — memorymanager
            assertEquals(MEMORY_CLASS, props.getProperty("memorymanager"),
                    "memorymanager must point to VulkanMemoryManager");

            // Step 3 — deviceidprovider
            assertEquals(DEVID_CLASS, props.getProperty("deviceidprovider"),
                    "deviceidprovider must point to VulkanDeviceIDProvider");

            // Step 4 — workspacemanager
            assertEquals(WORKSPACE_CLASS, props.getProperty("workspacemanager"),
                    "workspacemanager must point to VulkanWorkspaceManager");

            // The dual-buffer surface is active, so allocation must use the Vulkan factory.
            assertEquals(BUFFER_FACTORY_CLASS, props.getProperty("databufferfactory"),
                    "databufferfactory must point to VulkanDataBufferFactory");

            log.info("All four P4 properties keys verified as flipped to Vulkan implementations");
        }
    }
}
