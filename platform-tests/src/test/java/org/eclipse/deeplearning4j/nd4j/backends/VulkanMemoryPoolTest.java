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
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Verifies ADR-0111 Phase 2 VulkanMemoryPool gates:
 *
 *   (a) Allocation-count stress: allocate/free 8192 small device buffers via
 *       NativeOps mallocDevice/freeDevice.  Without block suballocation this
 *       would exhaust maxMemoryAllocationCount (~4096) and crash.  With the pool
 *       all 8192 allocations fit within a handful of VkDeviceMemory blocks.
 *
 *   (b) Host-to-device-to-host roundtrip through NativeOps memcpySync,
 *       verifying that opaque Vulkan allocation handles are never dereferenced
 *       as host pointers.
 *
 *   (c) Odd-sized device memset through NativeOps memsetSync, exercising the
 *       staging-copy path required when vkCmdFillBuffer alignment is insufficient.
 *
 *   (d) Exact-byte partial copy/fill semantics with guard-byte preservation.
 *
 *   (e) Free-memory accounting sanity while a device allocation is live.
 *
 *   (f) Direct real-device execution of the hierarchical exclusive integer scan,
 *       including boundary sizes, multiple recursion levels, and in-place aliasing.
 *
 * Tests use reflective access to the Vulkan NativeOps class so they do NOT
 * depend on Nd4j initialising with VulkanBackend as primary.  All tests skip
 * automatically when no Vulkan device is enumerated.
 *
 * Run with:
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Ptest-vulkan \
 *     -Dtest=VulkanMemoryPoolTest 2>&1 | tee /tmp/vulkan-pool-test.log
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("VulkanMemoryPool P2 gates (ADR-0111)")
public class VulkanMemoryPoolTest {

    private static final String VULKAN_BINDINGS_CLASS = "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";

    private static Object nativeOps;
    private static boolean vulkanDevicePresent = false;

    @BeforeAll
    static void initNativeOps() {
        try {
            Class<?> cls = Class.forName(VULKAN_BINDINGS_CLASS);
            nativeOps = cls.getDeclaredConstructor().newInstance();
            log.info("Loaded Vulkan NativeOps: {}", cls.getName());

            int count = (int) cls.getMethod("getAvailableDevices").invoke(nativeOps);
            log.info("Vulkan getAvailableDevices() = {}", count);
            vulkanDevicePresent = (count > 0);
        } catch (Exception e) {
            log.warn("VulkanMemoryPoolTest: Vulkan NativeOps unavailable: {} {}", e.getClass().getSimpleName(), e.getMessage());
            nativeOps = null;
            vulkanDevicePresent = false;
        }
    }

    // ── skip helpers ─────────────────────────────────────────────────────────

    private static void requireVulkanDevice() {
        assumeTrue(nativeOps != null, "Vulkan NativeOps class not available — nd4j-vulkan jar absent.");
        assumeTrue(vulkanDevicePresent,
                "getAvailableDevices() returned 0 — no Vulkan devices enumerated.");
    }

    // ── reflection helpers ────────────────────────────────────────────────────

    /** Call mallocDevice(memorySize, deviceId, flags) → Pointer. */
    private Pointer mallocDevice(long memorySize, int deviceId) throws Exception {
        Method m = nativeOps.getClass().getMethod("mallocDevice", long.class, int.class, int.class);
        return (Pointer) m.invoke(nativeOps, memorySize, deviceId, 0);
    }

    /** Call freeDevice(pointer, deviceId) → int (1 = success). */
    private int freeDevice(Pointer pointer, int deviceId) throws Exception {
        Method m = nativeOps.getClass().getMethod("freeDevice", Pointer.class, int.class);
        return (int) m.invoke(nativeOps, pointer, deviceId);
    }

    /** Call memcpySync(dst, src, size, flags, reserved) → int (1 = success). */
    private int memcpySync(Pointer dst, Pointer src, long size, int flags) throws Exception {
        Method m = nativeOps.getClass().getMethod(
                "memcpySync", Pointer.class, Pointer.class, long.class, int.class, Pointer.class);
        return (int) m.invoke(nativeOps, dst, src, size, flags, null);
    }

    /** Call memsetSync(dst, value, size, flags, reserved) → int (1 = success). */
    private int memsetSync(Pointer dst, int value, long size) throws Exception {
        Method m = nativeOps.getClass().getMethod(
                "memsetSync", Pointer.class, int.class, long.class, int.class, Pointer.class);
        return (int) m.invoke(nativeOps, dst, value, size, 0, null);
    }

    /** Call getDeviceFreeMemory(deviceId) → long. */
    private long getDeviceFreeMemory(int deviceId) throws Exception {
        return (long) nativeOps.getClass().getMethod("getDeviceFreeMemory", int.class)
                .invoke(nativeOps, deviceId);
    }

    private int streamSynchronize(Pointer stream) throws Exception {
        return (int) nativeOps.getClass().getMethod("streamSynchronize", Pointer.class)
                .invoke(nativeOps, stream);
    }

    private void prescanArrayRecursive(
            PointerPointer extras, IntPointer output, IntPointer input, int length) throws Exception {
        nativeOps.getClass().getMethod(
                        "prescanArrayRecursive", PointerPointer.class, IntPointer.class,
                        IntPointer.class, int.class, int.class)
                .invoke(nativeOps, extras, output, input, length, 0);
    }

    private void clearLastError() throws Exception {
        nativeOps.getClass().getMethod("clearLastError").invoke(nativeOps);
    }

    private int lastErrorCode() throws Exception {
        return (int) nativeOps.getClass().getMethod("lastErrorCode").invoke(nativeOps);
    }

    private String lastErrorMessage() throws Exception {
        return (String) nativeOps.getClass().getMethod("lastErrorMessage").invoke(nativeOps);
    }

    // ── Test (a): allocation-count stress ──────────────────────────────────────

    @Test
    @DisplayName("(a) Allocate/free 8192 small buffers without exhausting maxMemoryAllocationCount")
    void testAllocationCountStress() throws Exception {
        requireVulkanDevice();
        final int deviceId = 0;
        final int N = 8192;
        final long SIZE = 256;  // bytes each — tiny; all fit in 2 MiB

        List<Pointer> ptrs = new ArrayList<>(N);
        try {
            for (int i = 0; i < N; i++) {
                Pointer ptr = mallocDevice(SIZE, deviceId);
                assertTrue(ptr != null && ptr.address() != 0L,
                        "mallocDevice returned null at iteration " + i + " (allocation " + (i + 1) + " of " + N + ")");
                ptrs.add(ptr);
            }
            log.info("testAllocationCountStress: allocated {} buffers of {} bytes each (total ~{} MiB)",
                    N, SIZE, (N * SIZE) >> 20);
        } finally {
            // Free in reverse order and continue releasing even if one result is bad.
            int failedFrees = 0;
            for (int i = ptrs.size() - 1; i >= 0; i--) {
                if (freeDevice(ptrs.get(i), deviceId) != 1) {
                    failedFrees++;
                }
            }
            assertEquals(0, failedFrees, "Every freeDevice call must succeed");
            log.info("testAllocationCountStress: freed all {} buffers", ptrs.size());
        }
    }

    // ── Test (b): host-to-device-to-host roundtrip ─────────────────────────

    @Test
    @DisplayName("(b) Host-to-device-to-host roundtrip through Vulkan NativeOps")
    void testHostDeviceHostRoundtrip() throws Exception {
        requireVulkanDevice();
        final int deviceId = 0;
        final int size = 4096;

        Pointer device = mallocDevice(size, deviceId);
        assertTrue(device != null && device.address() != 0L, "mallocDevice returned null");

        try (BytePointer source = new BytePointer(size);
             BytePointer target = new BytePointer(size)) {
            for (int i = 0; i < size; i++) {
                source.put(i, (byte) (i * 31 + 7));
            }

            assertEquals(1, memcpySync(device, source, size, 1),
                    "Host-to-device memcpySync must succeed");
            assertEquals(1, memcpySync(target, device, size, 2),
                    "Device-to-host memcpySync must succeed");

            for (int i = 0; i < size; i++) {
                int expected = (byte) (i * 31 + 7) & 0xFF;
                int actual = target.get(i) & 0xFF;
                assertEquals(expected, actual, "Roundtrip mismatch at offset " + i);
            }
        } finally {
            assertEquals(1, freeDevice(device, deviceId), "freeDevice must succeed");
        }
    }

    // ── Test (c): odd-sized device fill ───────────────────────────────────────

    @Test
    @DisplayName("(c) Odd-sized Vulkan device memset is copied back correctly")
    void testOddSizedDeviceMemset() throws Exception {
        requireVulkanDevice();
        final int deviceId = 0;
        final int size = 4099;
        final int pattern = 0xA5;

        Pointer device = mallocDevice(size, deviceId);
        assertTrue(device != null && device.address() != 0L, "mallocDevice returned null");

        try (BytePointer target = new BytePointer(size)) {
            assertEquals(1, memsetSync(device, pattern, size),
                    "Device memsetSync must succeed");
            assertEquals(1, memcpySync(target, device, size, 2),
                    "Device-to-host memcpySync must succeed");

            for (int i = 0; i < size; i++) {
                assertEquals(pattern, target.get(i) & 0xFF,
                        "memset mismatch at offset " + i);
            }
        } finally {
            assertEquals(1, freeDevice(device, deviceId), "freeDevice must succeed");
        }
    }

    // ── Test (d): exact partial-byte transfer/fill semantics ──────────────────

    @Test
    @DisplayName("(d) Partial Vulkan copy and memset preserve adjacent guard bytes")
    void testPartialByteRangesPreserveGuards() throws Exception {
        requireVulkanDevice();
        final int deviceId = 0;
        final int allocationSize = 23;
        final int partialSize = 7;
        final int guard = 0x11;
        final int fill = 0x5A;

        Pointer sourceDevice = null;
        Pointer destinationDevice = null;
        try {
            sourceDevice = mallocDevice(allocationSize, deviceId);
            destinationDevice = mallocDevice(allocationSize, deviceId);
            assertTrue(sourceDevice != null && sourceDevice.address() != 0L,
                    "source mallocDevice returned null");
            assertTrue(destinationDevice != null && destinationDevice.address() != 0L,
                    "destination mallocDevice returned null");

            try (BytePointer guardBytes = new BytePointer(allocationSize);
                 BytePointer patchBytes = new BytePointer(partialSize);
                 BytePointer result = new BytePointer(allocationSize)) {
                for (int i = 0; i < allocationSize; i++) {
                    guardBytes.put(i, (byte) guard);
                }
                for (int i = 0; i < partialSize; i++) {
                    patchBytes.put(i, (byte) (0x80 + i));
                }

                assertEquals(1, memcpySync(destinationDevice, guardBytes, allocationSize, 1));
                assertEquals(1, memcpySync(destinationDevice, patchBytes, partialSize, 1));
                assertEquals(1, memcpySync(result, destinationDevice, allocationSize, 2));
                for (int i = 0; i < allocationSize; i++) {
                    int expected = i < partialSize ? 0x80 + i : guard;
                    assertEquals(expected, result.get(i) & 0xFF,
                            "H2D partial-copy guard mismatch at offset " + i);
                }

                assertEquals(1, memcpySync(sourceDevice, guardBytes, allocationSize, 1));
                assertEquals(1, memcpySync(sourceDevice, patchBytes, partialSize, 1));
                assertEquals(1, memcpySync(destinationDevice, guardBytes, allocationSize, 1));
                assertEquals(1, memcpySync(destinationDevice, sourceDevice, partialSize, 3));
                assertEquals(1, memcpySync(result, destinationDevice, allocationSize, 2));
                for (int i = 0; i < allocationSize; i++) {
                    int expected = i < partialSize ? 0x80 + i : guard;
                    assertEquals(expected, result.get(i) & 0xFF,
                            "D2D partial-copy guard mismatch at offset " + i);
                }

                assertEquals(1, memcpySync(destinationDevice, guardBytes, allocationSize, 1));
                assertEquals(1, memsetSync(destinationDevice, fill, partialSize));
                assertEquals(1, memcpySync(result, destinationDevice, allocationSize, 2));
                for (int i = 0; i < allocationSize; i++) {
                    int expected = i < partialSize ? fill : guard;
                    assertEquals(expected, result.get(i) & 0xFF,
                            "partial memset guard mismatch at offset " + i);
                }
            }
        } finally {
            if (destinationDevice != null && destinationDevice.address() != 0L) {
                assertEquals(1, freeDevice(destinationDevice, deviceId),
                        "destination freeDevice must succeed");
            }
            if (sourceDevice != null && sourceDevice.address() != 0L) {
                assertEquals(1, freeDevice(sourceDevice, deviceId),
                        "source freeDevice must succeed");
            }
        }
    }

    // ── Test (e): free-memory accounting sanity ───────────────────────────────

    @Test
    @DisplayName("(e) getDeviceFreeMemory remains valid across allocation and free")
    void testFreeMemoryAccounting() throws Exception {
        requireVulkanDevice();
        final int deviceId = 0;
        // Allocate 8 MiB — large enough to move the needle on the free-memory
        // counter, but small enough to succeed on any device with Vulkan support.
        final long ALLOC_SIZE = 8L * 1024 * 1024;

        long freeBefore = getDeviceFreeMemory(deviceId);
        log.info("testFreeMemoryAccounting: freeBefore={} MB", freeBefore >> 20);
        assertTrue(freeBefore >= 0, "getDeviceFreeMemory must be >= 0");

        Pointer ptr = mallocDevice(ALLOC_SIZE, deviceId);
        assertTrue(ptr != null && ptr.address() != 0L,
                "mallocDevice returned null for an 8 MiB device allocation");

        long freeDuring;
        try {
            freeDuring = getDeviceFreeMemory(deviceId);
            log.info("testFreeMemoryAccounting: freeDuring={} MB", freeDuring >> 20);
            assertTrue(freeDuring >= 0, "getDeviceFreeMemory during alloc must be >= 0");
            // On UMA devices DEVICE_LOCAL and HOST_VISIBLE share one heap, so the
            // budget drop may or may not be visible from the HOST_VISIBLE pool.
            // We assert non-negative accounting, not a strict decrease.
            // On discrete GPUs we expect freeDuring < freeBefore.
            log.info("testFreeMemoryAccounting: delta={} MB (positive = decreased as expected)",
                    (freeBefore - freeDuring) >> 20);
        } finally {
            int ok = freeDevice(ptr, deviceId);
            assertEquals(1, ok, "freeDevice must return 1 (success)");
        }

        long freeAfter = getDeviceFreeMemory(deviceId);
        log.info("testFreeMemoryAccounting: freeAfter={} MB", freeAfter >> 20);
        assertTrue(freeAfter >= 0, "getDeviceFreeMemory after free must be >= 0");
        // After free: accounting should recover.
        // VK_EXT_memory_budget: driver-reported budget should increase back.
        log.info("testFreeMemoryAccounting: recovery delta={} MB (positive = recovered)",
                (freeAfter - freeDuring) >> 20);
    }

    @Test
    @DisplayName("(f) Recursive exclusive scan executes on a real Vulkan stream")
    void testPrescanArrayRecursiveOnDevice() throws Exception {
        requireVulkanDevice();
        final int deviceId = 0;
        assertEquals(1, (int) nativeOps.getClass().getMethod("setDevice", int.class)
                .invoke(nativeOps, deviceId), "setDevice must select Vulkan device 0");

        // A null execution-stream handle is the CUDA-compatible default stream.
        // Vulkan resolves it to its backend-owned default VkQueue stream.
        try (PointerPointer extras = new PointerPointer(2)) {
            extras.put(0, (Pointer) null);
            extras.put(1, (Pointer) null);

            for (int length : new int[]{1, 255, 256, 257, 1025, 65537}) {
                assertPrescanResult(extras, deviceId, length, false);
            }
            assertPrescanResult(extras, deviceId, 1025, true);
        }
    }

    private void assertPrescanResult(
            PointerPointer extras, int deviceId, int length, boolean inPlace)
            throws Exception {
        final long bytes = (long) length * Integer.BYTES;
        Pointer inputDevice = null;
        Pointer outputDevice = null;

        try {
            inputDevice = mallocDevice(bytes, deviceId);
            assertTrue(inputDevice != null && inputDevice.address() != 0L,
                    "input mallocDevice returned null for length " + length);

            outputDevice = inPlace ? inputDevice : mallocDevice(bytes, deviceId);
            assertTrue(outputDevice != null && outputDevice.address() != 0L,
                    "output mallocDevice returned null for length " + length);

            try (IntPointer hostInput = new IntPointer(length);
                 IntPointer hostOutput = new IntPointer(length)) {
                int[] expected = new int[length];
                int running = 0;
                for (int i = 0; i < length; i++) {
                    int value = ((i * 17 + 3) % 29) - 14;
                    hostInput.put(i, value);
                    expected[i] = running;
                    running += value;
                }

                assertEquals(1, memcpySync(inputDevice, hostInput, bytes, 1),
                        "Host-to-device input copy must succeed for length " + length);

                // These are non-owning typed views. freeDevice below owns the
                // Vulkan allocation lifetime.
                IntPointer inputView = new IntPointer(inputDevice);
                IntPointer outputView = new IntPointer(outputDevice);
                inputView.capacity(length);
                outputView.capacity(length);

                clearLastError();
                // level is retained for CUDA ABI compatibility; Vulkan builds the full
                // hierarchy in one stream submission.
                prescanArrayRecursive(extras, outputView, inputView, length);
                int launchError = lastErrorCode();
                if (launchError != 0) {
                    fail("prescan launch failed for length " + length + ": "
                            + lastErrorMessage());
                }

                assertEquals(1, streamSynchronize(null),
                        "Vulkan default-stream synchronization must succeed for length " + length);
                int completionError = lastErrorCode();
                if (completionError != 0) {
                    fail("prescan completion failed for length " + length + ": "
                            + lastErrorMessage());
                }

                assertEquals(1, memcpySync(hostOutput, outputDevice, bytes, 2),
                        "Device-to-host prescan copy must succeed for length " + length);
                for (int i = 0; i < length; i++) {
                    assertEquals(expected[i], hostOutput.get(i),
                            "exclusive scan mismatch at index " + i + " of " + length
                                    + (inPlace ? " (in-place)" : ""));
                }
            }
        } finally {
            int failedFrees = 0;
            Exception cleanupFailure = null;
            if (!inPlace && outputDevice != null && outputDevice.address() != 0L) {
                try {
                    if (freeDevice(outputDevice, deviceId) != 1) {
                        failedFrees++;
                    }
                } catch (Exception e) {
                    cleanupFailure = e;
                }
            }
            if (inputDevice != null && inputDevice.address() != 0L) {
                try {
                    if (freeDevice(inputDevice, deviceId) != 1) {
                        failedFrees++;
                    }
                } catch (Exception e) {
                    if (cleanupFailure == null) {
                        cleanupFailure = e;
                    } else {
                        cleanupFailure.addSuppressed(e);
                    }
                }
            }
            if (cleanupFailure != null) {
                throw cleanupFailure;
            }
            assertEquals(0, failedFrees,
                    "Every prescan device allocation must be released for length " + length);
        }
    }
}
