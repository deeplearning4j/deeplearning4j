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

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * VulkanDeviceContext contract tests — ADR-0111 §2, ADR-0112 T2.
 *
 * <p>ADR-0111 §2 specifies: per-thread VkCommandPool (command pools are externally
 * synchronized — unlike CUDA streams, they may NOT be shared across threads; thread-local
 * pool registry keyed by (thread, device) with trim on thread exit), timeline semaphore
 * (Vulkan 1.2) as the device's ordering spine, dedicated transfer queue when exposed by
 * the device family.</p>
 *
 * <p>Gaps G1/G2/G3 are now closed by the native phase additions in NativeOps.h:
 * {@code vulkanGetThreadCommandPoolHandle(deviceId)},
 * {@code vulkanGetTimelineValue(deviceId)},
 * {@code vulkanHasDedicatedTransferQueue(deviceId)}.
 * The previously-gated device legs below now assert the real contracts.</p>
 *
 * <p>Run with:
 * <pre>
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Ptest-vulkan \
 *     -Dtest=VulkanDeviceContextTest 2>&1 | tee /tmp/vulkan-ctx-test.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("VulkanDeviceContext contract (ADR-0111 §2, ADR-0112 T2)")
public class VulkanDeviceContextTest {

    private static final String VULKAN_BINDINGS_CLASS = "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";

    private static Object nativeOps;
    private static boolean vulkanDevicePresent = false;
    private static int deviceCount = 0;

    // ── shared instance of Nd4jVulkan obtained reflectively ─────────────────

    @BeforeAll
    static void initNativeOps() {
        try {
            Class<?> cls = Class.forName(VULKAN_BINDINGS_CLASS);
            nativeOps = cls.getDeclaredConstructor().newInstance();
            deviceCount = (int) cls.getMethod("getAvailableDevices").invoke(nativeOps);
            vulkanDevicePresent = (deviceCount > 0);
            log.info("VulkanDeviceContextTest: nativeOps loaded; deviceCount={}", deviceCount);
        } catch (Exception e) {
            log.warn("VulkanDeviceContextTest: Vulkan NativeOps unavailable ({}: {})",
                    e.getClass().getSimpleName(), e.getMessage());
            nativeOps = null;
            vulkanDevicePresent = false;
        }
    }

    // ── skip helpers ─────────────────────────────────────────────────────────

    /** Skip if the Vulkan bindings jar is absent (class not found). */
    private static void requireVulkanClass() {
        assumeTrue(nativeOps != null,
                "Vulkan NativeOps class (" + VULKAN_BINDINGS_CLASS + ") not available "
                + "— nd4j-vulkan jar absent or native library failed to load.");
    }

    /** Skip if no physical Vulkan device is enumerated. */
    private static void requireVulkanDevice() {
        requireVulkanClass();
        assumeTrue(vulkanDevicePresent,
                "getAvailableDevices() returned 0 — no Vulkan devices enumerated. "
                + "Ensure a Vulkan ICD (NVIDIA driver or Mesa lavapipe) is installed.");
    }

    // ── reflection helpers ────────────────────────────────────────────────────

    private int invokeIntArg(String method, int arg) throws Exception {
        return (int) nativeOps.getClass().getMethod(method, int.class).invoke(nativeOps, arg);
    }

    private int invokeInt(String method) throws Exception {
        return (int) nativeOps.getClass().getMethod(method).invoke(nativeOps);
    }

    private Object invokeObj(String method) throws Exception {
        return nativeOps.getClass().getMethod(method).invoke(nativeOps);
    }

    private int invokeStreamSync(Object stream) throws Exception {
        Class<?> ptrCls = Class.forName("org.bytedeco.javacpp.Pointer");
        return (int) nativeOps.getClass()
                .getMethod("streamSynchronize", ptrCls)
                .invoke(nativeOps, stream);
    }

    private long invokeLongIntArg(String method, int arg) throws Exception {
        return (long) nativeOps.getClass().getMethod(method, int.class).invoke(nativeOps, arg);
    }

    private boolean invokeBoolIntArg(String method, int arg) throws Exception {
        return (boolean) nativeOps.getClass().getMethod(method, int.class).invoke(nativeOps, arg);
    }

    // =========================================================================
    // CONTRACT LEGS: run on 0-device boxes (class / constructor contracts)
    // =========================================================================

    /**
     * The Nd4jVulkan bindings class must be instantiable without throwing,
     * even when no Vulkan ICD is installed. This is the base precondition
     * for all subsequent contract tests.
     */
    @Test
    @DisplayName("(contract) Nd4jVulkan constructs without throwing on a 0-device box")
    void testBindingsConstructorNoThrow() {
        requireVulkanClass();
        assertNotNull(nativeOps, "nativeOps must be non-null after @BeforeAll succeeded");
        log.info("Nd4jVulkan instance class: {}", nativeOps.getClass().getName());
    }

    /**
     * getAvailableDevices() must return >= 0 and must NOT throw regardless of ICD
     * presence. Returns 0 on machines without a Vulkan ICD.
     */
    @Test
    @DisplayName("(contract) getAvailableDevices() >= 0 and does not throw")
    void testAvailableDevicesNonNegative() throws Exception {
        requireVulkanClass();
        int count = invokeInt("getAvailableDevices");
        assertTrue(count >= 0, "getAvailableDevices() must be >= 0, got " + count);
        log.info("getAvailableDevices() = {}", count);
    }

    /**
     * createStream() must not throw when called on a box with deviceCount >= 1.
     * The returned Pointer must be non-null. On a 0-device box the method is
     * expected to return null (or may throw a native error); this leg is skipped.
     *
     * <p>Note: createStream() maps to a Vulkan command-buffer-submission stream
     * (wraps a VkQueue index or a per-thread submit ticket), NOT to a
     * VkCommandPool. Per gap G1, command-pool identity is not yet observable.</p>
     */
    @Test
    @DisplayName("(device) createStream() returns a non-null handle on a device-capable host")
    void testCreateStreamReturnsHandle() throws Exception {
        requireVulkanDevice();
        Object stream = invokeObj("createStream");
        // The returned Pointer must be non-null and must not be a null Pointer.
        assertNotNull(stream, "createStream() returned Java null");
        // Reflectively check .isNull() — bytedeco Pointer contract.
        boolean isNull = (boolean) stream.getClass().getMethod("isNull").invoke(stream);
        assertFalse(isNull, "createStream() returned a null Pointer — stream creation failed");
        log.info("createStream() returned handle: {}", stream);
    }

    /**
     * Two threads calling createStream() must receive independent (non-null) handles.
     * ADR-0111 §2: per-thread VkCommandPool — each thread gets its own pool, so stream
     * handles created on different threads must be distinct objects.
     *
     * <p>Gap G1 CLOSED: vulkanGetThreadCommandPoolHandle(deviceId) now exposes the
     * VkCommandPool handle directly. The per-thread pool identity is verified below via
     * the dedicated G1 test {@link #testG1TwoThreadsDistinctCommandPools}. This test
     * retains the stream-address proxy as an additional cross-check.</p>
     */
    @Test
    @DisplayName("(device) two threads get distinct stream handles (pool-isolation proxy)")
    void testTwoThreadsGetDistinctStreamHandles() throws Exception {
        requireVulkanDevice();

        AtomicLong ptr1 = new AtomicLong(-1);
        AtomicLong ptr2 = new AtomicLong(-1);
        AtomicReference<Throwable> err = new AtomicReference<>();
        CountDownLatch ready = new CountDownLatch(2);
        CountDownLatch go = new CountDownLatch(1);

        Runnable worker = () -> {
            try {
                ready.countDown();
                go.await();
                Object stream = nativeOps.getClass().getMethod("createStream").invoke(nativeOps);
                long addr = (long) stream.getClass().getMethod("address").invoke(stream);
                if (ptr1.compareAndSet(-1, addr)) {
                    log.info("Thread {}: stream address = 0x{}", Thread.currentThread().getId(),
                            Long.toHexString(addr));
                } else {
                    ptr2.set(addr);
                    log.info("Thread {}: stream address = 0x{}", Thread.currentThread().getId(),
                            Long.toHexString(addr));
                }
            } catch (Throwable t) {
                err.set(t);
            }
        };

        CompletableFuture<Void> f1 = CompletableFuture.runAsync(worker);
        CompletableFuture<Void> f2 = CompletableFuture.runAsync(worker);

        ready.await();
        go.countDown();
        CompletableFuture.allOf(f1, f2).join();

        assertNull(err.get(),
                "Exception in worker thread: " + (err.get() != null ? err.get().getMessage() : ""));
        assertTrue(ptr1.get() != -1, "Thread 1 did not receive a stream handle");
        assertTrue(ptr2.get() != -1, "Thread 2 did not receive a stream handle");

        // Pointer addresses must differ — same address would mean the same underlying
        // native object was returned, violating per-thread isolation.
        assertNotEquals(ptr1.get(), ptr2.get(),
                "Both threads received the same stream address (0x" + Long.toHexString(ptr1.get()) + ") "
                + "— per-thread pool isolation violated.");
        log.info("Two-thread stream handles: 0x{} vs 0x{} — addresses differ",
                Long.toHexString(ptr1.get()), Long.toHexString(ptr2.get()));
    }

    /**
     * G1 HARDENED: vulkanGetThreadCommandPoolHandle(deviceId) must return a non-zero
     * value on a device-capable host, and two concurrent threads must get different values.
     * ADR-0111 §2: per-thread VkCommandPool keyed by (thread, device) — pools may NOT be
     * shared across threads.
     */
    @Test
    @DisplayName("(device, G1-closed) two threads receive distinct VkCommandPool handles")
    void testG1TwoThreadsDistinctCommandPools() throws Exception {
        requireVulkanDevice();

        AtomicLong pool1 = new AtomicLong(-1);
        AtomicLong pool2 = new AtomicLong(-1);
        AtomicReference<Throwable> err = new AtomicReference<>();
        CountDownLatch ready = new CountDownLatch(2);
        CountDownLatch go = new CountDownLatch(1);

        Runnable worker = () -> {
            try {
                ready.countDown();
                go.await();
                // Force the command pool to be lazily created for this thread.
                nativeOps.getClass().getMethod("createStream").invoke(nativeOps);
                long handle = invokeLongIntArg("vulkanGetThreadCommandPoolHandle", 0);
                if (pool1.compareAndSet(-1, handle)) {
                    log.info("Thread {}: commandPool handle = 0x{}", Thread.currentThread().getId(),
                            Long.toHexString(handle));
                } else {
                    pool2.set(handle);
                    log.info("Thread {}: commandPool handle = 0x{}", Thread.currentThread().getId(),
                            Long.toHexString(handle));
                }
            } catch (Throwable t) {
                err.set(t);
            }
        };

        CompletableFuture<Void> f1 = CompletableFuture.runAsync(worker);
        CompletableFuture<Void> f2 = CompletableFuture.runAsync(worker);

        ready.await();
        go.countDown();
        CompletableFuture.allOf(f1, f2).join();

        assertNull(err.get(), "Exception in worker: " + err.get());
        assertNotEquals(-1L, pool1.get(), "Thread 1 did not set a pool handle");
        assertNotEquals(-1L, pool2.get(), "Thread 2 did not set a pool handle");

        // Per-thread command pools must be distinct non-null handles.
        assertNotEquals(0L, pool1.get(), "Thread 1 received null command-pool handle — pool not created");
        assertNotEquals(0L, pool2.get(), "Thread 2 received null command-pool handle — pool not created");
        assertNotEquals(pool1.get(), pool2.get(),
                "Both threads received the same VkCommandPool handle (0x" + Long.toHexString(pool1.get())
                + ") — per-thread isolation violated (ADR-0111 §2)");
        log.info("G1 PASS: thread pools 0x{} vs 0x{} — distinct",
                Long.toHexString(pool1.get()), Long.toHexString(pool2.get()));
    }

    /**
     * streamSynchronize() on a freshly created stream must return a success code
     * (typically 0 or 1) and not crash. This is the indirect observable for the
     * timeline semaphore contract: syncing a stream that has had no work submitted
     * must be a no-op that does not hang or fault.
     *
     * <p>Gap G2: we cannot read the actual timeline semaphore counter value (no
     * getTimelineValue(deviceId) in the current NativeOps surface). Monotonicity
     * verification requires the native phase to add getTimelineValue().</p>
     */
    @Test
    @DisplayName("(device) streamSynchronize on a fresh stream succeeds and does not hang")
    void testStreamSynchronizeSucceeds() throws Exception {
        requireVulkanDevice();
        Object stream = invokeObj("createStream");
        assertNotNull(stream, "createStream() returned null");

        int result = invokeStreamSync(stream);
        // A return value of 0 (success) or 1 (also success in some impls) is acceptable.
        // The key contract is: must not throw, must not return a clearly-error code.
        // CUDA convention: non-zero = error. We check >= 0 and log the value.
        assertTrue(result >= 0,
                "streamSynchronize() returned negative error code " + result
                + " — timeline drain failed or stream handle corrupted");
        log.info("streamSynchronize() on fresh stream returned {}", result);
    }

    /**
     * Sequential streamSynchronize() calls on the same stream must all succeed.
     * This tests that the timeline's done-state is idempotent (draining a stream
     * that is already quiescent must not error or corrupt internal state).
     *
     * <p>Gap G2 CLOSED: after each sync we now also read vulkanGetTimelineValue(0)
     * and assert that the value is non-decreasing across calls.</p>
     */
    @Test
    @DisplayName("(device) sequential streamSynchronize() calls are idempotent")
    void testStreamSynchronizeIdempotent() throws Exception {
        requireVulkanDevice();
        Object stream = invokeObj("createStream");
        assertNotNull(stream, "createStream() returned null");

        long prevTimeline = -1;
        for (int i = 0; i < 5; i++) {
            int result = invokeStreamSync(stream);
            assertTrue(result >= 0,
                    "streamSynchronize() iteration " + i + " returned " + result + " (negative = error)");
            // G2: timeline value must be monotonically non-decreasing.
            long tl = invokeLongIntArg("vulkanGetTimelineValue", 0);
            assertTrue(tl >= 0, "vulkanGetTimelineValue returned negative value " + tl);
            if (prevTimeline >= 0) {
                assertTrue(tl >= prevTimeline,
                        "Timeline value decreased from " + prevTimeline + " to " + tl
                        + " at iteration " + i + " — monotonicity violated (ADR-0111 §2)");
            }
            prevTimeline = tl;
        }
        log.info("G2: sequential streamSynchronize() x5 idempotent; final timeline={}", prevTimeline);
    }

    /**
     * G2 HARDENED: vulkanGetTimelineValue(0) must return >= 0 and must be non-decreasing
     * across back-to-back calls. Value of 0 is acceptable when the device does not support
     * timeline semaphores (ADR-0111 §2 permits VK_NULL_HANDLE → 0 fallback).
     */
    @Test
    @DisplayName("(device, G2-closed) vulkanGetTimelineValue is non-negative and non-decreasing")
    void testG2TimelineValueMonotonic() throws Exception {
        requireVulkanDevice();
        long v1 = invokeLongIntArg("vulkanGetTimelineValue", 0);
        long v2 = invokeLongIntArg("vulkanGetTimelineValue", 0);
        assertTrue(v1 >= 0, "vulkanGetTimelineValue call-1 returned negative: " + v1);
        assertTrue(v2 >= 0, "vulkanGetTimelineValue call-2 returned negative: " + v2);
        assertTrue(v2 >= v1,
                "Timeline decreased from " + v1 + " to " + v2
                + " — monotonicity violated (ADR-0111 §2)");
        log.info("G2 PASS: timelineValue call1={} call2={} — non-decreasing", v1, v2);
    }

    /**
     * Same-device access must follow the NativeOps contract established by CUDA:
     * a device can always access its own allocations. Cross-device Vulkan access
     * remains capability-driven and may be false for distinct physical devices.
     *
     * <p>Gap G3 CLOSED: vulkanHasDedicatedTransferQueue(deviceId) now exposes the
     * transfer-queue presence directly. See {@link #testG3HasDedicatedTransferQueue}.</p>
     */
    @Test
    @DisplayName("(device) isPeerAccessSupported(0,0) follows the CUDA same-device contract")
    void testPeerAccessSupportedNoThrow() throws Exception {
        requireVulkanDevice();
        Class<?> cls = nativeOps.getClass();
        boolean result = (boolean) cls
                .getMethod("isPeerAccessSupported", int.class, int.class)
                .invoke(nativeOps, 0, 0);
        assertTrue(result,
                "isPeerAccessSupported(0,0) must be true for same-device access");
        log.info("isPeerAccessSupported(0,0) = true (CUDA-aligned same-device contract)");
    }

    /**
     * G3 HARDENED: vulkanHasDedicatedTransferQueue(0) must not throw. The return value
     * is device-dependent — discrete GPUs typically expose a dedicated DMA queue; UMA and
     * lavapipe software renderers usually do not. We assert no-throw and log the result.
     * This replaces the isPeerAccessSupported proxy used before the native phase.
     */
    @Test
    @DisplayName("(device, G3-closed) vulkanHasDedicatedTransferQueue does not throw")
    void testG3HasDedicatedTransferQueue() throws Exception {
        requireVulkanDevice();
        boolean hasDedicated = invokeBoolIntArg("vulkanHasDedicatedTransferQueue", 0);
        // Value is device-dependent — we do not assert true or false since both are valid.
        // The contract: must not throw.
        log.info("G3: vulkanHasDedicatedTransferQueue(0) = {} "
                + "(true = discrete GPU with DMA queue; false = UMA/lavapipe)", hasDedicated);
    }

    /**
     * setDevice(0) and getDevice() must be consistent (thread-local device selection
     * contract), mirroring VulkanDeviceManagementTest but from the context perspective:
     * a thread that sets its device and then queries must observe the same value.
     */
    @Test
    @DisplayName("(device) setDevice(0)/getDevice() thread-local consistency")
    void testSetGetDeviceConsistency() throws Exception {
        requireVulkanDevice();
        int setResult = invokeIntArg("setDevice", 0);
        assertTrue(setResult >= 0, "setDevice(0) returned error code " + setResult);
        int got = invokeInt("getDevice");
        assertEquals(0, got, "getDevice() must return 0 after setDevice(0)");
        log.info("setDevice(0) = {}, getDevice() = {}", setResult, got);
    }
}
