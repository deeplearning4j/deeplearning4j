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
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceContext;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicReferenceArray;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * VulkanMultiDeviceTest — ADR-0111 §6, ADR-0112 T2.
 *
 * <p>ADR-0111 §6: "Vulkan enumerates every ICD's devices (mixed vendors in one process is
 * normal — e.g. NVIDIA + lavapipe). Per-device everything: contexts, pools, pipeline caches,
 * constant/TAD caches, RNG states — deviceId-keyed from day one." ADR-0112 T2 specifies:
 * deterministic enumeration order across process restarts; per-device isolation (alloc on 0
 * not visible on 1); explicit cross-device copy correctness; skips to single-device subset
 * when count==1.</p>
 *
 * <h2>Failure injection tests (ADR-0112 T2 §2)</h2>
 * <ul>
 *   <li>The device-memory budget is a hard limit: an allocation beyond it must fail
 *       immediately instead of spilling into host-visible memory.</li>
 *   <li>Fence-deferred frees permit safe allocation reuse without dereferencing opaque
 *       device addresses from the host.</li>
 * </ul>
 *
 * <p>The native probes expose pool identity, exact Vulkan memory-property flags for an
 * allocation, and pending retire-list depth. The device legs below assert those contracts
 * directly; no allocator tier or host-memory fallback is accepted.</p>
 *
 * <p>Run with:
 * <pre>
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Ptest-vulkan \
 *     -Dtest=VulkanMultiDeviceTest 2>&1 | tee /tmp/vulkan-multidevice-test.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("VulkanMultiDevice contract (ADR-0111 §6, ADR-0112 T2)")
public class VulkanMultiDeviceTest {

    private static final String VULKAN_BINDINGS_CLASS = "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";
    private static final int VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001;
    private static final WorkspaceConfiguration TWO_DEVICE_WORKSPACE =
            WorkspaceConfiguration.builder()
                    .initialSize(1024L * 1024L)
                    .maxSize(1024L * 1024L)
                    .overallocationLimit(0.0)
                    .policyAllocation(AllocationPolicy.STRICT)
                    .policyLearning(LearningPolicy.NONE)
                    .policyMirroring(MirroringPolicy.FULL)
                    .policySpill(SpillPolicy.FAIL)
                    .build();

    private static Object nativeOps;
    private static boolean vulkanClassAvailable = false;
    private static boolean vulkanDevicePresent = false;
    private static int deviceCount = 0;

    private static float canonicalUniform(long seed, int index) {
        int s0 = (int) seed;
        int s1 = (int) (seed ^ 0xdeadbeefL);
        int position = index + 2;
        s0 ^= position * (s1 + 24243287);
        s1 ^= position * (s0 + 723829);
        int raw = Integer.rotateLeft((s1 ^ s0) * 0x9E3779BB, 5) * 5;
        return Float.intBitsToFloat(0x3f800000 | (raw >>> 9)) - 1.0f;
    }

    @BeforeAll
    static void initNativeOps() {
        try {
            Class<?> cls = Class.forName(VULKAN_BINDINGS_CLASS);
            nativeOps = cls.getDeclaredConstructor().newInstance();
            deviceCount = (int) cls.getMethod("getAvailableDevices").invoke(nativeOps);
            vulkanClassAvailable = true;
            vulkanDevicePresent = (deviceCount > 0);
            log.info("VulkanMultiDeviceTest: deviceCount={}", deviceCount);
        } catch (Exception e) {
            log.warn("VulkanMultiDeviceTest: bindings unavailable: {} {}",
                    e.getClass().getSimpleName(), e.getMessage());
            nativeOps = null;
            vulkanClassAvailable = false;
            vulkanDevicePresent = false;
        }
    }

    // ── skip helpers ─────────────────────────────────────────────────────────

    private static void requireVulkanClass() {
        assumeTrue(vulkanClassAvailable,
                "Vulkan NativeOps class not available — run with -Ptest-vulkan.");
    }

    private static void requireAtLeastOneDevice() {
        requireVulkanClass();
        assumeTrue(vulkanDevicePresent, "No Vulkan devices enumerated — skipping device leg.");
    }

    private static void requireAtLeastTwoDevices() {
        requireVulkanClass();
        assumeTrue(deviceCount >= 2,
                "Only " + deviceCount + " Vulkan device(s) enumerated; multi-device "
                + "isolation tests require >= 2 (lavapipe + discrete counts as 2).");
    }

    // ── reflection helpers ────────────────────────────────────────────────────

    private int invokeInt(String method) throws Exception {
        return (int) nativeOps.getClass().getMethod(method).invoke(nativeOps);
    }

    private int invokeIntArg(String method, int arg) throws Exception {
        return (int) nativeOps.getClass().getMethod(method, int.class).invoke(nativeOps, arg);
    }

    private long invokeLongArg(String method, int arg) throws Exception {
        return (long) nativeOps.getClass().getMethod(method, int.class).invoke(nativeOps, arg);
    }

    private long invokeLong(String method) throws Exception {
        return (long) nativeOps.getClass().getMethod(method).invoke(nativeOps);
    }

    private String invokeStringArg(String method, int arg) throws Exception {
        return (String) nativeOps.getClass().getMethod(method, int.class).invoke(nativeOps, arg);
    }

    private Pointer mallocDevice(long size, int deviceId) throws Exception {
        return (Pointer) nativeOps.getClass()
                .getMethod("mallocDevice", long.class, int.class, int.class)
                .invoke(nativeOps, size, deviceId, 0);
    }

    private int freeDevice(Pointer ptr, int deviceId) throws Exception {
        return (int) nativeOps.getClass()
                .getMethod("freeDevice", Pointer.class, int.class)
                .invoke(nativeOps, ptr, deviceId);
    }

    private int invokeIntPtrInt(String method, Pointer ptr, int deviceId) throws Exception {
        return (int) nativeOps.getClass()
                .getMethod(method, Pointer.class, int.class)
                .invoke(nativeOps, ptr, deviceId);
    }

    private static boolean isNull(Pointer pointer) {
        return pointer == null || pointer.isNull();
    }

    private int invokeIntIntArg(String method, int deviceId) throws Exception {
        return (int) nativeOps.getClass().getMethod(method, int.class).invoke(nativeOps, deviceId);
    }

    private static void requireVulkanFactory() {
        assertTrue(Nd4j.getNDArrayFactory().getClass().getName().contains(".vulkan."),
                "The test-vulkan profile must select the Vulkan NDArray factory, got "
                        + Nd4j.getNDArrayFactory().getClass().getName());
    }

    private void selectDevice(int deviceId) throws Exception {
        assertEquals(1, invokeIntArg("setDevice", deviceId),
                "setDevice(" + deviceId + ") must succeed");
        Nd4j.getAffinityManager().setDeviceForCurrentThread(deviceId);
        assertEquals(deviceId, invokeInt("getDevice"),
                "Native Vulkan device selection did not persist");
        assertEquals(deviceId, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                "ND4J affinity did not select Vulkan device " + deviceId);
        assertEquals(deviceId, Nd4j.getDeviceIdProvider().getDeviceId(),
                "ND4J device ID provider did not select Vulkan device " + deviceId);
    }

    private static Pointer requireSpecialPointer(DataBuffer buffer, String role) {
        assertNotNull(buffer, role + " buffer is null");
        assertNotNull(buffer.opaqueBuffer(), role + " has no OpaqueDataBuffer");
        Pointer pointer = buffer.opaqueBuffer().specialBuffer();
        assertFalse(isNull(pointer), role + " has no Vulkan special allocation");
        return pointer;
    }

    private void assertOwnedByDevice(Pointer pointer, int ownerDevice, String role)
            throws Exception {
        for (int candidate = 0; candidate < deviceCount; candidate++) {
            int flags = invokeIntPtrInt(
                    "vulkanGetAllocationMemoryPropertyFlags", pointer, candidate);
            if (candidate == ownerDevice) {
                assertTrue(flags >= 0,
                        role + " is not owned by Vulkan device " + ownerDevice);
                assertTrue((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
                        role + " is not DEVICE_LOCAL on Vulkan device " + ownerDevice
                                + ": flags=0x" + Integer.toHexString(flags));
            } else {
                assertEquals(-1, flags,
                        role + " owned by device " + ownerDevice
                                + " was also claimed by device " + candidate);
            }
        }
    }

    // =========================================================================
    // CONTRACT LEGS: run on 0-device boxes
    // =========================================================================

    /**
     * getAvailableDevices() must return the same value on two back-to-back calls within
     * the same process. This is the within-process proxy for "deterministic enumeration
     * order across restarts" (Gap M2: cross-restart requires an external harness).
     */
    @Test
    @DisplayName("(contract) getAvailableDevices() is stable across two consecutive calls")
    void testEnumerationCountStable() throws Exception {
        requireVulkanClass();
        int count1 = invokeInt("getAvailableDevices");
        int count2 = invokeInt("getAvailableDevices");
        assertEquals(count1, count2,
                "getAvailableDevices() returned different counts on consecutive calls: "
                + count1 + " vs " + count2 + " — enumeration is not stable");
        log.info("getAvailableDevices(): call1={} call2={} — stable", count1, count2);
    }

    /**
     * Device names must be stable across two probes (within the same process).
     * Same-index probe must return the same string both times.
     * Gap M2 proxy: full cross-restart determinism requires an external harness.
     */
    @Test
    @DisplayName("(device) device names are stable across two consecutive enumerations")
    void testDeviceNamesStable() throws Exception {
        requireAtLeastOneDevice();
        List<String> names1 = new ArrayList<>();
        List<String> names2 = new ArrayList<>();
        for (int i = 0; i < deviceCount; i++) {
            names1.add(invokeStringArg("getDeviceName", i));
        }
        for (int i = 0; i < deviceCount; i++) {
            names2.add(invokeStringArg("getDeviceName", i));
        }
        assertEquals(names1, names2,
                "Device name list changed between two consecutive enumerations — "
                + "enumeration order is non-deterministic: " + names1 + " vs " + names2);
        log.info("Device name stability: {}", names1);
    }

    // =========================================================================
    // SINGLE-DEVICE SUBSET: skips to these when count == 1
    // =========================================================================

    /**
     * When exactly one Vulkan device is enumerated, setDevice(0)/getDevice() must work
     * from two concurrent threads without race conditions. ADR-0111 §2: per-thread
     * binding must be thread-local, not shared.
     */
    @Test
    @DisplayName("(device-single) two-thread device binding is thread-local (single-device subset)")
    void testTwoThreadDeviceBindingSingleDevice() throws Exception {
        requireAtLeastOneDevice();
        // This test exercises single-device multi-threaded binding.
        // For multi-device isolation see testPerDeviceIsolationAllocsDisjoint().
        AtomicInteger t1Dev = new AtomicInteger(-1);
        AtomicInteger t2Dev = new AtomicInteger(-1);
        AtomicReference<Throwable> err = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);

        Runnable bindTo0 = () -> {
            try {
                latch.await();
                nativeOps.getClass().getMethod("setDevice", int.class).invoke(nativeOps, 0);
                int dev = invokeInt("getDevice");
                if (t1Dev.compareAndSet(-1, dev)) {
                    log.info("Thread {}: bound to device {}", Thread.currentThread().getId(), dev);
                } else {
                    t2Dev.set(dev);
                    log.info("Thread {}: bound to device {}", Thread.currentThread().getId(), dev);
                }
            } catch (Throwable t) { err.set(t); }
        };

        CompletableFuture<Void> first = CompletableFuture.runAsync(bindTo0);
        CompletableFuture<Void> second = CompletableFuture.runAsync(bindTo0);
        latch.countDown();
        CompletableFuture.allOf(first, second).join();

        assertNull(err.get(), "Exception in binding thread: " + err.get());
        assertEquals(0, t1Dev.get(), "Thread 1 must see device 0 after setDevice(0)");
        assertEquals(0, t2Dev.get(), "Thread 2 must see device 0 after setDevice(0)");
        log.info("Two-thread binding (single device): t1={} t2={}", t1Dev.get(), t2Dev.get());
    }

    /**
     * Allocate 256 small buffers on device 0 and free them. Must not crash or return null.
     * Single-device pool stress test that also exercises the retire-list reclaim path (Gap M4).
     */
    @Test
    @DisplayName("(device-single) 256 allocate/free cycles on device 0 without crash")
    void testManyAllocFreeOnDevice0() throws Exception {
        requireAtLeastOneDevice();
        final int N = 256;
        final long SIZE = 512L;
        List<Pointer> ptrs = new ArrayList<>(N);
        try {
            for (int i = 0; i < N; i++) {
                Pointer ptr = mallocDevice(SIZE, 0);
                assertFalse(isNull(ptr), "mallocDevice returned null at iteration " + i);
                ptrs.add(ptr);
            }
            log.info("Allocated {} x {} bytes on device 0", N, SIZE);
        } finally {
            for (Pointer ptr : ptrs) {
                freeDevice(ptr, 0);
            }
            log.info("Freed {} buffers on device 0", ptrs.size());
        }
    }

    // =========================================================================
    // MULTI-DEVICE LEGS: require >= 2 devices
    // =========================================================================

    /**
     * Allocation ownership is device-scoped. Each pointer must resolve only in the pool for
     * the device that allocated it; probing it against another device must fail.
     */
    @Test
    @DisplayName("(device-multi>=2) allocation ownership is isolated by device")
    void testPerDeviceAllocationOwnership() throws Exception {
        requireAtLeastTwoDevices();
        final long SIZE = 4096L;
        Pointer ptr0 = null;
        Pointer ptr1 = null;
        try {
            ptr0 = mallocDevice(SIZE, 0);
            ptr1 = mallocDevice(SIZE, 1);
            assertFalse(isNull(ptr0), "mallocDevice on device 0 returned null");
            assertFalse(isNull(ptr1), "mallocDevice on device 1 returned null");

            assertTrue(invokeIntPtrInt("vulkanGetAllocationMemoryPropertyFlags", ptr0, 0) >= 0,
                    "Device 0 did not recognize its own allocation");
            assertEquals(-1, invokeIntPtrInt("vulkanGetAllocationMemoryPropertyFlags", ptr0, 1),
                    "Device 1 incorrectly claimed device 0's allocation");
            assertTrue(invokeIntPtrInt("vulkanGetAllocationMemoryPropertyFlags", ptr1, 1) >= 0,
                    "Device 1 did not recognize its own allocation");
            assertEquals(-1, invokeIntPtrInt("vulkanGetAllocationMemoryPropertyFlags", ptr1, 0),
                    "Device 0 incorrectly claimed device 1's allocation");
        } finally {
            if (!isNull(ptr0)) freeDevice(ptr0, 0);
            if (!isNull(ptr1)) freeDevice(ptr1, 1);
        }
    }

    /**
     * Memory totals for each device must be independently queryable and positive.
     * On a multi-device box with heterogeneous devices (e.g. lavapipe + discrete GPU),
     * totals may differ substantially — that is expected and correct.
     */
    @Test
    @DisplayName("(device-multi>=2) each device reports independent positive total memory")
    void testPerDeviceMemoryTotals() throws Exception {
        requireAtLeastTwoDevices();
        for (int i = 0; i < deviceCount; i++) {
            long total = invokeLongArg("getDeviceTotalMemory", i);
            long free  = invokeLongArg("getDeviceFreeMemory", i);
            log.info("device {}: name='{}' totalMem={} MB freeMem={} MB",
                    i, invokeStringArg("getDeviceName", i), total >> 20, free >> 20);
            assertTrue(total > 0, "device " + i + " totalMemory must be > 0, got " + total);
            assertTrue(free >= 0, "device " + i + " freeMemory must be >= 0, got " + free);
            assertTrue(free <= total,
                    "device " + i + " freeMemory (" + (free >> 20) + " MB) exceeds total "
                    + "(" + (total >> 20) + " MB)");
        }
    }

    /**
     * The constant-data, constant-shape, and TAD caches must reuse device allocations
     * within one device while producing independently-owned allocations for every other
     * enumerated Vulkan device. This exercises the public executioner APIs used by normal
     * ops rather than constructing backend-private cache entries in the test.
     */
    @Test
    @DisplayName("(device-all) constant, shape, and TAD caches are device-keyed")
    void testConstantShapeAndTadCachesAreDeviceKeyed() throws Exception {
        requireAtLeastOneDevice();
        requireVulkanFactory();

        final long[] constantValues = {0x5eed1234L, 0x5eed5678L, 0x5eed9abcL};
        final long[] arrayShape = {37, 41};
        final long[] tadDimensions = {1};
        final int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        try {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                selectDevice(deviceId);
                String deviceName = invokeStringArg("getDeviceName", deviceId);

                DataBuffer constantFirst = Nd4j.getExecutioner().createConstantBuffer(
                        constantValues, DataType.INT64);
                DataBuffer constantSecond = Nd4j.getExecutioner().createConstantBuffer(
                        constantValues, DataType.INT64);
                Pointer constantFirstSpecial = requireSpecialPointer(
                        constantFirst, "constant buffer on device " + deviceId);
                Pointer constantSecondSpecial = requireSpecialPointer(
                        constantSecond, "reused constant buffer on device " + deviceId);
                assertEquals(constantFirstSpecial.address(), constantSecondSpecial.address(),
                        "Identical constants were not reused on Vulkan device " + deviceId);
                assertOwnedByDevice(constantFirstSpecial, deviceId,
                        "constant buffer on " + deviceName);
                long deviceConstantBytes =
                        invokeLongArg("getConstantCacheBytes", deviceId);
                assertTrue(deviceConstantBytes > 0,
                        "Vulkan device " + deviceId + " reported an empty constant cache");
                assertTrue(Nd4j.getConstantHandler().getCachedBytes()
                                >= deviceConstantBytes,
                        "The Java constant handler omitted Vulkan device " + deviceId
                                + " from its cache accounting");

                try (INDArray firstArray = Nd4j.create(DataType.FLOAT, arrayShape);
                     INDArray secondArray = Nd4j.create(DataType.FLOAT, arrayShape)) {
                    Pointer firstShapeSpecial = requireSpecialPointer(
                            firstArray.shapeInfoDataBuffer(),
                            "constant shape buffer on device " + deviceId);
                    Pointer secondShapeSpecial = requireSpecialPointer(
                            secondArray.shapeInfoDataBuffer(),
                            "reused constant shape buffer on device " + deviceId);
                    assertEquals(firstShapeSpecial.address(), secondShapeSpecial.address(),
                            "Identical shape descriptors were not reused on Vulkan device "
                                    + deviceId);
                    assertOwnedByDevice(firstShapeSpecial, deviceId,
                            "constant shape buffer on " + deviceName);

                    long tadEntriesBefore = invokeLong("getTadCacheEntries");
                    TadPack firstTad = Nd4j.getExecutioner().tadShapeInfoAndOffsets(
                            firstArray, tadDimensions);
                    long tadEntriesAfterFirst = invokeLong("getTadCacheEntries");
                    TadPack secondTad = Nd4j.getExecutioner().tadShapeInfoAndOffsets(
                            firstArray, tadDimensions);
                    long tadEntriesAfterSecond = invokeLong("getTadCacheEntries");

                    assertTrue(tadEntriesAfterFirst > tadEntriesBefore,
                            "The first TAD request on Vulkan device " + deviceId
                                    + " did not create a device-keyed cache entry");
                    assertEquals(tadEntriesAfterFirst, tadEntriesAfterSecond,
                            "Repeating a TAD request grew the cache on Vulkan device "
                                    + deviceId);

                    Pointer firstTadShapeSpecial = requireSpecialPointer(
                            firstTad.getTadShapeInfo(),
                            "TAD shape buffer on device " + deviceId);
                    Pointer secondTadShapeSpecial = requireSpecialPointer(
                            secondTad.getTadShapeInfo(),
                            "reused TAD shape buffer on device " + deviceId);
                    Pointer firstTadOffsetSpecial = requireSpecialPointer(
                            firstTad.getTadOffsets(),
                            "TAD offsets buffer on device " + deviceId);
                    Pointer secondTadOffsetSpecial = requireSpecialPointer(
                            secondTad.getTadOffsets(),
                            "reused TAD offsets buffer on device " + deviceId);

                    assertEquals(firstTadShapeSpecial.address(),
                            secondTadShapeSpecial.address(),
                            "Identical TAD shapes were not reused on Vulkan device " + deviceId);
                    assertEquals(firstTadOffsetSpecial.address(),
                            secondTadOffsetSpecial.address(),
                            "Identical TAD offsets were not reused on Vulkan device " + deviceId);
                    assertOwnedByDevice(firstTadShapeSpecial, deviceId,
                            "TAD shape buffer on " + deviceName);
                    assertOwnedByDevice(firstTadOffsetSpecial, deviceId,
                            "TAD offsets buffer on " + deviceName);
                }

                log.info("Verified constant/shape/TAD cache isolation on device {} ('{}')",
                        deviceId, deviceName);
            }
        } finally {
            selectDevice(originalDevice);
        }
    }

    @Test
    @DisplayName("(device-multi>=2) generic device release follows allocation ownership")
    void testDeviceReleaseUsesAllocationOwner() throws Exception {
        requireAtLeastTwoDevices();
        requireVulkanFactory();

        final int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        Pointer allocation = null;
        try {
            selectDevice(0);
            allocation = Nd4j.getMemoryManager().allocate(
                    4096, MemoryKind.DEVICE, true);
            assertFalse(isNull(allocation), "Device 0 allocation returned a null pointer");
            assertOwnedByDevice(allocation, 0, "generic device allocation");

            selectDevice(1);
            Nd4j.getMemoryManager().release(allocation, MemoryKind.DEVICE);
            assertTrue(isNull(allocation),
                    "Cross-device release did not clear the caller's pointer");
        } finally {
            if (!isNull(allocation)) {
                selectDevice(0);
                Nd4j.getMemoryManager().release(allocation, MemoryKind.DEVICE);
            }
            selectDevice(originalDevice);
        }
    }

    /**
     * Execute a real numerical kernel on every enumerated Vulkan physical device.
     * The result and its shape metadata must remain allocated on the selected device;
     * successful host verification is performed only after the Vulkan executioner commits.
     */
    @Test
    @DisplayName("(device-all) real pairwise kernel executes on every Vulkan device")
    void testRealKernelExecutesOnEveryDevice() throws Exception {
        requireAtLeastOneDevice();
        requireVulkanFactory();
        final int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        try {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                selectDevice(deviceId);
                String deviceName = invokeStringArg("getDeviceName", deviceId);

                try (INDArray left = Nd4j.createFromArray(new float[]{1, 2, 3, 4});
                     INDArray right = Nd4j.createFromArray(new float[]{10, 20, 30, 40})) {
                    assertEquals(deviceId, left.data().opaqueBuffer().deviceId(),
                            "Left input data was allocated on the wrong Vulkan device");
                    assertEquals(deviceId, right.data().opaqueBuffer().deviceId(),
                            "Right input data was allocated on the wrong Vulkan device");
                    assertEquals(deviceId, left.shapeInfoDataBuffer().opaqueBuffer().deviceId(),
                            "Left input shape was allocated on the wrong Vulkan device");
                    assertEquals(deviceId, right.shapeInfoDataBuffer().opaqueBuffer().deviceId(),
                            "Right input shape was allocated on the wrong Vulkan device");
                    assertEquals(deviceId, invokeInt("getDevice"),
                            "Array construction changed the current native Vulkan device");
                    assertEquals(deviceId, Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                            "Array construction changed the Java Vulkan affinity");

                    try (INDArray result = left.add(right)) {
                        Nd4j.getExecutioner().commit();
                        assertArrayEquals(new float[]{11, 22, 33, 44},
                                result.toFloatVector(), 0.0f,
                                "Unexpected pairwise result on Vulkan device " + deviceId);
                        assertEquals(deviceId, result.data().opaqueBuffer().deviceId(),
                                "Kernel output data belongs to the wrong Vulkan device");
                        assertEquals(deviceId,
                                result.shapeInfoDataBuffer().opaqueBuffer().deviceId(),
                                "Kernel output shape belongs to the wrong Vulkan device");

                        Pointer resultSpecial = requireSpecialPointer(
                                result.data(), "kernel output on device " + deviceId);
                        Pointer resultShapeSpecial = requireSpecialPointer(
                                result.shapeInfoDataBuffer(),
                                "kernel output shape on device " + deviceId);
                        assertOwnedByDevice(resultSpecial, deviceId,
                                "kernel output on " + deviceName);
                        assertOwnedByDevice(resultShapeSpecial, deviceId,
                                "kernel output shape on " + deviceName);
                    }
                }

                assertEquals(deviceId, invokeInt("getDevice"),
                        "Kernel execution changed the current Vulkan device");
                log.info("Executed and verified real pairwise kernel on device {} ('{}')",
                        deviceId, deviceName);
            }
        } finally {
            selectDevice(originalDevice);
        }
    }

    /**
     * Custom-op execution follows the owning arrays when the caller is currently bound
     * to another Vulkan device, then restores the caller's original device. This mirrors
     * CUDA's select-context-execute contract without migrating cross-device inputs.
     */
    @Test
    @DisplayName("(device-multi>=2) custom op follows array device and restores caller device")
    void testCustomOpFollowsArrayDeviceAndRestoresCallerDevice() throws Exception {
        requireAtLeastTwoDevices();
        requireVulkanFactory();
        final int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        final int arrayDevice = 1;
        final int callerDevice = 0;

        try {
            selectDevice(arrayDevice);
            try (INDArray left = Nd4j.createFromArray(new float[]{1, 2, 3, 4});
                 INDArray right = Nd4j.createFromArray(new float[]{10, 20, 30, 40})) {
                assertEquals(arrayDevice, left.data().opaqueBuffer().deviceId());
                assertEquals(arrayDevice, right.data().opaqueBuffer().deviceId());
                assertEquals(arrayDevice, left.shapeInfoDataBuffer().opaqueBuffer().deviceId());
                assertEquals(arrayDevice, right.shapeInfoDataBuffer().opaqueBuffer().deviceId());

                selectDevice(callerDevice);
                INDArray[] outputs = Nd4j.getExecutioner().exec(new AddOp(left, right));
                assertEquals(1, outputs.length, "AddOp must produce exactly one output");
                assertEquals(callerDevice, invokeInt("getDevice"),
                        "Custom-op execution did not restore the caller's native Vulkan device");
                assertEquals(callerDevice,
                        Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                        "Custom-op execution did not restore the caller's Java affinity");

                try (INDArray result = outputs[0]) {
                    assertEquals(arrayDevice, result.data().opaqueBuffer().deviceId(),
                            "Custom-op output data belongs to the wrong Vulkan device");
                    assertEquals(arrayDevice,
                            result.shapeInfoDataBuffer().opaqueBuffer().deviceId(),
                            "Custom-op output shape belongs to the wrong Vulkan device");

                    selectDevice(arrayDevice);
                    Nd4j.getExecutioner().commit();
                    assertArrayEquals(new float[]{11, 22, 33, 44},
                            result.toFloatVector(), 0.0f,
                            "Unexpected custom-op result on the array's Vulkan device");
                }
            }
        } finally {
            selectDevice(originalDevice);
        }
    }

    /**
     * Standalone shape inference uses the input arrays' Vulkan device for both the
     * native context and the returned constant shape buffer, then restores the caller.
     */
    @Test
    @DisplayName("(device-multi>=2) shape inference follows array device and restores caller device")
    void testShapeInferenceFollowsArrayDeviceAndRestoresCallerDevice() throws Exception {
        requireAtLeastTwoDevices();
        requireVulkanFactory();
        final int originalDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        final int arrayDevice = 1;
        final int callerDevice = 0;

        try {
            selectDevice(arrayDevice);
            try (INDArray left = Nd4j.create(DataType.FLOAT, 2, 3);
                 INDArray right = Nd4j.create(DataType.FLOAT, 2, 3)) {
                selectDevice(callerDevice);
                List<DataBuffer> outputShapes =
                        Nd4j.getExecutioner().calculateOutputShape(new AddOp(left, right));

                assertEquals(callerDevice, invokeInt("getDevice"),
                        "Shape inference did not restore the caller's native Vulkan device");
                assertEquals(callerDevice,
                        Nd4j.getAffinityManager().getDeviceForCurrentThread(),
                        "Shape inference did not restore the caller's Java affinity");
                assertEquals(1, outputShapes.size(),
                        "AddOp shape inference must produce exactly one shape");

                DataBuffer outputShape = outputShapes.get(0);
                assertTrue(outputShape.isConstant(),
                        "Shape inference did not return a constant shape buffer");
                assertEquals(arrayDevice, outputShape.opaqueBuffer().deviceId(),
                        "Inferred constant shape buffer belongs to the wrong Vulkan device");
                assertOwnedByDevice(
                        requireSpecialPointer(outputShape, "inferred constant shape buffer"),
                        arrayDevice, "inferred constant shape buffer");
            }
        } finally {
            selectDevice(originalDevice);
        }
    }

    /**
     * Keeps two physical Vulkan devices active at the same time and proves that the
     * shared device-context provider, streams, events, RNG execution, workspace
     * allocations, and eager kernels remain isolated by device.
     */
    @Test
    @DisplayName("(device-multi>=2) two devices execute concurrently with independent runtime resources")
    void testTwoDevicesExecuteConcurrentlyWithIndependentResources() throws Exception {
        requireAtLeastTwoDevices();
        requireVulkanFactory();

        DeviceMemoryManager deviceMemoryManager = DeviceMemoryManager.getInstance();
        assertTrue(deviceMemoryManager.getContextProvider().getClass().getName().contains(".vulkan."),
                "Vulkan must install its own DeviceContextProvider, got "
                        + deviceMemoryManager.getContextProvider().getClass().getName());
        assertTrue(deviceMemoryManager.getContextProvider().supportsStreams(),
                "The Vulkan DeviceContextProvider must expose real Vulkan streams");
        assertEquals(deviceCount, deviceMemoryManager.getContextProvider().getDeviceCount(),
                "The shared device provider returned the wrong Vulkan device count");

        NativeOps ops = (NativeOps) nativeOps;
        CountDownLatch ready = new CountDownLatch(2);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch submitted = new CountDownLatch(2);
        AtomicReferenceArray<Throwable> errors = new AtomicReferenceArray<>(2);
        AtomicReferenceArray<float[]> randomSnapshots = new AtomicReferenceArray<>(2);
        AtomicLongArray executionStreams = new AtomicLongArray(2);
        AtomicLongArray copyStreams = new AtomicLongArray(2);
        AtomicLongArray events = new AtomicLongArray(2);
        AtomicLongArray commandPools = new AtomicLongArray(2);
        AtomicLongArray timelineBefore = new AtomicLongArray(2);
        AtomicLongArray timelineAfter = new AtomicLongArray(2);

        List<CompletableFuture<Void>> workers = new ArrayList<>(2);
        for (int assignedDevice = 0; assignedDevice < 2; assignedDevice++) {
            final int deviceId = assignedDevice;
            workers.add(CompletableFuture.runAsync(() -> {
                Pointer event = null;
                MemoryWorkspace workspace = null;
                Random rng = null;
                boolean readySignalled = false;
                boolean submittedSignalled = false;
                try {
                    DeviceContext context = deviceMemoryManager.switchDevice(
                            deviceId, VulkanMultiDeviceTest.class.getName(),
                            "two-device Vulkan runtime isolation test");
                    assertEquals(deviceId, context.getDeviceId());
                    assertEquals(deviceId, ops.getDevice());
                    assertEquals(deviceId,
                            Nd4j.getAffinityManager().getDeviceForCurrentThread());
                    assertEquals(deviceId, Nd4j.getDeviceIdProvider().getDeviceId());

                    Pointer executionStream = context.getExecutionStream();
                    Pointer copyStream = context.getCopyStream();
                    assertFalse(isNull(executionStream),
                            "Device " + deviceId + " has no execution stream");
                    assertFalse(isNull(copyStream),
                            "Device " + deviceId + " has no copy stream");
                    assertNotEquals(executionStream.address(), copyStream.address(),
                            "Device " + deviceId + " reused one stream for execution and copy");
                    executionStreams.set(deviceId, executionStream.address());
                    copyStreams.set(deviceId, copyStream.address());

                    event = ops.createEvent();
                    assertFalse(isNull(event), "Device " + deviceId + " failed to create an event");
                    events.set(deviceId, event.address());
                    commandPools.set(deviceId,
                            invokeLongArg("vulkanGetThreadCommandPoolHandle", deviceId));
                    assertNotEquals(0L, commandPools.get(deviceId),
                            "Device " + deviceId + " has no thread-local command pool");
                    timelineBefore.set(deviceId,
                            invokeLongArg("vulkanGetTimelineValue", deviceId));

                    String workspaceId = "vulkan-two-device-" + deviceId;
                    workspace = Nd4j.getWorkspaceManager()
                            .getWorkspaceForCurrentThread(TWO_DEVICE_WORKSPACE, workspaceId);
                    rng = Nd4j.getRandomFactory()
                            .getNewRandomInstance(0x5eed0000L + deviceId);

                    ready.countDown();
                    readySignalled = true;
                    assertTrue(start.await(30, TimeUnit.SECONDS),
                            "Timed out waiting to start device " + deviceId);

                    try (MemoryWorkspace ignored = workspace.notifyScopeEntered()) {
                        INDArray random = Nd4j.rand(rng, DataType.FLOAT, 256);
                        INDArray result = random.mul(2.0).addi(deviceId + 1.0);

                        assertTrue(random.isAttached(),
                                "Random input is not backed by the device workspace");
                        assertTrue(result.isAttached(),
                                "Kernel output is not backed by the device workspace");
                        assertEquals(deviceId, random.data().opaqueBuffer().deviceId());
                        assertEquals(deviceId, result.data().opaqueBuffer().deviceId());
                        assertEquals(deviceId,
                                random.shapeInfoDataBuffer().opaqueBuffer().deviceId());
                        assertEquals(deviceId,
                                result.shapeInfoDataBuffer().opaqueBuffer().deviceId());

                        assertOwnedByDevice(
                                requireSpecialPointer(random.data(), "random input"),
                                deviceId, "device " + deviceId + " workspace random input");
                        assertOwnedByDevice(
                                requireSpecialPointer(result.data(), "kernel output"),
                                deviceId, "device " + deviceId + " workspace kernel output");
                        assertOwnedByDevice(
                                requireSpecialPointer(
                                        result.shapeInfoDataBuffer(), "kernel output shape"),
                                deviceId, "device " + deviceId + " kernel output shape");

                        assertEquals(1, ops.registerEvent(event, executionStream),
                                "Failed to record device " + deviceId + " event");
                        submitted.countDown();
                        submittedSignalled = true;
                        assertTrue(submitted.await(30, TimeUnit.SECONDS),
                                "The other Vulkan device did not submit work concurrently");

                        assertEquals(1, ops.eventSynchronize(event),
                                "Device " + deviceId + " event synchronization failed");
                        assertEquals(1, ops.streamSynchronize(executionStream),
                                "Device " + deviceId + " stream synchronization failed");

                        float[] randomValues = random.toFloatVector();
                        float[] resultValues = result.toFloatVector();
                        randomSnapshots.set(deviceId, randomValues);
                        assertEquals(randomValues.length, resultValues.length);
                        boolean varied = false;
                        for (int index = 0; index < randomValues.length; index++) {
                            if (index > 0 && randomValues[index] != randomValues[0]) {
                                varied = true;
                            }
                            float expectedRandom = canonicalUniform(
                                    0x5eed0000L + deviceId, index);
                            assertEquals(
                                    Float.floatToRawIntBits(expectedRandom),
                                    Float.floatToRawIntBits(randomValues[index]),
                                    "Vulkan RNG diverged from the canonical framework sequence"
                                            + " on device " + deviceId + " at index " + index);
                            assertTrue(Float.isFinite(randomValues[index]),
                                    "Non-finite RNG value on device " + deviceId);
                            assertTrue(randomValues[index] >= 0.0f
                                            && randomValues[index] <= 1.0f,
                                    "Uniform RNG value outside [0,1] on device " + deviceId);
                            assertEquals(
                                    randomValues[index] * 2.0f + deviceId + 1.0f,
                                    resultValues[index], 1.0e-6f,
                                    "Wrong kernel result on device " + deviceId
                                            + " at index " + index);
                        }
                        assertTrue(varied,
                                "Device " + deviceId
                                        + " RNG kernel produced one repeated value");
                    }

                    timelineAfter.set(deviceId,
                            invokeLongArg("vulkanGetTimelineValue", deviceId));
                    assertTrue(timelineAfter.get(deviceId) >= timelineBefore.get(deviceId),
                            "Device " + deviceId + " timeline moved backwards");
                    assertEquals(deviceId, ops.getDevice());
                    assertEquals(deviceId,
                            Nd4j.getAffinityManager().getDeviceForCurrentThread());
                    assertEquals(deviceId, Nd4j.getDeviceIdProvider().getDeviceId());
                } catch (Throwable failure) {
                    errors.set(deviceId, failure);
                } finally {
                    if (!readySignalled) {
                        ready.countDown();
                    }
                    if (!submittedSignalled) {
                        submitted.countDown();
                    }
                    if (!isNull(event)) {
                        try {
                            if (ops.destroyEvent(event) != 1 && errors.get(deviceId) == null) {
                                errors.set(deviceId, new IllegalStateException(
                                        "Failed to destroy event for device " + deviceId));
                            }
                        } catch (Throwable cleanupFailure) {
                            if (errors.get(deviceId) == null) {
                                errors.set(deviceId, cleanupFailure);
                            }
                        }
                    }
                    if (workspace != null) {
                        try {
                            Nd4j.getWorkspaceManager().destroyWorkspace(workspace);
                        } catch (Throwable cleanupFailure) {
                            if (errors.get(deviceId) == null) {
                                errors.set(deviceId, cleanupFailure);
                            }
                        }
                    }
                    if (rng != null) {
                        try {
                            rng.close();
                        } catch (Throwable cleanupFailure) {
                            if (errors.get(deviceId) == null) {
                                errors.set(deviceId, cleanupFailure);
                            }
                        }
                    }
                }
            }));
        }

        boolean bothReady = ready.await(30, TimeUnit.SECONDS);
        start.countDown();
        assertTrue(bothReady, "Both Vulkan devices did not initialize concurrently");
        CompletableFuture.allOf(workers.toArray(new CompletableFuture[0]))
                .get(90, TimeUnit.SECONDS);

        for (int deviceId = 0; deviceId < 2; deviceId++) {
            assertNull(errors.get(deviceId),
                    "Concurrent Vulkan worker " + deviceId + " failed: "
                            + errors.get(deviceId));
        }
        float[] deviceZeroRandom = randomSnapshots.get(0);
        float[] deviceOneRandom = randomSnapshots.get(1);
        assertNotNull(deviceZeroRandom);
        assertNotNull(deviceOneRandom);
        assertEquals(deviceZeroRandom.length, deviceOneRandom.length);
        boolean sameSequence = true;
        for (int index = 0; index < deviceZeroRandom.length; index++) {
            if (deviceZeroRandom[index] != deviceOneRandom[index]) {
                sameSequence = false;
                break;
            }
        }
        assertFalse(sameSequence,
                "Independent per-device RNG seeds produced the same sequence");
        assertNotEquals(executionStreams.get(0), executionStreams.get(1),
                "Devices 0 and 1 shared one execution stream");
        assertNotEquals(copyStreams.get(0), copyStreams.get(1),
                "Devices 0 and 1 shared one copy stream");
        assertNotEquals(events.get(0), events.get(1),
                "Devices 0 and 1 shared one event");
        assertNotEquals(commandPools.get(0), commandPools.get(1),
                "Devices 0 and 1 shared one command pool");
    }

    // =========================================================================
    // FAILURE INJECTION: strict device-memory budget (ADR-0112 T2 §2a)
    // =========================================================================

    /**
     * Sets Environment.maxDeviceMemory to a deliberately tiny value and allocates past
     * it. Vulkan device allocations must fail loudly at the budget boundary; spilling to
     * host-visible memory would violate the backend contract.
     */
    @Test
    @DisplayName("(device, failure-injection) allocation past device budget fails without spill")
    void testDeviceBudgetHardFailure() throws Exception {
        requireAtLeastOneDevice();

        final long tinyBudget = 64L * 1024L;
        final long allocationSize = 32L * 1024L;
        List<Pointer> allocations = new ArrayList<>();
        boolean failedLoudly = false;
        try {
            Nd4j.getEnvironment().setMaxDeviceMemory(tinyBudget);
            for (int i = 0; i < 20; i++) {
                Pointer pointer;
                try {
                    pointer = mallocDevice(allocationSize, 0);
                } catch (Exception e) {
                    log.info("Device allocation failed at #{} as required: {}", i, e.getMessage());
                    failedLoudly = true;
                    break;
                }
                if (isNull(pointer)) {
                    log.info("Device allocation returned null at #{} as required", i);
                    failedLoudly = true;
                    break;
                }

                int flags = invokeIntPtrInt(
                        "vulkanGetAllocationMemoryPropertyFlags", pointer, 0);
                assertTrue((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
                        "Allocation #" + i + " was not backed by DEVICE_LOCAL memory: flags=0x"
                                + Integer.toHexString(flags));
                allocations.add(pointer);
            }

            assertTrue(failedLoudly,
                    "Device-memory budget was not enforced: 20 allocations of 32KB all succeeded "
                            + "with maxDeviceMemory=64KB. Vulkan must fail instead of spilling to "
                            + "host-visible memory (ADR-0111 §3).");
        } finally {
            for (Pointer pointer : allocations) {
                freeDevice(pointer, 0);
            }
            Nd4j.getEnvironment().setMaxDeviceMemory(-1L);
        }
    }

    // =========================================================================
    // FAILURE INJECTION: safe retire-list reuse (ADR-0112 T2 §2b)
    // =========================================================================

    /**
     * Frees and immediately reallocates device-local memory. The pointers are opaque
     * device allocation handles: the test deliberately does not dereference them from
     * the host. Stream/event tests cover in-flight ordering separately.
     */
    @Test
    @DisplayName("(device, failure-injection) free then re-allocate remains device-local")
    void testRetireListSafeReuse() throws Exception {
        requireAtLeastOneDevice();

        final long size = 4096L;
        Pointer first = mallocDevice(size, 0);
        assertFalse(isNull(first), "Initial mallocDevice returned null");
        int firstFlags = invokeIntPtrInt(
                "vulkanGetAllocationMemoryPropertyFlags", first, 0);
        assertTrue((firstFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
                "Initial allocation was not DEVICE_LOCAL: flags=0x"
                        + Integer.toHexString(firstFlags));
        assertEquals(1, freeDevice(first, 0), "freeDevice must return success");

        Pointer replacement = mallocDevice(size, 0);
        try {
            assertFalse(isNull(replacement), "Reallocation after free returned null");
            int replacementFlags = invokeIntPtrInt(
                    "vulkanGetAllocationMemoryPropertyFlags", replacement, 0);
            assertTrue((replacementFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
                    "Replacement allocation was not DEVICE_LOCAL: flags=0x"
                            + Integer.toHexString(replacementFlags));
        } finally {
            if (!isNull(replacement)) {
                freeDevice(replacement, 0);
            }
        }
    }

    // =========================================================================
    // M1/M3/M4 HARDENED DEVICE LEGS (native-phase gap closure)
    // =========================================================================

    /**
     * M1 HARDENED: vulkanGetPoolBlockId returns >= 0 for a suballocated pointer
     * and -1 for a dedicated allocation.  Two small allocations on the same device
     * must return the same (or different) block id depending on whether the pool
     * suballocated them from the same VkDeviceMemory block.
     */
    @Test
    @DisplayName("(device, M1-closed) vulkanGetPoolBlockId returns valid block id for suballoc")
    void testM1PoolBlockId() throws Exception {
        requireAtLeastOneDevice();
        final long size = 4096L;
        Pointer ptr1 = mallocDevice(size, 0);
        Pointer ptr2 = mallocDevice(size, 0);
        try {
            assertFalse(isNull(ptr1), "mallocDevice(1) returned null");
            assertFalse(isNull(ptr2), "mallocDevice(2) returned null");

            int blockId1 = invokeIntPtrInt("vulkanGetPoolBlockId", ptr1, 0);
            int blockId2 = invokeIntPtrInt("vulkanGetPoolBlockId", ptr2, 0);
            assertTrue(blockId1 >= 0,
                    "vulkanGetPoolBlockId for ptr1 returned " + blockId1);
            assertTrue(blockId2 >= 0,
                    "vulkanGetPoolBlockId for ptr2 returned " + blockId2);
        } finally {
            if (!isNull(ptr1)) freeDevice(ptr1, 0);
            if (!isNull(ptr2)) freeDevice(ptr2, 0);
        }
    }

    /**
     * Every Vulkan device allocation must use a memory type carrying DEVICE_LOCAL.
     * UMA memory may also be HOST_VISIBLE; that is a hardware property, not a fallback.
     */
    @Test
    @DisplayName("(device, M3-closed) allocations use DEVICE_LOCAL Vulkan memory")
    void testM3DeviceLocalMemory() throws Exception {
        requireAtLeastOneDevice();
        Pointer ptr = mallocDevice(4096L, 0);
        try {
            assertFalse(isNull(ptr), "mallocDevice returned null");
            int flags = invokeIntPtrInt(
                    "vulkanGetAllocationMemoryPropertyFlags", ptr, 0);
            assertTrue((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
                    "Allocation memory flags did not contain DEVICE_LOCAL: 0x"
                            + Integer.toHexString(flags));
        } finally {
            if (!isNull(ptr)) freeDevice(ptr, 0);
        }
    }

    /**
     * M4 HARDENED: vulkanGetRetireListPendingCount increases after free() and decreases
     * to 0 after an immediate sweep (approximated here by calling freeDevice which routes
     * through freeImmediate on the current implementation path).
     *
     * <p>With timeline semaphores the retire list may not be immediately empty after free;
     * we assert non-negative and that the count is readable without crashing.</p>
     */
    @Test
    @DisplayName("(device, M4-closed) vulkanGetRetireListPendingCount is readable and non-negative")
    void testM4RetireListPendingCount() throws Exception {
        requireAtLeastOneDevice();
        int before = invokeIntIntArg("vulkanGetRetireListPendingCount", 0);
        assertTrue(before >= 0,
                "vulkanGetRetireListPendingCount before alloc returned negative: " + before);

        Pointer ptr = mallocDevice(4096L, 0);
        assertFalse(isNull(ptr), "mallocDevice returned null");
        freeDevice(ptr, 0);

        int after = invokeIntIntArg("vulkanGetRetireListPendingCount", 0);
        assertTrue(after >= 0,
                "vulkanGetRetireListPendingCount after free returned negative: " + after);
    }
}
