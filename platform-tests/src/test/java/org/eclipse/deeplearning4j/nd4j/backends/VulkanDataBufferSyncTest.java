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
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * VulkanDataBufferSyncTest — ADR-0111 §4, ADR-0112 T2 ("actuality state machine parity").
 *
 * <p>ADR-0111 §4 specifies: "allocateSpecial() from VulkanMemoryPool; syncToSpecial()/
 * syncToPrimary() as recorded copies on the transfer queue with fence/timeline completion;
 * actuality ticks (read/writePrimary, read/writeSpecial) preserved exactly — the DSP
 * layer's staleness logic already depends on them."</p>
 *
 * <p>The spec says device-leg tests mirror CUDA sync tests (DspDeviceBudgetEnforcementTest,
 * CrossDeviceTransferTest patterns).  The Java-observable surface for the tick/state machine
 * is {@link OpaqueDataBuffer#syncToSpecial()} and {@link OpaqueDataBuffer#syncToPrimary()},
 * plus {@link OpaqueDataBuffer#primaryBuffer()} / {@link OpaqueDataBuffer#specialBuffer()}.
 * We test these through owner-scoped buffer operations after verifying the backend is Vulkan.</p>
 *
 * <p>The native actuality probes expose primary/special state, while
 * {@code dbAllocateSpecialBuffer(OpaqueDataBuffer)} forces eager device allocation.
 * Hardened device legs below use those APIs without consulting another backend.</p>
 *
 * <p>Run with:
 * <pre>
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Ptest-vulkan \
 *     -Dtest=VulkanDataBufferSyncTest 2>&1 | tee /tmp/vulkan-databuffer-sync-test.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("VulkanDataBuffer sync/actuality state machine (ADR-0111 §4, ADR-0112 T2)")
public class VulkanDataBufferSyncTest {

    private static final String VULKAN_BINDINGS_CLASS = "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";

    /** True when the Nd4jVulkan class is on the classpath (i.e. -Ptest-vulkan is active). */
    private static boolean vulkanClassAvailable = false;

    /** True when at least one Vulkan device is enumerated. */
    private static boolean vulkanDevicePresent = false;

    @BeforeAll
    static void probeVulkan() {
        try {
            Class<?> cls = Class.forName(VULKAN_BINDINGS_CLASS);
            Object nativeOps = cls.getDeclaredConstructor().newInstance();
            int count = (int) cls.getMethod("getAvailableDevices").invoke(nativeOps);
            vulkanClassAvailable = true;
            vulkanDevicePresent = (count > 0);
            log.info("VulkanDataBufferSyncTest: vulkanClassAvailable=true, deviceCount={}", count);
        } catch (Exception e) {
            log.warn("VulkanDataBufferSyncTest: Vulkan bindings unavailable ({}: {})",
                    e.getClass().getSimpleName(), e.getMessage());
            vulkanClassAvailable = false;
            vulkanDevicePresent = false;
        }
    }

    // ── skip helpers ─────────────────────────────────────────────────────────

    private static void requireVulkanClass() {
        assumeTrue(vulkanClassAvailable,
                "Vulkan NativeOps class not available — run with -Ptest-vulkan.");
    }

    private static void requireVulkanDevice() {
        requireVulkanClass();
        assumeTrue(vulkanDevicePresent,
                "getAvailableDevices() returned 0 — no Vulkan device enumerated. "
                + "Ensure a Vulkan ICD is installed.");
    }

    private static boolean invokeBooleanBuffer(String method, OpaqueDataBuffer buffer)
            throws Exception {
        Object ownerOps = buffer.backendOwner().nativeOps();
        return (boolean) ownerOps.getClass()
                .getMethod(method, OpaqueDataBuffer.class)
                .invoke(ownerOps, buffer);
    }

    // =========================================================================
    // DEVICE LEGS: require a real Vulkan device
    // =========================================================================

    @Test
    @DisplayName("(device) Vulkan OpaqueDataBuffer allocation returns a live handle")
    void testOpaqueBufferAllocatesOnDeviceBackend() {
        requireVulkanDevice();
        OpaqueDataBuffer buf = OpaqueDataBuffer.allocateDataBuffer(16, DataType.FLOAT, false);
        try {
            assertNotNull(buf, "allocateDataBuffer returned null");
            assertFalse(buf.isNull(), "allocateDataBuffer returned null-pointer handle");
        } finally {
            buf.close();
        }
    }

    /**
     * Matches CUDA's allocateBoth contract: true allocates device storage and host staging.
     */
    @Test
    @DisplayName("(device) allocateBoth exposes a non-null primary staging buffer")
    void testAllocateBothProvidesPrimaryBuffer() {
        requireVulkanDevice();
        OpaqueDataBuffer buf = OpaqueDataBuffer.allocateDataBuffer(8, DataType.FLOAT, true);
        try {
            Pointer primary = buf.primaryBuffer();
            assertNotNull(primary, "primaryBuffer() must not be Java null when allocateBoth=true");
            assertFalse(primary.isNull(), "primaryBuffer() must not be a null Pointer when allocateBoth=true");
        } finally {
            buf.close();
        }
    }

    /**
     * Mirrors CUDA's dual-buffer state machine: write primary, tick host write, copy H2D,
     * mark the device authoritative, and copy D2H. The same contract applies to discrete
     * and UMA devices; Vulkan memory properties do not change DataBuffer semantics.
     */
    @Test
    @DisplayName("(device) host-write → H2D → D2H preserves values and actuality")
    void testSyncRoundtripPreservesValues() {
        requireVulkanDevice();

        final int length = 64;
        final float seed = 3.14159f;
        OpaqueDataBuffer buf =
                OpaqueDataBuffer.allocateDataBuffer(length, DataType.FLOAT, true);
        try {
            Pointer primary = buf.primaryBuffer();
            assertNotNull(primary, "primaryBuffer() must not be null");
            assertFalse(primary.isNull(), "primaryBuffer() must not be a null Pointer");
            FloatPointer host = new FloatPointer(primary);

            for (int i = 0; i < length; i++) {
                host.put(i, seed + i);
            }
            buf.backendOwner().nativeOps().dbTickHostWrite(buf);
            buf.syncToSpecial();

            for (int i = 0; i < length; i++) {
                host.put(i, 0.0f);
            }
            buf.backendOwner().nativeOps().dbTickDeviceWrite(buf);
            buf.syncToPrimary();

            for (int i = 0; i < length; i++) {
                assertEquals(seed + i, host.get(i), 1e-5f,
                        "sync roundtrip mismatch at index " + i);
            }
        } finally {
            buf.close();
        }
    }

    /**
     * allocateDataBuffer with allocateBoth=true must expose the Vulkan device allocation
     * through specialBuffer(). The handle is opaque and is never host-dereferenced.
     */
    @Test
    @DisplayName("(device) specialBuffer() is non-null after allocateBoth=true on a device host")
    void testSpecialBufferNonNullOnDevice() {
        requireVulkanDevice();
        OpaqueDataBuffer buf = OpaqueDataBuffer.allocateDataBuffer(32, DataType.FLOAT, true);
        try {
            assertNotNull(buf, "allocateDataBuffer returned null");
            Pointer special = buf.specialBuffer();
            assertNotNull(special, "specialBuffer() must not be Java null (allocateBoth=true, device present)");
            assertFalse(special.isNull(), "specialBuffer() must not be a null Pointer");
            log.info("specialBuffer address: 0x{} (primary: 0x{})",
                    Long.toHexString(special.address()),
                    Long.toHexString(buf.primaryBuffer().address()));
        } finally {
            buf.close();
        }
    }

    /**
     * deviceId() must return a valid (>= 0) device id for a buffer allocated when a
     * device is present. This verifies that the pool attribution registry (pointer →
     * deviceId) is wired correctly per ADR-0111 §3.
     */
    @Test
    @DisplayName("(device) deviceId() returns >= 0 for a device-allocated buffer")
    void testDeviceIdNonNegativeOnDevice() {
        requireVulkanDevice();
        OpaqueDataBuffer buf = OpaqueDataBuffer.allocateDataBuffer(16, DataType.FLOAT, true);
        try {
            assertNotNull(buf, "allocateDataBuffer returned null");
            int deviceId = buf.deviceId();
            assertTrue(deviceId >= 0,
                    "deviceId() must be >= 0 for a device-allocated buffer, got " + deviceId);
            log.info("OpaqueDataBuffer.deviceId() = {}", deviceId);
        } finally {
            buf.close();
        }
    }

    /**
     * Multiple allocate/sync/free cycles must not crash or produce inconsistent results.
     * This is a lightweight stress test of the retire-list and pool reclaim path.
     * Five cycles of: allocate → H2D → D2H → free.
     */
    @Test
    @DisplayName("(device) multiple allocate/sync/free cycles preserve data")
    void testMultipleSyncFreeCycles() {
        requireVulkanDevice();
        final int length = 128;
        for (int cycle = 0; cycle < 5; cycle++) {
            final float marker = cycle * 100.0f + 1.0f;
            OpaqueDataBuffer buf =
                    OpaqueDataBuffer.allocateDataBuffer(length, DataType.FLOAT, true);
            try {
                FloatPointer host = new FloatPointer(buf.primaryBuffer());
                host.put(0, marker);
                buf.backendOwner().nativeOps().dbTickHostWrite(buf);
                buf.syncToSpecial();

                host.put(0, 0.0f);
                buf.backendOwner().nativeOps().dbTickDeviceWrite(buf);
                buf.syncToPrimary();
                assertEquals(marker, host.get(0), 0.0f,
                        "cycle " + cycle + " did not restore device data");
            } finally {
                buf.close();
            }
        }
    }

    // =========================================================================
    // S1/S2 HARDENED DEVICE LEGS: actuality state machine (Gap S1/S2 closed)
    // =========================================================================

    /**
     * Verifies the same primary/special actuality transitions used by CUDA.
     */
    @Test
    @DisplayName("(device, S1-closed) actuality transitions follow H2D and D2H")
    void testS1ActualityAfterSync() throws Exception {
        requireVulkanDevice();

        OpaqueDataBuffer buf = OpaqueDataBuffer.allocateDataBuffer(16, DataType.FLOAT, true);
        try {
            assertFalse(invokeBooleanBuffer("dbIsPrimaryActual", buf),
                    "CUDA-style allocation makes special authoritative initially");
            assertTrue(invokeBooleanBuffer("dbIsSpecialActual", buf),
                    "special must be actual immediately after allocation");

            buf.backendOwner().nativeOps().dbTickHostWrite(buf);
            buf.syncToSpecial();
            assertTrue(invokeBooleanBuffer("dbIsPrimaryActual", buf),
                    "primary must remain actual after H2D");
            assertTrue(invokeBooleanBuffer("dbIsSpecialActual", buf),
                    "special must become actual after H2D");

            buf.backendOwner().nativeOps().dbTickDeviceWrite(buf);
            assertFalse(invokeBooleanBuffer("dbIsPrimaryActual", buf),
                    "device write must make primary stale");
            assertTrue(invokeBooleanBuffer("dbIsSpecialActual", buf),
                    "device write must keep special actual");

            buf.syncToPrimary();
            assertTrue(invokeBooleanBuffer("dbIsPrimaryActual", buf),
                    "D2H must make primary actual");
        } finally {
            buf.close();
        }
    }

    /**
     * Matches CUDA's special-first allocation contract and verifies that the owner-scoped
     * NativeOps allocation call is idempotent when the device allocation already exists.
     */
    @Test
    @DisplayName("(device, S2-closed) dbAllocateSpecial is idempotent for special-only allocation")
    void testS2SpecialBufferAllocationIsIdempotent() {
        requireVulkanDevice();

        // allocateBoth=false matches CUDA: special is eager and primary is absent.
        OpaqueDataBuffer buf = OpaqueDataBuffer.allocateDataBuffer(16, DataType.FLOAT, false);
        try {
            assertNotNull(buf, "allocateDataBuffer returned null");

            Pointer primary = buf.primaryBuffer();
            assertTrue(primary == null || primary.isNull(),
                    "primaryBuffer() must remain absent when allocateBoth=false");

            Pointer before = buf.specialBuffer();
            assertNotNull(before,
                    "specialBuffer() must be allocated eagerly when allocateBoth=false");
            assertFalse(before.isNull(),
                    "specialBuffer() must not be a null Pointer when allocateBoth=false");

            assertDoesNotThrow(
                    () -> buf.backendOwner().nativeOps().dbAllocateSpecialBuffer(buf),
                    "dbAllocateSpecialBuffer must not throw for an existing device allocation");

            Pointer special = buf.specialBuffer();
            assertNotNull(special,
                    "specialBuffer() must not be Java null after dbAllocateSpecial");
            assertFalse(special.isNull(),
                    "specialBuffer() must not be a null Pointer after dbAllocateSpecial");
            assertEquals(before.address(), special.address(),
                    "dbAllocateSpecialBuffer must preserve the existing Vulkan allocation");
            log.info("S2 PASS: specialBuffer address 0x{} preserved by dbAllocateSpecial",
                    Long.toHexString(special.address()));
        } finally {
            buf.close();
        }
    }

}
