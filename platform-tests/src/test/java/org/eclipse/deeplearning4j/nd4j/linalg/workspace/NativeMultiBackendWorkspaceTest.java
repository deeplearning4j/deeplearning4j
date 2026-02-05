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

package org.eclipse.deeplearning4j.nd4j.linalg.workspace;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.memory.abstracts.NativeMultiBackendWorkspace;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@Tag(TagNames.WORKSPACES)
@NativeTag
public class NativeMultiBackendWorkspaceTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCreateAndDestroy(Nd4jBackend backend) {
        // Test basic creation and destruction via NativeOps
        try (NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                4096, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0)) {
            assertNotNull(ws.getNativeHandle());
        }
        // After close, workspace should be cleaned up (no crash = success)
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllocationAndScopeCycling(Nd4jBackend backend) {
        try (NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                8192, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0)) {

            ws.scopeIn();

            // Allocate some bytes
            Pointer ptr = ws.allocateBytes(1024);
            assertNotNull(ptr);
            assertFalse(ptr.isNull());

            ws.scopeOut();

            // After scope cycle, can allocate again
            ws.scopeIn();
            Pointer ptr2 = ws.allocateBytes(1024);
            assertNotNull(ptr2);
            assertFalse(ptr2.isNull());
            ws.scopeOut();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCoherenceStateTransitions(Nd4jBackend backend) {
        try (NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                4096, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0)) {

            // CPU device should have a valid coherence state after creation
            int cpuCoherence = ws.getCoherenceState(
                    NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0);

            // After creation with CPU as primary, CPU should be EXCLUSIVE or MODIFIED
            assertTrue(cpuCoherence == NativeMultiBackendWorkspace.COHERENCE_EXCLUSIVE ||
                       cpuCoherence == NativeMultiBackendWorkspace.COHERENCE_MODIFIED,
                    "Expected EXCLUSIVE or MODIFIED coherence for primary device, got: " + cpuCoherence);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTotalAllocatedSize(Nd4jBackend backend) {
        try (NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                4096, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0)) {

            long totalSize = ws.getTotalAllocatedSize();
            // Should have allocated at least the initial size
            assertTrue(totalSize >= 4096,
                    "Expected total size >= 4096, got: " + totalSize);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testClosedWorkspaceThrows(Nd4jBackend backend) {
        NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                4096, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0);
        ws.close();

        // Operations after close should throw
        assertThrows(IllegalStateException.class, () -> ws.allocateBytes(1024));
        assertThrows(IllegalStateException.class, () -> ws.scopeIn());
        assertThrows(IllegalStateException.class, () -> ws.scopeOut());
        assertThrows(IllegalStateException.class, () ->
                ws.getCoherenceState(NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0));
        assertThrows(IllegalStateException.class, () -> ws.getTotalAllocatedSize());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultipleScopeCycles(Nd4jBackend backend) {
        try (NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                16384, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0)) {

            // Run many scope cycles to verify no memory leaks or crashes
            for (int i = 0; i < 100; i++) {
                ws.scopeIn();

                Pointer ptr = ws.allocateBytes(512);
                assertNotNull(ptr);
                assertFalse(ptr.isNull());

                ws.scopeOut();
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCudaDeviceAllocation(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        try (NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                1024 * 1024, NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA, 0)) {

            ws.scopeIn();

            Pointer ptr = ws.allocateBytes(4096);
            assertNotNull(ptr, "CUDA allocation should return non-null pointer");
            assertFalse(ptr.isNull(), "CUDA allocation pointer should not be null");

            // Verify coherence state reflects CUDA ownership
            int cudaCoherence = ws.getCoherenceState(
                    NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA, 0);
            assertTrue(cudaCoherence == NativeMultiBackendWorkspace.COHERENCE_EXCLUSIVE ||
                            cudaCoherence == NativeMultiBackendWorkspace.COHERENCE_MODIFIED,
                    "CUDA coherence should be EXCLUSIVE or MODIFIED, got: " + cudaCoherence);

            // Verify device-specific allocation tracking (may not be implemented yet)
            try {
                long cudaAllocated = ws.getAllocatedSizeOnDevice(
                        NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA, 0);
                assertTrue(cudaAllocated > 0, "CUDA allocated size should be > 0");
            } catch (UnsupportedOperationException e) {
                log.info("getAllocatedSizeOnDevice not yet implemented: {}", e.getMessage());
            }

            ws.scopeOut();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTransferCpuToCuda(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        // Allocate on CPU, transfer to CUDA, verify coherence transitions
        try (NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                1024 * 1024, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0)) {

            ws.scopeIn();
            ws.allocateBytes(4096);

            // Before transfer: CPU should own exclusively
            int cpuBefore = ws.getCoherenceState(
                    NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0);
            assertTrue(cpuBefore == NativeMultiBackendWorkspace.COHERENCE_EXCLUSIVE ||
                            cpuBefore == NativeMultiBackendWorkspace.COHERENCE_MODIFIED,
                    "CPU should be EXCLUSIVE or MODIFIED before transfer, got: " + cpuBefore);

            // Transfer CPU -> CUDA
            ws.transferTo(
                    NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0,
                    NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA, 0);

            // After transfer: both should be SHARED
            int cpuAfter = ws.getCoherenceState(
                    NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0);
            int cudaAfter = ws.getCoherenceState(
                    NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA, 0);

            assertEquals(NativeMultiBackendWorkspace.COHERENCE_SHARED, cpuAfter,
                    "CPU should be SHARED after transfer to CUDA");
            assertEquals(NativeMultiBackendWorkspace.COHERENCE_SHARED, cudaAfter,
                    "CUDA should be SHARED after transfer from CPU");

            ws.scopeOut();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCudaSyncDevice(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        try (NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                1024 * 1024, NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA, 0)) {

            ws.scopeIn();
            ws.allocateBytes(4096);

            // syncDevice/syncAllDevices may not be implemented yet
            try {
                ws.syncDevice(NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA, 0);
                ws.syncAllDevices();
            } catch (UnsupportedOperationException e) {
                log.info("syncDevice not yet implemented: {}", e.getMessage());
            }

            ws.scopeOut();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllocateBytesOnSpecificDevice(Nd4jBackend backend) {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        // Test explicit device-targeted allocation (may not be implemented yet)
        try (NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                1024 * 1024, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0)) {

            ws.scopeIn();

            try {
                Pointer cudaPtr = ws.allocateBytesOnDevice(4096,
                        NativeMultiBackendWorkspace.DEVICE_TYPE_CUDA, 0);
                assertNotNull(cudaPtr, "Device-specific CUDA allocation should succeed");

                Pointer cpuPtr = ws.allocateBytesOnDevice(4096,
                        NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0);
                assertNotNull(cpuPtr, "Device-specific CPU allocation should succeed");
            } catch (UnsupportedOperationException e) {
                log.info("allocateBytesOnDevice not yet implemented: {}", e.getMessage());
            }

            ws.scopeOut();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCurrentOffset(Nd4jBackend backend) {
        try (NativeMultiBackendWorkspace ws = new NativeMultiBackendWorkspace(
                16384, NativeMultiBackendWorkspace.DEVICE_TYPE_CPU, 0)) {

            ws.scopeIn();

            try {
                long offsetBefore = ws.getCurrentOffset();

                ws.allocateBytes(1024);

                long offsetAfter = ws.getCurrentOffset();
                assertTrue(offsetAfter > offsetBefore,
                        "Offset should increase after allocation: before=" + offsetBefore + " after=" + offsetAfter);

                ws.scopeOut();

                // After scope out, offset should reset
                ws.scopeIn();
                long offsetAfterReset = ws.getCurrentOffset();
                assertTrue(offsetAfterReset <= offsetBefore + 64,
                        "Offset should reset after scope cycle: " + offsetAfterReset);
                ws.scopeOut();
            } catch (UnsupportedOperationException e) {
                log.info("getCurrentOffset not yet implemented: {}", e.getMessage());
                ws.scopeOut();
            }
        }
    }
}
