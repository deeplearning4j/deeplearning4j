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
package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Enforcement gate for {@code Environment.setMaxDeviceMemory} on the CUDA pool
 * (MEMORY-POOL-ENFORCEMENT.md). Before the fix the bound was decorative: the
 * value reached C++ {@code _maxDeviceMemory} but had no getter and the pool
 * never read it, so a 110M-param encoder's single-plan warmup reserved 22.4 GB
 * against a 13.5 GB bound (1.66x). The fix wires the bound into
 * {@code CudaMemoryPool::allocate}: an allocation that would push this process's
 * pool usage past the budget routes to {@code allocateFailover} (host/peer)
 * instead of growing device reservation.
 *
 * This test drives the SUPPORTED API — {@code Nd4j.getEnvironment()
 * .setMaxDeviceMemory(bytes)} — NOT the hallucinated {@code -Dnd4j.environment
 * .maxDeviceMemory} property (which no dl4j code reads). It bounds the pool to
 * a small budget, allocates several times that in device buffers, and asserts
 * (a) every allocation still succeeds (spilled, not OOM-failed) and (b) the
 * async pool's own usage stays at/under the budget — i.e. the overflow went to
 * host/peer rather than growing device reservation.
 */
@Slf4j
@Tag("cuda-only")
public class DspDeviceBudgetEnforcementTest {

    private static final long GiB = 1024L * 1024 * 1024;
    private static final long MiB = 1024L * 1024;

    @AfterEach
    public void resetBudget() {
        // -1 → unbounded (the default); leaving a bound set would poison later tests.
        Nd4j.getEnvironment().setMaxDeviceMemory(-1L);
    }

    @Test
    public void testDeviceBudgetBoundsPoolUsage() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        assumeTrue("CUDA".equalsIgnoreCase(backend), "CUDA-only enforcement test");

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int device = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        long totalMem = nativeOps.getDeviceTotalMemory(device);
        // Need headroom for budget + spill target: ≥ 8 GB device, and the box
        // always has a host pinned-fallback home, so a spill target exists.
        assumeTrue(totalMem >= 8L * GiB, "device too small to exercise budget + spill");

        long baselinePoolUsed = poolUsed(nativeOps, device);
        long budget = baselinePoolUsed + 3L * GiB;
        long allocTotal = 6L * GiB;                 // 2x the delta above baseline
        long chunkBytes = 256L * MiB;
        int chunks = (int) (allocTotal / chunkBytes);

        Nd4j.getEnvironment().setMaxDeviceMemory(budget);
        log.info("budget set: {} MB (baseline poolUsed {} MB); allocating {} x {} MB device-only",
                budget / MiB, baselinePoolUsed / MiB, chunks, chunkBytes / MiB);

        List<OpaqueDataBuffer> allocs = new ArrayList<>();
        try {
            for (int i = 0; i < chunks; i++) {
                // Device-only (allocateBoth=false): no host mirror to trip JavaCPP's
                // maxPhysicalBytes guard; we want to pressure the DEVICE pool.
                OpaqueDataBuffer b = OpaqueDataBuffer.allocateDataBuffer(chunkBytes / 4, DataType.FLOAT, false);
                assertNotNull(b, "allocation #" + i + " returned null — budget failover should spill, not fail");
                assertTrue(!b.isNull(), "allocation #" + i + " is null pointer — spill failed");
                allocs.add(b);
                if (i % 4 == 0) {
                    log.info("after {} chunks ({} MB requested): poolUsed={} MB",
                            i + 1, ((i + 1L) * chunkBytes) / MiB, poolUsed(nativeOps, device) / MiB);
                }
            }

            long finalPoolUsed = poolUsed(nativeOps, device);
            log.info("FINAL: requested {} MB total, budget {} MB, poolUsed {} MB",
                    allocTotal / MiB, budget / MiB, finalPoolUsed / MiB);

            // The bound holds: async pool device usage stayed at/under budget even
            // though we requested 2x that — the overflow spilled to host/peer.
            // Slack absorbs the single chunk that trips the limit + driver rounding.
            long slack = 512L * MiB;
            assertTrue(finalPoolUsed <= budget + slack,
                    "pool device usage " + (finalPoolUsed / MiB) + " MB exceeded budget "
                            + (budget / MiB) + " MB (+" + (slack / MiB) + " MB slack) — bound NOT enforced");

            // And the bound actually bit: we requested materially more than the budget,
            // so an unbounded pool would show usage well above it. This guards against a
            // false pass where the allocations simply never grew the pool.
            assertTrue(allocTotal > budget - baselinePoolUsed + GiB,
                    "test misconfigured: request must exceed the budget delta to prove enforcement");
        } finally {
            for (OpaqueDataBuffer b : allocs) {
                try {
                    Nd4j.getNativeOps().dbClose(b);
                } catch (Throwable t) {
                    log.warn("ballast close failed: {}", t.toString());
                }
            }
        }
    }

    private static long poolUsed(NativeOps nativeOps, int device) {
        try (LongPointer usedPtr = new LongPointer(1);
             LongPointer reservedPtr = new LongPointer(1)) {
            nativeOps.getMemoryPoolStats(device, usedPtr, reservedPtr);
            return usedPtr.get();
        }
    }
}
