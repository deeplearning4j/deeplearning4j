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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Frees the native memory of objects dropped without close() once memory runs low, instead of
 * waiting for Java heap pressure to collect them.
 *
 * <p>A graph's plans hold device memory the Java heap never sees, so the heap-pressure collection
 * of ADR 0070 does not run while dropped graphs fill the device. Before a graph takes a native
 * plan cache, {@link #reclaimIfLow} collects and frees what was dropped when less than a quarter
 * of the device is free (of the host, for a backend without device memory), or when the process
 * is past three quarters of JavaCPP's physical memory limit.</p>
 *
 * <p>One thread reclaims at a time. A reclaim that leaves memory low doubles the wait before the
 * next one, up to a minute, so memory filled by live data costs at most one collection a
 * minute.</p>
 */
@Slf4j
public final class LowMemoryReclaim {

    private static final long MIN_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(1);
    private static final long MAX_INTERVAL_NANOS = TimeUnit.MINUTES.toNanos(1);

    private static final AtomicBoolean RECLAIMING = new AtomicBoolean();
    private static final AtomicLong RECLAIMS = new AtomicLong();
    private static volatile long nextReclaimNanos = System.nanoTime();
    /** Written only by the thread that holds {@link #RECLAIMING}. */
    private static long intervalNanos = MIN_INTERVAL_NANOS;

    private LowMemoryReclaim() {
    }

    /** The number of reclaims run so far. */
    public static long reclaims() {
        return RECLAIMS.get();
    }

    /** Lets the next {@link #reclaimIfLow} run without waiting out the last one's interval (tests). */
    static void resetSchedule() {
        nextReclaimNanos = System.nanoTime();
    }

    /**
     * Collects and frees native owners that were dropped without close() when memory on
     * {@code device} is low; returns at once while another thread reclaims or the wait after the
     * last reclaim has not passed.
     */
    public static void reclaimIfLow(NativeOps nativeOps, int device) {
        if (System.nanoTime() - nextReclaimNanos < 0 || !isLow(nativeOps, device)
                || !RECLAIMING.compareAndSet(false, true)) {
            return;
        }
        try {
            RECLAIMS.incrementAndGet();
            Nd4j.getMemoryManager().invokeGc();
            int freed = Nd4j.getDeallocatorService().flushCollectedReferences();
            boolean stillLow = isLow(nativeOps, device);
            intervalNanos = stillLow ? Math.min(MAX_INTERVAL_NANOS, intervalNanos * 2) : MIN_INTERVAL_NANOS;
            nextReclaimNanos = System.nanoTime() + intervalNanos;
            log.info("Low memory on device {}: freed {} collected native registrations; {}", device, freed,
                    stillLow ? "still low, next reclaim in " + TimeUnit.NANOSECONDS.toSeconds(intervalNanos) + " s"
                            : "no longer low");
        } finally {
            RECLAIMING.set(false);
        }
    }

    /**
     * Less than a quarter of the device free (pool-aware), of the host for a backend without device
     * memory, or the process past three quarters of JavaCPP's physical memory limit.
     */
    static boolean isLow(NativeOps nativeOps, int device) {
        long maxPhysical = Pointer.maxPhysicalBytes();
        if (maxPhysical > 0 && Pointer.physicalBytes() > maxPhysical / 4 * 3) {
            return true;
        }
        long deviceTotal = nativeOps.getDeviceTotalMemory(device);
        if (deviceTotal > 0) {
            return DeviceMemoryManager.getInstance().getPoolAwareFreeMemory(device) < deviceTotal / 4;
        }
        long hostTotal = Pointer.totalPhysicalBytes();
        return hostTotal > 0 && Pointer.availablePhysicalBytes() < hostTotal / 4;
    }
}
