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

package org.nd4j.linalg.framework.device;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JSystemProperties;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Transfer tracking subsystem.
 *
 * Records device transfer events with zero overhead when disabled.
 * Uses a lock-free ring buffer for recent events and concurrent accumulators
 * for per-variable stats.
 *
 * Thread-safe: ring buffer uses atomic index, per-variable stats use
 * ConcurrentHashMap with AtomicLong accumulators.
 *
 * @author ND4J Team
 */
@Slf4j
public final class TransferSubsystem {

    private static final int RING_BUFFER_CAPACITY = 4096;

    private final AtomicBoolean enabled;
    private final TransferEvent[] ringBuffer;
    private final AtomicLong ringIndex;
    private final ConcurrentHashMap<String, VariableAccumulator> perVariableAccum;
    private final AtomicLong totalTransferCount;
    private final AtomicLong totalBytes;
    private final AtomicLong windowStartMs;

    public TransferSubsystem() {
        String prop = System.getProperty(ND4JSystemProperties.DEVICE_TRANSFER_TRACKING);
        this.enabled = new AtomicBoolean(prop != null && Boolean.parseBoolean(prop));
        this.ringBuffer = new TransferEvent[RING_BUFFER_CAPACITY];
        this.ringIndex = new AtomicLong(0);
        this.perVariableAccum = new ConcurrentHashMap<>();
        this.totalTransferCount = new AtomicLong(0);
        this.totalBytes = new AtomicLong(0);
        this.windowStartMs = new AtomicLong(System.currentTimeMillis());
    }

    public boolean isEnabled() {
        return enabled.get();
    }

    public void setEnabled(boolean enabled) {
        this.enabled.set(enabled);
    }

    /**
     * Record a transfer event. Zero-overhead when disabled.
     */
    public void record(TransferEvent event) {
        if (!enabled.get() || event == null) {
            return;
        }

        totalTransferCount.incrementAndGet();
        totalBytes.addAndGet(event.getBytes());

        String varName = event.getVariableName();
        if (varName != null && !varName.isEmpty()) {
            VariableAccumulator accum = perVariableAccum.computeIfAbsent(varName,
                k -> new VariableAccumulator(k));
            accum.add(event);
        }

        long idx = ringIndex.getAndIncrement();
        ringBuffer[(int) (idx % RING_BUFFER_CAPACITY)] = event;
    }

    /**
     * Get statistics for a specific variable.
     */
    public TransferStats getStats(String varName) {
        VariableAccumulator accum = perVariableAccum.get(varName);
        return accum != null ? accum.snapshot() : null;
    }

    /**
     * Get a full snapshot report.
     */
    public TransferReport getReport() {
        long nowMs = System.currentTimeMillis();
        long windowDurationMs = nowMs - windowStartMs.get();

        Map<String, TransferStats> statsSnapshot = new HashMap<>();
        for (Map.Entry<String, VariableAccumulator> entry : perVariableAccum.entrySet()) {
            statsSnapshot.put(entry.getKey(), entry.getValue().snapshot());
        }

        List<TransferEvent> recentEvents = getRecentEvents(RING_BUFFER_CAPACITY);

        return TransferReport.builder()
            .timestampMs(nowMs)
            .perVariableStats(statsSnapshot)
            .totalTransferCount(totalTransferCount.get())
            .totalBytes(totalBytes.get())
            .windowDurationMs(windowDurationMs)
            .recentEvents(recentEvents)
            .build();
    }

    /**
     * Get the last N events from the ring buffer.
     */
    public List<TransferEvent> getRecentEvents(int count) {
        long currentIdx = ringIndex.get();
        long totalRecorded = currentIdx;
        int available = (int) Math.min(count, Math.min(totalRecorded, RING_BUFFER_CAPACITY));
        List<TransferEvent> result = new ArrayList<>(available);

        long startIdx = Math.max(0, currentIdx - available);
        for (long i = startIdx; i < currentIdx; i++) {
            TransferEvent event = ringBuffer[(int) (i % RING_BUFFER_CAPACITY)];
            if (event != null) {
                result.add(event);
            }
        }

        return result;
    }

    public void reset() {
        ringIndex.set(0);
        for (int i = 0; i < RING_BUFFER_CAPACITY; i++) {
            ringBuffer[i] = null;
        }
        perVariableAccum.clear();
        totalTransferCount.set(0);
        totalBytes.set(0);
        windowStartMs.set(System.currentTimeMillis());
    }

    public long getTotalTransferCount() {
        return totalTransferCount.get();
    }

    public long getTotalBytes() {
        return totalBytes.get();
    }

    /**
     * Per-variable accumulator using atomic counters for lock-free concurrent updates.
     * Avoids rebuilding TransferStats objects on every record() call.
     */
    private static final class VariableAccumulator {
        private final String variableName;
        private final AtomicLong transfers = new AtomicLong(0);
        private final AtomicLong bytes = new AtomicLong(0);
        private final AtomicLong durationNanos = new AtomicLong(0);
        private final ConcurrentHashMap<TransferDirection, AtomicLong> countByDirection = new ConcurrentHashMap<>();
        private final ConcurrentHashMap<TransferDirection, AtomicLong> bytesByDirection = new ConcurrentHashMap<>();

        VariableAccumulator(String variableName) {
            this.variableName = variableName;
        }

        void add(TransferEvent event) {
            transfers.incrementAndGet();
            bytes.addAndGet(event.getBytes());
            durationNanos.addAndGet(event.getDurationNanos());

            TransferDirection dir = event.getDirection();
            if (dir != null) {
                countByDirection.computeIfAbsent(dir, k -> new AtomicLong(0)).incrementAndGet();
                bytesByDirection.computeIfAbsent(dir, k -> new AtomicLong(0)).addAndGet(event.getBytes());
            }
        }

        TransferStats snapshot() {
            Map<TransferDirection, Long> countSnap = new HashMap<>();
            for (Map.Entry<TransferDirection, AtomicLong> e : countByDirection.entrySet()) {
                countSnap.put(e.getKey(), e.getValue().get());
            }
            Map<TransferDirection, Long> bytesSnap = new HashMap<>();
            for (Map.Entry<TransferDirection, AtomicLong> e : bytesByDirection.entrySet()) {
                bytesSnap.put(e.getKey(), e.getValue().get());
            }

            return TransferStats.builder()
                .variableName(variableName)
                .totalTransfers(transfers.get())
                .totalBytes(bytes.get())
                .totalDurationNanos(durationNanos.get())
                .countByDirection(countSnap)
                .bytesByDirection(bytesSnap)
                .build();
        }
    }
}
