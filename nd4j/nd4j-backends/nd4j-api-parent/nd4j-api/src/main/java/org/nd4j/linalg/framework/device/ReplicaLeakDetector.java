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

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Tracks replica arrays across devices and detects leaks.
 *
 * Replicas are created when arrays are replicated across devices for
 * multi-GPU execution. If not properly cleaned up, these replicas
 * can leak GPU memory.
 *
 * @author ND4J Team
 */
@Slf4j
public final class ReplicaLeakDetector {

    /**
     * Enabled flag - controlled by system property nd4j.device.replica.leak.detection
     */
    private final boolean enabled;

    /**
     * Tracked replicas - maps array uniqueId to metadata
     */
    private final ConcurrentHashMap<Long, ReplicaInfo> trackedReplicas;

    /**
     * Total replicas created
     */
    private final AtomicLong totalReplicasCreated;

    /**
     * Total replicas cleaned up
     */
    private final AtomicLong totalReplicasCleaned;

    /**
     * Constructor - reads system property to determine if enabled.
     */
    public ReplicaLeakDetector() {
        String prop = System.getProperty(ND4JSystemProperties.DEVICE_REPLICA_LEAK_DETECTION);
        this.enabled = prop != null && Boolean.parseBoolean(prop);
        this.trackedReplicas = new ConcurrentHashMap<>();
        this.totalReplicasCreated = new AtomicLong(0);
        this.totalReplicasCleaned = new AtomicLong(0);
    }

    /**
     * Check if replica leak detection is enabled.
     */
    public boolean isEnabled() {
        return enabled;
    }

    /**
     * Register a replica for tracking.
     *
     * @param replica The replica array
     * @param varName Variable name (e.g., "weights", "input")
     * @param srcDevice Source device ID
     * @param dstDevice Target device ID
     */
    public void registerReplica(INDArray replica, String varName, int srcDevice, int dstDevice) {
        if (!enabled) {
            return;
        }

        if (replica == null) {
            return;
        }

        long arrayId = replica.getId();
        long sizeBytes = replica.length() * replica.dataType().width();

        ReplicaInfo info = ReplicaInfo.builder()
            .arrayId(arrayId)
            .variableName(varName)
            .sourceDevice(srcDevice)
            .targetDevice(dstDevice)
            .sizeBytes(sizeBytes)
            .createdAtMs(System.currentTimeMillis())
            .build();

        trackedReplicas.put(arrayId, info);
        totalReplicasCreated.incrementAndGet();

        if (log.isDebugEnabled()) {
            log.debug("Registered replica '{}' (id={}): device {} -> {}, {} bytes",
                varName, arrayId, srcDevice, dstDevice, sizeBytes);
        }
    }

    /**
     * Unregister a replica (called when replica is cleaned up).
     *
     * @param arrayId Array unique ID
     */
    public void unregisterReplica(long arrayId) {
        if (!enabled) {
            return;
        }

        ReplicaInfo removed = trackedReplicas.remove(arrayId);
        if (removed != null) {
            totalReplicasCleaned.incrementAndGet();

            if (log.isDebugEnabled()) {
                log.debug("Unregistered replica '{}' (id={})",
                    removed.getVariableName(), arrayId);
            }
        }
    }

    /**
     * Detect leaked replicas (older than threshold).
     *
     * @param ageThresholdMs Age threshold in milliseconds
     * @return List of potential leaked replicas
     */
    public List<PotentialReplicaLeak> detectLeakedReplicas(long ageThresholdMs) {
        List<PotentialReplicaLeak> leaks = new ArrayList<>();

        if (!enabled) {
            return leaks;
        }

        long currentTime = System.currentTimeMillis();

        for (Map.Entry<Long, ReplicaInfo> entry : trackedReplicas.entrySet()) {
            ReplicaInfo info = entry.getValue();
            long ageMs = currentTime - info.getCreatedAtMs();

            if (ageMs > ageThresholdMs) {
                PotentialReplicaLeak leak = PotentialReplicaLeak.builder()
                    .arrayId(info.getArrayId())
                    .variableName(info.getVariableName())
                    .sourceDevice(info.getSourceDevice())
                    .targetDevice(info.getTargetDevice())
                    .sizeBytes(info.getSizeBytes())
                    .ageMs(ageMs)
                    .build();

                leaks.add(leak);

                if (log.isDebugEnabled()) {
                    log.debug("Detected leaked replica '{}' (id={}): age={}ms, {} bytes",
                        info.getVariableName(), info.getArrayId(), ageMs, info.getSizeBytes());
                }
            }
        }

        return leaks;
    }

    /**
     * Get a summary report of all tracked replicas.
     *
     * @return ReplicaReport with current state
     */
    public ReplicaReport getReplicaReport() {
        long currentTime = System.currentTimeMillis();
        List<ReplicaInfoSnapshot> snapshots = new ArrayList<>();

        for (ReplicaInfo info : trackedReplicas.values()) {
            long ageMs = currentTime - info.getCreatedAtMs();
            snapshots.add(ReplicaInfoSnapshot.builder()
                .arrayId(info.getArrayId())
                .variableName(info.getVariableName())
                .sourceDevice(info.getSourceDevice())
                .targetDevice(info.getTargetDevice())
                .sizeBytes(info.getSizeBytes())
                .ageMs(ageMs)
                .build());
        }

        long totalTrackedBytes = snapshots.stream()
            .mapToLong(ReplicaInfoSnapshot::getSizeBytes)
            .sum();

        return ReplicaReport.builder()
            .timestampMs(currentTime)
            .totalReplicasTracked(snapshots.size())
            .totalTrackedBytes(totalTrackedBytes)
            .totalReplicasCreated(totalReplicasCreated.get())
            .totalReplicasCleaned(totalReplicasCleaned.get())
            .replicas(snapshots)
            .build();
    }

    /**
     * Get count of currently tracked replicas.
     */
    public int getTrackedReplicaCount() {
        return trackedReplicas.size();
    }

    /**
     * Get total bytes of tracked replicas.
     */
    public long getTrackedReplicaBytes() {
        return trackedReplicas.values().stream()
            .mapToLong(ReplicaInfo::getSizeBytes)
            .sum();
    }

    /**
     * Clear all tracked replicas.
     */
    public void clear() {
        trackedReplicas.clear();
        if (log.isDebugEnabled()) {
            log.debug("Cleared all tracked replicas");
        }
    }

    /**
     * Replica information record.
     */
    @Data
    @Builder
    public static class ReplicaInfo {
        private long arrayId;
        private String variableName;
        private int sourceDevice;
        private int targetDevice;
        private long sizeBytes;
        private long createdAtMs;
    }

    /**
     * Snapshot of replica info for reporting.
     */
    @Data
    @Builder
    public static class ReplicaInfoSnapshot {
        private long arrayId;
        private String variableName;
        private int sourceDevice;
        private int targetDevice;
        private long sizeBytes;
        private long ageMs;
    }

    /**
     * Potential replica leak record.
     */
    @Data
    @Builder
    public static class PotentialReplicaLeak {
        private long arrayId;
        private String variableName;
        private int sourceDevice;
        private int targetDevice;
        private long sizeBytes;
        private long ageMs;

        public String getSizeHumanReadable() {
            long bytes = sizeBytes;
            if (bytes < 1024) {
                return bytes + " B";
            } else if (bytes < 1024 * 1024) {
                return String.format("%.1f KB", bytes / 1024.0);
            } else if (bytes < 1024 * 1024 * 1024) {
                return String.format("%.1f MB", bytes / (1024.0 * 1024.0));
            } else {
                return String.format("%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
            }
        }

        public String getAgeHumanReadable() {
            long seconds = ageMs / 1000;
            if (seconds < 60) {
                return seconds + "s";
            } else if (seconds < 3600) {
                return (seconds / 60) + "m";
            } else {
                return (seconds / 3600) + "h";
            }
        }
    }

    /**
     * Replica tracking report.
     */
    @Data
    @Builder
    public static class ReplicaReport {
        private long timestampMs;
        private int totalReplicasTracked;
        private long totalTrackedBytes;
        private long totalReplicasCreated;
        private long totalReplicasCleaned;
        private List<ReplicaInfoSnapshot> replicas;

        public String summary() {
            return String.format(
                "Replica Report: %d tracked (%d bytes), %d created, %d cleaned",
                totalReplicasTracked, totalTrackedBytes,
                totalReplicasCreated, totalReplicasCleaned);
        }
    }
}
