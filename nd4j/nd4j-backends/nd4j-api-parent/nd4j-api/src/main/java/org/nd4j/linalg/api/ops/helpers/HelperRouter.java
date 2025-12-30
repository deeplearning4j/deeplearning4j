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

package org.nd4j.linalg.api.ops.helpers;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Routes operations to platform-specific helpers (MKLDNN, CUDNN, etc.)
 * based on device type, operation type, and performance history.
 */
@Slf4j
public class HelperRouter {

    private static volatile HelperRouter instance;
    private static final Object INSTANCE_LOCK = new Object();

    // Registered helpers by device type
    private final Map<DeviceType, List<PlatformHelperDescriptor>> helpersByDevice = new ConcurrentHashMap<>();

    // Performance tracking per helper per op
    private final Map<String, Map<String, PerformanceStats>> performanceHistory = new ConcurrentHashMap<>();

    // Default helper preferences
    private final Map<DeviceType, String> defaultHelpers = new ConcurrentHashMap<>();

    private HelperRouter() {
        // Initialize default preferences
        defaultHelpers.put(DeviceType.CPU, "mkldnn");
        defaultHelpers.put(DeviceType.CUDA_GPU, "cudnn");
    }

    /**
     * Get the singleton instance
     * @return the helper router instance
     */
    public static HelperRouter getInstance() {
        if (instance == null) {
            synchronized (INSTANCE_LOCK) {
                if (instance == null) {
                    instance = new HelperRouter();
                }
            }
        }
        return instance;
    }

    /**
     * Register a platform helper
     * @param descriptor the helper descriptor
     */
    public void registerHelper(PlatformHelperDescriptor descriptor) {
        helpersByDevice.computeIfAbsent(descriptor.getTargetDevice(), k -> new ArrayList<>())
                .add(descriptor);

        // Sort by priority (descending)
        helpersByDevice.get(descriptor.getTargetDevice())
                .sort((a, b) -> Integer.compare(b.getPriority(), a.getPriority()));

        log.debug("Registered helper: {}", descriptor);
    }

    /**
     * Select the best helper for an operation on a device
     * @param opName the operation name
     * @param device the target device
     * @return the selected helper, or null if none available
     */
    public PlatformHelperDescriptor selectHelper(String opName, DeviceDescriptor device) {
        List<PlatformHelperDescriptor> helpers = helpersByDevice.get(device.getDeviceType());
        if (helpers == null || helpers.isEmpty()) {
            return null;
        }

        // Find helpers that support this operation
        List<PlatformHelperDescriptor> candidates = new ArrayList<>();
        for (PlatformHelperDescriptor helper : helpers) {
            if (helper.isAvailable() && helper.supportsOp(opName)) {
                candidates.add(helper);
            }
        }

        if (candidates.isEmpty()) {
            return null;
        }

        // If only one candidate, use it
        if (candidates.size() == 1) {
            return candidates.get(0);
        }

        // Select based on performance history
        return selectByPerformance(opName, candidates);
    }

    /**
     * Select helper based on performance history
     */
    private PlatformHelperDescriptor selectByPerformance(String opName, List<PlatformHelperDescriptor> candidates) {
        Map<String, PerformanceStats> opStats = performanceHistory.get(opName);

        if (opStats == null || opStats.isEmpty()) {
            // No history, use highest priority
            return candidates.get(0);
        }

        PlatformHelperDescriptor best = null;
        double bestScore = Double.MAX_VALUE;

        for (PlatformHelperDescriptor helper : candidates) {
            PerformanceStats stats = opStats.get(helper.getHelperId());
            if (stats != null) {
                double avgTime = stats.getAverageTime();
                // Lower time = better
                double score = avgTime / (1.0 + helper.getPriority() * 0.1);
                if (score < bestScore) {
                    bestScore = score;
                    best = helper;
                }
            }
        }

        return best != null ? best : candidates.get(0);
    }

    /**
     * Select helper for a standard operation
     * @param op the operation
     * @param device the target device
     * @return the selected helper
     */
    public PlatformHelperDescriptor selectHelper(Op op, DeviceDescriptor device) {
        return selectHelper(op.opName(), device);
    }

    /**
     * Select helper for a custom operation
     * @param op the custom operation
     * @param device the target device
     * @return the selected helper
     */
    public PlatformHelperDescriptor selectHelper(CustomOp op, DeviceDescriptor device) {
        return selectHelper(op.opName(), device);
    }

    /**
     * Record execution performance for a helper
     * @param opName the operation name
     * @param helperId the helper ID
     * @param executionTimeNanos execution time in nanoseconds
     */
    public void recordPerformance(String opName, String helperId, long executionTimeNanos) {
        performanceHistory.computeIfAbsent(opName, k -> new ConcurrentHashMap<>())
                .computeIfAbsent(helperId, k -> new PerformanceStats())
                .record(executionTimeNanos);
    }

    /**
     * Get all registered helpers
     * @return map of device type to helper list
     */
    public Map<DeviceType, List<PlatformHelperDescriptor>> getAllHelpers() {
        return Collections.unmodifiableMap(helpersByDevice);
    }

    /**
     * Get helpers for a specific device type
     * @param deviceType the device type
     * @return list of helpers
     */
    public List<PlatformHelperDescriptor> getHelpers(DeviceType deviceType) {
        List<PlatformHelperDescriptor> helpers = helpersByDevice.get(deviceType);
        return helpers != null ? Collections.unmodifiableList(helpers) : Collections.emptyList();
    }

    /**
     * Set the default helper for a device type
     * @param deviceType the device type
     * @param helperId the helper ID to use by default
     */
    public void setDefaultHelper(DeviceType deviceType, String helperId) {
        defaultHelpers.put(deviceType, helperId);
    }

    /**
     * Get performance statistics
     * @return map of op name to helper performance stats
     */
    public Map<String, Map<String, Double>> getPerformanceStats() {
        Map<String, Map<String, Double>> result = new HashMap<>();

        for (Map.Entry<String, Map<String, PerformanceStats>> opEntry : performanceHistory.entrySet()) {
            Map<String, Double> helperStats = new HashMap<>();
            for (Map.Entry<String, PerformanceStats> helperEntry : opEntry.getValue().entrySet()) {
                helperStats.put(helperEntry.getKey(), helperEntry.getValue().getAverageTime());
            }
            result.put(opEntry.getKey(), helperStats);
        }

        return result;
    }

    /**
     * Clear performance history
     */
    public void clearPerformanceHistory() {
        performanceHistory.clear();
    }

    /**
     * Internal class for tracking performance statistics
     */
    private static class PerformanceStats {
        private final AtomicLong count = new AtomicLong(0);
        private volatile double movingAverage = 0.0;
        private static final double ALPHA = 0.3;  // Smoothing factor

        synchronized void record(long timeNanos) {
            if (count.getAndIncrement() == 0) {
                movingAverage = timeNanos;
            } else {
                movingAverage = ALPHA * timeNanos + (1 - ALPHA) * movingAverage;
            }
        }

        double getAverageTime() {
            return movingAverage;
        }

        long getCount() {
            return count.get();
        }
    }
}
