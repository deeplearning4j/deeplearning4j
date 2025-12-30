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

package org.nd4j.linalg.api.ops.routing;

import org.nd4j.linalg.api.buffer.HybridDataBuffer;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.CustomOp;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Routing policy that selects the device with the best estimated performance.
 * Uses historical execution times and device capabilities to estimate total time.
 */
public class PerformancePolicy implements RoutingPolicy {

    // Track execution times per operation per device
    private final Map<String, Map<String, ExecutionStats>> executionHistory = new ConcurrentHashMap<>();

    // Smoothing factor for exponential moving average
    private final double alpha;

    public PerformancePolicy() {
        this(0.3);  // Default smoothing
    }

    public PerformancePolicy(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public String getName() {
        return "Performance";
    }

    @Override
    public RoutingDecision selectDevice(Op op, List<DeviceDescriptor> availableDevices) {
        String opKey = getOpKey(op);
        List<INDArray> inputs = new ArrayList<>();
        if (op.x() != null) inputs.add(op.x());
        if (op.y() != null) inputs.add(op.y());
        return selectBestDevice(opKey, inputs, availableDevices);
    }

    @Override
    public RoutingDecision selectDevice(CustomOp op, List<DeviceDescriptor> availableDevices) {
        String opKey = op.opName();
        List<INDArray> inputs = op.inputArguments() != null ?
                op.inputArguments() : Collections.emptyList();
        return selectBestDevice(opKey, inputs, availableDevices);
    }

    private RoutingDecision selectBestDevice(String opKey, List<INDArray> inputs,
                                              List<DeviceDescriptor> availableDevices) {
        if (availableDevices == null || availableDevices.isEmpty()) {
            throw new IllegalArgumentException("No available devices");
        }

        DeviceDescriptor bestDevice = null;
        long bestTotalTime = Long.MAX_VALUE;
        long bestExecTime = 0;
        long bestTransferTime = 0;
        double bestConfidence = 0.0;

        for (DeviceDescriptor device : availableDevices) {
            // Estimate execution time
            long execTime = estimateExecutionTime(opKey, device, inputs);

            // Calculate transfer time
            long transferTime = calculateTransferTime(inputs, device, availableDevices);

            long totalTime = execTime + transferTime;

            if (totalTime < bestTotalTime) {
                bestTotalTime = totalTime;
                bestDevice = device;
                bestExecTime = execTime;
                bestTransferTime = transferTime;

                // Confidence based on whether we have history
                Map<String, ExecutionStats> deviceHistory = executionHistory.get(device.getDeviceId());
                if (deviceHistory != null && deviceHistory.containsKey(opKey)) {
                    bestConfidence = Math.min(0.95, 0.5 + deviceHistory.get(opKey).count.get() * 0.05);
                } else {
                    bestConfidence = 0.3;  // Low confidence for estimates
                }
            }
        }

        if (bestDevice == null) {
            bestDevice = availableDevices.get(0);
        }

        // Build decision with transfers
        RoutingDecision.Builder builder = RoutingDecision.builder(bestDevice)
                .estimatedExecutionTime(bestExecTime)
                .estimatedTransferTime(bestTransferTime)
                .confidence(bestConfidence)
                .reason("lowest estimated total time: " + bestTotalTime + "ns");

        // Add transfer info
        addTransferInfo(builder, inputs, bestDevice, availableDevices);

        return builder.build();
    }

    private long estimateExecutionTime(String opKey, DeviceDescriptor device, List<INDArray> inputs) {
        // Check history
        Map<String, ExecutionStats> deviceHistory = executionHistory.get(device.getDeviceId());
        if (deviceHistory != null) {
            ExecutionStats stats = deviceHistory.get(opKey);
            if (stats != null) {
                return (long) stats.movingAverage;
            }
        }

        // Estimate based on device capabilities and input size
        long totalElements = 0;
        for (INDArray input : inputs) {
            if (input != null) {
                totalElements += input.length();
            }
        }

        // Base estimate: operations per element / FLOPS
        long flops = device.getEstimatedFlops();
        if (flops <= 0) flops = 1_000_000_000L;  // 1 GFLOP default

        // Assume ~10 FLOPs per element for typical operations
        long estimatedOps = totalElements * 10;
        long execTimeNanos = (estimatedOps * 1_000_000_000L) / flops;

        // GPU operations have overhead but are faster for large arrays
        if (device.getDeviceType().isGpu()) {
            // Kernel launch overhead (~10 microseconds)
            execTimeNanos += 10_000;

            // GPUs are much faster for large operations
            if (totalElements > 10000) {
                execTimeNanos /= 10;
            }
        }

        return Math.max(1000, execTimeNanos);  // At least 1 microsecond
    }

    private long calculateTransferTime(List<INDArray> inputs, DeviceDescriptor targetDevice,
                                        List<DeviceDescriptor> availableDevices) {
        long totalTransferTime = 0;
        DeviceDescriptor defaultDevice = availableDevices.get(0);

        for (INDArray input : inputs) {
            if (input == null || input.data() == null) continue;

            DeviceDescriptor sourceDevice;
            if (input.data().isHybrid()) {
                sourceDevice = input.data().asHybrid().getOwnerDevice();
            } else {
                sourceDevice = defaultDevice;
            }

            if (sourceDevice != null && !sourceDevice.getDeviceId().equals(targetDevice.getDeviceId())) {
                long bytes = input.length() * input.data().getElementSize();
                long bandwidth = sourceDevice.getTransferBandwidthTo(targetDevice);
                if (bandwidth > 0) {
                    totalTransferTime += (bytes * 1_000_000_000L) / bandwidth;
                }
            }
        }

        return totalTransferTime;
    }

    private void addTransferInfo(RoutingDecision.Builder builder, List<INDArray> inputs,
                                  DeviceDescriptor targetDevice, List<DeviceDescriptor> availableDevices) {
        DeviceDescriptor defaultDevice = availableDevices.get(0);

        for (INDArray input : inputs) {
            if (input == null || input.data() == null) continue;

            DeviceDescriptor sourceDevice;
            if (input.data().isHybrid()) {
                sourceDevice = input.data().asHybrid().getOwnerDevice();
            } else {
                sourceDevice = defaultDevice;
            }

            if (sourceDevice != null && !sourceDevice.getDeviceId().equals(targetDevice.getDeviceId())) {
                builder.addTransfer(input, sourceDevice, targetDevice);
            }
        }
    }

    private String getOpKey(Op op) {
        return op.opName() + "_" + op.getClass().getSimpleName();
    }

    @Override
    public void recordExecution(Op op, DeviceDescriptor device, long executionTimeNanos) {
        recordExecutionInternal(getOpKey(op), device, executionTimeNanos);
    }

    @Override
    public void recordExecution(CustomOp op, DeviceDescriptor device, long executionTimeNanos) {
        recordExecutionInternal(op.opName(), device, executionTimeNanos);
    }

    private void recordExecutionInternal(String opKey, DeviceDescriptor device, long executionTimeNanos) {
        executionHistory.computeIfAbsent(device.getDeviceId(), k -> new ConcurrentHashMap<>())
                .computeIfAbsent(opKey, k -> new ExecutionStats())
                .record(executionTimeNanos, alpha);
    }

    @Override
    public void reset() {
        executionHistory.clear();
    }

    @Override
    public int getPriority() {
        return 50;  // Medium priority
    }

    /**
     * Statistics tracker for execution times
     */
    private static class ExecutionStats {
        final AtomicLong count = new AtomicLong(0);
        volatile double movingAverage = 0.0;

        synchronized void record(long timeNanos, double alpha) {
            if (count.getAndIncrement() == 0) {
                movingAverage = timeNanos;
            } else {
                movingAverage = alpha * timeNanos + (1 - alpha) * movingAverage;
            }
        }
    }
}
