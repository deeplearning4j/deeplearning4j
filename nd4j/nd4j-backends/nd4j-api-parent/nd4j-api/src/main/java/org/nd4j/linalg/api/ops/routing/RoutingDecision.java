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

import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 * Represents the result of a routing decision.
 * Contains the target device, required transfers, and metadata.
 */
public class RoutingDecision {

    private final DeviceDescriptor targetDevice;
    private final List<TransferInfo> requiredTransfers;
    private final long estimatedExecutionTime;
    private final long estimatedTransferTime;
    private final double confidence;
    private final String reason;

    private RoutingDecision(Builder builder) {
        this.targetDevice = builder.targetDevice;
        this.requiredTransfers = Collections.unmodifiableList(new ArrayList<>(builder.transfers));
        this.estimatedExecutionTime = builder.estimatedExecutionTime;
        this.estimatedTransferTime = builder.estimatedTransferTime;
        this.confidence = builder.confidence;
        this.reason = builder.reason;
    }

    /**
     * Get the target device for execution
     * @return target device
     */
    public DeviceDescriptor getTargetDevice() {
        return targetDevice;
    }

    /**
     * Get the list of required data transfers
     * @return list of transfers
     */
    public List<TransferInfo> getRequiredTransfers() {
        return requiredTransfers;
    }

    /**
     * Get estimated execution time on target device (excluding transfers)
     * @return time in nanoseconds
     */
    public long getEstimatedExecutionTime() {
        return estimatedExecutionTime;
    }

    /**
     * Get estimated total transfer time
     * @return time in nanoseconds
     */
    public long getEstimatedTransferTime() {
        return estimatedTransferTime;
    }

    /**
     * Get total estimated time (execution + transfers)
     * @return time in nanoseconds
     */
    public long getTotalEstimatedTime() {
        return estimatedExecutionTime + estimatedTransferTime;
    }

    /**
     * Get the confidence level of this decision
     * @return confidence between 0.0 and 1.0
     */
    public double getConfidence() {
        return confidence;
    }

    /**
     * Get the reason for this routing decision
     * @return human-readable reason
     */
    public String getReason() {
        return reason;
    }

    /**
     * Check if any transfers are required
     * @return true if transfers are needed
     */
    public boolean requiresTransfers() {
        return !requiredTransfers.isEmpty();
    }

    /**
     * Get the number of required transfers
     * @return transfer count
     */
    public int getTransferCount() {
        return requiredTransfers.size();
    }

    /**
     * Get total bytes that need to be transferred
     * @return total bytes
     */
    public long getTotalTransferBytes() {
        return requiredTransfers.stream()
                .mapToLong(TransferInfo::getBytes)
                .sum();
    }

    @Override
    public String toString() {
        return String.format("RoutingDecision[target=%s, transfers=%d, execTime=%dns, transferTime=%dns, confidence=%.2f, reason=%s]",
                targetDevice.getDeviceId(),
                requiredTransfers.size(),
                estimatedExecutionTime,
                estimatedTransferTime,
                confidence,
                reason);
    }

    /**
     * Create a simple decision with just a target device
     * @param device the target device
     * @return routing decision
     */
    public static RoutingDecision simple(DeviceDescriptor device) {
        return new Builder(device).build();
    }

    /**
     * Create a builder for a routing decision
     * @param targetDevice the target device
     * @return builder
     */
    public static Builder builder(DeviceDescriptor targetDevice) {
        return new Builder(targetDevice);
    }

    /**
     * Builder for RoutingDecision
     */
    public static class Builder {
        private final DeviceDescriptor targetDevice;
        private final List<TransferInfo> transfers = new ArrayList<>();
        private long estimatedExecutionTime = 0;
        private long estimatedTransferTime = 0;
        private double confidence = 1.0;
        private String reason = "default";

        public Builder(DeviceDescriptor targetDevice) {
            this.targetDevice = Objects.requireNonNull(targetDevice);
        }

        public Builder addTransfer(INDArray array, DeviceDescriptor from, DeviceDescriptor to) {
            transfers.add(new TransferInfo(array, from, to));
            return this;
        }

        public Builder addTransfer(TransferInfo transfer) {
            transfers.add(transfer);
            return this;
        }

        public Builder estimatedExecutionTime(long timeNanos) {
            this.estimatedExecutionTime = timeNanos;
            return this;
        }

        public Builder estimatedTransferTime(long timeNanos) {
            this.estimatedTransferTime = timeNanos;
            return this;
        }

        public Builder confidence(double confidence) {
            this.confidence = Math.max(0.0, Math.min(1.0, confidence));
            return this;
        }

        public Builder reason(String reason) {
            this.reason = reason;
            return this;
        }

        public RoutingDecision build() {
            return new RoutingDecision(this);
        }
    }

    /**
     * Information about a required data transfer
     */
    public static class TransferInfo {
        private final INDArray array;
        private final DeviceDescriptor sourceDevice;
        private final DeviceDescriptor targetDevice;
        private final long bytes;

        public TransferInfo(INDArray array, DeviceDescriptor sourceDevice, DeviceDescriptor targetDevice) {
            this.array = array;
            this.sourceDevice = sourceDevice;
            this.targetDevice = targetDevice;
            this.bytes = array != null ? array.length() * array.data().getElementSize() : 0;
        }

        public INDArray getArray() {
            return array;
        }

        public DeviceDescriptor getSourceDevice() {
            return sourceDevice;
        }

        public DeviceDescriptor getTargetDevice() {
            return targetDevice;
        }

        public long getBytes() {
            return bytes;
        }

        @Override
        public String toString() {
            return String.format("Transfer[%s -> %s, %d bytes]",
                    sourceDevice.getDeviceId(),
                    targetDevice.getDeviceId(),
                    bytes);
        }
    }
}
