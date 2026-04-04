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

package org.nd4j.linalg.api.device;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Configurable fake device for testing multi-device scenarios on single-GPU machines.
 * Extends BaseDeviceDescriptor with mutable available memory and configurable peer topology.
 *
 * <p>Usage:
 * <pre>{@code
 * StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
 *     .totalMemory(24L * 1024 * 1024 * 1024)
 *     .peerDevices(Set.of(1))
 *     .build();
 * }</pre>
 */
public class StubDeviceDescriptor extends BaseDeviceDescriptor {

    private final long totalMemory;
    private volatile long availableMemory;
    private final long memoryBandwidth;
    private final double computeCapability;
    private final int computeUnits;
    private final long estimatedFlops;
    private final Set<Integer> peerDevices;
    private final Map<Integer, Long> peerBandwidthOverrides;

    private StubDeviceDescriptor(Builder builder) {
        super("stub", builder.deviceType, builder.deviceIndex,
                builder.deviceName != null ? builder.deviceName :
                        builder.deviceType.getIdentifier() + "-stub-" + builder.deviceIndex,
                builder.isDefault);
        this.totalMemory = builder.totalMemory;
        this.availableMemory = builder.availableMemory > 0 ? builder.availableMemory : (long) (totalMemory * 0.8);
        this.memoryBandwidth = builder.memoryBandwidth;
        this.computeCapability = builder.computeCapability;
        this.computeUnits = builder.computeUnits;
        this.estimatedFlops = builder.estimatedFlops;
        this.peerDevices = Collections.unmodifiableSet(new HashSet<>(builder.peerDevices));
        this.peerBandwidthOverrides = Collections.unmodifiableMap(new HashMap<>(builder.peerBandwidthOverrides));
    }

    /**
     * Create a builder for a stub device.
     *
     * @param deviceType  the device type
     * @param deviceIndex the device index
     * @return new builder
     */
    public static Builder builder(DeviceType deviceType, int deviceIndex) {
        return new Builder(deviceType, deviceIndex);
    }

    @Override
    public long getTotalMemory() {
        return totalMemory;
    }

    @Override
    public long getAvailableMemory() {
        return availableMemory;
    }

    /**
     * Set the available memory. Thread-safe (volatile write).
     *
     * @param bytes new available memory in bytes
     */
    public void setAvailableMemory(long bytes) {
        this.availableMemory = Math.max(0, bytes);
    }

    /**
     * Consume memory, reducing available memory by the specified amount.
     *
     * @param bytes bytes to consume
     * @return actual bytes consumed (may be less if not enough available)
     */
    public long consumeMemory(long bytes) {
        long current = this.availableMemory;
        long consumed = Math.min(current, bytes);
        this.availableMemory = current - consumed;
        return consumed;
    }

    @Override
    public long getMemoryBandwidth() {
        return memoryBandwidth;
    }

    @Override
    public double getComputeCapability() {
        return computeCapability;
    }

    @Override
    public int getComputeUnits() {
        return computeUnits;
    }

    @Override
    public long getEstimatedFlops() {
        return estimatedFlops;
    }

    /**
     * Get the set of peer device indices that have direct (P2P) access.
     */
    public Set<Integer> getPeerDevices() {
        return peerDevices;
    }

    /**
     * Check if this device has peer access to another device index.
     */
    public boolean hasPeerAccessTo(int otherDeviceIndex) {
        return peerDevices.contains(otherDeviceIndex);
    }

    @Override
    protected long computeTransferBandwidth(DeviceDescriptor target) {
        if (target instanceof StubDeviceDescriptor) {
            int targetIdx = target.getDeviceIndex();
            // Check for explicit bandwidth override
            Long override = peerBandwidthOverrides.get(targetIdx);
            if (override != null) {
                return override;
            }
            // P2P peer → NVLink-like bandwidth
            if (peerDevices.contains(targetIdx)) {
                return 300L * 1024 * 1024 * 1024; // ~300 GB/s NVLink
            }
            // Non-peer GPU → PCIe bandwidth
            if (target.getDeviceType().isGpu()) {
                return 12L * 1024 * 1024 * 1024; // ~12 GB/s PCIe
            }
        }
        // GPU to CPU or default
        return 16L * 1024 * 1024 * 1024; // ~16 GB/s PCIe
    }

    @Override
    public boolean canTransferTo(DeviceDescriptor target) {
        return target != null && !target.getDeviceId().equals(this.getDeviceId());
    }

    @Override
    public String toString() {
        return String.format("StubDevice[%s:%d, %s, %d MB free/%d MB total, peers=%s]",
                getDeviceType().getIdentifier(), getDeviceIndex(), getDeviceName(),
                availableMemory / (1024 * 1024), totalMemory / (1024 * 1024), peerDevices);
    }

    /**
     * Builder for StubDeviceDescriptor.
     */
    public static class Builder {
        private final DeviceType deviceType;
        private final int deviceIndex;
        private String deviceName;
        private boolean isDefault;
        private long totalMemory = 10L * 1024 * 1024 * 1024; // 10 GB default
        private long availableMemory = -1; // -1 means 80% of total
        private long memoryBandwidth = 760L * 1024 * 1024 * 1024; // ~760 GB/s
        private double computeCapability = 8.6;
        private int computeUnits = 68;
        private long estimatedFlops = 30_000_000_000_000L; // 30 TFLOPS
        private Set<Integer> peerDevices = new HashSet<>();
        private Map<Integer, Long> peerBandwidthOverrides = new HashMap<>();

        private Builder(DeviceType deviceType, int deviceIndex) {
            this.deviceType = deviceType;
            this.deviceIndex = deviceIndex;
        }

        public Builder deviceName(String name) {
            this.deviceName = name;
            return this;
        }

        public Builder isDefault(boolean isDefault) {
            this.isDefault = isDefault;
            return this;
        }

        public Builder totalMemory(long bytes) {
            this.totalMemory = bytes;
            return this;
        }

        public Builder availableMemory(long bytes) {
            this.availableMemory = bytes;
            return this;
        }

        public Builder memoryBandwidth(long bytesPerSec) {
            this.memoryBandwidth = bytesPerSec;
            return this;
        }

        public Builder computeCapability(double cc) {
            this.computeCapability = cc;
            return this;
        }

        public Builder computeUnits(int units) {
            this.computeUnits = units;
            return this;
        }

        public Builder estimatedFlops(long flops) {
            this.estimatedFlops = flops;
            return this;
        }

        public Builder peerDevices(Set<Integer> peers) {
            this.peerDevices = new HashSet<>(peers);
            return this;
        }

        public Builder addPeerDevice(int deviceIndex) {
            this.peerDevices.add(deviceIndex);
            return this;
        }

        public Builder peerBandwidth(int peerDeviceIndex, long bandwidthBytesPerSec) {
            this.peerBandwidthOverrides.put(peerDeviceIndex, bandwidthBytesPerSec);
            return this;
        }

        public StubDeviceDescriptor build() {
            return new StubDeviceDescriptor(this);
        }
    }
}
