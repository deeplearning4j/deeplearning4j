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

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base implementation of DeviceDescriptor providing common functionality
 * for all device types.
 */
public abstract class BaseDeviceDescriptor implements DeviceDescriptor {

    protected final String backendId;
    protected final DeviceType deviceType;
    protected final int deviceIndex;
    protected final String deviceName;
    protected final boolean isDefault;

    protected final Set<String> capabilities = ConcurrentHashMap.newKeySet();
    protected final Map<String, Object> properties = new ConcurrentHashMap<>();

    // Cache for transfer bandwidths to other devices
    protected final Map<String, Long> transferBandwidthCache = new ConcurrentHashMap<>();

    protected BaseDeviceDescriptor(String backendId, DeviceType deviceType, int deviceIndex,
                                   String deviceName, boolean isDefault) {
        this.backendId = backendId;
        this.deviceType = deviceType;
        this.deviceIndex = deviceIndex;
        this.deviceName = deviceName;
        this.isDefault = isDefault;
    }

    @Override
    public String getDeviceId() {
        return backendId + ":" + deviceType.getIdentifier() + ":" + deviceIndex;
    }

    @Override
    public DeviceType getDeviceType() {
        return deviceType;
    }

    @Override
    public int getDeviceIndex() {
        return deviceIndex;
    }

    @Override
    public String getDeviceName() {
        return deviceName;
    }

    @Override
    public String getBackendId() {
        return backendId;
    }

    @Override
    public boolean isDefault() {
        return isDefault;
    }

    @Override
    public boolean hasCapability(String capability) {
        return capabilities.contains(capability);
    }

    @Override
    public Set<String> getCapabilities() {
        return Collections.unmodifiableSet(capabilities);
    }

    @Override
    public Map<String, Object> getProperties() {
        return Collections.unmodifiableMap(properties);
    }

    @Override
    public Object getProperty(String key) {
        return properties.get(key);
    }

    /**
     * Add a capability to this device
     * @param capability the capability to add
     */
    protected void addCapability(String capability) {
        capabilities.add(capability);
    }

    /**
     * Set a property on this device
     * @param key the property key
     * @param value the property value
     */
    protected void setProperty(String key, Object value) {
        properties.put(key, value);
    }

    @Override
    public long getTransferBandwidthTo(DeviceDescriptor target) {
        if (target == null || !canTransferTo(target)) {
            return 0;
        }

        String cacheKey = target.getDeviceId();
        return transferBandwidthCache.computeIfAbsent(cacheKey, k -> computeTransferBandwidth(target));
    }

    /**
     * Compute the transfer bandwidth to another device
     * Subclasses should override for device-specific calculations
     * @param target the target device
     * @return estimated bandwidth in bytes per second
     */
    protected long computeTransferBandwidth(DeviceDescriptor target) {
        // Default: PCIe Gen3 x16 bandwidth (~16 GB/s)
        if (target.getDeviceType() != this.deviceType) {
            return 16L * 1024 * 1024 * 1024;
        }
        // Same device type - likely system memory bandwidth
        return getMemoryBandwidth();
    }

    @Override
    public boolean canTransferTo(DeviceDescriptor target) {
        // Default: can always transfer via host memory
        return target != null && !target.getDeviceId().equals(this.getDeviceId());
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BaseDeviceDescriptor that = (BaseDeviceDescriptor) o;
        return Objects.equals(getDeviceId(), that.getDeviceId());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getDeviceId());
    }

    @Override
    public String toString() {
        return String.format("%s[id=%s, name=%s, memory=%dMB, compute=%.2f]",
                getClass().getSimpleName(),
                getDeviceId(),
                deviceName,
                getTotalMemory() / (1024 * 1024),
                getComputeCapability());
    }
}
