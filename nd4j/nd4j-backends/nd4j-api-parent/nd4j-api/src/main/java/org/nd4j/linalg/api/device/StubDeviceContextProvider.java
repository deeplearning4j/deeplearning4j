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

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Fake DeviceContextProvider for testing multi-device scenarios without real GPU hardware.
 * Manages a list of StubDeviceDescriptor devices, tracks device switches, and returns
 * null-stream DeviceContext instances (CPU-like, no real CUDA streams).
 *
 * <p>Records all device switch calls as {@link DeviceSwitchRecord} for test inspection.</p>
 */
public class StubDeviceContextProvider implements DeviceContextProvider {

    private final List<StubDeviceDescriptor> devices;
    private final ThreadLocal<Integer> currentDevice = ThreadLocal.withInitial(() -> 0);
    private final CopyOnWriteArrayList<DeviceSwitchRecord> switchHistory = new CopyOnWriteArrayList<>();

    /**
     * Create a provider with the given stub devices.
     *
     * @param devices list of stub device descriptors (index in list = device ID)
     */
    public StubDeviceContextProvider(List<StubDeviceDescriptor> devices) {
        if (devices == null || devices.isEmpty()) {
            throw new IllegalArgumentException("Must provide at least one stub device");
        }
        this.devices = new ArrayList<>(devices);
    }

    @Override
    public DeviceContext getCurrentContext() {
        int deviceId = currentDevice.get();
        return new DeviceContext(deviceId, null, null);
    }

    @Override
    public DeviceContext switchDevice(int deviceId, String caller, String reason) {
        if (deviceId < 0 || deviceId >= devices.size()) {
            throw new IllegalArgumentException("Invalid device ID: " + deviceId +
                    ", valid range: 0-" + (devices.size() - 1));
        }
        int previousDevice = currentDevice.get();
        currentDevice.set(deviceId);

        switchHistory.add(new DeviceSwitchRecord(
                previousDevice, deviceId, caller, reason, System.nanoTime()));

        return new DeviceContext(deviceId, null, null);
    }

    @Override
    public Pointer getFreshExecutionStream() {
        return null; // No real streams in stub
    }

    @Override
    public int getCurrentDeviceId() {
        return currentDevice.get();
    }

    @Override
    public int getDeviceCount() {
        return devices.size();
    }

    @Override
    public void syncExecutionStream() {
        // no-op in stub
    }

    @Override
    public boolean supportsStreams() {
        return false;
    }

    @Override
    public void ensureHostAccess(INDArray array) {
        // no-op in stub — data is always on host
    }

    /**
     * Get the stub device at the given index.
     */
    public StubDeviceDescriptor getDevice(int index) {
        return devices.get(index);
    }

    /**
     * Get all stub devices.
     */
    public List<StubDeviceDescriptor> getDevices() {
        return Collections.unmodifiableList(devices);
    }

    /**
     * Get the full device switch history.
     */
    public List<DeviceSwitchRecord> getSwitchHistory() {
        return Collections.unmodifiableList(switchHistory);
    }

    /**
     * Clear the device switch history.
     */
    public void clearSwitchHistory() {
        switchHistory.clear();
    }

    /**
     * Record of a device switch event for test inspection.
     */
    public static class DeviceSwitchRecord {
        private final int previousDeviceId;
        private final int newDeviceId;
        private final String caller;
        private final String reason;
        private final long timestampNanos;

        public DeviceSwitchRecord(int previousDeviceId, int newDeviceId,
                                  String caller, String reason, long timestampNanos) {
            this.previousDeviceId = previousDeviceId;
            this.newDeviceId = newDeviceId;
            this.caller = caller;
            this.reason = reason;
            this.timestampNanos = timestampNanos;
        }

        public int getPreviousDeviceId() { return previousDeviceId; }
        public int getNewDeviceId() { return newDeviceId; }
        public String getCaller() { return caller; }
        public String getReason() { return reason; }
        public long getTimestampNanos() { return timestampNanos; }

        @Override
        public String toString() {
            return String.format("DeviceSwitch[%d->%d, caller=%s, reason=%s]",
                    previousDeviceId, newDeviceId, caller, reason);
        }
    }
}
