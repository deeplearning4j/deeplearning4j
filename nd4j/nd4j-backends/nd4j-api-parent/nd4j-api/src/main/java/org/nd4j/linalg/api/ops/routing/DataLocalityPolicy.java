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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.CustomOp;

import java.util.*;

/**
 * Routing policy that selects the device with the most input data.
 * Minimizes data movement by executing where data already resides.
 */
public class DataLocalityPolicy implements RoutingPolicy {

    @Override
    public String getName() {
        return "DataLocality";
    }

    @Override
    public RoutingDecision selectDevice(Op op, List<DeviceDescriptor> availableDevices) {
        List<INDArray> inputs = new ArrayList<>();
        if (op.x() != null) inputs.add(op.x());
        if (op.y() != null) inputs.add(op.y());
        return selectDeviceForInputs(inputs, availableDevices);
    }

    @Override
    public RoutingDecision selectDevice(CustomOp op, List<DeviceDescriptor> availableDevices) {
        List<INDArray> inputs = new ArrayList<>();
        if (op.inputArguments() != null) {
            inputs.addAll(op.inputArguments());
        }
        return selectDeviceForInputs(inputs, availableDevices);
    }

    private RoutingDecision selectDeviceForInputs(List<INDArray> inputs, List<DeviceDescriptor> availableDevices) {
        if (availableDevices == null || availableDevices.isEmpty()) {
            throw new IllegalArgumentException("No available devices");
        }

        // If no inputs, use the first device
        if (inputs.isEmpty()) {
            return RoutingDecision.builder(availableDevices.get(0))
                    .reason("no inputs, using default device")
                    .confidence(0.5)
                    .build();
        }

        // Count bytes per device
        Map<String, Long> bytesPerDevice = new HashMap<>();
        Map<String, DeviceDescriptor> deviceById = new HashMap<>();

        for (DeviceDescriptor device : availableDevices) {
            bytesPerDevice.put(device.getDeviceId(), 0L);
            deviceById.put(device.getDeviceId(), device);
        }

        // Analyze input locations
        for (INDArray input : inputs) {
            if (input == null || input.data() == null) continue;

            long bytes = input.length() * input.data().getElementSize();

            if (input.data().isHybrid()) {
                HybridDataBuffer hybrid = input.data().asHybrid();
                DeviceDescriptor owner = hybrid.getEffectiveDevice();
                if (owner != null && bytesPerDevice.containsKey(owner.getDeviceId())) {
                    bytesPerDevice.merge(owner.getDeviceId(), bytes, Long::sum);
                }
            } else {
                // Non-hybrid buffer - assume it's on the first/default device
                String defaultId = availableDevices.get(0).getDeviceId();
                bytesPerDevice.merge(defaultId, bytes, Long::sum);
            }
        }

        // Find device with most data
        String bestDeviceId = null;
        long maxBytes = -1;
        for (Map.Entry<String, Long> entry : bytesPerDevice.entrySet()) {
            if (entry.getValue() > maxBytes) {
                maxBytes = entry.getValue();
                bestDeviceId = entry.getKey();
            }
        }

        DeviceDescriptor targetDevice = deviceById.get(bestDeviceId);
        if (targetDevice == null) {
            targetDevice = availableDevices.get(0);
        }

        // Build decision with transfer info
        RoutingDecision.Builder builder = RoutingDecision.builder(targetDevice)
                .reason("device has most input data (" + maxBytes + " bytes)")
                .confidence(maxBytes > 0 ? 0.9 : 0.5);

        // Calculate required transfers
        long totalTransferTime = 0;
        for (INDArray input : inputs) {
            if (input == null || input.data() == null) continue;

            DeviceDescriptor sourceDevice = null;
            if (input.data().isHybrid()) {
                sourceDevice = input.data().asHybrid().getEffectiveDevice();
            } else {
                sourceDevice = availableDevices.get(0);  // Assume default
            }

            if (sourceDevice != null && !sourceDevice.getDeviceId().equals(targetDevice.getDeviceId())) {
                builder.addTransfer(input, sourceDevice, targetDevice);
                long bytes = input.length() * input.data().getElementSize();
                long bandwidth = sourceDevice.getTransferBandwidthTo(targetDevice);
                if (bandwidth > 0) {
                    totalTransferTime += (bytes * 1_000_000_000L) / bandwidth;
                }
            }
        }

        builder.estimatedTransferTime(totalTransferTime);
        return builder.build();
    }

    @Override
    public int getPriority() {
        return 100;  // High priority - data locality is usually important
    }
}
