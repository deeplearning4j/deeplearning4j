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

package org.nd4j.linalg.api.ops.executioner;

import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.nativeblas.NativeOps;

import java.util.List;

/**
 * Strategy interface for routing operations to different backends based on
 * device location of inputs and configuration preferences.
 *
 * <p>Implementations determine which NativeOps backend should execute an operation
 * based on factors such as:
 * <ul>
 *   <li>Device location of input arrays</li>
 *   <li>Op characteristics (memory bound vs compute bound)</li>
 *   <li>Available devices and their memory</li>
 *   <li>User preferences and configuration</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public interface BackendRoutingStrategy {

    /**
     * Select the target device for executing an operation.
     *
     * @param op the operation to execute
     * @return the device where the operation should run
     */
    DeviceDescriptor selectTargetDevice(Op op);

    /**
     * Select the target device for executing a custom operation.
     *
     * @param op the custom operation to execute
     * @return the device where the operation should run
     */
    DeviceDescriptor selectTargetDevice(CustomOp op);

    /**
     * Select the target device based on input arrays.
     *
     * @param inputs the input arrays
     * @return the optimal device for these inputs
     */
    DeviceDescriptor selectTargetDevice(List<INDArray> inputs);

    /**
     * Get the NativeOps for a specific device.
     *
     * @param device the target device
     * @return the NativeOps implementation for that device
     */
    NativeOps getNativeOpsForDevice(DeviceDescriptor device);

    /**
     * Get the NativeOps for a device type.
     *
     * @param deviceType the device type
     * @return the NativeOps implementation
     */
    NativeOps getNativeOpsForDeviceType(DeviceType deviceType);

    /**
     * Check if a backend is available for a device type.
     *
     * @param deviceType the device type
     * @return true if a backend is available
     */
    boolean isBackendAvailable(DeviceType deviceType);

    /**
     * Get the current device for the calling thread.
     *
     * @return the current device
     */
    DeviceDescriptor getCurrentDevice();

    /**
     * Set the current device for the calling thread.
     *
     * @param device the device to use
     */
    void setCurrentDevice(DeviceDescriptor device);

    /**
     * Transfer an array to a target device if needed.
     *
     * @param array the array to transfer
     * @param targetDevice the target device
     * @return the array on the target device (may be same instance if already there)
     */
    INDArray ensureOnDevice(INDArray array, DeviceDescriptor targetDevice);

    /**
     * Get routing statistics.
     *
     * @return routing statistics info
     */
    String getRoutingStats();
}
