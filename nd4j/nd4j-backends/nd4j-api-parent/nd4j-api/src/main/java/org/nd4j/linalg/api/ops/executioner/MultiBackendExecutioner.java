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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.routing.RoutingDecision;
import org.nd4j.linalg.api.ops.routing.RoutingPolicy;

import java.util.List;
import java.util.Map;

/**
 * Interface for multi-backend operation execution.
 * Extends OpExecutioner with device routing and multi-backend coordination.
 */
public interface MultiBackendExecutioner extends OpExecutioner {

    /**
     * Set the routing policy for device selection
     * @param policy the routing policy to use
     */
    void setRoutingPolicy(RoutingPolicy policy);

    /**
     * Get the current routing policy
     * @return the current policy
     */
    RoutingPolicy getRoutingPolicy();

    /**
     * Execute an operation on a specific device
     * @param op the operation
     * @param device the target device
     * @return the result
     */
    INDArray exec(Op op, DeviceDescriptor device);

    /**
     * Execute a custom operation on a specific device
     * @param op the custom operation
     * @param device the target device
     * @return the results
     */
    INDArray[] exec(CustomOp op, DeviceDescriptor device);

    /**
     * Get the routing decision for an operation
     * @param op the operation
     * @return the routing decision
     */
    RoutingDecision route(Op op);

    /**
     * Get the routing decision for a custom operation
     * @param op the custom operation
     * @return the routing decision
     */
    RoutingDecision route(CustomOp op);

    /**
     * Get all available devices
     * @return list of devices
     */
    List<DeviceDescriptor> getAvailableDevices();

    /**
     * Get the executioner for a specific device
     * @param device the device
     * @return the executioner
     */
    OpExecutioner getExecutioner(DeviceDescriptor device);

    /**
     * Transfer an array to a specific device
     * @param array the array to transfer
     * @param device the target device
     * @return the array (possibly with updated buffer location)
     */
    INDArray transferTo(INDArray array, DeviceDescriptor device);

    /**
     * Prefetch an array to a specific device
     * @param array the array to prefetch
     * @param device the target device
     */
    void prefetch(INDArray array, DeviceDescriptor device);

    /**
     * Synchronize all pending operations
     */
    void syncAll();

    /**
     * Synchronize operations on a specific device
     * @param device the device
     */
    void sync(DeviceDescriptor device);

    /**
     * Get execution statistics
     * @return map of statistic name to value
     */
    Map<String, Object> getStatistics();

    /**
     * Reset execution statistics
     */
    void resetStatistics();

    /**
     * Check if multi-backend execution is enabled
     * @return true if enabled
     */
    boolean isMultiBackendEnabled();

    /**
     * Enable or disable multi-backend execution
     * @param enabled true to enable
     */
    void setMultiBackendEnabled(boolean enabled);
}
