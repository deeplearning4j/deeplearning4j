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
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.CustomOp;

import java.util.List;

/**
 * Interface for operation routing policies.
 * Policies determine which device should execute a given operation
 * based on various criteria like data locality, performance, or manual selection.
 */
public interface RoutingPolicy {

    /**
     * Get the name of this routing policy
     * @return policy name
     */
    String getName();

    /**
     * Select the best device for executing an operation
     * @param op the operation to execute
     * @param availableDevices list of available devices
     * @return routing decision with target device and transfer info
     */
    RoutingDecision selectDevice(Op op, List<DeviceDescriptor> availableDevices);

    /**
     * Select the best device for executing a custom operation
     * @param op the custom operation to execute
     * @param availableDevices list of available devices
     * @return routing decision with target device and transfer info
     */
    RoutingDecision selectDevice(CustomOp op, List<DeviceDescriptor> availableDevices);

    /**
     * Check if this policy can handle the given operation
     * @param op the operation
     * @return true if this policy can route this operation
     */
    default boolean canHandle(Op op) {
        return true;
    }

    /**
     * Check if this policy can handle the given custom operation
     * @param op the custom operation
     * @return true if this policy can route this operation
     */
    default boolean canHandle(CustomOp op) {
        return true;
    }

    /**
     * Get the priority of this policy (higher = checked first)
     * @return priority value
     */
    default int getPriority() {
        return 0;
    }

    /**
     * Called when an operation execution completes
     * Allows policies to track performance for future decisions
     * @param op the operation that was executed
     * @param device the device that executed it
     * @param executionTimeNanos actual execution time
     */
    default void recordExecution(Op op, DeviceDescriptor device, long executionTimeNanos) {
        // Default: no tracking
    }

    /**
     * Called when a custom operation execution completes
     * @param op the custom operation that was executed
     * @param device the device that executed it
     * @param executionTimeNanos actual execution time
     */
    default void recordExecution(CustomOp op, DeviceDescriptor device, long executionTimeNanos) {
        // Default: no tracking
    }

    /**
     * Reset any learned state in this policy
     */
    default void reset() {
        // Default: nothing to reset
    }
}
