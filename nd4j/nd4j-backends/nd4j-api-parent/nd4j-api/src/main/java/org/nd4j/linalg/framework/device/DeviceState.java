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

package org.nd4j.linalg.framework.device;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Device subsystem state snapshot.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DeviceState {
    
    /**
     * Number of devices
     */
    private int numDevices;
    
    /**
     * Current device ID
     */
    private int currentDeviceId;
    
    /**
     * Default device ID
     */
    private int defaultDeviceId;
    
    /**
     * Device routing policy
     */
    private String routingPolicy;
    
    /**
     * Whether auto fallback is enabled
     */
    private boolean autoFallbackEnabled;
    
    /**
     * Memory pressure threshold (0.0-1.0)
     */
    private double memoryPressureThreshold;
    
    /**
     * CUDA memory pool size (MB)
     */
    private int cudaMemoryPoolSizeMb;

    /**
     * Whether GPU is available
     */
    private boolean hasGpu;

    /**
     * Total number of transfers tracked
     */
    @Builder.Default
    private long totalTransfers = 0L;

    /**
     * Total bytes transferred
     */
    @Builder.Default
    private long totalTransferBytes = 0L;

    /**
     * Number of pinned variables
     */
    @Builder.Default
    private int pinnedVariableCount = 0;

    /**
     * Number of frozen buffers (non-migratable)
     */
    @Builder.Default
    private int frozenBufferCount = 0;

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(
            "Devices: %d%s, Current: %d, Default: %d, Routing: %s, " +
            "Fallback: %s, Pressure: %.0f%%, CUDA Pool: %d MB",
            numDevices,
            hasGpu ? " (GPU)" : " (CPU only)",
            currentDeviceId,
            defaultDeviceId,
            routingPolicy,
            autoFallbackEnabled ? "ON" : "OFF",
            memoryPressureThreshold * 100,
            cudaMemoryPoolSizeMb
        ));
        if (totalTransfers > 0) {
            sb.append(String.format(", Transfers: %d (%d bytes)", totalTransfers, totalTransferBytes));
        }
        if (pinnedVariableCount > 0) {
            sb.append(String.format(", Pinned: %d", pinnedVariableCount));
        }
        if (frozenBufferCount > 0) {
            sb.append(String.format(", Frozen: %d", frozenBufferCount));
        }
        return sb.toString();
    }
}
