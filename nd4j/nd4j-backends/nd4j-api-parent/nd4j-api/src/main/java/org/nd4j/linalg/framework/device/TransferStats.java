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

import lombok.Builder;
import lombok.Data;

import java.util.HashMap;
import java.util.Map;

/**
 * Aggregated transfer statistics for a variable.
 *
 * @author ND4J Team
 */
@Data
@Builder
public class TransferStats {
    /**
     * Variable name
     */
    private String variableName;

    /**
     * Total number of transfers
     */
    private long totalTransfers;

    /**
     * Total bytes transferred
     */
    private long totalBytes;

    /**
     * Total duration in nanoseconds
     */
    private long totalDurationNanos;

    /**
     * Count by direction
     */
    @Builder.Default
    private Map<TransferDirection, Long> countByDirection = new HashMap<>();

    /**
     * Bytes by direction
     */
    @Builder.Default
    private Map<TransferDirection, Long> bytesByDirection = new HashMap<>();

    /**
     * Calculate average bandwidth in bytes per second.
     */
    public double averageBandwidthBytesPerSec() {
        if (totalDurationNanos <= 0) {
            return 0.0;
        }
        return (totalBytes * 1_000_000_000.0) / totalDurationNanos;
    }
}
