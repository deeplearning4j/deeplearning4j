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

/**
 * Records a single device transfer event.
 *
 * @author ND4J Team
 */
@Data
@Builder
public class TransferEvent {
    /**
     * Timestamp in nanoseconds (System.nanoTime())
     */
    private long timestampNanos;

    /**
     * Variable name (null if unknown)
     */
    private String variableName;

    /**
     * Source device ID
     */
    private int sourceDeviceId;

    /**
     * Destination device ID
     */
    private int destDeviceId;

    /**
     * Transfer direction (H2D, D2H, D2D)
     */
    private TransferDirection direction;

    /**
     * Reason for the transfer
     */
    private TransferReason reason;

    /**
     * Number of bytes transferred
     */
    private long bytes;

    /**
     * Duration in nanoseconds
     */
    private long durationNanos;

    /**
     * Caller context (e.g., "DSP.executeNative", "replicateToDevice")
     */
    private String callerContext;
}
