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

package org.nd4j.linalg.framework;

import lombok.Builder;
import lombok.Data;

/**
 * Health status report for ND4J framework subsystems.
 */
@Data
@Builder
public class HealthStatus {
    
    /** Overall health status */
    private final boolean healthy;
    
    /** Memory subsystem health */
    private final boolean memoryHealthy;
    
    /** Execution subsystem health */
    private final boolean executionHealthy;
    
    /** Device subsystem health */
    private final boolean deviceHealthy;
    
    /** Number of detected issues */
    private final int issueCount;
    
    /** Summary message */
    private final String summary;
    
    /**
     * Create a healthy status.
     */
    public static HealthStatus healthy() {
        return HealthStatus.builder()
            .healthy(true)
            .memoryHealthy(true)
            .executionHealthy(true)
            .deviceHealthy(true)
            .issueCount(0)
            .summary("All subsystems healthy")
            .build();
    }
    
    /**
     * Create an unhealthy status with a message.
     */
    public static HealthStatus unhealthy(String reason) {
        return HealthStatus.builder()
            .healthy(false)
            .memoryHealthy(true)
            .executionHealthy(true)
            .deviceHealthy(true)
            .issueCount(1)
            .summary(reason)
            .build();
    }
}
