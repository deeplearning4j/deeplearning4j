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

package org.nd4j.linalg.framework.leak;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Information about a potential memory leak.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PotentialLeak {
    
    /**
     * Unique ID of the array
     */
    private long arrayId;
    
    /**
     * Size in bytes
     */
    private long sizeBytes;
    
    /**
     * Data type
     */
    private String dataType;
    
    /**
     * Shape (comma-separated)
     */
    private String shape;
    
    /**
     * Age in milliseconds (time since creation)
     */
    private long ageMs;
    
    /**
     * Allocation site stack trace (if available)
     */
    private StackTraceElement[] stackTrace;
    
    /**
     * Reason this is flagged as potential leak
     */
    public enum LeakReason {
        OLD_AGE,              // Object older than threshold
        LARGE_SIZE,           // Unusually large allocation
        GROWING_TREND,        // Part of growing memory trend
        NO_SIMILAR_FREED,     // No similar objects freed recently
        CONSTANT_REFERENCE,   // Reference held constant unusually long
        LEAKED_REPLICA        // Replica array not properly cleaned up
    }
    
    /**
     * Reason for flagging
     */
    private LeakReason leakReason;

    /**
     * Confidence score (0.0 - 1.0)
     */
    private double confidence;

    /**
     * Get stack trace as formatted string
     */
    public String getStackTraceAsString() {
        if (stackTrace == null || stackTrace.length == 0) {
            return "Stack trace not available";
        }
        
        StringBuilder sb = new StringBuilder();
        for (StackTraceElement element : stackTrace) {
            sb.append("\tat ").append(element).append("\n");
        }
        return sb.toString();
    }
    
    /**
     * Get size in human-readable format
     */
    public String getSizeHumanReadable() {
        return formatBytes(sizeBytes);
    }
    
    /**
     * Get age in human-readable format
     */
    public String getAgeHumanReadable() {
        long seconds = ageMs / 1000;
        if (seconds < 60) {
            return seconds + "s";
        } else if (seconds < 3600) {
            return (seconds / 60) + "m";
        } else {
            return (seconds / 3600) + "h";
        }
    }
    
    /**
     * Format bytes to human-readable string
     */
    private String formatBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        } else if (bytes < 1024 * 1024) {
            return String.format("%.1f KB", bytes / 1024.0);
        } else if (bytes < 1024 * 1024 * 1024) {
            return String.format("%.1f MB", bytes / (1024.0 * 1024.0));
        } else {
            return String.format("%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
        }
    }
}
