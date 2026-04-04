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

package org.nd4j.linalg.framework.stats;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Profiling statistics for op execution.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ProfilingStats {
    
    /**
     * Total operations executed
     */
    private long totalOps;
    
    /**
     * Number of unique operation types
     */
    private int uniqueOps;
    
    /**
     * Total execution time (nanoseconds)
     */
    private long totalTimeNanos;
    
    /**
     * Percentage of ops using helper libraries (cuDNN, oneDNN, etc.)
     */
    private double helperUsagePercent;
    
    /**
     * Get total execution time in milliseconds
     */
    public double getTotalTimeMillis() {
        return totalTimeNanos / 1_000_000.0;
    }
    
    /**
     * Get total execution time in seconds
     */
    public double getTotalTimeSeconds() {
        return totalTimeNanos / 1_000_000_000.0;
    }
    
    /**
     * Get average execution time per op (microseconds)
     */
    public double getAvgTimeMicros() {
        if (totalOps <= 0) return 0.0;
        return (totalTimeNanos / 1000.0) / totalOps;
    }
}
