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

package org.nd4j.linalg.framework.exec;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Execution subsystem state snapshot.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExecutionState {
    
    /**
     * Whether DSP is enabled
     */
    private boolean dspEnabled;
    
    /**
     * DSP diagnostics categories
     */
    private String dspDiagnostics;
    
    /**
     * DSP diagnostics level
     */
    private String dspDiagnosticsLevel;
    
    /**
     * Whether op profiling is enabled
     */
    private boolean opProfilingEnabled;
    
    /**
     * Preferred backend
     */
    private String preferredBackend;
    
    /**
     * Operations executed
     */
    private long operationsExecuted;
    
    /**
     * Total execution time (nanos)
     */
    private long totalExecutionTimeNanos;
    
    /**
     * Executioner type
     */
    private String executionerType;
    
    /**
     * Get DSP status string
     */
    public String getDspStatus() {
        if (!dspEnabled) return "OFF";
        return dspDiagnostics + "/" + dspDiagnosticsLevel;
    }
    
    /**
     * Get total execution time in seconds
     */
    public double getTotalExecutionTimeSeconds() {
        return totalExecutionTimeNanos / 1_000_000_000.0;
    }
    
    /**
     * Get average execution time per op (microseconds)
     */
    public double getAvgExecutionTimeMicros() {
        if (operationsExecuted <= 0) return 0.0;
        return (totalExecutionTimeNanos / 1000.0) / operationsExecuted;
    }
    
    @Override
    public String toString() {
        return String.format(
            "DSP: %s, Profiling: %s, Backend: %s, Ops: %d (%.2f s, %.1f µs/op), Type: %s",
            getDspStatus(),
            opProfilingEnabled ? "ON" : "OFF",
            preferredBackend,
            operationsExecuted,
            getTotalExecutionTimeSeconds(),
            getAvgExecutionTimeMicros(),
            executionerType
        );
    }
}
