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
 * Execution statistics.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExecutionStats {
    
    /**
     * Number of operations executed
     */
    private long operationsExecuted;
    
    /**
     * Total execution time in nanoseconds
     */
    private long totalExecutionTimeNanos;
    
    /**
     * Executioner type
     */
    private String executionerType;
    
    /**
     * Get total execution time in milliseconds
     */
    public double getTotalExecutionTimeMillis() {
        return totalExecutionTimeNanos / 1_000_000.0;
    }
    
    /**
     * Get average execution time per op (microseconds)
     */
    public double getAvgExecutionTimeMicros() {
        if (operationsExecuted <= 0) return 0.0;
        return (totalExecutionTimeNanos / 1000.0) / operationsExecuted;
    }
}
